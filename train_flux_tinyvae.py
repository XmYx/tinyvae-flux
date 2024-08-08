import torch
import torch.nn as nn
from diffusers.image_processor import VaeImageProcessor
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
from diffusers import AutoencoderKL
from accelerate import Accelerator
from tqdm import tqdm

from datasets import load_dataset

def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class CustomDataset(Dataset):
    def __init__(self, images, encoded_latents, transform=None):
        self.images = images
        self.encoded_latents = encoded_latents
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        encoded_latent = self.encoded_latents[idx]

        if self.transform:
            image = self.transform(image)

        return image, encoded_latent
class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3

class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))

class Encoder(nn.Module):
    def __init__(self, channels, latent_channels):
        super().__init__()
        layers = [conv(3, channels[0]), Block(channels[0], channels[0])]
        for i in range(len(channels) - 1):
            layers += [conv(channels[i], channels[i + 1], stride=2, bias=False), Block(channels[i + 1], channels[i + 1])]
        layers += [conv(channels[-1], latent_channels, stride=1, bias=False)]
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, channels, latent_channels):
        super().__init__()
        layers = [Clamp(), conv(latent_channels, channels[0]), nn.ReLU()]
        for i in range(len(channels) - 1):
            layers += [Block(channels[i], channels[i]), nn.Upsample(scale_factor=2), conv(channels[i], channels[i + 1], bias=False)]
        layers += [Block(channels[-1], channels[-1]), nn.Upsample(scale_factor=1), conv(channels[-1], 3)]
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)

class TinyAutoEncoder(nn.Module):
    latent_magnitude = 0.3611
    latent_shift = 0.1159

    def __init__(self, encoder_path=None, decoder_path=None, latent_channels=16, channels=None):
        super().__init__()
        if channels is None:
            channels = [128, 256, 512, 1024]
        self.encoder = Encoder(channels, latent_channels)
        self.decoder = Decoder(channels[::-1], latent_channels)
        if encoder_path is not None:
            self.encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
        if decoder_path is not None:
            self.decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))

    @staticmethod
    def scale_latents(x):
        return x.div(2 * TinyAutoEncoder.latent_magnitude).add(TinyAutoEncoder.latent_shift).clamp(0, 1)

    @staticmethod
    def unscale_latents(x):
        return x.sub(TinyAutoEncoder.latent_shift).mul(2 * TinyAutoEncoder.latent_magnitude)

class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('jpg', 'jpeg', 'png'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB").resize((size, size), resample=Image.Resampling.LANCZOS)
        if self.transform:
            image = self.transform(image)
        return {"image":image}

def postprocess(tensor):
    tensor = (tensor * 0.5 + 0.5) * 255
    tensor = tensor.permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy()
    return Image.fromarray(tensor)

class CyclicalAnnealingScheduler:
    def __init__(self, n_epochs, n_cycles, ratio=0.5):
        self.n_epochs = n_epochs
        self.n_cycles = n_cycles
        self.ratio = ratio
        self.epoch_per_cycle = n_epochs // n_cycles
        self.cycle_length = int(self.epoch_per_cycle * ratio)

    def get_beta(self, epoch):
        cycle_epoch = epoch % self.epoch_per_cycle
        if cycle_epoch < self.cycle_length:
            return cycle_epoch / self.cycle_length
        else:
            return 1.0

def encode_dataset(dataset, vae, device, batch_size, size):
    processor = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=16)
    vae.eval()
    encoded_images = []
    preprocessed_images = []
    with torch.no_grad():
        for images in DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True):
            images = images['image'].to(device)
            preprocessed = processor.preprocess(images, height=size, width=size)
            preprocessed_images.append(preprocessed.cpu())
            latents = vae.encode(preprocessed.half()).latent_dist.sample().cpu()
            encoded_images.append(latents)
    return torch.cat(preprocessed_images), torch.cat(encoded_images)

def train_epoch(model, dataloader, optimizer, criterion, device, beta, scaler, accelerator):
    model.train()
    total_encoder_loss = 0
    total_decoder_loss = 0
    for images, ground_truth_latents in dataloader:
        images = images.to(device)
        ground_truth_latents = ground_truth_latents.to(device)
        # print(images.shape, ground_truth_latents.shape)
        optimizer.zero_grad()

        with accelerator.autocast():
            encoded = model.encoder(images)
            decoded = model.decoder(ground_truth_latents)
            encoder_loss = criterion(encoded, ground_truth_latents) * beta
            decoder_loss = criterion(decoded, images) * beta

            # Additional adversarial loss
            pooled_real_images = nn.functional.avg_pool2d(images, 8)
            pooled_decoded_images = nn.functional.avg_pool2d(decoded, 8)
            adversarial_loss = criterion(pooled_decoded_images, pooled_real_images)

            loss = encoder_loss + decoder_loss + adversarial_loss

        accelerator.backward(scaler.scale(loss))
        scaler.step(optimizer)
        scaler.update()
        total_encoder_loss += encoder_loss.item()
        total_decoder_loss += decoder_loss.item()

    return total_encoder_loss / len(dataloader), total_decoder_loss / len(dataloader)


def main(data_folder, output_folder, epochs=100000, batch_size=8, learning_rate=0.0003, n_cycles=10):
    accelerator = Accelerator(mixed_precision='fp16')
    device = accelerator.device
    size = 1024  # Assuming the image size is 512x512
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.ColorJitter(hue=0.1, saturation=0.1)  # Color augmentation
    ])

    use_folder = False

    if use_folder:
        dataset = ImageFolderDataset(data_folder, transform)

    else:
        dataset = load_dataset("fantasyfish/laion-art", split='train')

    # Apply the function to each example in the dataset
    dataset = dataset.shuffle(seed=42).select(range(500))
    # dataset = load_dataset("fantasyfish/laion-art", split='train')

    # Define a function to load and transform the images
    def load_and_transform_image(example):
        # Load the image from the URL
        image = example['image'].convert("RGB")

        # Apply the transformations
        image = transform(image)
        return image

    # Apply the function to each example in the dataset
    dataset = dataset.map(lambda example: {'image': load_and_transform_image(example)}, batched=False, num_proc=1)

    # Remove the other columns (optional)
    dataset.set_format(type='torch', columns=['image'])

    # Define the size variants
    size_variants = {
        'zepto': [1,2,2,4],
        'yocto': [16,8,8,16],
        'zeptometer': [2, 4, 8, 16],
        'yoctometer': [4, 8, 8, 16],
        'angstrom': [4, 8, 16, 32],
        'attometer': [8, 16, 16, 32],
        'nanometer': [8, 16, 32, 64],
        'micrometer': [16, 32, 32, 64],
        'pico': [16, 32, 64, 64],
        'nano': [16, 32, 64, 128],
        'micro': [24, 48, 96, 192],
        'tiny': [32, 64, 128, 256],
        'small': [64, 128, 256, 512],
        'normal': [128, 256, 512, 1024],
        'large': [256, 512, 1024, 2048],
    }
    var = 'tiny'
    channels = size_variants.get(var, size_variants['normal'])

    model = TinyAutoEncoder(channels=channels).to(device)

    # model.decoder.load_state_dict(torch.load('/home/mix/Playground/flux_base/output_nanometer/nanometer_decoder_flux_130.pth'))
    # model.encoder.load_state_dict(torch.load('/home/mix/Playground/flux_base/output_nanometer/nanometer_encoder_flux_130.pth'))

    vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-schnell", subfolder='vae', torch_dtype=torch.float16).to(device)
    processor = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=16)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.9), amsgrad=True)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = CyclicalAnnealingScheduler(epochs, n_cycles)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    save_per_epoch = 10
    vae = vae.to(device, dtype=torch.float16)
    # Encode the dataset
    preprocessed_images, encoded_images = encode_dataset(dataset, vae, device, 4, size)
    # Create custom dataset with preprocessed images and encoded latents
    custom_dataset = CustomDataset(preprocessed_images, encoded_images, transform=None)
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    epoch_progress = tqdm(range(epochs), desc="Training", unit="epoch")

    for epoch in epoch_progress:
        beta = scheduler.get_beta(epoch)
        train_encoder_loss, train_decoder_loss = train_epoch(model, dataloader, optimizer, criterion, device, beta, scaler, accelerator)
        epoch_progress.set_postfix(encoder_loss=train_encoder_loss, decoder_loss=train_decoder_loss, beta=beta)

        if (epoch + 1) % save_per_epoch == 0:
            torch.save(model.encoder.state_dict(), os.path.join(output_folder, f"{var}_encoder_flux_{epoch+1}.pth"))
            torch.save(model.decoder.state_dict(), os.path.join(output_folder, f"{var}_decoder_flux_{epoch+1}.pth"))
        with torch.no_grad():
            # sample_img, _ = next(iter(dataloader))
            # sample_img = sample_img.to(device)
            pil_img = Image.open('input.jpg')
            preprocessed = processor.preprocess(pil_img, width=512, height=512).to(device)
            encoded_sample = model.encoder(preprocessed)
            decoded_sample = model.decoder(encoded_sample)
            postprocessed_image = postprocess(decoded_sample[0])
            postprocessed_image.save(os.path.join(output_folder, f"decoded_image_{var}_{epoch+1}.png"))

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python script.py <data_folder> <output_folder>")
    else:
        main(sys.argv[1], sys.argv[2])
