import torch
import torch.nn as nn
from diffusers.image_processor import VaeImageProcessor
from torch.cuda import amp
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
from diffusers import AutoencoderKL
from accelerate import Accelerator

size = 1024

def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)
def compute_kl_divergence(mean, logvar):
    kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return kl_div

def new_kl_divergence(posterior_means, posterior_logvars, batch_size):
    global_mean = torch.mean(posterior_means, dim=0)
    global_var = torch.mean(posterior_logvars.exp(), dim=0)
    kl_div = 0.5 * torch.sum(global_var + global_mean.pow(2) - 1 - torch.log(global_var))
    return kl_div / batch_size
def regularization_term(logvars):
    reg_loss = torch.sum(logvars.exp() - 1 - logvars)
    return reg_loss
class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, stride=1, padding=1)
        )

    def forward(self, img):
        return self.model(img)

def compute_loss(vae, discriminator, real_images, recon_images, latent_mean, latent_logvar, beta):
    # Reconstruction loss
    recon_loss = nn.MSELoss()(recon_images, real_images)

    # KL divergence
    kl_loss = new_kl_divergence(latent_mean, latent_logvar, real_images.size(0))

    # Regularization term
    reg_loss = regularization_term(latent_logvar)

    # Discriminator loss
    real_validity = discriminator(real_images)
    fake_validity = discriminator(recon_images.detach())
    d_loss_real = nn.BCEWithLogitsLoss()(real_validity, torch.ones_like(real_validity))
    d_loss_fake = nn.BCEWithLogitsLoss()(fake_validity, torch.zeros_like(fake_validity))
    d_loss = (d_loss_real + d_loss_fake) / 2

    # Generator loss (VAE)
    g_loss = nn.BCEWithLogitsLoss()(discriminator(recon_images), torch.ones_like(fake_validity))

    elbo_loss = recon_loss + beta * (kl_loss + reg_loss) + g_loss
    return elbo_loss, d_loss

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

def Encoder(latent_channels=16, size_variant='normal'):
    if size_variant == 'pico':
        channels = [4, 8, 16, 32]
    elif size_variant == 'nano':
        channels = [16, 32, 64, 128]
    elif size_variant == 'micro':
        channels = [24, 48, 96, 192]
    elif size_variant == 'tiny':
        channels = [32, 64, 128, 256]
    elif size_variant == 'small':
        channels = [64, 128, 256, 512]
    else:  # 'normal'
        channels = [128, 256, 512, 1024]
    return nn.Sequential(
        conv(3, channels[0]), Block(channels[0], channels[0]),
        conv(channels[0], channels[1], stride=2, bias=False), Block(channels[1], channels[1]),
        conv(channels[1], channels[2], stride=2, bias=False), Block(channels[2], channels[2]),
        conv(channels[2], channels[3], stride=2, bias=False), Block(channels[3], channels[3]),
        conv(channels[3], latent_channels, stride=1, bias=False)
    )

def Decoder(latent_channels=16, size_variant='normal'):
    if size_variant == 'pico':
        channels = [32, 16, 8, 4]

    elif size_variant == 'nano':
        channels = [128, 64, 32, 16]
    elif size_variant == 'micro':
        channels = [192, 96, 48, 24]

    elif size_variant == 'tiny':
        channels = [256, 128, 64, 32]
    elif size_variant == 'small':
        channels = [512, 256, 128, 64]
    else:  # 'normal'
        channels = [1024, 512, 256, 128]
    return nn.Sequential(
        Clamp(), conv(latent_channels, channels[0]), nn.ReLU(),
        Block(channels[0], channels[0]), nn.Upsample(scale_factor=2), conv(channels[0], channels[1], bias=False),
        Block(channels[1], channels[1]), nn.Upsample(scale_factor=2), conv(channels[1], channels[2], bias=False),
        Block(channels[2], channels[2]), nn.Upsample(scale_factor=2), conv(channels[2], channels[3], bias=False),
        Block(channels[3], channels[3]), nn.Upsample(scale_factor=1), conv(channels[3], 3),
    )

class TinyAutoEncoder(nn.Module):
    latent_magnitude = 0.3611
    latent_shift = 0.1159

    def __init__(self, encoder_path=None, decoder_path=None, latent_channels=16, size_variant='normal'):
        super().__init__()
        self.encoder = Encoder(latent_channels, size_variant)
        self.decoder = Decoder(latent_channels, size_variant)
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
        return image

def load_matching_layers(tiny_model, vae):
    vae_state_dict = vae.state_dict()
    tiny_encoder_dict = tiny_model.encoder.state_dict()
    tiny_decoder_dict = tiny_model.decoder.state_dict()
    for name, param in tiny_encoder_dict.items():
        for vae_name, vae_param in vae_state_dict.items():
            if param.shape == vae_param.shape:
                print(f"Loading encoder layer: {name} from VAE layer: {vae_name}")
                tiny_encoder_dict[name].copy_(vae_param)
                break
    for name, param in tiny_decoder_dict.items():
        for vae_name, vae_param in vae_state_dict.items():
            if param.shape == vae_param.shape:
                print(f"Loading decoder layer: {name} from VAE layer: {vae_name}")
                tiny_decoder_dict[name].copy_(vae_param)
                break
    tiny_model.encoder.load_state_dict(tiny_encoder_dict)
    tiny_model.decoder.load_state_dict(tiny_decoder_dict)

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

def encode_dataset(dataset, model, device):
    processor = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=16)
    model.eval()
    encoded_images = []
    with torch.no_grad():
        for images in DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True):
            images = images.to(device)
            preprocessed_images = processor.preprocess(images, height=size, width=size)
            encoded_images.append(preprocessed_images.cpu())
    return torch.cat(encoded_images, dim=0)


def train_epoch(model, discriminator, dataloader, encoded_images, batch_size, optimizer, optimizer_d, criterion, device,
                vae, processor, beta, scaler, accelerator):
    model.train()
    discriminator.train()
    total_encoder_loss = 0
    total_decoder_loss = 0
    for i, images in enumerate(dataloader):
        preprocessed_images = encoded_images[i * batch_size:(i + 1) * batch_size].to(device)
        with torch.inference_mode():
            ground_truth_latents = []
            for batch in DataLoader(preprocessed_images, batch_size=1):
                ground_truth_latents.append(vae.encode(batch.half().to(device)).latent_dist.sample().cpu().detach())
            ground_truth_latents = torch.cat(ground_truth_latents).to(device)

        optimizer.zero_grad()
        optimizer_d.zero_grad()

        with accelerator.autocast():
            encoded = model.encoder(preprocessed_images)
            decoded = model.decoder(encoded)

            elbo_loss, d_loss = compute_loss(model, discriminator, preprocessed_images, decoded, encoded,
                                             ground_truth_latents, beta)

        accelerator.backward(scaler.scale(elbo_loss))
        scaler.step(optimizer)
        scaler.update()

        accelerator.backward(scaler.scale(d_loss))
        scaler.step(optimizer_d)
        scaler.update()

        total_encoder_loss += elbo_loss.item()
        total_decoder_loss += d_loss.item()

    return total_encoder_loss / len(dataloader), total_decoder_loss / len(dataloader)


def main(data_folder, output_folder, epochs=100000, batch_size=8, learning_rate=0.0005, n_cycles=10):
    accelerator = Accelerator(mixed_precision='fp16')
    device = accelerator.device
    size = 512
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    dataset = ImageFolderDataset(data_folder, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    model = TinyAutoEncoder(size_variant='tiny').to(device)
    discriminator = PatchDiscriminator().to(device)

    # model.decoder.load_state_dict(torch.load('/home/mix/Playground/flux_base/output_100k/tiny_decoder_epoch_360.pth'))
    # model.encoder.load_state_dict(torch.load('/home/mix/Playground/flux_base/output_100k/tiny_encoder_epoch_360.pth'))

    vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-schnell", subfolder='vae',
                                        torch_dtype=torch.float16).to(device)
    processor = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=16)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = CyclicalAnnealingScheduler(epochs, n_cycles)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    save_per_epoch = 10

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    vae = vae.to(device, dtype=torch.float16)

    encoded_images = encode_dataset(dataset, vae, device)

    for epoch in range(epochs):
        beta = scheduler.get_beta(epoch)
        train_encoder_loss, train_decoder_loss = train_epoch(model, discriminator, dataloader, encoded_images,
                                                             batch_size, optimizer, optimizer_d, criterion, device, vae,
                                                             processor, beta, scaler, accelerator)
        print(
            f"Epoch {epoch + 1}/{epochs}, Train Encoder Loss: {train_encoder_loss}, Train Decoder Loss: {train_decoder_loss}, Beta: {beta}")
        if (epoch + 1) % save_per_epoch == 0:
            torch.save(model.encoder.state_dict(), os.path.join(output_folder, f"tiny_encoder_epoch_{epoch + 1}.pth"))
            torch.save(model.decoder.state_dict(), os.path.join(output_folder, f"tiny_decoder_epoch_{epoch + 1}.pth"))
        with torch.no_grad():
            sample_img = next(iter(dataloader)).to(device)
            preprocessed = processor.preprocess(sample_img, width=size, height=size)
            encoded_sample = model.encoder(preprocessed)
            decoded_sample = model.decoder(encoded_sample)
            postprocessed_image = postprocess(decoded_sample[0])
            postprocessed_image.save(os.path.join(output_folder, f"decoded_image_epoch_{epoch + 1}.png"))


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python script.py <data_folder> <output_folder>")
    else:
        main(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python script.py <data_folder> <output_folder>")
    else:
        main(sys.argv[1], sys.argv[2])
