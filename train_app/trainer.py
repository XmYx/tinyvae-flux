import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.image_processor import VaeImageProcessor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
from diffusers import AutoencoderKL
from accelerate import Accelerator
from tqdm import tqdm
from datasets import load_dataset
from transformers import Adafactor, AdamW
from bitsandbytes.optim import AdamW as AdamW8bit  # For 8-bit AdamW
from prodigyopt import Prodigy
from torchmetrics.functional.image import structural_similarity_index_measure as ssim


def get_optimizer(optimizer_name, model_parameters, learning_rate):
    if optimizer_name == "adam":
        return torch.optim.Adam(model_parameters, lr=learning_rate, betas=(0.9, 0.9), amsgrad=True)
    elif optimizer_name == "adamw":
        return AdamW(model_parameters, lr=learning_rate)
    elif optimizer_name == "adamw8bit":
        return AdamW8bit(model_parameters, lr=learning_rate)
    elif optimizer_name == "adamw_bf16":
        return AdamW(model_parameters, lr=learning_rate, betas=(0.9, 0.9), amsgrad=True, torch_dtype=torch.bfloat16)
    elif optimizer_name == "adafactor":
        return Adafactor(model_parameters, lr=learning_rate, scale_parameter=False, relative_step=False)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(model_parameters, lr=learning_rate, momentum=0.9)
    elif optimizer_name == "rmsprop":
        return torch.optim.RMSprop(model_parameters, lr=learning_rate, momentum=0.9)
    elif optimizer_name == "prodigy":
        # Assuming you have the Prodigy optimizer installed and configured
        return Prodigy(model_parameters, lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


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

    def encode(self, x):
        encoded = self.encoder(x)
        return encoded

    def decode(self, latent):
        decoded = self.decoder(latent)
        return decoded
class DynamicAutoencoder(nn.Module):
    def __init__(self, latent_scale, latent_channels, in_blocks, out_blocks):
        super(DynamicAutoencoder, self).__init__()

        self.latent_scale = latent_scale
        self.latent_channels = latent_channels

        # Encoder
        encoder_layers = []
        in_channels = 3  # Assuming input images have 3 channels (RGB)
        for i, out_channels in enumerate(in_blocks):
            if i < 3:  # Apply downsampling only in the first 3 layers
                encoder_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
            else:
                encoder_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            encoder_layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.encoder = nn.Sequential(*encoder_layers)

        # Adjust the latent_conv layer to ensure it matches the latent_channels
        self.latent_conv = nn.Conv2d(in_channels, latent_channels, kernel_size=1)

        # Decoder
        decoder_layers = []
        in_channels = latent_channels
        for i, out_channels in enumerate(out_blocks):
            # Upsample first, then apply the convolution for better control
            if i < 3:  # Upscale in the first 3 layers to match the downscaling in the encoder
                decoder_layers.append(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
                )
            else:  # If no more upscaling is needed, just apply regular convolution
                decoder_layers.append(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
                )
            decoder_layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        # Final layer to output 3 channels (RGB) and match the original image size
        decoder_layers.append(nn.Conv2d(in_channels, 3, kernel_size=3, padding=1))  # Output to 3 channels (RGB)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        latent = self.encode(x)
        decoded = self.decode(latent)
        return latent, decoded

    def encode(self, x):
        encoded = self.encoder(x)
        latent = self.latent_conv(encoded)  # Ensure correct transformation to latent channels
        return latent

    def decode(self, latent):
        decoded = self.decoder(latent)
        return decoded
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):
        # Flatten the input
        inputs_flattened = inputs.view(-1, self.embedding_dim)

        # Calculate distances to embeddings
        distances = (
            torch.sum(inputs_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight ** 2, dim=1)
            - 2 * torch.matmul(inputs_flattened, self.embeddings.weight.t())
        )

        # Find the nearest embeddings
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embeddings.weight).view_as(inputs)

        # Compute the loss for embedding updates
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()

        # Preserve quantization for backpropagation
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity

class DynamicVQVAE(nn.Module):
    def __init__(self, latent_scale, latent_channels, in_blocks, out_blocks, num_embeddings, commitment_cost=0.25):
        super(DynamicVQVAE, self).__init__()

        self.latent_scale = latent_scale
        self.latent_channels = latent_channels
        self.commitment_cost = commitment_cost

        # Encoder
        encoder_layers = []
        in_channels = 3  # Assuming input images have 3 channels (RGB)
        for i, out_channels in enumerate(in_blocks):
            if i < 3:  # Apply downsampling only in the first 3 layers
                encoder_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
            else:
                encoder_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            encoder_layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent representation
        self.latent_conv = nn.Conv2d(in_channels, latent_channels, kernel_size=1)
        self.quantizer = VectorQuantizer(num_embeddings, latent_channels, commitment_cost)

        # Decoder
        decoder_layers = []
        in_channels = latent_channels
        for i, out_channels in enumerate(out_blocks):
            # Upsample first, then apply the convolution for better control
            if i < 3:  # Upscale in the first 3 layers to match the downscaling in the encoder
                decoder_layers.append(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
                )
            else:  # If no more upscaling is needed, just apply regular convolution
                decoder_layers.append(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
                )
            decoder_layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        # Final layer to output 3 channels (RGB) and match the original image size
        decoder_layers.append(nn.Conv2d(in_channels, 3, kernel_size=3, padding=1))  # Output to 3 channels (RGB)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encode(x)
        quantized, vq_loss, perplexity = self.quantizer(encoded)
        decoded = self.decode(quantized)
        return decoded, vq_loss, perplexity

    def encode(self, x):
        encoded = self.encoder(x)
        latent = self.latent_conv(encoded)  # Ensure correct transformation to latent channels
        return latent

    def decode(self, quantized):
        decoded = self.decoder(quantized)
        return decoded
class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, size, transform=None, use_crops=False):
        self.folder_path = folder_path
        self.transform = transform
        self.size = size
        self.use_crops = use_crops
        self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('jpg', 'jpeg', 'png'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.use_crops:
            # Randomly crop the image to (size, size)
            width, height = image.size
            left = random.randint(0, max(0, width - self.size))
            top = random.randint(0, max(0, height - self.size))
            image = image.crop((left, top, left + self.size, top + self.size))
        else:
            image = image.resize((self.size, self.size), resample=Image.Resampling.LANCZOS)

        if self.transform:
            image = self.transform(image)
        return {"image": image}

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
    total_consistency_loss = 0  # New consistency loss tracker

    for images, ground_truth_latents in dataloader:
        images = images.to(device)
        ground_truth_latents = ground_truth_latents.to(device)
        optimizer.zero_grad()

        with accelerator.autocast():
            encoded = model.encode(images)
            decoded = model.decode(encoded)
            decoded_ground_truth = model.decode(ground_truth_latents)

            encoder_loss = criterion(encoded, ground_truth_latents) * beta
            decoder_loss = criterion(decoded, images) * beta
            consistency_loss = criterion(decoded_ground_truth, images) * beta

            # Additional adversarial loss
            pooled_real_images = nn.functional.avg_pool2d(images, 8)
            pooled_decoded_images = nn.functional.avg_pool2d(decoded, 8)
            adversarial_loss = criterion(pooled_decoded_images, pooled_real_images) * 0.1  # Adjust as needed

            # Total loss
            loss = encoder_loss + decoder_loss + consistency_loss + adversarial_loss

        # Backward and optimize
        accelerator.backward(scaler.scale(loss))
        scaler.step(optimizer)
        scaler.update()

        # Gradient clipping
        clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Track the losses
        total_encoder_loss += encoder_loss.item()
        total_decoder_loss += decoder_loss.item()
        total_consistency_loss += consistency_loss.item()

    # Return average losses per epoch
    avg_encoder_loss = total_encoder_loss / len(dataloader)
    avg_decoder_loss = total_decoder_loss / len(dataloader)
    avg_consistency_loss = total_consistency_loss / len(dataloader)

    return avg_encoder_loss, avg_decoder_loss, avg_consistency_loss

def train_epoch_vq(model, dataloader, optimizer, criterion, device, beta, scaler, accelerator):
    model.train()
    total_reconstruction_loss = 0
    total_vq_loss = 0
    total_perplexity = 0
    total_ground_truth_loss = 0  # Track the ground truth latents decoding loss

    for images, ground_truth_latents in dataloader:
        images = images.to(device)
        ground_truth_latents = ground_truth_latents.to(device)
        optimizer.zero_grad()

        with accelerator.autocast():
            # Forward pass: Get the decoded output and VQ loss from the model
            decoded_images, vq_loss, perplexity = model(images)

            # Decode the ground truth latents (assumes model can decode latents directly)
            decoded_ground_truth = model.decode(ground_truth_latents)

            # Compute reconstruction loss for images and decoded ground truth latents
            reconstruction_loss = criterion(decoded_images, images) * beta
            ground_truth_loss = criterion(decoded_ground_truth, images) * beta

            # Total loss is a combination of all losses
            total_loss = reconstruction_loss + vq_loss + ground_truth_loss

        # Backward pass and optimization
        accelerator.backward(scaler.scale(total_loss))
        scaler.step(optimizer)
        scaler.update()

        # Track the losses
        total_reconstruction_loss += reconstruction_loss.item()
        total_vq_loss += vq_loss.item()
        total_perplexity += perplexity.item()
        total_ground_truth_loss += ground_truth_loss.item()

    # Return average losses per epoch
    avg_encoder_loss = total_reconstruction_loss / len(dataloader)
    avg_decoder_loss = total_vq_loss / len(dataloader)
    avg_consistency_loss = total_perplexity / len(dataloader)
    avg_ground_truth_loss = total_ground_truth_loss / len(dataloader)

    return avg_encoder_loss, avg_decoder_loss, avg_consistency_loss, avg_ground_truth_loss


def save_model(model, output_folder, var, epoch):
    model_save_path = os.path.join(output_folder, f"{var}_vae_flux_{epoch+1}.pth")
    torch.save(model.state_dict(), model_save_path)


# def decode_and_save_image(model, processor, output_folder, epoch, device):
#     with torch.no_grad():
#         pil_img = Image.open('input/input.jpg')  # Replace with the actual path
#         preprocessed = processor.preprocess(pil_img, width=512, height=512).to(device)
#         encoded_sample = model.encode(preprocessed)
#
#
#
#         decoded_sample = model.decode(encoded_sample)
#         postprocessed_image = postprocess(decoded_sample[0])
#         image_path = os.path.join(output_folder, f"decoded_image_epoch_{epoch+1}.png")
#         postprocessed_image.save(image_path)
#         return image_path
def decode_and_save_image(model, processor, output_folder, epoch, device, dataset):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Sample a random image from the dataset
        random_index = random.randint(0, len(dataset) - 1)
        random_sample = dataset[random_index]
        # print(random_sample)
        # pil_img = random_sample['image']
        #
        # # Preprocess the image and encode/decode
        # preprocessed = processor.preprocess(pil_img.unsqueeze(0).to(device), width=512, height=512)
        # encoded_sample = model.encode(preprocessed)

        encoded_sample = random_sample[1].to(device).float()  # Assuming dataset returns (image, encoded_latent)

        decoded_sample = model.decode(encoded_sample)
        # Post-process and save the decoded image
        postprocessed_image = postprocess(decoded_sample)


        image_path = os.path.join(output_folder, f"decoded_image_epoch_{epoch+1}.png")
        postprocessed_image.save(image_path)
        return image_path
def train_model(data_folder, output_folder, var, size, epochs=100000, batch_size=8, learning_rate=0.0003, n_cycles=10, use_folder=False, optimizer_name="adam"):
    accelerator = Accelerator(mixed_precision='fp16')
    device = accelerator.device
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.ColorJitter(hue=0.1, saturation=0.2)  # Color augmentation
    ])

    if use_folder:
        dataset = ImageFolderDataset(data_folder, size, transform, use_crops=True)
    else:
        dataset = load_dataset("fantasyfish/laion-art", split='train')

        def load_and_transform_image(example):
            image = example['image'].convert("RGB")
            if True:
                # Randomly crop the image to (size, size)
                width, height = image.size
                left = random.randint(0, max(0, width - size))
                top = random.randint(0, max(0, height - size))
                image = image.crop((left, top, left + size, top + size))
            else:
                image = image.resize((size, size), resample=Image.Resampling.LANCZOS)

            image = transform(image)
            return image

        dataset = dataset.shuffle(seed=34).select(range(100))
        dataset = dataset.map(lambda example: {'image': load_and_transform_image(example)}, batched=False, num_proc=16)
        dataset.set_format(type='torch', columns=['image'])

    size_variants = {
        'millizepto': [32, 64, 128],
        'decazepto': [48, 96, 192],
        'zepto': [4,8,8,16],
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
        'mega': [32, 64, 128, 256, 512, 1024],
    }
    channels = size_variants.get(var, size_variants['normal'])

    #
    latent_scale = 8
    latent_channels = 16
    out_blocks = copy.deepcopy(channels)
    out_blocks.reverse()
    use_dyna_vae = True
    if use_dyna_vae:
        model = DynamicVQVAE(latent_scale, latent_channels, channels, out_blocks, num_embeddings=512).to(device)
        model.load_state_dict(torch.load('/home/mix/Playground/flux_base/.experiments/tests/millizepto/millizepto_vae_flux_70.pth'))
    else:
        model = TinyAutoEncoder(channels=channels).to(device)

    vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-schnell", subfolder='vae', torch_dtype=torch.float16).to(device)
    processor = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=16)
    criterion = nn.MSELoss()
    optimizer = get_optimizer(optimizer_name, model.parameters(), learning_rate)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = CyclicalAnnealingScheduler(epochs, n_cycles)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    preprocessed_images, encoded_images = encode_dataset(dataset, vae, device, 4, size)
    custom_dataset = CustomDataset(preprocessed_images, encoded_images, transform=None)
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    epoch_progress = tqdm(range(epochs), desc="Training", unit="epoch")
    vae.to('cpu')
    torch.cuda.empty_cache()
    for epoch in epoch_progress:
        beta = scheduler.get_beta(epoch)
        train_encoder_loss, train_decoder_loss, consistency_loss, avg_ground_truth_loss = train_epoch_vq(model, dataloader, optimizer, criterion, device, beta, scaler, accelerator)
        epoch_progress.set_postfix(encoder_loss=train_encoder_loss, decoder_loss=train_decoder_loss, beta=beta, cons=consistency_loss, avg_ground_truth_loss=avg_ground_truth_loss)

        if (epoch + 1) % 10 == 0:
            save_model(model, output_folder, var, epoch)
            image_path = decode_and_save_image(model, processor, output_folder, epoch, device,
                                               custom_dataset)  # Pass the dataset here
            yield (epoch, image_path)
    model.to('cpu')
    del model, vae, dataset, preprocessed_images, encoded_images
    torch.cuda.empty_cache()

