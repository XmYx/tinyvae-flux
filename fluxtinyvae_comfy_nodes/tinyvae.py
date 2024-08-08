import torch
import torch.nn as nn

size = 512

def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

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
            channels = [32, 64, 128, 256]
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