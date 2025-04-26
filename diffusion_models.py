import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np

# Define U-Net Model for Diffusion Process
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Define Diffusion Model
class DiffusionModel:
    def __init__(self, unet, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.unet = unet
        self.timesteps = timesteps
        self.beta = torch.linspace(beta_start, beta_end, timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
    
    def forward_diffusion(self, x0, t):
        noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
        return xt, noise

    def sample(self, shape):
        x = torch.randn(shape)
        for t in reversed(range(self.timesteps)):
            z = torch.randn_like(x) if t > 0 else 0
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            beta_t = self.beta[t]
            noise_pred = self.unet(x)
            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * noise_pred) + torch.sqrt(beta_t) * z
        return x


def load_audio_as_spectrogram(file_path):
    waveform, sr = torchaudio.load(file_path)
    spectrogram = torchaudio.transforms.MelSpectrogram()(waveform)
    spectrogram = torch.log(spectrogram + 1e-6)
    return spectrogram.unsqueeze(0)  # Add channel dimension

# Instantiate Model
unet = UNet()
diffusion = DiffusionModel(unet)

# Example Usage
audio_spectrogram = load_audio_as_spectrogram("example.wav")
diffused_spectrogram, noise = diffusion.forward_diffusion(audio_spectrogram, torch.tensor([500]))
restored_spectrogram = diffusion.sample(audio_spectrogram.shape)
