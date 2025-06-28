# This is an older UNet architecture that ultimately wasn't up to the job at hand.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel


# ---------------------------
# Pretrained CLIP Embedder
# ---------------------------
class CLIPTextEmbedder(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    def forward(self, text):
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}
        return self.model(**tokens).last_hidden_state.mean(dim=1)  # (batch, embedding_dim)

# ---------------------------
# Simple Autoencoder (VAE-style)
# ---------------------------
class SimpleAutoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()

        # --- Encoder ---
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),   # 32 → 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, 2, 1), # 16 → 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, 2, 1),# 8 → 4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, latent_dim, 3, 1, 1),  # 4x4 feature map with latent_dim channels
        )

        # --- Decoder ---
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 3, 1, 1),  # 4x4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 4 → 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 8 → 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),     # 16 → 32
            nn.Tanh()
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

# ---------------------------
# UNet for Latent Denoising
# ---------------------------
class UNet(nn.Module):
    def __init__(self, latent_dim=8, embed_dim=512, time_embed_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.conv1 = nn.Conv2d(latent_dim + embed_dim + time_embed_dim, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, latent_dim, 3, padding=1)  # Final layer outputs noise prediction

    def forward(self, x, t, cond):
        if t.dim() == 1:
            t = t.float().unsqueeze(-1)  # [B, 1]
        t_embed = self.time_embed(t).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3])
        cond = cond.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3])

        x = torch.cat([x, cond, t_embed], dim=1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return self.conv5(x)



# ---------------------------
# Latent Diffusion Model
# ---------------------------
class LatentDiffusionModel(nn.Module):
    def __init__(self, autoencoder, unet, text_embedder, noise_scheduler):
        super().__init__()
        self.autoencoder = autoencoder
        self.unet = unet
        self.text_embedder = text_embedder
        self.noise_scheduler = noise_scheduler

    def forward(self, img, text, t, return_reconstruction=False):
        with torch.no_grad():
            latents = self.autoencoder.encode(img)

        noise = torch.randn_like(latents)
        noised_latents = self.noise_scheduler.add_noise(latents, noise, t)

        cond = self.text_embedder(text)
        pred_noise = self.unet(noised_latents, t, cond)

        loss = F.mse_loss(pred_noise, noise)

        if return_reconstruction:
            with torch.no_grad():
                recon = self.autoencoder.decode(latents)
            return loss, recon

        return loss


# ---------------------------
# Noise Scheduler (Beta Schedule)
# ---------------------------
class NoiseScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x, noise, t):
        alpha_cumprod = self.alpha_cumprod.to(t.device)
        sqrt_alpha = alpha_cumprod[t]**0.5
        sqrt_one_minus_alpha = (1 - alpha_cumprod[t])**0.5
        return sqrt_alpha.view(-1, 1, 1, 1) * x + sqrt_one_minus_alpha.view(-1, 1, 1, 1) * noise

