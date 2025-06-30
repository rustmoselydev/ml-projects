# ---------------------------
# A Latent Diffusion Model for generating 32x32 pixel art characters from prompts
# ---------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel
import math
import numpy as np
from params import guidance_scale, guidance_dropout

# ---------------------------
# Pretrained CLIP Embedder
# ---------------------------
class CLIPTextEmbedder(nn.Module):
    def __init__(self, device="mps"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    def forward(self, text):
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}
        return self.model(**tokens).last_hidden_state.mean(dim=1)

# ---------------------------
# Simple Autoencoder (VAE-style)
# ---------------------------
class SimpleAutoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, latent_dim, 3, 1, 1),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

# ---------------------------
# Sinusoidal Positional Embedding
# ---------------------------
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

# ---------------------------
# Residual Block
# ---------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_proj = nn.Linear(time_embed_dim, out_channels)

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_embed):
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(t_embed).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)

# ---------------------------
# Attention Block
# ---------------------------
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)
        h, _ = self.attn(h, h, h)
        h = self.proj(h)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        return x + h

# ---------------------------
# Downsample/Upsample
# ---------------------------
class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)

# ---------------------------
# UNet
# ---------------------------
class UNet(nn.Module):
    def __init__(self, latent_dim=32, embed_dim=512, base_channels=256, time_embed_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        self.text_proj = nn.Linear(embed_dim, time_embed_dim)
        self.init_conv = nn.Conv2d(latent_dim, base_channels, 3, padding=1)
        self.time_cond_proj = nn.Linear(time_embed_dim * 2, time_embed_dim)

        # Downsampling
        self.down1 = ResidualBlock(base_channels, base_channels, time_embed_dim)
        self.attn1 = AttentionBlock(base_channels)

        self.down2 = ResidualBlock(base_channels, base_channels * 2, time_embed_dim)
        self.downsample1 = Downsample(base_channels * 2)

        self.down3 = ResidualBlock(base_channels * 2, base_channels * 4, time_embed_dim)
        self.downsample2 = Downsample(base_channels * 4)

        self.mid1 = ResidualBlock(base_channels * 4, base_channels * 4, time_embed_dim)
        self.mid_attn = AttentionBlock(base_channels * 4)
        self.mid2 = ResidualBlock(base_channels * 4, base_channels * 4, time_embed_dim)
        self.mid3 = ResidualBlock(base_channels * 4, base_channels * 4, time_embed_dim)

        # Upsampling
        self.upsample2 = Upsample(base_channels * 4)
        self.up3 = ResidualBlock(base_channels * 4 + base_channels * 4, base_channels * 2, time_embed_dim)

        self.upsample1 = Upsample(base_channels * 2)
        self.up2 = ResidualBlock(base_channels * 2 + base_channels, base_channels, time_embed_dim)

        self.attn2 = AttentionBlock(base_channels)
        self.up1 = ResidualBlock(base_channels, base_channels, time_embed_dim)

        # Final
        self.final_norm = nn.GroupNorm(8, base_channels)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(base_channels, latent_dim, 3, padding=1)

    def forward(self, x, t, cond):
        t_embed = self.time_mlp(t.float())
        cond_embed = self.text_proj(cond)
        time_cond = self.time_cond_proj(torch.cat([t_embed, cond_embed], dim=1))

        # Encoding path
        x = self.init_conv(x)                     # [B, base, 4, 4]
        d1 = self.down1(x, time_cond)
        d1 = self.attn1(d1)

        d2 = self.down2(d1, time_cond)            # [B, base*2, 4, 4]
        d3 = self.downsample1(d2)                 # [B, base*2, 2, 2]

        d3 = self.down3(d3, time_cond)            # [B, base*4, 2, 2]
        d4 = self.downsample2(d3)                 # [B, base*4, 1, 1]

        m = self.mid1(d4, time_cond)
        m = self.mid_attn(m)
        m = self.mid2(m, time_cond)
        m = self.mid3(m, time_cond)

        # Decoding path
        u3 = self.upsample2(m)                    # [B, base*4, 2, 2]
        u3 = self.up3(torch.cat([u3, d3], dim=1), time_cond)

        u2 = self.upsample1(u3)                   # [B, base*2, 4, 4]
        d1_resized = F.interpolate(d1, size=u2.shape[-2:], mode='nearest')
        u2 = self.up2(torch.cat([u2, d1_resized], dim=1), time_cond)

        u1 = self.attn2(u2)
        u1 = self.up1(u1, time_cond)

        out = self.final_conv(self.final_act(self.final_norm(u1)))
        return out

# ---------------------------
# Latent Diffusion Model
# ---------------------------
class LatentDiffusionModel(nn.Module):
    def __init__(self, autoencoder, unet, text_embedder, noise_scheduler, predict_velocity=True):
        super().__init__()
        self.autoencoder = autoencoder
        self.unet = unet
        self.text_embedder = text_embedder
        self.noise_scheduler = noise_scheduler
        self.predict_velocity = predict_velocity

    def forward(self, img, text, t, return_reconstruction=False):
        with torch.no_grad():
            latents = self.autoencoder.encode(img)

        noise = torch.randn_like(latents)
        noised_latents = self.noise_scheduler.add_noise(latents, noise, t)

        # Duplicate latents and timesteps for cond/uncond
        latent_input = torch.cat([noised_latents, noised_latents], dim=0)
        t_input = torch.cat([t, t], dim=0)

        # Cond/uncond embeddings
        cond = self.text_embedder(text)

        # Guidance dropout to force the model to learn from the prompt
        if self.training and torch.rand(1).item() < guidance_dropout:
            cond = torch.zeros_like(cond)
        uncond = torch.zeros_like(cond)
        cond_embeds = torch.cat([uncond, cond], dim=0)

        pred_noise = self.unet(latent_input, t_input, cond_embeds)
        pred_uncond, pred_cond = pred_noise.chunk(2, dim=0)

        pred_guided = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

        loss = F.mse_loss(pred_guided, noise)

        if return_reconstruction:
            with torch.no_grad():
                recon = self.autoencoder.decode(latents)
            return loss, recon

        return loss

# ---------------------------
# Noise Scheduler (Beta Schedule)
# ---------------------------
class NoiseScheduler:
    def __init__(self, timesteps=100):
        self.timesteps = timesteps
        steps = np.arange(timesteps + 1)
        alpha_bar = np.cos(((steps / timesteps) + 0.008) / 1.008 * np.pi / 2) ** 2
        self.alpha_cumprod = torch.tensor(alpha_bar[:-1] / alpha_bar[0], dtype=torch.float32)

    def add_noise(self, x, noise, t):
        alpha_cumprod = self.alpha_cumprod.to(t.device)
        sqrt_alpha = alpha_cumprod[t] ** 0.5
        sqrt_one_minus_alpha = (1 - alpha_cumprod[t]) ** 0.5
        return sqrt_alpha.view(-1, 1, 1, 1) * x + sqrt_one_minus_alpha.view(-1, 1, 1, 1) * noise

    def get_velocity(self, x0, noise, t):
        alpha_cumprod = self.alpha_cumprod.to(t.device)
        sqrt_alpha = alpha_cumprod[t] ** 0.5
        sqrt_one_minus_alpha = (1 - alpha_cumprod[t]) ** 0.5
        return sqrt_alpha.view(-1, 1, 1, 1) * noise - sqrt_one_minus_alpha.view(-1, 1, 1, 1) * x0

    def velocity_to_x0(self, x_t, v, t):
        alpha_cumprod = self.alpha_cumprod.to(t.device)
        sqrt_alpha = alpha_cumprod[t] ** 0.5
        sqrt_one_minus_alpha = (1 - alpha_cumprod[t]) ** 0.5
        return (x_t - sqrt_one_minus_alpha.view(-1, 1, 1, 1) * v) / sqrt_alpha.view(-1, 1, 1, 1)