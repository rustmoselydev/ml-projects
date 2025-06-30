import torch
from PIL import Image
from torchvision import transforms
from model import LatentDiffusionModel, SimpleAutoencoder, UNet, CLIPTextEmbedder, NoiseScheduler
from params import guidance_scale, eta, num_inference_steps, latent_dim, embed_dim, image_size, device

# --- Configuration ---
checkpoint_path = "./ldm_checkpoint_v.pt"
autoencoder_ckpt_path = "./autoencoder_checkpoint.pt"

# --- Load Model Components ---
autoencoder = SimpleAutoencoder(latent_dim=latent_dim).to(device)
autoencoder_ckpt = torch.load(autoencoder_ckpt_path, map_location=device)
autoencoder.load_state_dict(autoencoder_ckpt['model_state_dict'])
autoencoder.eval()
for param in autoencoder.parameters():
    param.requires_grad = False
unet = UNet(latent_dim=latent_dim, embed_dim=embed_dim).to(device)
text_embedder = CLIPTextEmbedder(device=device)
noise_scheduler = NoiseScheduler(timesteps=num_inference_steps)

model = LatentDiffusionModel(autoencoder, unet, text_embedder, noise_scheduler).to(device)
model.eval()

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

def generate_image(prompt, save_path=None):
    with torch.no_grad():
        cond = text_embedder([prompt])  # (1, embed_dim)
        z = torch.randn(1, latent_dim, image_size // 8, image_size // 8).to(device)

        for t in reversed(range(num_inference_steps)):
            t_tensor = torch.tensor([t], device=device, dtype=torch.long)

            alpha_cumprod = noise_scheduler.alpha_cumprod.to(z.device)
            alpha_t = alpha_cumprod[t]
            alpha_prev = alpha_cumprod[t - 1 if t > 0 else 0]

            # Create both conditional and unconditional embeddings
            cond_embed = cond
            uncond_embed = torch.zeros_like(cond_embed)

            # Predict noise for both
            z_dup = torch.cat([z, z], dim=0)
            t_dup = torch.cat([t_tensor, t_tensor], dim=0)
            conds = torch.cat([uncond_embed, cond_embed], dim=0)
            noise_pred = model.unet(z_dup, t_dup, conds)

            noise_uncond, noise_cond = noise_pred.chunk(2, dim=0)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            pred_x0 = (z - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()

            if t > 0:
                sigma = eta * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)).sqrt()
                noise = torch.randn_like(z) if eta > 0 else 0
                z = alpha_prev.sqrt() * pred_x0 + sigma * noise
            else:
                z = pred_x0


        recon = model.autoencoder.decode(z)
        recon = recon.squeeze(0).cpu() * 0.5 + 0.5  # Rescale from [-1, 1] to [0, 1]
        img = transforms.ToPILImage()(recon)

        if save_path:
            img.save(save_path)
            print(f"Image saved to {save_path}")

        return img


# --- Example usage ---
if __name__ == "__main__":
    prompt = "The character has striking blue hair styled in a way that complements their gray skin. They are dressed in a pink garment, possibly a robe or tunic. They hold fiery elements in both hands, casting a warm glow around them. Their appearance suggests they might belong to a mystical or magical class, such as a sorcerer or mage."
    img = generate_image(prompt, save_path="sample_output.png")
    img.show()
