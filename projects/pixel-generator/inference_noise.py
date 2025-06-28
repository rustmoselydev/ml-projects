import torch
from PIL import Image
from torchvision import transforms
from model import LatentDiffusionModel, SimpleAutoencoder, UNet, CLIPTextEmbedder, NoiseScheduler

# --- Configuration ---
checkpoint_path = "./ldm_checkpoint.pt"
autoencoder_ckpt_path = "./autoencoder_checkpoint.pt"
image_size = 32
latent_dim = 32
embed_dim = 512
num_inference_steps = 1000
device = torch.device("mps")

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

model = LatentDiffusionModel(autoencoder, unet, text_embedder, noise_scheduler, predict_velocity=False).to(device)
model.eval()

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# --- Inference Function (DDIM-style Sampling) ---
def generate_image(prompt, save_path=None):
    with torch.no_grad():
        cond = text_embedder([prompt])  # (1, embed_dim)
        z = torch.randn(1, latent_dim, image_size // 8, image_size // 8).to(device)

        for t in reversed(range(num_inference_steps)):
            t_tensor = torch.tensor([t], device=device, dtype=torch.long)
            alpha_cumprod = noise_scheduler.alpha_cumprod.to(z.device)
            alpha_t = alpha_cumprod[t]
            alpha_prev = alpha_cumprod[t - 1 if t > 0 else 0]

            noise_pred = model.unet(z, t_tensor, cond)
            pred_x0 = (z - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()

            if t > 0:
                z = alpha_prev.sqrt() * pred_x0 + (1 - alpha_prev).sqrt() * noise_pred
            else:
                z = pred_x0

        recon = model.autoencoder.decode(z)
        recon = recon.squeeze(0).cpu() * 0.5 + 0.5  # now in [0, 1]
        img = transforms.ToPILImage()(recon)

        if save_path:
            img.save(save_path)
            print(f"Image saved to {save_path}")

        return img

# --- Example usage ---
if __name__ == "__main__":
    prompt = "The character is bald, has blue eyes, and is wearing a red top with a belt and blue pants. The character appears to be barefoot. The style and clothing suggest a soldier."
    img = generate_image(prompt, save_path="sample_output.png")
    img.show()
