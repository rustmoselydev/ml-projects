import torch
from torchvision import transforms
from PIL import Image
from model import LatentDiffusionModel, SimpleAutoencoder, UNet, CLIPTextEmbedder, NoiseScheduler

# --- Configuration ---
checkpoint_path = "./ldm_checkpoint.pt"
image_path = "../../training-data/character-forward/0.png"
image_size = 32
latent_dim = 32
embed_dim = 512
num_inference_steps = 1000
device = torch.device("mps")  # or "cuda" or "cpu"

# --- Load and preprocess the image ---
preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),  # Match training normalization
])

img = Image.open(image_path).convert("RGB")
x = preprocess(img).unsqueeze(0).to(device)  # (1, 3, 32, 32)

# --- Load model and checkpoint ---
autoencoder = SimpleAutoencoder(latent_dim=latent_dim).to(device)
unet = UNet(latent_dim=latent_dim, embed_dim=embed_dim).to(device)
text_embedder = CLIPTextEmbedder(device=device)
noise_scheduler = NoiseScheduler(timesteps=num_inference_steps)
model = LatentDiffusionModel(autoencoder, unet, text_embedder, noise_scheduler).to(device)

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

# --- Encode → Decode (Autoencoder test) ---
with torch.no_grad():
    z = model.autoencoder.encode(x)
    recon = model.autoencoder.decode(z)

    recon = recon.clamp(-1, 1).squeeze(0).cpu()
    recon = recon * 0.5 + 0.5  # Denormalize from [-1, 1] to [0, 1]

# --- Show Original vs Reconstructed ---
original = x.squeeze(0).cpu()
original = original * 0.5 + 0.5  # Denormalize

transforms.ToPILImage()(original).show(title="Original Image")
transforms.ToPILImage()(recon).show(title="Reconstructed Image")
