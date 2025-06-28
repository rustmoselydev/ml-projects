import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch
from tqdm import tqdm
from model import LatentDiffusionModel, SimpleAutoencoder, UNet, CLIPTextEmbedder, NoiseScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from perceptual_loss import VGGPerceptualLoss
from params import learning_rate, num_inference_steps, epochs, device, core_loss_ratio, perc_loss_ratio, latent_dim, embed_dim, image_size, batch_size


# --- Config ---
image_folder = "../../training-data/character-forward"
caption_csv = "./generated_captions.csv"
checkpoint_path = "./ldm_checkpoint_v.pt"
autoencoder_ckpt_path = "autoencoder_checkpoint.pt"


# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


# --- Dataset ---
class PixelArtDataset(Dataset):
    def __init__(self, image_folder, caption_csv, transform):
        self.image_folder = image_folder
        self.transform = transform
        self.data = pd.read_csv(caption_csv)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]['image']
        caption = self.data.iloc[idx]['caption']

        try:
            image = Image.open(f"../../training-data/character-forward/{image_path}.png").convert("RGB")
            return self.transform(image), caption
        except Exception as e:
            # Skip this item by returning another valid one instead
            new_idx = (idx + 1) % len(self)
            return self.__getitem__(new_idx)

# --- DataLoaders ---
dataset = PixelArtDataset(image_folder, caption_csv, transform)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


# --- Models ---
if __name__ == "__main__":
    perceptual_loss_fn = VGGPerceptualLoss().to(device)
    # Load trained autoencoder
    autoencoder = SimpleAutoencoder(latent_dim=latent_dim).to(device)
    autoencoder_ckpt = torch.load(autoencoder_ckpt_path, map_location=device)
    autoencoder.load_state_dict(autoencoder_ckpt['model_state_dict'])
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False

    # Initialize rest of the LDM stack
    unet = UNet(latent_dim=latent_dim, embed_dim=embed_dim).to(device)
    text_embedder = CLIPTextEmbedder(device=device)
    noise_scheduler = NoiseScheduler(timesteps=num_inference_steps)

    model = LatentDiffusionModel(autoencoder, unet, text_embedder, noise_scheduler).to(device)


    # --- Optimizer and Scheduler ---
    optimizer = None
    scheduler = None
    best_val_loss = float('inf')
    start_epoch = 0

    # Resume training if checkpoint exists
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_loss = checkpoint['best_val_loss']
        start_epoch = checkpoint['epoch'] + 1
        #scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"]) 
        print(f"Resuming from epoch {start_epoch} with best val loss {best_val_loss:.4f}")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        #scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)


    # --- Training Loop ---
    for epoch in range(start_epoch, epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")
        model.train()
        train_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]")

        for step, (img, captions) in enumerate(progress):
            img = img.to(device)
            t = torch.randint(0, noise_scheduler.timesteps, (img.size(0),), device=device).long()

            core_loss, recon = model(img, captions, t, return_reconstruction=True)
            perc_loss = perceptual_loss_fn(recon, img)
            loss = core_loss_ratio * core_loss + perc_loss_ratio * perc_loss

            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img, captions in val_loader:
                img = img.to(device)
                t = torch.randint(0, noise_scheduler.timesteps, (img.size(0),), device=device).long()
                core_loss, recon = model(img, captions, t, return_reconstruction=True)
                perc_loss = perceptual_loss_fn(recon, img)
                loss = core_loss_ratio * core_loss + perc_loss_ratio * perc_loss
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
        scheduler.step()

        # Save model if it improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch}")
        else:
            print("Model didn't improve. Not saving.")
