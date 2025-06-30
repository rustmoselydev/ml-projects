# ---------------------------
# The autoencoder must be trained to reconstruct images from the latent space before the model will work
# ---------------------------

import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from model import SimpleAutoencoder
from torch.optim.lr_scheduler import StepLR
from perceptual_loss import VGGPerceptualLoss

# --- Configuration ---
image_folder = "../../training-data/character-forward"
caption_csv = "./generated_captions.csv"  # Only need image column
image_size = 32
batch_size = 64
epochs = 200
latent_dim = 32
learning_rate = 1e-3
checkpoint_path = "autoencoder_checkpoint.pt"
device = torch.device("mps")  # Change to 'cuda' or 'cpu' as needed

# --- Dataset ---
class PixelImageDataset(Dataset):
    def __init__(self, image_folder, caption_csv, transform):
        self.image_folder = image_folder
        self.transform = transform
        self.data = pd.read_csv(caption_csv)

        # Determine style by numeric filename
        self.data["image"] = self.data["image"].astype(int)
        majority = self.data[self.data["image"] <= 911]
        diverse = self.data[self.data["image"] > 911]

        # My sample set is not wholly diverse- balance it to lower the homogeneous bias
        total_size = len(majority) + len(diverse)
        diverse_target = int(total_size * 0.6)
        majority_target = total_size - diverse_target

        # Upsample diverse if needed
        if len(diverse) < diverse_target:
            diverse = pd.concat(
                [diverse] * (diverse_target // len(diverse)) +
                [diverse.sample(n=diverse_target % len(diverse), replace=True, random_state=42)],
                ignore_index=True
            )
        else:
            diverse = diverse.sample(n=diverse_target, random_state=42)

        # Downsample majority if needed
        if len(majority) > majority_target:
            majority = majority.sample(n=majority_target, random_state=42)
        else:
            majority = pd.concat(
                [majority] * (majority_target // len(majority)) +
                [majority.sample(n=majority_target % len(majority), replace=True, random_state=42)],
                ignore_index=True
            )

        # Combine and shuffle
        self.data = pd.concat([majority, diverse], ignore_index=True)
        self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Loop until a valid file is found or the dataset is exhausted
        attempts = 0
        while attempts < len(self):
            image_name = str(self.data.iloc[idx]['image']) + ".png"
            image_path = os.path.join(self.image_folder, image_name)

            if os.path.exists(image_path):
                # Load with alpha preserved
                image = Image.open(image_path).convert("RGBA")

                bg = Image.new("RGBA", image.size, (127, 127, 127, 255))
                image = Image.alpha_composite(bg, image)

                # Convert back to RGB for transform
                image = image.convert("RGB")

                return self.transform(image)
            else:
                # Try next index (wrap around with modulo)
                idx = (idx + 1) % len(self)
                attempts += 1

        raise RuntimeError("No valid images found in dataset.")

transform = transforms.Compose([
    transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = PixelImageDataset(image_folder, caption_csv, transform)
# sample = dataset[1090] * 0.5 + 0.5  # Denormalize
# plt.imshow(sample.permute(1, 2, 0).numpy())
# plt.title("Transformed Sample (Resized w/ NEAREST)")
# plt.axis("off")
# plt.show()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

if __name__ == "__main__":
    perceptual_loss_fn = VGGPerceptualLoss().to(device)
    autoencoder = SimpleAutoencoder(latent_dim=latent_dim).to(device)
    criterion = nn.L1Loss()
    optimizer = None
    scheduler = None


    best_loss = float("inf")
    start_epoch = 0

    # Load checkpoint if it exists
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        autoencoder.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.Adam(autoencoder.parameters())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_loss = checkpoint['best_loss']
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch} with best loss {best_loss:.4f}")
        print(f"LR: {str(optimizer.param_groups[0]['lr'])}")
    else:
        optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in range(start_epoch, epochs):
        autoencoder.train()
        total_loss = 0

        for imgs in dataloader:
            imgs = imgs.to(device)
            recon = autoencoder.decode(autoencoder.encode(imgs)).clamp(-1, 1)
            l1 = criterion(recon, imgs)
            percep = perceptual_loss_fn(recon, imgs)
            loss = 0.9 * l1 + 0.1 * percep

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Recon Loss: {avg_loss:.4f}")
        scheduler.step()
        if epoch >= epochs - 1:
            autoencoder.eval()
            with torch.no_grad():
                sample = next(iter(dataloader))[:8].to(device)
                recon = autoencoder.decode(autoencoder.encode(sample)).clamp(-1, 1)
                recon = recon * 0.5 + 0.5
                original = sample * 0.5 + 0.5
                grid = make_grid(torch.cat([original, recon], dim=0), nrow=8)
                plt.figure(figsize=(16, 4))
                plt.imshow(grid.permute(1, 2, 0).cpu())
                plt.title("Original (top) vs Reconstruction (bottom)")
                plt.axis("off")
                plt.show()

        # Save checkpoint if improved
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': autoencoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1} with improved loss {best_loss:.4f}")
        else:
            print("Model did not improve. Skipping save.")

