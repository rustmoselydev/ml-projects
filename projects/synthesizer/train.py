import os
import pandas as pd
from torchvision.io import decode_image
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, random_split
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import params
from models import Unet
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.set_default_dtype(torch.float32)



def normalize_db(spectrogram, min_db=-80.0, max_db=0.0):
    return (spectrogram - min_db) / (max_db - min_db) * 2 - 1  # Scale to [-1, 1]

class CustomImageDataset(Dataset):
    # Initialize parameters in class
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    # Return length of samples in dataset
    def __len__(self):
        return len(self.img_labels)

    # Get an image/labels from the dataset
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        img_path = img_path.replace(".png", ".npy")
        file_content = np.load(img_path).astype(np.float32)
        image = normalize_db(file_content)
        # Grab the entire row of labels
        labelsFrame = self.img_labels.iloc[idx]
        labels = torch.tensor([labelsFrame["pitch"], labelsFrame["velocity"], labelsFrame["source"], labelsFrame["family"], labelsFrame["quality_bright"], labelsFrame["quality_dark"], labelsFrame["quality_distortion"], labelsFrame["quality_fast_decay"], labelsFrame["quality_long_release"], labelsFrame["quality_multiphonic"], labelsFrame["quality_nonlinear_env"], labelsFrame["quality_percussive"], labelsFrame["quality_reverb"], labelsFrame["quality_tempo_synced"]], dtype=torch.float)
        if self.transform:
            image = self.transform(image)
        # We shouldn't need to transform the labels
        # if self.target_transform:
        #     label = self.target_transform(label)
        return image, labels

## Helper functions
from tqdm import trange, tqdm

# How much noise we add at each step; the variance/noise schedule
# Could use linear, but this only removes a lil noise at the end and a lot at the beginning
def cosine_alphas_bar(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alphas_bar = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    return alphas_bar[:timesteps]

## DataLoaders
tfms = transforms.Compose([
    #Resize image
    transforms.Resize([params.image_size, params.image_size]),
])
dataset = CustomImageDataset('data.csv', 'data/specs', transform=tfms)

generator = torch.Generator().manual_seed(42)
training_data, test_data = random_split(dataset, [0.7, 0.3])

train_dataloader = DataLoader(training_data, batch_size=16, shuffle=True, num_workers=1)
test_dataloader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=1)
if __name__ == '__main__':
    # Create a dataloader iterable object
    dataiter = iter(train_dataloader)
    # Sample from the iterable object
    images, labels = next(dataiter)

    # UNet network
    u_net = Unet(channel_in=15, ch=16, blocks=(1, 2, 4, 8), timesteps=params.timesteps, num_labels=params.num_labels).to(params.device).to(torch.float32)
    resume_epoch = -1
    # Adam optimizer
    optimizer = optim.Adam(u_net.parameters(), lr=params.lr)
    # Reduce the LR if it plateaus for too long
    lr_schedule = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    # if we're picking up from a saved checkpoint
    best_val_loss_so_far = 1000000
    model_state = None
    if params.do_reload:
        print("Picking up where we left off from saved checkpoint")
        checkpoint = torch.load("diffusion_checkpoint.pth", map_location=params.device)
        u_net.load_state_dict(checkpoint["model_state_dict"])
        model_state = checkpoint["model_state_dict"]
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        resume_epoch = checkpoint["epoch"]  # Resume from last saved epoch
        best_val_loss_so_far = checkpoint.get('best_val_loss', float('inf'))
        if "scheduler_state_dict" in checkpoint:
            lr_schedule.load_state_dict(checkpoint["scheduler_state_dict"]) 
            lr_schedule.step(checkpoint["best_val_loss"])
        
        print("We will resume at epoch " + str(resume_epoch + 1))
    

    loss_log = []
    mean_loss = 0

    alphas = torch.flip(cosine_alphas_bar(params.timesteps), (0,)).to(params.device).to(torch.float32)


# Training
    pbar = trange(params.train_epoch, leave=False, desc="Epoch")    
    u_net.train()
    for epoch in pbar:
        if epoch > resume_epoch:
            current_lr = optimizer.param_groups[0]['lr']
            print("Epoch runnin: " + str(epoch) + ", Learning Rate: " + str(current_lr) + ", Last saved val loss: " + str(round(best_val_loss_so_far, 6)))
            pbar.set_postfix_str('Loss: %.4f' % (mean_loss/len(train_dataloader)))
            mean_loss = 0

            for num_iter, (images, labels) in enumerate(tqdm(train_dataloader, leave=False)):
                images = images.to(params.device).to(torch.float32)
                labels = labels.to(params.device).to(torch.float32)
                
                #the size of the current minibatch
                bs = images.shape[0]
                # Randomly sample a timestamp
                rand_index = torch.randint(params.timesteps, (bs, ), device=params.device)
                #Randomly sample an image
                random_sample = torch.randn_like(images).to(torch.float32)
                # Use our cosine alpha to choose the noise factor
                alpha_batch = alphas[rand_index].reshape(bs, 1, 1, 1)
                # Ensure images has shape [batch, 1, height, width]
                images = images.unsqueeze(1)

                # Expand labels to match spatial dimensions
                labels_expanded = labels.view(labels.shape[0], labels.shape[1], 1, 1).expand(-1, -1, images.shape[2], images.shape[3])

                # Combine spectrogram (1 channel) + labels (1 extra channel)
                input_tensor = torch.cat([images, labels_expanded], dim=1)
                # SQRT to make sure STD is fixed over training
                noise_input = alpha_batch.sqrt() * input_tensor + (1 - alpha_batch).sqrt() * random_sample.unsqueeze(1)
                img_pred = u_net(noise_input, rand_index, labels)
                # Calculate loss by quality of prediction
                # L2 loss would work, but L1 helps with noisy training
                loss = F.l1_loss(img_pred, images)
                
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient Clipping (avoid losses going off the rails)
                torch.nn.utils.clip_grad_norm_(u_net.parameters(), max_norm=0.5)
                optimizer.step()
                
                #log the generator training loss
                loss_log.append(loss.item())
                mean_loss += loss.item()
                pbar.update(1)
                
            #Validation loop
            print("Starting validation batches")
            u_net.eval()
            val_loss = 0
            with torch.no_grad():  # No gradients needed for validation
                for images, labels in test_dataloader:
                    images = images.to(params.device).to(torch.float32)
                    labels = labels.to(params.device).to(torch.float32)

                    bs = images.shape[0]
                    rand_index = torch.randint(params.timesteps, (bs,), device=params.device)
                    random_sample = torch.randn_like(images).to(torch.float32)
                    alpha_batch = alphas[rand_index].reshape(bs, 1, 1, 1)
                    images = images.unsqueeze(1)
                    labels_expanded = labels.view(labels.shape[0], labels.shape[1], 1, 1).expand(-1, -1, images.shape[2], images.shape[3])
                    input_tensor = torch.cat([images, labels_expanded], dim=1)
                    noise_input = alpha_batch.sqrt() * input_tensor + (1 - alpha_batch).sqrt() * random_sample.unsqueeze(1)
                    
                    img_pred = u_net(noise_input, rand_index, labels)
                    loss = F.l1_loss(img_pred, images)

                    val_loss += loss.item()

            val_loss /= len(test_dataloader)
            print(f"Validation Loss: {val_loss:.4f}")
            lr_schedule.step(val_loss)
            save_dict = {
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_schedule.state_dict(),
                'best_val_loss': best_val_loss_so_far,
                'model_state_dict': model_state
            }
            if val_loss < best_val_loss_so_far:
                print("Model improved! Saving full checkpoint.")
                backup_model = {
                    'model_state_dict': u_net.state_dict(),
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_schedule.state_dict(),
                    'best_val_loss': val_loss
                }
                torch.save(backup_model, "backup_model.pth")
                save_dict['model_state_dict'] = u_net.state_dict()
                save_dict['best_val_loss'] = val_loss
                torch.save(save_dict, "diffusion_checkpoint.pth")
                best_val_loss_so_far = val_loss
            else:
                print("Validation loss did not improve. Saving optimizer/scheduler only.")
                torch.save(save_dict, "diffusion_checkpoint.pth")
        else:
            print("Skipping epoch " + str(epoch))
   