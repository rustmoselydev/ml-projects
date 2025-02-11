import os
import pandas as pd
from torchvision.io import decode_image
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, random_split
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import upscaleparams as params
from upscalemodel import Unet
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F


## Custom Datset

class CustomImageDataset(Dataset):
    # Initialize parameters in class
    def __init__(self, img_dir, transform, target_transform):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    # Return length of samples in dataset
    def __len__(self):
        return len(self.img_labels)

    # Get an image/labels from the dataset
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        sourceTensor = decode_image(img_path)
        image = sourceTensor.clone().detach().to(dtype=torch.float)
        lowres_image = image
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            lowres_image = self.target_transform(image)
        return image, lowres_image

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
                               transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]),
                           ])

tgt_tfms = tfms = transforms.Compose([
                               #Resize image
                               transforms.Resize([params.image_size / 4, params.image_size / 4]),
                               transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]),
                           ])

dataset = CustomImageDataset('../synthesizer/data/specs', tfms, tgt_tfms)

generator = torch.Generator().manual_seed(42)
training_data, test_data = random_split(dataset, [0.7, 0.3])

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True, num_workers=1)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=1)
if __name__ == '__main__':
    # Create a dataloader iterable object
    dataiter = iter(train_dataloader)
    # Sample from the iterable object
    images, lowres_images = next(dataiter)

    # UNet network
    u_net = Unet(channel_in=4, ch=64, blocks=(1, 2, 4, 8), timesteps=params.timesteps).to(params.device)
    #A fixed latent noise vector so we can see the improvement over the epochs
    fixed_latent_noise = torch.randn(images.shape[0], 4, params.image_size, params.image_size).to(params.device)

    
    resume_epoch = -1
    # Adam optimizer
    optimizer = optim.Adam(u_net.parameters(), lr=params.lr)
    # if we're picking up from a saved checkpoint
    if params.do_reload:
        checkpoint = torch.load("diffusion_checkpoint.pth", map_location=params.device)
        u_net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        resume_epoch = checkpoint["epoch"]  # Resume from last saved epoch
    # CosineAnnealer to slowly drop the learning rate
    lr_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.train_epoch, eta_min=0)

    loss_log = []
    mean_loss = 0

    alphas = torch.flip(cosine_alphas_bar(params.timesteps), (0,)).to(params.device)


# Training

    pbar = trange(params.train_epoch, leave=False, desc="Epoch")    
    u_net.train()
    for epoch in pbar:
        if epoch >= resume_epoch:
            print("Epoch runnin: " + str(epoch))
            pbar.set_postfix_str('Loss: %.4f' % (mean_loss/len(train_dataloader)))
            mean_loss = 0

            for num_iter, (images, labels) in enumerate(tqdm(train_dataloader, leave=False)):
                images = images.to(params.device)
                labels = labels.to(params.device)
                
                #the size of the current minibatch
                bs = images.shape[0]
                # Randomly sample a timestamp
                rand_index = torch.randint(params.timesteps, (bs, ), device=params.device)
                #Randomly sample an image
                random_sample = torch.randn_like(images)
                # Use our cosine alpha to choose the noise factor
                alpha_batch = alphas[rand_index].reshape(bs, 1, 1, 1)
                # SQRT to make sure STD is fixed over training
                noise_input = alpha_batch.sqrt() * images + (1 - alpha_batch).sqrt() * random_sample

                img_pred = u_net(noise_input, rand_index, labels)
                # Calculate loss by quality of prediction
                # L2 loss would work, but L1 helps with noisy training
                loss = F.l1_loss(img_pred, images)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                #log the generator training loss
                loss_log.append(loss.item())
                mean_loss += loss.item()
                pbar.update(1)
                
            lr_schedule.step()
            torch.save({
                'epoch': epoch,
                'model_state_dict': u_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, "diffusion_checkpoint.pth")
            print("Model saved with checkpoint!")


    plt.figure(figsize = (20,10))
    out = vutils.make_grid(torch.clamp(noise_input, -1, 1).detach().cpu(), nrow=16, normalize=True)
    _ = plt.imshow(out.numpy().transpose((1, 2, 0)))

    plt.figure(figsize = (20,10))
    out = vutils.make_grid(img_pred.detach().cpu(), nrow=16, normalize=True)
    _ = plt.imshow(out.numpy().transpose((1, 2, 0)))

    plt.plot(loss_log)

    # Save model weights
    # torch.save(u_net.state_dict(), "diffusion_unet.pth")
   