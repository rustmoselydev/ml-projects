# An older style UNet model designed to replicate audio spectrograms based on a multi-hot vector
# The spectrograms are huge, and with my home hardware training this is not super feasible.
# It's also much less efficient than an LDM. I may revisit this at a later date

import torch.nn as nn
import math
import torch
import torch.nn.functional as F

# Conditional normalization of our data
# Makes data harder for the model to ignore, though this isn't strictly necessary in diffusion
class ConditionalNorm2d(nn.Module):
    def __init__(self, channels, num_features, norm_type="gn"):
        super(ConditionalNorm2d, self).__init__()
        self.channels = channels
        if norm_type == "bn":
            self.norm = nn.BatchNorm2d(channels, affine=False, eps=1e-3)
        elif norm_type == "gn":
            self.norm = nn.GroupNorm(8, channels, affine=False, eps=1e-3)
        else:
            raise ValueError("Normalisation type not recognised.")
        # Remember: num_features is the static index-keeping parameter from our UNet
        #Weight/Scale
        self.fcw = nn.Linear(num_features, channels)
        #Bias
        self.fcb = nn.Linear(num_features, channels)

    def forward(self, x, features):
        out = self.norm(x)
        # Our weight and bias are thus connected to our positional embedding
        w = self.fcw(features)
        b = self.fcb(features)

        out = w.view(-1, self.channels, 1, 1) * out + b.view(-1, self.channels, 1, 1)
        return out

# Halve the resolution
class ResDown(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, num_features=128):
        super(ResDown, self).__init__()
        self.norm1 = ConditionalNorm2d(channel_in, num_features=num_features)

        self.conv1 = nn.Conv2d(channel_in, channel_out, kernel_size, 2, kernel_size // 2)
        self.norm2 = ConditionalNorm2d(channel_out, num_features=num_features)

        self.conv2 = nn.Conv2d(channel_out, channel_out, kernel_size, 1, kernel_size // 2)

        self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size, 2, kernel_size // 2)

        self.act_fnc = nn.ELU()

    def forward(self, x, features):
        x = self.act_fnc(self.norm1(x, features))

        skip = self.conv3(x)

        x = self.act_fnc(self.norm2(self.conv1(x), features))
        x = self.conv2(x)

        return x + skip

# Double the resolution
class ResUp(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, scale_factor=2, num_features=128):
        super(ResUp, self).__init__()
        self.norm1 = ConditionalNorm2d(channel_in, num_features=num_features)

        self.conv1 = nn.Conv2d(channel_in, channel_in, kernel_size, 1, kernel_size // 2)
        self.norm2 = ConditionalNorm2d(channel_in, num_features=num_features)

        self.conv2 = nn.Conv2d(channel_in, channel_out, kernel_size, 1, kernel_size // 2)

        self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size, 1, kernel_size // 2)

        self.up_nn = nn.Upsample(scale_factor=scale_factor, mode="nearest")

        self.act_fnc = nn.ELU()

    def forward(self, x, features):
        x = self.up_nn(self.act_fnc(self.norm1(x, features)))

        skip = self.conv3(x)

        x = self.act_fnc(self.norm2(self.conv1(x), features))
        x = self.conv2(x)

        return x + skip


class ResBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, num_features=128):
        super(ResBlock, self).__init__()
        self.norm1 = ConditionalNorm2d(channel_in, num_features=num_features)

        self.conv1 = nn.Conv2d(channel_in, channel_in, kernel_size, 1, kernel_size // 2)
        self.norm2 = ConditionalNorm2d(channel_in, num_features=num_features)

        self.conv2 = nn.Conv2d(channel_in, channel_out, kernel_size, 1, kernel_size // 2)

        if not channel_in == channel_out:
            self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size, 1, kernel_size // 2)
        else:
            self.conv3 = None

        self.act_fnc = nn.ELU()

    def forward(self, x_in, features):
        x = self.act_fnc(self.norm1(x_in, features))

        if self.conv3 is not None:
            skip = self.conv3(x)
        else:
            skip = x_in

        x = self.act_fnc(self.norm2(self.conv1(x), features))
        x = self.conv2(x)

        return x + skip

# Make noisier
class Encoder(nn.Module):

    def __init__(self, channels, ch=16, blocks=(1, 2, 4, 8), num_features=128):
        super(Encoder, self).__init__()
        self.conv_in = nn.Conv2d(channels, blocks[0] * ch, 3, 1, 1)

        widths_in = list(blocks)
        widths_out = list(blocks[1:]) + [blocks[-1]]

        self.layer_blocks = nn.ModuleList([])

        for w_in, w_out in zip(widths_in, widths_out):
            self.layer_blocks.append(ResDown(w_in * ch, w_out * ch, num_features=num_features))

        self.block_out = ResBlock(w_out * ch, w_out * ch, num_features=num_features)

        # self.act_fnc = nn.ELU()

    def forward(self, x, index_features):
        x = self.conv_in(x)
        skip_list = [x]

        for block in self.layer_blocks:
            x = block(x, index_features)
            skip_list.append(x)
        
        x = self.block_out(x, index_features)
        return x, skip_list

# Make less noisy
class Decoder(nn.Module):

    def __init__(self, channels, ch=16, blocks=(1, 2, 4, 8), num_features=128):
        super(Decoder, self).__init__()
        
        widths_out = list(blocks)[::-1]
        widths_in = (list(blocks[1:]) + [blocks[-1]])[::-1]

        self.block_in = ResBlock(blocks[-1] * ch, blocks[-1] * ch, num_features=num_features)
        
        self.layer_blocks = nn.ModuleList([])

        for w_in, w_out in zip(widths_in, widths_out):
            self.layer_blocks.append(ResUp(w_in * ch * 2, w_out * ch, num_features=num_features))

        self.conv_out = nn.Conv2d(blocks[0] * ch * 2, 1, 3, 1, 1)
        # self.act_fnc = nn.ELU()

    def forward(self, x_in, skip_list, index_features):
        x = self.block_in(x_in, index_features)
        
        for block in self.layer_blocks:
            skip = skip_list.pop()
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat((x, skip), 1)
            x = block(x, index_features)
            
        skip = skip_list.pop()
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, skip), 1)
        return self.conv_out(x)

 # Fixed index embedding for keeping track of our timestamp (index) in process   
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# Unet is a U shaped architecture of encoding/decoding
class Unet(nn.Module):
    def __init__(self, channel_in=2, ch=16, blocks=(1, 2, 4, 8), timesteps=20, num_features=128, num_labels=10):
        super(Unet, self).__init__()
        # Timestep embedding        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(num_features),
            nn.Linear(num_features, 2 * num_features),
            nn.GELU(),
            nn.Linear(2 * num_features, num_features),
            nn.GELU()
        )

         # Label embedding
        self.label_mlp = nn.Sequential(
            nn.Linear(num_labels, num_features),  
            nn.GELU(),
            nn.Linear(num_features, num_features),
            nn.GELU()
        )
        
        self.encoder = Encoder(channel_in, ch=ch, blocks=blocks, num_features=num_features)
        self.decoder = Decoder(channel_in, ch=ch, blocks=blocks, num_features=num_features)

    def forward(self, x, index, labels):
        index_features = self.time_mlp(index)
        label_features = self.label_mlp(labels) 

        # Combine time and label embeddings
        cond_features = (index_features / index_features.norm()) + (label_features / label_features.norm())
        
        bottleneck, skip_list = self.encoder(x, cond_features)
        recon_img = self.decoder(bottleneck, skip_list, cond_features)
        return recon_img
