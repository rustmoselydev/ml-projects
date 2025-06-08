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
            self.norm = nn.BatchNorm2d(channels, affine=False, eps=1e-4)
        elif norm_type == "gn":
            self.norm = nn.GroupNorm(3, channels, affine=False, eps=1e-4)
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

        self.conv2 = nn.Conv2d(channel_in, channel_out * 4, kernel_size, 1, kernel_size // 2)
        # Pixel shuffle is better than nearest-neighbor for fine details, which we need in this case
        self.pixel_shuffle = nn.PixelShuffle(2)  # Upscales spatially by 2x

        self.conv_skip = nn.Conv2d(channel_in, channel_out, kernel_size, 1, kernel_size // 2)
        # Ensure residual matches output shape
        self.conv_residual = nn.Conv2d(channel_in, channel_out, 1) if channel_in != channel_out else nn.Identity()
        self.act_fnc = nn.ELU()

    def forward(self, x, features):
        residual = self.conv_residual(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        x = self.act_fnc(self.norm1(x, features))  # Normalize & activate
        skip = self.conv_skip(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))  # Upscale skip branch

        x = self.act_fnc(self.norm2(self.conv1(x), features))
        x = self.conv2(x)  # Prepare for Pixel Shuffle
        x = self.pixel_shuffle(x)  # Apply sub-pixel upsampling

        return x + skip + residual  # Merge with skip connection


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

    def __init__(self, channels, ch=3, blocks=(16, 8), num_features=128):
        super(Encoder, self).__init__()
        self.conv_in = nn.Conv2d(channels, blocks[0] * ch, 3, 1, 1)

        # Define two ResDown layers (since we need to scale twice)
        # self.layer_blocks = nn.ModuleList([
        #     ResDown(blocks[-1] * ch * 2, blocks[-2] * ch, num_features=num_features),  # 256x256 -> 128x128
        #     ResDown(blocks[-2] * ch * 2, blocks[-1] * ch, num_features=num_features)   # 128x128 ->  64x64
        # ])
        widths_in = list(blocks)
        widths_out = list(blocks[1:]) + [blocks[-1]]
        self.layer_blocks = nn.ModuleList([])

        for w_in, w_out in zip(widths_in, widths_out):
            self.layer_blocks.append(ResDown(w_in * ch, w_out * ch, num_features=num_features))

        self.block_out = ResBlock(blocks[1] * ch, blocks[1] * ch, num_features=num_features)

        self.act_fnc = nn.ELU()

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

    def __init__(self, channels, ch=3, blocks=(16, 8), num_features=128):
        super(Decoder, self).__init__()

        self.block_in = ResBlock(blocks[-1] * ch, blocks[-1] * ch, num_features=num_features)
        widths_out = list(blocks)[::-1]
        widths_in = (list(blocks[1:]) + [blocks[-1]])[::-1]

        self.layer_blocks = nn.ModuleList([])

        for w_in, w_out in zip(widths_in, widths_out):
            self.layer_blocks.append(ResUp(w_in * ch * 2, w_out * ch, num_features=num_features))

        self.conv_out = nn.Conv2d(blocks[0] * ch * 2, 3, 3, 1, 1)
        self.act_fnc = nn.ELU()
        self.final_conv = nn.Conv2d(blocks[0] * ch * 2, blocks[0] * ch * 2, padding=1, kernel_size=3)
        # self.conv_refine = nn.Conv2d(blocks[0] * ch * 2, blocks[0] * ch * 2, padding=1, kernel_size=1)
        self.final_norm = ConditionalNorm2d(96, num_features=num_features)

    def forward(self, x_in, skip_list, index_features):
        x = self.block_in(x_in, index_features)

        for block in self.layer_blocks:
            skip = skip_list.pop()
            x = torch.cat((x, skip), 1)
            x = block(x, index_features)
            
        skip = skip_list.pop()
        if skip.shape[2:] != x.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat((x, skip), 1)
        
        x = self.final_norm(x, index_features)
        x = F.elu(x)
        x = self.final_conv(x)
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
    def __init__(self, channel_in=6, ch=3, timesteps=20, num_features=128):
        super(Unet, self).__init__()
        # Timestep embedding        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(num_features),
            nn.Linear(num_features, 2 * num_features),
            nn.GELU(),
            nn.Linear(2 * num_features, num_features),
            nn.GELU()
        )

        self.low_res_conv = nn.Conv2d(3, ch, kernel_size=3, padding=1)
        
        
        self.encoder = Encoder(channel_in, ch=ch, num_features=num_features)
        self.decoder = Decoder(channel_in, ch=ch, num_features=num_features)

    def forward(self, x, index, small_img):
        index_features = self.time_mlp(index)

        # Encode the low-res image
        low_res_features = self.low_res_conv(small_img)
        low_res_features = F.interpolate(low_res_features, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
        # Combine the two embeddings
        cond_features = torch.cat((x, low_res_features), dim=1)
        
        bottleneck, skip_list = self.encoder(cond_features, index_features)
        recon_img = self.decoder(bottleneck, skip_list, index_features)
        return recon_img
