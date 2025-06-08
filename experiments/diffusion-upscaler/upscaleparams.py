import torch

# PARAMETERS
batch_size = 16
lr = 1e-4
train_epoch = 50
image_size = 512
# Uncomment if you're on windows/nvda, and change use_mps to False
use_cuda = False #torch.cuda.is_available()
use_mps = torch.mps.is_available()
gpu_idx = 0
device = torch.device(gpu_idx if use_mps or use_cuda else "cpu")
timesteps = 50
do_reload = False