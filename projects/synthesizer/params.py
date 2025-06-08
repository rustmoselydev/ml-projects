import torch

# PARAMETERS
batch_size = 16
lr = 1e-4
train_epoch = 100
image_size = 256
use_cuda = False #torch.cuda.is_available()
use_mps = torch.mps.is_available()
gpu_idx = 0
device = torch.device(gpu_idx if use_mps or use_cuda else "cpu")
timesteps = 50
num_labels = 14
do_reload = True