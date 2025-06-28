import torch

learning_rate = 1e-4
num_inference_steps = 1000
epochs = 150
device = torch.device("mps")
core_loss_ratio = 0.85
perc_loss_ratio = 0.15
guidance_scale = 10.0
guidance_dropout = 0.05
eta = 0.1
latent_dim = 32
embed_dim = 512
image_size = 32
batch_size = 64