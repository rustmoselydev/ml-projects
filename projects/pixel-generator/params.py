import torch

learning_rate = 1e-4
num_inference_steps = 1000
epochs = 200
device = torch.device("mps")
core_loss_ratio = 0.9
perc_loss_ratio = 0.1
guidance_scale = 5.0
guidance_dropout = 0.05
eta = 0.1
latent_dim = 32
embed_dim = 512
image_size = 32
batch_size = 64