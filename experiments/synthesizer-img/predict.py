import torch
import matplotlib.pyplot as plt
import params
from models import Unet

# Loading the model
# Recreate the model architecture
u_net = Unet(channel_in=4, ch=64, blocks=(1, 2, 4, 8), 
             timesteps=params.timesteps, num_labels=params.num_labels).to(params.device)

# Load saved model weights
checkpoint = torch.load("diffusion_unet.pth", map_location=params.device)
u_net.load_state_dict(checkpoint["model_state_dict"])
u_net.eval()  # Set to evaluation mode
print("Model loaded successfully!")


# Image generation function
def generate_image(u_net, labels, timesteps, image_size, device):
    u_net.eval()  # Set model to evaluation mode

    # Ensure labels are on the correct device
    labels = labels.to(device).float()

    # Initialize random noise image
    bs = 1  # Generate one image at a time
    channel_in = 4  # Assuming RGB images
    noisy_image = torch.randn(bs, channel_in, image_size, image_size).to(device)

    # Reverse diffusion: denoise step by step
    for t in reversed(range(timesteps)):
        # Create tensor for the current timestep
        timestep_tensor = torch.tensor([t], device=device)

        # Predict the denoised image at this step
        with torch.no_grad():
            pred_image = u_net(noisy_image, timestep_tensor, labels)

        # Update noisy image (this should be adjusted based on diffusion process)
        noisy_image = pred_image  # Simplified update (actual process may involve more noise handling)

    # Convert to numpy for visualization
    generated_image = noisy_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    generated_image = (generated_image - generated_image.min()) / (generated_image.max() - generated_image.min())  # Normalize for visualization

    # Display image
    plt.imshow(generated_image)
    plt.axis("off")
    plt.show()

#Generate an image with desired labels
desired_labels = torch.tensor([86, 25, 1, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float)  # Example multi-hot label vector

generate_image(u_net, desired_labels, timesteps=params.timesteps, image_size=params.image_size, device=params.device)