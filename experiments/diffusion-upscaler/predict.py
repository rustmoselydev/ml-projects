import torch
import matplotlib.pyplot as plt
import upscaleparams
from upscalemodel import Unet
from PIL import Image
import torchvision.transforms.v2 as transforms
import librosa
import soundfile as sf
from io import BytesIO
import numpy as np

# Loading the model
# Recreate the model architecture
u_net = Unet(channel_in=6, ch=3, timesteps=upscaleparams.timesteps).to(upscaleparams.device)

# Load saved model weights
checkpoint = torch.load("diffusion_checkpoint.pth", map_location=upscaleparams.device)
u_net.load_state_dict(checkpoint["model_state_dict"])
u_net.eval()  # Set to evaluation mode
print("Model loaded successfully!")


# Image generation function
def generate_image(u_net, lores, timesteps, image_size, device):
    u_net.eval()  # Set model to evaluation mode

    # Initialize random noise image
    bs = 1  # Generate one image at a time
    channel_in = 3  # Assuming RGB images
    noisy_image = torch.randn(bs, channel_in, image_size, image_size).to(device)

    # Reverse diffusion: denoise step by step
    for t in reversed(range(timesteps)):
        # Create tensor for the current timestep
        timestep_tensor = torch.tensor([t], device=device)

        # Predict the denoised image at this step
        with torch.no_grad():
            pred_image = u_net(noisy_image, timestep_tensor, lores)
        pred_image = (pred_image * 0.5 + 0.5).clamp(0, 1)
        # Update noisy image (this should be adjusted based on diffusion process)
        noisy_image = pred_image  # Simplified update (actual process may involve more noise handling)
    
    # Convert to numpy for visualization
    generated_image = noisy_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    generated_image = (generated_image - generated_image.min()) / (generated_image.max() - generated_image.min())  # Normalize for visualization
    
    arr = (generated_image * 255).astype(np.uint8)
    # bytes_img = BytesIO(arr)
    pil_img = Image.fromarray(arr)
    pil_img = pil_img.convert('RGB')
    pil_img = pil_img.resize([640, 480])
    pil_img.show()
    pil_img.save("./test.png")
    lib_img = np.array(pil_img).astype(np.float64)
    audio_signal = librosa.feature.inverse.mel_to_audio(lib_img, sr=22050)
    sf.write("test.wav", np.ravel(audio_signal), 22050)
    # Display image
    plt.imshow(lib_img)
    plt.axis("off")
    plt.show()
    

#Generate an image with desired labels
lores_source = Image.open("./ex-lo.png").convert("RGB")  # Example
tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
lores_tfmd = tfms(lores_source).to(upscaleparams.device).float()
lores_tfmd = lores_tfmd.unsqueeze(0)

generate_image(u_net, lores_tfmd, timesteps=upscaleparams.timesteps, image_size=upscaleparams.image_size, device=upscaleparams.device)