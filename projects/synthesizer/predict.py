import torch
import matplotlib.pyplot as plt
import params
from models import Unet
import librosa
import soundfile as sf
import numpy as np
import cv2

# Loading the model
# Recreate the model architecture
u_net = Unet(channel_in=15, timesteps=params.timesteps, num_labels=params.num_labels).to(params.device)

# Load saved model weights
checkpoint = torch.load("diffusion_checkpoint.pth", map_location=params.device)
u_net.load_state_dict(checkpoint["model_state_dict"])
u_net.eval()  # Set to evaluation mode
print("Model loaded successfully!")

def denormalize_db(norm_spec, min_db=-80.0, max_db=0.0):
    return ((norm_spec + 1) / 2) * (max_db - min_db) + min_db

# Image generation function
def generate_image(u_net, labels, timesteps, image_size, device):
    u_net.eval()

    # Ensure labels have batch dimension and are on the correct device
    labels = labels.to(device).float().unsqueeze(0)

    bs = 1  # Batch size
    channel_in = 1  # Spectrogram is single channel

    # Initialize random noise with the correct shape
    noisy_image = torch.randn(bs, channel_in, image_size, image_size).to(device)

   
    # Reverse diffusion
    for t in reversed(range(timesteps)):
         # Expand labels to match spatial dimensions
        labels_expanded = labels.view(bs, labels.shape[1], 1, 1).expand(-1, -1, image_size, image_size)

        # Concatenate along the channel dimension
        input_tensor = torch.cat([noisy_image, labels_expanded], dim=1) 
        timestep_tensor = torch.full((bs,), t, device=device, dtype=torch.long)  # Match batch size
        with torch.no_grad(), torch.autocast(device_type="mps", dtype=torch.float16):
            pred_image = u_net(input_tensor, timestep_tensor, labels)

        input_tensor = pred_image

    # Extract the spectrogram channel
    generated_image = input_tensor[:, 0, :, :].squeeze(0).cpu().numpy()
    generated_image = generated_image.astype('float32')

    # Normalize and resize
    
    generated_image = cv2.resize(generated_image, (173, 1025), interpolation=cv2.INTER_LINEAR)
    testing_image = (generated_image - generated_image.min()) / (generated_image.max() - generated_image.min() + 1e-8)
    print("generated_image shape:", generated_image.shape)
    print("Min/max:", generated_image.min(), generated_image.max())
    plt.figure(figsize=(10, 5))  # Set figure size
    plt.imshow(testing_image, cmap="magma", origin="lower", aspect="auto")
    plt.colorbar(label="Magnitude (dB)")  # Add color bar
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency Bins")
    plt.title("Spectrogram Visualization")
    plt.show()

    denormalized_db = denormalize_db(generated_image, min_db=-80.0, max_db=0.0)
    # Convert to amplitude and audio signal
    audio_signal = librosa.db_to_amplitude(denormalized_db)
    audio_signal = librosa.griffinlim(audio_signal, n_iter=64)

    # Normalize audio signal
    audio_signal = audio_signal / np.max(np.abs(audio_signal))

    

    sf.write("foolin2.wav", audio_signal, 22050)
    print("Audio generated successfully!")


#Generate an image with desired labels
desired_labels = torch.tensor([86, 25, 1, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float)  # Example multi-hot label vector

generate_image(u_net, desired_labels, timesteps=params.timesteps, image_size=params.image_size, device=params.device)