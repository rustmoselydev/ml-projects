import librosa
import soundfile as sf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

plt.figure(figsize=(3, 3), frameon=False)
y, sr = librosa.load("./foolin.wav")
print(sr)
abs_spectrogram = librosa.stft(y, n_fft=2048, hop_length=512)
abs_spectrogram = librosa.amplitude_to_db(np.abs(abs_spectrogram), ref=np.max)
print(abs_spectrogram.shape)
# print(np.min(np.abs(abs_spectrogram)), np.max(np.abs(abs_spectrogram)))
# print("Spectrogram dB range:", abs_spectrogram.min(), abs_spectrogram.max())
np.save("foolin.npy", abs_spectrogram)


# print("spec saved")
img = np.load("./foolin.npy")
# print("spec loaded")


audio_signal = librosa.db_to_amplitude(img)
# print("Min/Max of loaded spectrogram:", np.min(audio_signal), np.max(audio_signal))
audio_signal = librosa.griffinlim(audio_signal, n_iter=64)
# audio_signal = np.clip(audio_signal, -1.0, 1.0)
# normalize
max_peak = np.max(np.abs(audio_signal))
ratio = 1 / max_peak
audio_signal = audio_signal * ratio

# audio_signal = librosa.feature.inverse.mel_to_audio(lib_img, sr=sr)
sf.write("foolin2.wav", audio_signal, sr)