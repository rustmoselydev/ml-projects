# This file converts audio to spectrograms
# Though it's counterintuitive, this challenge is visual multi-categorization on the spectrograms
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import soundfile as sf
import os

directory_path = "./audio"

with os.scandir(directory_path) as entries:
    for entry in entries:
        if entry.is_file():
            print(entry.name)
            # Load the audio file
            y, sr = librosa.load(f"{directory_path}/{entry.name}")

            # Compute the spectrogram
            D = librosa.stft(y)

            # Convert to decibels
            D_db = librosa.amplitude_to_db(abs(D), ref=np.max)

            # Save the file
            fig = plt.Figure()
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            p = librosa.display.specshow(librosa.amplitude_to_db(D_db, ref=np.max), ax=ax, y_axis='log', x_axis='time')
            fig.savefig(f"./specs/{entry.name}.png")
            fig.clear()
            print(f"saved f{entry.name}.png")
            # If you want to listen to the sound it produces, run this code
            # Note: it doesn't invert perfectly, and for me, that's fun audio
            # spec = librosa.feature.melspectrogram(y=y,sr=sr)
            # audio_signal = librosa.feature.inverse.mel_to_audio(spec, sr=sr)
            # pitch_fix = librosa.effects.pitch_shift(audio_signal, sr=sr, n_steps=12) 
            # sf.write("test.wav", pitch_fix, sr)
print("done!")