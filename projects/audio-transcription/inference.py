import subprocess
from pathlib import Path
from basic_pitch.inference import predict

CANDIDATES = [
    "mscore4portable",           # MuseScore 4 AppImage (ChromeOS/Linux)
    "mscore",                    # Linux/macOS
    "musescore",                 # some Linux distros
    r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe",  # Windows
]

def find_musescore():
    for c in CANDIDATES:
        try:
            subprocess.run([c, "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return c
        except Exception:
            pass
    raise FileNotFoundError("MuseScore CLI not found in PATH. If on ChromeOS, use the AppImage and call `mscore4portable`.")

def audio_to_pdf(audio_rel_path: str):
    audio = Path(audio_rel_path).expanduser().resolve()
    model_output, midi_data, note_events = predict(str(audio))
    midi_path = audio.with_suffix(audio.suffix + ".mid")
    midi_data.write(str(midi_path))
    mscore = find_musescore()
    pdf_path = midi_path.with_suffix(".pdf")
    subprocess.run([mscore, "-o", str(pdf_path), str(midi_path)], check=True)
    print(f"Saved sheet music PDF: {pdf_path}")

if __name__ == "__main__":
    p = input("File path: ").strip()
    audio_to_pdf(p)