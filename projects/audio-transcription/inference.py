import subprocess
from pathlib import Path
import numpy as np
import pretty_midi as pm
from basic_pitch.inference import predict
from basic_pitch.postprocessing import model_output_to_notes, notes_to_midi

# ---- Tunables ----
ONSET_THRESHOLD = 0.6         # stricter onsets => fewer spurious notes
FRAME_THRESHOLD = 0.3
MIN_NOTE_LENGTH_SEC = 0.08    # 80 ms; raise to merge tiny blips
QUANT_GRID = 1/8              # quantize to 8th-notes; try 1/16 for finer
TEMPO_BPM = 120               # used for quantization grid in seconds
# Optionally constrain range (Hz); set to None to disable
MIN_FREQ_HZ = None
MAX_FREQ_HZ = None
# -------------------

CANDIDATES = [
    "mscore4portable",           # MuseScore 4 AppImage (ChromeOS/Linux)
    "musescore4", "mscore4",
    "mscore", "musescore",       # distro package (often MuseScore 3)
    r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe",
]

def find_musescore():
    for c in CANDIDATES:
        try:
            subprocess.run([c, "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            return c
        except Exception:
            pass
    raise FileNotFoundError("MuseScore CLI not found on PATH. Install MuseScore or expose the AppImage as `mscore4portable`.")

def quantize_midi(in_mid: Path, out_mid: Path, grid_fraction=QUANT_GRID, tempo_bpm=TEMPO_BPM):
    midi = pm.PrettyMIDI(str(in_mid))
    sec_per_beat = 60.0 / float(tempo_bpm)
    q = float(grid_fraction) * sec_per_beat

    def snap(x): return np.round(x / q) * q

    for inst in midi.instruments:
        for n in inst.notes:
            n.start = snap(n.start)
            n.end = max(n.start + q, snap(n.end))  # avoid zero-length
    midi.remove_invalid_notes()
    midi.write(str(out_mid))

def audio_to_pdf(audio_rel_path: str):
    audio = Path(audio_rel_path).expanduser().resolve()
    if not audio.exists():
        raise FileNotFoundError(f"Audio not found: {audio}")

    # ---- (1) Basic Pitch with stricter decoding -> "clean.mid"
    print("Running Basic Pitch with stricter thresholds…")
    model_output, _midi_unused, _events_unused = predict(
        str(audio),
        onset_threshold=ONSET_THRESHOLD,
        frame_threshold=FRAME_THRESHOLD,
        minimum_note_length=MIN_NOTE_LENGTH_SEC,
    )

    notes, _ = model_output_to_notes(
        model_output,
        onset_threshold=ONSET_THRESHOLD,
        frame_threshold=FRAME_THRESHOLD,
        minimum_note_length=MIN_NOTE_LENGTH_SEC,
        minimum_frequency=MIN_FREQ_HZ,
        maximum_frequency=MAX_FREQ_HZ,
        infer_onsets=True,
    )

    pm_obj = notes_to_midi(notes)
    clean_mid = audio.with_suffix(audio.suffix + ".clean.mid")
    pm_obj.write(str(clean_mid))
    print(f"Saved cleaned MIDI: {clean_mid}")

    # ---- (2) Quantize MIDI before engraving -> "clean_q.mid"
    clean_q_mid = audio.with_suffix(audio.suffix + ".clean_q.mid")
    print(f"Quantizing to grid={QUANT_GRID} at {TEMPO_BPM} bpm…")
    quantize_midi(clean_mid, clean_q_mid, QUANT_GRID, TEMPO_BPM)
    print(f"Saved quantized MIDI: {clean_q_mid}")

    # ---- MuseScore export to PDF
    mscore = find_musescore()
    pdf_path = audio.with_suffix(".pdf")  # final artifact
    print("Engraving with MuseScore…")
    subprocess.run([mscore, "-o", str(pdf_path), str(clean_q_mid)], check=True)
    print(f"Saved sheet music PDF: {pdf_path}")

if __name__ == "__main__":
    p = input("File path: ").strip()
    audio_to_pdf(p)