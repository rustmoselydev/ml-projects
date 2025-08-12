import subprocess
from pathlib import Path
import numpy as np
import pretty_midi as pm
from basic_pitch.inference import predict

# ---- Tunables ----
ONSET_THRESHOLD = 0.6         # stricter onsets => fewer spurious notes
FRAME_THRESHOLD = 0.3
MIN_NOTE_LENGTH_SEC = 0.08    # merge tiny blips
QUANT_GRID = 1/8              # quantize to 8th-notes
TEMPO_BPM = 120               # for quantization grid in seconds
# -------------------

CANDIDATES = [
    "mscore4portable", "musescore4", "mscore4",
    "mscore", "musescore",
    r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe",
]

def find_musescore():
    for c in CANDIDATES:
        try:
            subprocess.run([c, "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            return c
        except Exception:
            pass
    raise FileNotFoundError("MuseScore CLI not found in PATH.")

def quantize_midi(in_mid: Path, out_mid: Path, grid_fraction=QUANT_GRID, tempo_bpm=TEMPO_BPM):
    midi = pm.PrettyMIDI(str(in_mid))
    sec_per_beat = 60.0 / float(tempo_bpm)
    q = float(grid_fraction) * sec_per_beat

    def snap(x): return np.round(x / q) * q

    for inst in midi.instruments:
        for n in inst.notes:
            n.start = snap(n.start)
            n.end = max(n.start + q, snap(n.end))
    midi.remove_invalid_notes()
    midi.write(str(out_mid))

def note_events_to_midi(note_events, out_path: Path):
    """Build PrettyMIDI from Basic Pitch note_events list."""
    midi = pm.PrettyMIDI()
    inst = pm.Instrument(program=0)  # Acoustic Grand Piano
    for onset, offset, pitch, confidence in note_events:
        inst.notes.append(pm.Note(
            velocity=int(confidence * 127),
            pitch=int(pitch),
            start=float(onset),
            end=float(offset)
        ))
    midi.instruments.append(inst)
    midi.write(str(out_path))

def audio_to_pdf(audio_rel_path: str):
    audio = Path(audio_rel_path).expanduser().resolve()
    if not audio.exists():
        raise FileNotFoundError(f"Audio not found: {audio}")

    # ---- Step 1: Basic Pitch with stricter thresholds
    print("Running Basic Pitch with stricter thresholds…")
    model_output, midi_data, note_events = predict(
        str(audio),
        onset_threshold=ONSET_THRESHOLD,
        frame_threshold=FRAME_THRESHOLD,
        minimum_note_length=MIN_NOTE_LENGTH_SEC,
    )

    clean_mid = audio.with_suffix(audio.suffix + ".clean.mid")
    note_events_to_midi(note_events, clean_mid)
    print(f"Saved cleaned MIDI: {clean_mid}")

    # ---- Step 2: Quantize
    clean_q_mid = audio.with_suffix(audio.suffix + ".clean_q.mid")
    print(f"Quantizing to grid={QUANT_GRID} at {TEMPO_BPM} bpm…")
    quantize_midi(clean_mid, clean_q_mid)
    print(f"Saved quantized MIDI: {clean_q_mid}")

    # ---- Step 3: MuseScore export
    mscore = find_musescore()
    pdf_path = audio.with_suffix(".pdf")
    print("Engraving with MuseScore…")
    subprocess.run([mscore, "-o", str(pdf_path), str(clean_q_mid)], check=True)
    print(f"Saved sheet music PDF: {pdf_path}")

if __name__ == "__main__":
    p = input("File path: ").strip()
    audio_to_pdf(p)