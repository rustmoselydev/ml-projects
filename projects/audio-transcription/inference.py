import subprocess
from pathlib import Path
import numpy as np
import pretty_midi as pm
from basic_pitch.inference import predict

# ---- Tunables ----
ONSET_THRESHOLD = 0.6
FRAME_THRESHOLD = 0.3
MIN_NOTE_LENGTH_SEC = 0.08
QUANT_GRID = 1/8
TEMPO_BPM = 120
# Optional: constrain range (MIDI notes); set to None to disable
MIN_MIDI = None
MAX_MIDI = None
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
            n.end = max(n.start + q, snap(n.end))  # avoid zero-length
    midi.remove_invalid_notes()
    midi.write(str(out_mid))

def normalize_note_events(note_events):
    """Return list of (onset, offset, midi, confidence) as floats/ints."""
    norm = []
    for ev in note_events:
        if isinstance(ev, dict):
            onset  = ev.get("onset", ev.get("onset_time", ev.get("start", ev.get("start_time"))))
            offset = ev.get("offset", ev.get("offset_time", ev.get("end", ev.get("end_time"))))
            pitch  = ev.get("pitch", ev.get("midi_pitch", ev.get("note", ev.get("note_number"))))
            conf   = ev.get("confidence", ev.get("amplitude", 0.8))
        elif isinstance(ev, (list, tuple)):
            # handle [onset, offset, pitch] or [onset, offset, pitch, conf, ...]
            onset, offset, pitch = ev[0], ev[1], ev[2]
            conf = ev[3] if len(ev) >= 4 else 0.9
        else:
            # unknown shape; skip
            continue

        if onset is None or offset is None or pitch is None:
            continue

        # cast & clamp
        onset = float(onset)
        offset = float(offset)
        midi = int(round(pitch))
        conf = float(conf)
        if MIN_MIDI is not None and midi < MIN_MIDI: 
            continue
        if MAX_MIDI is not None and midi > MAX_MIDI: 
            continue
        if offset <= onset:
            continue

        norm.append((onset, offset, midi, max(0, min(1, conf))))
    return norm

def note_events_to_midi(note_events, out_path: Path):
    midi = pm.PrettyMIDI()
    inst = pm.Instrument(program=0)  # Acoustic Grand Piano
    for onset, offset, midi_pitch, confidence in normalize_note_events(note_events):
        inst.notes.append(pm.Note(
            velocity=int(40 + confidence * 87),  # 40..127
            pitch=int(midi_pitch),
            start=float(onset),
            end=float(offset),
        ))
    midi.instruments.append(inst)
    midi.write(str(out_path))

def audio_to_pdf(audio_rel_path: str):
    audio = Path(audio_rel_path).expanduser().resolve()
    if not audio.exists():
        raise FileNotFoundError(f"Audio not found: {audio}")

    # 1) Basic Pitch with stricter thresholds
    print("Running Basic Pitch with stricter thresholds…")
    model_output, _midi_unused, note_events = predict(
        str(audio),
        onset_threshold=ONSET_THRESHOLD,
        frame_threshold=FRAME_THRESHOLD,
        minimum_note_length=MIN_NOTE_LENGTH_SEC,
    )

    clean_mid = audio.with_suffix(audio.suffix + ".clean.mid")
    note_events_to_midi(note_events, clean_mid)
    print(f"Saved cleaned MIDI: {clean_mid}")

    # 2) Quantize
    clean_q_mid = audio.with_suffix(audio.suffix + ".clean_q.mid")
    print(f"Quantizing to grid={QUANT_GRID} at {TEMPO_BPM} bpm…")
    quantize_midi(clean_mid, clean_q_mid)
    print(f"Saved quantized MIDI: {clean_q_mid}")

    # 3) MuseScore -> PDF
    mscore = find_musescore()
    pdf_path = audio.with_suffix(".pdf")
    print("Engraving with MuseScore…")
    subprocess.run([mscore, "-o", str(pdf_path), str(clean_q_mid)], check=True)
    print(f"Saved sheet music PDF: {pdf_path}")

if __name__ == "__main__":
    p = input("File path: ").strip()
    audio_to_pdf(p)