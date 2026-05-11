"""Dataset scanning and metadata utilities.

Functions here scan a dataset directory for audio files, try to infer
emotion labels from folder names or filenames, and generate a `metadata.csv`
that lists all samples with associated labels and optional transcripts.
"""
from pathlib import Path
import csv
import re
from typing import List, Optional, Tuple

KNOWN_EMOTIONS = [
    "angry",
    "anger",
    "disgust",
    "fear",
    "happy",
    "pleasant_surprise",
    "surprise",
    "sad",
    "neutral",
]

AUDIO_EXT = {"wav", "mp3", "m4a", "flac", "ogg"}


def _find_audio_files(root: Path) -> List[Path]:
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower().lstrip('.') in AUDIO_EXT:
            files.append(p)
    return sorted(files)


def _infer_emotion_from_path(p: Path) -> Optional[str]:
    # check parent folders first
    parts = [s.lower() for s in p.parts]
    for part in parts[::-1]:
        for emo in KNOWN_EMOTIONS:
            if emo in part:
                # normalize common variants
                if emo == 'anger':
                    return 'angry'
                if emo == 'pleasant_surprise' or emo == 'surprise':
                    return 'surprise'
                return emo

    # fallback: look into filename tokens
    name = p.stem.lower()
    tokens = re.split(r'[^a-zA-Z]+', name)
    for t in tokens:
        for emo in KNOWN_EMOTIONS:
            if emo in t:
                if emo == 'anger':
                    return 'angry'
                if emo == 'pleasant_surprise' or emo == 'surprise':
                    return 'surprise'
                return emo
    return None


def scan_dataset(root_dir: str, out_csv: str) -> Tuple[int, Path]:
    """Scan a dataset directory and write metadata CSV.

    CSV columns: audio_path, label, transcript_path (optional)
    Returns: (num_files, Path(out_csv))
    """
    root = Path(root_dir)
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    audio_files = _find_audio_files(root)

    with out.open('w', newline='', encoding='utf8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['audio_path', 'label', 'transcript_path'])
        for p in audio_files:
            label = _infer_emotion_from_path(p) or 'unknown'
            # try to find transcript with same stem
            transcript = p.with_suffix('.txt')
            if not transcript.exists():
                transcript = ''
            else:
                transcript = str(transcript.resolve())
            writer.writerow([str(p.resolve()), label, transcript])

    return (len(audio_files), out.resolve())
