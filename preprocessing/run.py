"""Command-line preprocessing runner.

Usage example:
  python -m preprocessing.run --dataset ./data/raw/TESS --out ./data/processed --sr 16000

This script scans the dataset, creates metadata.csv, processes audio files
to extract MFCC and log-mel features, cleans transcripts if available, and
saves outputs under the provided output directory.
"""
import argparse
from pathlib import Path
import logging
import sys
import csv

import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocessing import utils
from preprocessing import audio as audio_utils
from preprocessing import text as text_utils

LOG = logging.getLogger("mer.preprocessing.run")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def process(args):
    ds_root = Path(args.dataset)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_csv = out_dir / 'metadata.csv'
    num_files, _ = utils.scan_dataset(str(ds_root), str(metadata_csv))
    LOG.info(f"Found {num_files} audio files. Metadata written to {metadata_csv}")

    df = pd.read_csv(metadata_csv)

    # prepare folders
    feat_mfcc_dir = out_dir / 'features' / 'mfcc'
    feat_mel_dir = out_dir / 'features' / 'mel'
    texts_dir = out_dir / 'texts'
    feat_mfcc_dir.mkdir(parents=True, exist_ok=True)
    feat_mel_dir.mkdir(parents=True, exist_ok=True)
    texts_dir.mkdir(parents=True, exist_ok=True)

    skipped = []

    sample_shapes = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Processing'):
        audio_path = Path(row['audio_path'])
        label = row.get('label', 'unknown')
        transcript_path = row.get('transcript_path', '')

        try:
            y, sr = audio_utils.load_audio(str(audio_path), sr=args.sr)
            y = audio_utils.trim_silence(y, top_db=args.top_db)
            y = audio_utils.normalize_audio(y)

            mfcc = audio_utils.extract_mfcc(y, sr, n_mfcc=args.n_mfcc, hop_length=args.hop_length)
            mel = audio_utils.extract_log_mel(y, sr, n_mels=args.n_mels, n_fft=args.n_fft, hop_length=args.hop_length)

            # save features
            base = audio_path.stem
            mfcc_path = feat_mfcc_dir / f"{base}.npy"
            mel_path = feat_mel_dir / f"{base}.npy"
            np.save(mfcc_path, mfcc.astype(np.float32))
            np.save(mel_path, mel.astype(np.float32))

            sample_shapes.append({'audio': str(audio_path), 'mfcc_shape': mfcc.shape, 'mel_shape': mel.shape, 'label': label})

            # process transcript if exists
            if isinstance(transcript_path, str) and transcript_path:
                try:
                    with open(transcript_path, 'r', encoding='utf8') as f:
                        txt = f.read()
                except Exception:
                    LOG.exception(f"Failed to read transcript {transcript_path}")
                    txt = ''
                cleaned = text_utils.clean_text(txt)
                tokens = text_utils.tokenize(cleaned)
                out_txt = texts_dir / f"{base}.clean.txt"
                with out_txt.open('w', encoding='utf8') as f:
                    f.write(cleaned)
            else:
                cleaned = ''

        except Exception:
            LOG.exception(f"Failed processing {audio_path}; skipping")
            skipped.append(str(audio_path))

    # write simple summary CSV for sample shapes
    shapes_df = pd.DataFrame(sample_shapes)
    shapes_csv = out_dir / 'sample_feature_shapes.csv'
    if not shapes_df.empty:
        shapes_df.to_csv(shapes_csv, index=False)
        LOG.info(f"Saved sample feature shapes to {shapes_csv}")

    LOG.info(f"Processing complete. Processed: {len(sample_shapes)}. Skipped: {len(skipped)}")
    if skipped:
        LOG.info("Skipped files:\n" + "\n".join(skipped[:20]))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Preprocess TESS dataset: extract features and clean transcripts')
    parser.add_argument('--dataset', required=True, help='Path to raw dataset folder (contains audio files)')
    parser.add_argument('--out', default='data/processed', help='Output folder for processed data')
    parser.add_argument('--sr', type=int, default=16000, help='Target sampling rate')
    parser.add_argument('--top_db', type=int, default=20, help='Silence trim threshold in dB')
    parser.add_argument('--n_mfcc', type=int, default=13, help='Number of MFCC coefficients')
    parser.add_argument('--n_mels', type=int, default=64, help='Number of mel bins')
    parser.add_argument('--n_fft', type=int, default=2048, help='FFT window size')
    parser.add_argument('--hop_length', type=int, default=512, help='Hop length for spectrograms')
    args = parser.parse_args(argv)

    process(args)


if __name__ == '__main__':
    main()
