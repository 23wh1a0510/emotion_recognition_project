"""Sanity check utilities after preprocessing.

Prints metadata summary, label distribution, sample feature shapes and value ranges.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import logging

LOG = logging.getLogger('mer.sanity')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_dir', default='data/processed')
    parser.add_argument('--feature', choices=['mfcc','mel'], default='mfcc')
    parser.add_argument('--max_samples', type=int, default=5)
    args = parser.parse_args(argv)

    processed = Path(args.processed_dir)
    meta = processed / 'metadata.csv'
    if not meta.exists():
        LOG.error(f"Metadata not found at {meta}")
        return

    df = pd.read_csv(meta)
    total = len(df)
    print(f"Total samples in metadata: {total}")
    print("Label distribution:")
    print(df['label'].value_counts())

    shapes_file = processed / 'sample_feature_shapes.csv'
    if shapes_file.exists():
        s_df = pd.read_csv(shapes_file)
        print(f"Sample feature shapes (first {args.max_samples}):")
        print(s_df.head(args.max_samples))
    else:
        print("No sample_feature_shapes.csv found. Will inspect raw .npy files.")
        feat_dir = processed / 'features' / args.feature
        files = list(feat_dir.glob('*.npy'))
        print(f"Found {len(files)} feature files in {feat_dir}")
        for f in files[:args.max_samples]:
            arr = np.load(f)
            print(f"{f.name}: shape={arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")


if __name__ == '__main__':
    main()
