"""Load a trained model and print prediction probabilities for sample files.
"""
import argparse
from pathlib import Path
import joblib
import numpy as np
import logging
import tensorflow as tf

from models.speech_pipeline import data as data_utils

LOG = logging.getLogger('mer.predict_samples')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--label_encoder', required=False)
    parser.add_argument('--processed_dir', default='data/processed')
    parser.add_argument('--feature', choices=['mfcc','mel'], default='mfcc')
    parser.add_argument('--max_frames', type=int, default=200)
    parser.add_argument('--n_samples', type=int, default=5)
    args = parser.parse_args(argv)

    model = tf.keras.models.load_model(args.model)
    LOG.info('Model loaded')

    X, y, le, files = data_utils.load_features_and_labels(args.processed_dir, feature_type=args.feature, max_frames=args.max_frames)
    LOG.info(f'Loaded {len(X)} samples; will print predictions for first {args.n_samples}')

    preds = model.predict(X[:args.n_samples])
    for i in range(min(args.n_samples, len(preds))):
        probs = preds[i]
        top_idx = int(np.argmax(probs))
        label = le.inverse_transform([top_idx])[0]
        print(f"File: {files[i]}")
        print(f"  Predicted: {label} (idx={top_idx})")
        print(f"  Probabilities: {probs}")


if __name__ == '__main__':
    main()
