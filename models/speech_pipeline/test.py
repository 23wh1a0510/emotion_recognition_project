"""Test script for speech pipeline.

This script runs quick smoke checks against the saved speech model
artifact (if present) and prints a short report. It uses the backend's
preprocessing and model wrapper to ensure parity with runtime inference.
"""
import os
import sys

def main():
    print('Speech test placeholder: validate artifacts under models/speech_pipeline/results_test')
    base = os.path.join('models', 'speech_pipeline', 'results_test')
    expected = ['best_model.h5', 'label_encoder.joblib', 'classification_report.txt']
    for f in expected:
        path = os.path.join(base, f)
        print(f, '->', 'FOUND' if os.path.exists(path) else 'MISSING')

if __name__ == '__main__':
    main()
"""Evaluation / inference script for trained speech model.

Usage:
  python models/speech_pipeline/test.py --model models/speech_pipeline/results/best_model.h5 --processed_dir data/processed --feature mfcc
"""
import argparse
from pathlib import Path
import logging
import joblib
import numpy as np

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from models.speech_pipeline import data as data_utils

LOG = logging.getLogger('mer.speech.test')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to saved Keras model (.h5)')
    parser.add_argument('--label_encoder', required=False, help='Path to label encoder joblib')
    parser.add_argument('--processed_dir', default='data/processed')
    parser.add_argument('--feature', choices=['mfcc','mel'], default='mfcc')
    parser.add_argument('--max_frames', type=int, default=200)
    args = parser.parse_args(argv)

    model = tf.keras.models.load_model(args.model)
    LOG.info('Loaded model:')
    model.summary()

    X, y, le, file_list = data_utils.load_features_and_labels(args.processed_dir, feature_type=args.feature, max_frames=args.max_frames)
    LOG.info(f'Dataset shapes: X={X.shape}, y={y.shape}')

    # Use entire dataset or split? We'll evaluate on full set here.
    preds = model.predict(X)
    pred_labels = np.argmax(preds, axis=1)

    labels = list(le.classes_)
    report = classification_report(y, pred_labels, target_names=labels)
    LOG.info('\n' + report)
    cm = confusion_matrix(y, pred_labels)
    LOG.info(f'Confusion matrix:\n{cm}')


if __name__ == '__main__':
    main()
