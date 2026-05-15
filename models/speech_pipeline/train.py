"""Train script placeholder for speech pipeline.

This wrapper documents how to train the speech model and provides a
convenient CLI entrypoint for automated runs. The original training
code (if present) should be called from here. For now this file simply
prints instructions so it is safe to include in the deliverable structure.
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description='Speech pipeline trainer (placeholder)')
    parser.add_argument('--out_dir', default='models/speech_pipeline/results_test', help='Output directory for artifacts')
    args = parser.parse_args()

    print('Speech training placeholder. Replace this with your training code.')
    print('Suggested: call the original training script and save artifacts to', args.out_dir)


if __name__ == '__main__':
    main()
"""Training script for speech-only emotion recognition model.

Usage:
  python models/speech_pipeline/train.py --processed_dir data/processed --feature mfcc
"""
import argparse
from pathlib import Path
import logging
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

from models.speech_pipeline import data as data_utils
from models.speech_pipeline import model as model_builder
from sklearn.metrics import confusion_matrix, classification_report

LOG = logging.getLogger('mer.speech.train')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def plot_history(history, out_dir: Path):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'training_curves.png')
    plt.close()


def plot_confusion(y_true, y_pred, labels, out_dir: Path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.title('Confusion Matrix')
    plt.savefig(out_dir / 'confusion_matrix.png')
    plt.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_dir', default='data/processed', help='Processed data folder (metadata.csv and features)')
    parser.add_argument('--feature', choices=['mfcc','mel'], default='mfcc')
    parser.add_argument('--max_frames', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--out_dir', default='models/speech_pipeline/results')
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load features and labels
    X, y, le, file_list = data_utils.load_features_and_labels(args.processed_dir, feature_type=args.feature, max_frames=args.max_frames)
    LOG.info(f"Dataset shapes: X={X.shape}, y={y.shape}")
    LOG.info(f"Label classes: {list(le.classes_)}")
    # Save label encoder
    le_path = out_dir / 'label_encoder.joblib'
    data_utils.save_label_encoder(le, str(le_path))

    # Train/val/test split
    X_train, X_val, X_test, y_train, y_val, y_test = data_utils.train_val_test_split(X, y, test_size=0.15, val_size=0.15)

    # Build model
    input_shape = X_train.shape[1], X_train.shape[2]
    num_classes = len(le.classes_)
    model = model_builder.build_model(input_shape, num_classes)
    LOG.info(model.summary())
    # Callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint(str(out_dir / 'best_model.h5'), monitor='val_loss', save_best_only=True)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    csv_logger = tf.keras.callbacks.CSVLogger(str(out_dir / 'training_log.csv'))

    # Fit
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[checkpoint, earlystop, csv_logger]
    )

    # Save final model and config
    model.save(out_dir / 'final_model.h5')
    with open(out_dir / 'config.json', 'w') as fh:
        json.dump(
            {
                'processed_dir': args.processed_dir,
                'feature': args.feature,
                'max_frames': args.max_frames,
                'sr': 16000,
                'top_db': 20,
                'hop_length': 512,
                'n_mfcc': 13,
                'n_mels': 64,
                'n_fft': 2048,
            },
            fh,
            indent=2,
        )

    # Plots
    plot_history(history, out_dir)

    # Evaluate on test set
    preds = model.predict(X_test)
    pred_labels = np.argmax(preds, axis=1)
    LOG.info('Sample prediction probabilities (first 5):')
    for i in range(min(5, len(preds))):
        LOG.info(f"File: {file_list[i]}, probs: {preds[i].tolist()}")

    labels = list(le.classes_)
    report = classification_report(y_test, pred_labels, target_names=labels)
    LOG.info('\n' + report)
    with open(out_dir / 'classification_report.txt', 'w') as fh:
        fh.write(report)

    plot_confusion(y_test, pred_labels, labels, out_dir)

    LOG.info(f"Training complete. Results in {out_dir}")


if __name__ == '__main__':
    main()
