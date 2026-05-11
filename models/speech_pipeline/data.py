"""Data loading and preprocessing utilities for speech model.

This module loads precomputed features from `data/processed/features/{mfcc,mel}`
and metadata.csv created by `preprocessing.run`. It handles padding/truncation
to a fixed number of frames, label encoding, and creating train/val/test splits.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import logging
from typing import Tuple, List

LOG = logging.getLogger("mer.speech.data")


def load_metadata(processed_dir: str) -> pd.DataFrame:
    p = Path(processed_dir) / 'metadata.csv'
    if not p.exists():
        raise FileNotFoundError(f"Metadata not found at {p}; run preprocessing first")
    df = pd.read_csv(p)
    LOG.info(f"Loaded metadata with {len(df)} rows from {p}")
    return df


def _feature_path_from_audio(audio_path: str, processed_dir: str, feature_type: str) -> Path:
    # audio_path is absolute path to raw audio; features are saved by stem
    stem = Path(audio_path).stem
    feat_dir = Path(processed_dir) / 'features' / feature_type
    return feat_dir / f"{stem}.npy"


def pad_truncate(feature: np.ndarray, max_frames: int) -> np.ndarray:
    """Pad (with zeros) or truncate feature along time axis (frames).

    Expect feature shape: (n_features, T) or (T, n_features). Normalize to (T, n_features).
    """
    # Ensure shape is (T, F)
    if feature.ndim != 2:
        raise ValueError("Feature must be 2D array")
    # if shape is (F, T) (common for librosa), transpose
    if feature.shape[0] < feature.shape[1] and feature.shape[0] <= 64:
        # Heuristic: mfcc often 13 x T where 13 < T
        feature = feature.T

    T, F = feature.shape
    if T == max_frames:
        return feature
    if T > max_frames:
        return feature[:max_frames, :]
    # pad
    pad_width = ((0, max_frames - T), (0, 0))
    return np.pad(feature, pad_width, mode='constant', constant_values=0.0)


def load_features_and_labels(processed_dir: str, feature_type: str = 'mfcc', max_frames: int = 200) -> Tuple[np.ndarray, np.ndarray, LabelEncoder, List[str]]:
    """Load all features and labels. Returns X (N, T, F), y (N,), label_encoder, file_list.
    """
    df = load_metadata(processed_dir)
    X_list = []
    y_list = []
    file_list = []
    skipped = 0
    for _, row in df.iterrows():
        audio_path = row['audio_path']
        label = row.get('label', 'unknown')
        feat_path = _feature_path_from_audio(audio_path, processed_dir, feature_type)
        if not feat_path.exists():
            LOG.warning(f"Feature missing for {audio_path}: expected {feat_path}; skipping")
            skipped += 1
            continue
        arr = np.load(feat_path)
        # arr might be shape (F, T)
        arr = pad_truncate(arr, max_frames)
        X_list.append(arr.astype(np.float32))
        y_list.append(label)
        file_list.append(str(feat_path))

    if not X_list:
        raise RuntimeError("No features loaded. Check processed_dir and that features exist")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)

    LOG.info(f"Loaded features: X.shape={X.shape}; labels={len(y)}; skipped={skipped}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    return X, y_enc, le, file_list


def train_val_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42):
    # First split off test
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    # Then split train/val
    val_frac_of_trainval = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_frac_of_trainval, stratify=y_trainval, random_state=random_state)
    LOG.info(f"Split data: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_label_encoder(le: LabelEncoder, out_path: str):
    joblib.dump(le, out_path)
    LOG.info(f"Saved label encoder to {out_path}")
