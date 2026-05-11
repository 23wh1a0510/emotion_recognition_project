"""Model loading and inference utilities for FastAPI backend.

This module loads a Keras model and LabelEncoder (joblib) and provides an
`infer_audio` function that mirrors preprocessing used during training.
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import joblib
import tensorflow as tf

from preprocessing import audio as audio_utils
from preprocessing import text as text_utils

LOG = logging.getLogger("mer_backend.model")


class MERModel:
    def __init__(self, model_path: str, label_encoder_path: str, config_path: str):
        self.model_path = Path(model_path)
        self.label_encoder_path = Path(label_encoder_path)
        self.config_path = Path(config_path)
        self.model = None
        self.label_encoder = None
        self.config = None

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not self.label_encoder_path.exists():
            raise FileNotFoundError(f"Label encoder not found: {self.label_encoder_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        LOG.info(f"Loading Keras model from {self.model_path}")
        self.model = tf.keras.models.load_model(str(self.model_path))
        LOG.info("Loading label encoder")
        self.label_encoder = joblib.load(str(self.label_encoder_path))
        LOG.info("Loading config")
        with open(self.config_path, 'r', encoding='utf8') as fh:
            self.config = json.load(fh)

    def infer_audio(self, audio_path: str) -> Dict[str, Any]:
        """Run preprocessing + inference for a single audio file.

        Returns a dict with predicted label, confidence, and full probability map.
        """
        start_total = time.perf_counter()
        # Preprocessing params — ensure these match training config
        max_frames = int(self.config.get('max_frames', 200))
        feature = self.config.get('feature', 'mfcc')
        top_db = int(self.config.get('top_db', 20))

        # Load audio with same SR used during preprocessing/run
        sr = int(self.config.get('sr', 16000)) if 'sr' in self.config else 16000
        LOG.info(f"Inference starting for {audio_path}; feature={feature}; max_frames={max_frames}; sr={sr}")
        audio_started = time.perf_counter()
        y, _ = audio_utils.load_audio(audio_path, sr=sr, mono=True)
        y = audio_utils.trim_silence(y, top_db=top_db)
        y = audio_utils.normalize_audio(y)
        LOG.info(
            "Audio loaded and normalized in %.3fs; samples=%d",
            time.perf_counter() - audio_started,
            len(y),
        )

        # Extract feature
        feature_started = time.perf_counter()
        if feature == 'mfcc':
            n_mfcc = int(self.config.get('n_mfcc', 13))
            hop_length = int(self.config.get('hop_length', 512))
            arr = audio_utils.extract_mfcc(y, sr, n_mfcc=n_mfcc, hop_length=hop_length)
        else:
            n_mels = int(self.config.get('n_mels', 64))
            hop_length = int(self.config.get('hop_length', 512))
            n_fft = int(self.config.get('n_fft', 2048))
            arr = audio_utils.extract_log_mel(y, sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        LOG.info("Feature extraction completed in %.3fs; raw_shape=%s", time.perf_counter() - feature_started, tuple(arr.shape))

        # Ensure shape (T, F)
        if arr.ndim == 2 and arr.shape[0] < arr.shape[1] and arr.shape[0] <= 64:
            arr = arr.T

        T, F = arr.shape
        LOG.info(f"Extracted feature shape (T,F) = ({T},{F}) for {audio_path}")

        # Pad/truncate
        pad_started = time.perf_counter()
        if T > max_frames:
            in_arr = arr[:max_frames, :]
        else:
            pad_width = ((0, max_frames - T), (0, 0))
            in_arr = np.pad(arr, pad_width, mode='constant', constant_values=0.0)
        LOG.info("Feature padding completed in %.3fs; final_shape=%s", time.perf_counter() - pad_started, tuple(in_arr.shape))

        # Model expects shape (1, T, F)
        inp = np.expand_dims(in_arr.astype(np.float32), axis=0)
        expected_shape = tuple(self.model.input_shape[1:])
        actual_shape = tuple(inp.shape[1:])
        LOG.info("Model input shape check: expected=%s actual=%s", expected_shape, actual_shape)
        if expected_shape != actual_shape:
            raise ValueError(f"Input feature shape mismatch: expected {expected_shape}, got {actual_shape}")

        predict_started = time.perf_counter()
        LOG.info("model.predict() started")
        probs = self.model.predict(inp, verbose=0)[0]
        LOG.info("model.predict() completed in %.3fs", time.perf_counter() - predict_started)
        top_idx = int(np.argmax(probs))
        label = str(self.label_encoder.inverse_transform([top_idx])[0])

        # Map probabilities to label names
        label_names = [str(x) for x in list(self.label_encoder.classes_)]
        prob_map = {name: float(probs[i]) for i, name in enumerate(label_names)}

        return {
            'predicted_emotion': label,
            'confidence': float(probs[top_idx]),
            'all_probabilities': prob_map,
            'feature_shape': [int(T), int(F)],
            'input_shape': [int(actual_shape[0]), int(actual_shape[1])],
            'elapsed_seconds': round(time.perf_counter() - start_total, 4),
        }


class TextEmotionModel:
    """Runtime wrapper for text emotion inference artifacts."""

    def __init__(self, artifact_path: str):
        self.artifact_path = Path(artifact_path)
        self.vectorizer = None
        self.classifier = None
        self.label_encoder = None
        self.config = None

    def load(self):
        if not self.artifact_path.exists():
            raise FileNotFoundError(f"Text model artifact not found: {self.artifact_path}")

        LOG.info("Loading text model artifact from %s", self.artifact_path)
        artifact = joblib.load(str(self.artifact_path))
        self.vectorizer = artifact.get('vectorizer')
        self.classifier = artifact.get('classifier')
        self.label_encoder = artifact.get('label_encoder')
        self.config = artifact.get('config', {})

        if self.vectorizer is None or self.classifier is None or self.label_encoder is None:
            raise ValueError("Text artifact missing vectorizer/classifier/label_encoder")

    def infer_text(self, text: str) -> Dict[str, Any]:
        start_total = time.perf_counter()
        cleaned = text_utils.clean_text(text)
        tokens = text_utils.tokenize(cleaned)
        if not cleaned:
            raise ValueError("Text is empty after preprocessing")

        vector_started = time.perf_counter()
        x = self.vectorizer.transform([cleaned])
        LOG.info("Text vectorization completed in %.3fs", time.perf_counter() - vector_started)

        predict_started = time.perf_counter()
        probs = self.classifier.predict_proba(x)[0]
        LOG.info("Text model predict_proba completed in %.3fs", time.perf_counter() - predict_started)

        top_idx = int(np.argmax(probs))
        predicted_label = str(self.label_encoder.inverse_transform([top_idx])[0])
        label_names = [str(x) for x in list(self.label_encoder.classes_)]
        prob_map = {name: float(probs[i]) for i, name in enumerate(label_names)}

        return {
            'predicted_emotion': predicted_label,
            'confidence': float(probs[top_idx]),
            'all_probabilities': prob_map,
            'cleaned_text': cleaned,
            'token_count': len(tokens),
            'elapsed_seconds': round(time.perf_counter() - start_total, 4),
        }
