"""Audio preprocessing and feature extraction utilities.

Functions provided:
- load_audio(path, sr=16000): load, resample, convert to mono
- trim_silence(y): trim leading/trailing silence
- normalize_audio(y): peak normalization
- extract_mfcc(y, sr): returns mfcc (n_mfcc x T)
- extract_log_mel(y, sr): returns log-mel (n_mels x T)
"""
from pathlib import Path
import time
from math import gcd
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import resample_poly
import logging

LOG = logging.getLogger("mer.preprocessing.audio")


def load_audio(path: str, sr: int = 16000, mono: bool = True, duration: float | None = None):
    """Load audio file with librosa, resample to sr and convert to mono.

    Returns: y (np.ndarray), sr (int)
    """
    try:
        read_started = time.perf_counter()
        y, read_sr = sf.read(path, dtype="float32", always_2d=False)
        LOG.debug("soundfile read completed in %.3fs; path=%s; samples=%d; sr=%d", time.perf_counter() - read_started, path, len(y), read_sr)
        if mono and y.ndim > 1:
            y = np.mean(y, axis=1)
        if duration is not None:
            max_samples = max(1, int(duration * read_sr))
            y = y[:max_samples]
        if read_sr != sr:
            resample_started = time.perf_counter()
            rate_gcd = gcd(int(read_sr), int(sr))
            up = int(sr // rate_gcd)
            down = int(read_sr // rate_gcd)
            y = resample_poly(y, up=up, down=down).astype(np.float32, copy=False)
            LOG.debug("soundfile resample completed in %.3fs; target_sr=%d", time.perf_counter() - resample_started, sr)
        LOG.debug(f"Loaded {path} with soundfile (orig_sr={read_sr} -> sr={sr}) len={len(y)}")
        return y.astype(np.float32, copy=False), sr
    except Exception as exc:
        LOG.debug(f"soundfile load failed for {path} ({exc}); falling back to librosa.load")
        y, orig_sr = librosa.load(path, sr=sr, mono=mono, duration=duration)
        LOG.debug(f"Loaded {path} (orig_sr={orig_sr} -> sr={sr}) len={len(y)}")
        return y.astype(np.float32, copy=False), sr


def trim_silence(y: np.ndarray, top_db: int = 20):
    if y.size == 0:
        return y
    peak = float(np.max(np.abs(y)))
    if peak <= 0.0:
        return y
    threshold = peak * (10.0 ** (-float(top_db) / 20.0))
    non_silent = np.flatnonzero(np.abs(y) > threshold)
    if non_silent.size == 0:
        LOG.debug(f"Trimmed silence: from {len(y)} to {len(y)} samples")
        return y
    start = int(non_silent[0])
    end = int(non_silent[-1]) + 1
    yt = y[start:end]
    LOG.debug(f"Trimmed silence: from {len(y)} to {len(yt)} samples")
    return yt


def normalize_audio(y: np.ndarray):
    maxv = np.max(np.abs(y))
    if maxv > 0:
        y = y / maxv
    return y


def extract_mfcc(y: np.ndarray, sr: int, n_mfcc: int = 13, hop_length: int = 512):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    LOG.debug(f"MFCC shape: {mfcc.shape}")
    return mfcc


def extract_log_mel(y: np.ndarray, sr: int, n_mels: int = 64, n_fft: int = 2048, hop_length: int = 512):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    logS = librosa.power_to_db(S, ref=np.max)
    LOG.debug(f"Log-mel shape: {logS.shape}")
    return logS
