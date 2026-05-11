"""Direct speech-model inference debug runner.

This bypasses FastAPI so we can time the pure model path on CPU and verify
feature shapes before hitting the API.
"""

import argparse
import logging
import time
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.model import MERModel
from preprocessing import audio as audio_utils


LOG = logging.getLogger("mer.debug_inference")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("mer.preprocessing.audio").setLevel(logging.DEBUG)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio",
        default=r"data/TESS/TESS Toronto emotional speech set data/OAF_angry/OAF_back_angry.wav",
        help="Audio file to test",
    )
    parser.add_argument("--model_dir", default=r"models/speech_pipeline/results_test")
    parser.add_argument("--max_seconds", type=float, default=2.5, help="Load only the first N seconds to reduce input size")
    args = parser.parse_args(argv)

    LOG.info("TensorFlow version: %s", tf.__version__)
    LOG.info("GPUs visible: %s", tf.config.list_physical_devices("GPU"))
    LOG.info("CPUs visible: %s", tf.config.list_physical_devices("CPU"))
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        LOG.info("TensorFlow threading set to 1/1 for faster CPU debug runs")
    except Exception as exc:
        LOG.warning("Unable to set TensorFlow threading: %s", exc)

    model_dir = Path(args.model_dir)
    model_path = model_dir / "best_model.h5"
    label_encoder_path = model_dir / "label_encoder.joblib"
    config_path = model_dir / "config.json"

    model = MERModel(str(model_path), str(label_encoder_path), str(config_path))
    load_started = time.perf_counter()
    model.load()
    LOG.info("Model and artifacts loaded in %.3fs", time.perf_counter() - load_started)
    LOG.info("Loaded model input shape: %s", model.model.input_shape)
    LOG.info("Loaded config: %s", model.config)

    sample_path = Path(args.audio)
    if not sample_path.exists():
        raise FileNotFoundError(sample_path)

    sr = int(model.config.get("sr", 16000))
    feature = model.config.get("feature", "mfcc")
    top_db = int(model.config.get("top_db", 20))
    hop_length = int(model.config.get("hop_length", 512))
    n_mfcc = int(model.config.get("n_mfcc", 13))

    load_started = time.perf_counter()
    y, _ = audio_utils.load_audio(str(sample_path), sr=sr, mono=True, duration=args.max_seconds)
    LOG.info("Direct audio load completed in %.3fs; samples=%d", time.perf_counter() - load_started, len(y))

    trim_started = time.perf_counter()
    y = audio_utils.trim_silence(y, top_db=top_db)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    LOG.info("Direct trim+normalize completed in %.3fs; samples=%d", time.perf_counter() - trim_started, len(y))

    if feature != "mfcc":
        raise ValueError(f"This debug runner currently expects mfcc training, got feature={feature}")

    feat_started = time.perf_counter()
    mfcc = audio_utils.extract_mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    LOG.info("Direct MFCC extraction completed in %.3fs; raw_shape=%s", time.perf_counter() - feat_started, tuple(mfcc.shape))

    if mfcc.ndim == 2 and mfcc.shape[0] < mfcc.shape[1] and mfcc.shape[0] <= 64:
        mfcc = mfcc.T

    T, F = mfcc.shape
    max_frames = int(model.config.get("max_frames", 200))
    pad_started = time.perf_counter()
    if T > max_frames:
        mfcc = mfcc[:max_frames, :]
    else:
        mfcc = np.pad(mfcc, ((0, max_frames - T), (0, 0)), mode="constant", constant_values=0.0)
    LOG.info("Direct padding completed in %.3fs; final_shape=%s", time.perf_counter() - pad_started, tuple(mfcc.shape))

    inp = np.expand_dims(mfcc.astype(np.float32), axis=0)
    expected_shape = tuple(model.model.input_shape[1:])
    actual_shape = tuple(inp.shape[1:])
    LOG.info("Training-vs-inference shape check: expected=%s actual=%s", expected_shape, actual_shape)
    if expected_shape != actual_shape:
        raise ValueError(f"Input feature shape mismatch: expected {expected_shape}, got {actual_shape}")

    pred_started = time.perf_counter()
    probs = model.model.predict(inp, verbose=0)[0]
    LOG.info("Direct model.predict() completed in %.3fs", time.perf_counter() - pred_started)
    top_idx = int(np.argmax(probs))
    label = str(model.label_encoder.inverse_transform([top_idx])[0])
    LOG.info("Predicted label=%s confidence=%.4f", label, float(probs[top_idx]))
    LOG.info("Probabilities=%s", probs.tolist())


if __name__ == "__main__":
    main()