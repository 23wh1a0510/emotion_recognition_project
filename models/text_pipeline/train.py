"""Training script for real text emotion classification.

Chosen approach: TF-IDF + Logistic Regression
Reason: stable, fast on CPU, and reliable for a college demo.

It also logs full preprocessing artifacts:
- text cleaning
- tokenization
- vocabulary creation
- sequence tokenization + padding metadata
- label encoding

Usage:
  python models/text_pipeline/train.py --out_dir models/text_pipeline/results_test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import tokenizer_from_json

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.text_pipeline import data as data_utils

LOG = logging.getLogger("mer.text.train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def plot_confusion(y_true, y_pred, labels, out_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title("Text Emotion Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="models/text_pipeline/results_test")
    parser.add_argument("--samples_per_emotion", type=int, default=240)
    parser.add_argument("--max_features", type=int, default=7000)
    parser.add_argument("--max_words", type=int, default=8000)
    parser.add_argument("--max_len", type=int, default=24)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build labeled text dataset
    df = data_utils.build_synthetic_dataset(
        samples_per_emotion=args.samples_per_emotion,
        random_state=args.seed,
    )

    # 2) Text preprocessing + tokenization
    cleaned_texts, tokenized = data_utils.preprocess_texts(df["text"].tolist())
    token_strings = [" ".join(tokens) for tokens in tokenized]

    # 3) Sequence tokenizer/vocabulary/padding artifacts (for reproducibility)
    seq_artifacts = data_utils.build_sequence_artifacts(
        cleaned_texts,
        max_words=args.max_words,
        max_len=args.max_len,
    )

    # 4) Label encoding
    le = LabelEncoder()
    y = le.fit_transform(df["label"].to_numpy())

    # Save preprocessed dataset snapshot
    dataset_snapshot = pd.DataFrame(
        {
            "raw_text": df["text"],
            "cleaned_text": cleaned_texts,
            "tokens": [" ".join(toks) for toks in tokenized],
            "label": df["label"],
        }
    )
    dataset_snapshot.to_csv(out_dir / "training_dataset_snapshot.csv", index=False)

    x_train, x_val, x_test, y_train, y_val, y_test = data_utils.split_data(
        token_strings,
        y,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed,
    )

    # 5) TF-IDF vocabulary + vectorization
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=args.max_features,
        sublinear_tf=True,
    )
    x_train_vec = vectorizer.fit_transform(x_train)
    x_val_vec = vectorizer.transform(x_val)
    x_test_vec = vectorizer.transform(x_test)

    # 6) Classifier
    clf = LogisticRegression(
        max_iter=600,
        random_state=args.seed,
        class_weight="balanced",
        multi_class="auto",
    )
    clf.fit(x_train_vec, y_train)

    # 7) Validation / testing
    y_val_pred = clf.predict(x_val_vec)
    y_test_pred = clf.predict(x_test_vec)

    val_acc = float(accuracy_score(y_val, y_val_pred))
    test_acc = float(accuracy_score(y_test, y_test_pred))

    labels = list(le.classes_)
    report = classification_report(y_test, y_test_pred, target_names=labels)
    LOG.info("Validation accuracy: %.4f", val_acc)
    LOG.info("Test accuracy: %.4f", test_acc)
    LOG.info("\n%s", report)

    # 8) Save logs + metrics
    pd.DataFrame(
        [
            {"split": "validation", "accuracy": val_acc},
            {"split": "test", "accuracy": test_acc},
        ]
    ).to_csv(out_dir / "training_log.csv", index=False)

    with (out_dir / "classification_report.txt").open("w", encoding="utf8") as fh:
        fh.write(report)

    plot_confusion(y_test, y_test_pred, labels, out_dir / "confusion_matrix.png")

    vocab_terms = vectorizer.get_feature_names_out()
    with (out_dir / "vocabulary.txt").open("w", encoding="utf8") as fh:
        for term in vocab_terms:
            fh.write(term + "\n")

    # Save sequence tokenizer artifacts
    with (out_dir / "tokenizer.json").open("w", encoding="utf8") as fh:
        fh.write(seq_artifacts.tokenizer.to_json())

    sequence_debug = {
        "max_len": seq_artifacts.max_len,
        "vocab_size": seq_artifacts.vocab_size,
        "sequence_shape": list(seq_artifacts.padded_sequences.shape),
        "example_first_sequence": seq_artifacts.padded_sequences[0].tolist(),
    }
    with (out_dir / "sequence_debug.json").open("w", encoding="utf8") as fh:
        json.dump(sequence_debug, fh, indent=2)

    # Save inference artifact used by backend
    artifact = {
        "vectorizer": vectorizer,
        "classifier": clf,
        "label_encoder": le,
        "config": {
            "model_type": "tfidf_logreg",
            "max_features": args.max_features,
            "max_words": args.max_words,
            "max_len": args.max_len,
            "classes": labels,
        },
    }
    joblib.dump(artifact, out_dir / "text_model.joblib")

    # Sanity predictions to verify behavior with distinct emotional texts
    sanity_inputs = [
        "I am so happy today",
        "I feel terrible and alone",
        "Why did you do this?!",
    ]
    sanity_cleaned, _ = data_utils.preprocess_texts(sanity_inputs)
    sanity_x = vectorizer.transform(sanity_cleaned)
    sanity_probs = clf.predict_proba(sanity_x)
    sanity_preds = le.inverse_transform(np.argmax(sanity_probs, axis=1))

    sanity_rows = []
    for text, pred, probs in zip(sanity_inputs, sanity_preds, sanity_probs):
        sanity_rows.append(
            {
                "text": text,
                "predicted_emotion": str(pred),
                "confidence": float(np.max(probs)),
            }
        )
    pd.DataFrame(sanity_rows).to_csv(out_dir / "sanity_predictions.csv", index=False)

    LOG.info("Saved text model artifacts to %s", out_dir)

    # Validate tokenizer JSON load path (guards against corrupted artifact)
    with (out_dir / "tokenizer.json").open("r", encoding="utf8") as fh:
        _ = tokenizer_from_json(fh.read())


if __name__ == "__main__":
    main()
