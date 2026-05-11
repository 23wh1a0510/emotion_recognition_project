"""Data utilities for text emotion classification.

This module provides:
- Synthetic-yet-meaningful labeled sentence generation for emotion classes
- Text preprocessing helpers (cleaning + tokenization)
- Train/validation/test splits
- Sequence tokenization and padding artifacts for reproducibility
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from preprocessing import text as text_utils


EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


@dataclass
class SequenceArtifacts:
    tokenizer: Tokenizer
    padded_sequences: np.ndarray
    max_len: int
    vocab_size: int


def _manual_seed_samples() -> List[Tuple[str, str]]:
    """Seed examples, including user-provided acceptance sentences."""
    return [
        ("I am so happy today", "happy"),
        ("I feel terrible and alone", "sad"),
        ("Why did you do this?!", "angry"),
        ("This smells awful and makes me sick", "disgust"),
        ("I am scared something bad will happen", "fear"),
        ("It is just an ordinary day", "neutral"),
        ("Wow, I never expected this gift", "surprise"),
    ]


def _emotion_lexicon() -> Dict[str, Dict[str, List[str]]]:
    return {
        "happy": {
            "feeling": ["happy", "joyful", "great", "amazing", "delighted", "excited", "glad", "cheerful"],
            "event": ["today", "this morning", "at the party", "after class", "with my friends", "right now"],
            "reaction": ["smiling", "celebrating", "laughing", "feeling grateful", "full of energy"],
        },
        "sad": {
            "feeling": ["sad", "lonely", "heartbroken", "down", "miserable", "hopeless", "empty", "upset"],
            "event": ["today", "tonight", "after the call", "since yesterday", "all week", "right now"],
            "reaction": ["crying", "feeling alone", "wanting to hide", "without motivation", "losing hope"],
        },
        "angry": {
            "feeling": ["angry", "furious", "mad", "annoyed", "irritated", "frustrated", "outraged", "upset"],
            "event": ["right now", "because of this", "after that message", "during this argument", "again today"],
            "reaction": ["shouting", "clenching my fists", "ready to argue", "unable to stay calm", "boiling inside"],
        },
        "fear": {
            "feeling": ["afraid", "scared", "terrified", "nervous", "anxious", "worried", "frightened", "uneasy"],
            "event": ["about tomorrow", "in the dark", "during the storm", "before the exam", "walking alone"],
            "reaction": ["my heart is racing", "I cannot relax", "I want to run", "I feel unsafe", "I am shaking"],
        },
        "disgust": {
            "feeling": ["disgusted", "sick", "grossed out", "revolted", "nauseous", "repulsed", "uncomfortable", "appalled"],
            "event": ["by that smell", "by this mess", "after seeing that", "because of this food", "from that behavior"],
            "reaction": ["I want to leave", "I feel nauseous", "this is unbearable", "I cannot look at it", "it makes me ill"],
        },
        "surprise": {
            "feeling": ["surprised", "shocked", "amazed", "stunned", "astonished", "speechless", "startled", "impressed"],
            "event": ["by the news", "by this result", "by your message", "by that gift", "by what happened"],
            "reaction": ["I did not expect this", "my jaw dropped", "this came out of nowhere", "I am in disbelief", "wow"],
        },
        "neutral": {
            "feeling": ["fine", "okay", "normal", "calm", "steady", "balanced", "regular", "ordinary"],
            "event": ["today", "right now", "at work", "in class", "this afternoon", "this week"],
            "reaction": ["nothing special happened", "just doing my tasks", "everything is routine", "I feel stable", "same as usual"],
        },
    }


def build_synthetic_dataset(samples_per_emotion: int = 240, random_state: int = 42) -> pd.DataFrame:
    """Build a deterministic synthetic dataset with meaningful emotional sentences."""
    rng = np.random.default_rng(random_state)
    lexicon = _emotion_lexicon()

    templates = [
        "I feel {feeling} {event}",
        "I am {feeling} and {reaction}",
        "This makes me {feeling} {event}",
        "Right now I am {feeling} because {event}",
        "Honestly I feel {feeling} and {reaction}",
        "I have been {feeling} {event} and {reaction}",
        "My mood is {feeling}; {reaction}",
        "I am feeling {feeling} {event}",
    ]

    rows: List[Tuple[str, str]] = []
    rows.extend(_manual_seed_samples())

    for emotion in EMOTIONS:
        slots = lexicon[emotion]
        combos = list(
            itertools.product(
                templates,
                slots["feeling"],
                slots["event"],
                slots["reaction"],
            )
        )
        rng.shuffle(combos)
        for tpl, feeling, event, reaction in combos[:samples_per_emotion]:
            sentence = tpl.format(feeling=feeling, event=event, reaction=reaction)
            rows.append((sentence, emotion))

    df = pd.DataFrame(rows, columns=["text", "label"])
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


def preprocess_texts(texts: List[str]) -> Tuple[List[str], List[List[str]]]:
    cleaned = [text_utils.clean_text(t) for t in texts]
    tokenized = [text_utils.tokenize(t) for t in cleaned]
    return cleaned, tokenized


def split_data(
    texts: List[str],
    labels: np.ndarray,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
):
    """Create stratified train/val/test split."""
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        texts, labels, test_size=test_size, stratify=labels, random_state=random_state
    )

    val_frac_of_trainval = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval,
        y_trainval,
        test_size=val_frac_of_trainval,
        stratify=y_trainval,
        random_state=random_state,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def build_sequence_artifacts(
    cleaned_texts: List[str],
    max_words: int = 8000,
    max_len: int = 24,
) -> SequenceArtifacts:
    """Build Keras tokenizer vocabulary and padded sequences (for reproducibility/debug)."""
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(cleaned_texts)
    sequences = tokenizer.texts_to_sequences(cleaned_texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
    vocab_size = min(max_words, len(tokenizer.word_index) + 1)
    return SequenceArtifacts(
        tokenizer=tokenizer,
        padded_sequences=padded,
        max_len=max_len,
        vocab_size=vocab_size,
    )
