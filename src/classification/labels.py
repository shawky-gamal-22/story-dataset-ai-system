# src/classification/labels.py

from __future__ import annotations

from sklearn.preprocessing import LabelEncoder


def build_genre_label_encoder(genres: list[str]) -> LabelEncoder:
    """
    Build the genre label encoder used by the classification pipeline.

    The encoder learns a deterministic alphabetical mapping between
    genre names and integer class IDs.
    """
    encoder = LabelEncoder()
    encoder.fit(genres)
    return encoder


def get_genre_labels(genres: list[str]) -> list[str]:
    """Return the sorted set of valid genre labels."""
    encoder = build_genre_label_encoder(genres)
    return encoder.classes_.tolist()
