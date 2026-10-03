# src/classification/prompts.py

from __future__ import annotations


SYSTEM_PROMPT = """You are a story genre classifier.
Choose exactly one genre from the provided list of allowed genres.
Output only the genre name and nothing else."""


def build_classification_messages(
    story: str,
    genre_labels: list[str],
) -> list[dict[str, str]]:
    labels = "\n".join(f"- {label}" for label in genre_labels)

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"Allowed genres:\n{labels}\n\n" f"Story:\n{story}\n\n" "Genre:"
            ),
        },
    ]
