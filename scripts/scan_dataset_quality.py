import re

import pandas as pd


STORIES_PATH = "data/artifacts/retrieval/stories.parquet"
OUTPUT_PATH = "experiments/task1/data_quality_issues.csv"


# Patterns that strongly suggest chat/generation-template contamination.
SUSPICIOUS_PATTERNS = {
    "im_start": r"<\|im_start\|>",
    "im_end": r"<\|im_end\|>",
    "system_tag": r"<\|system\|>",
    "assistant_tag": r"<\|assistant\|>",
    "user_tag": r"<\|user\|>",
    "generation_instruction": (
        r"\b(write|generate|create)\b.{0,80}" r"\b(story|stories)\b"
    ),
    "word_count_instruction": (r"\b(at least|minimum of)\b.{0,30}\bwords?\b"),
}


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def scan_row(row: pd.Series) -> dict:
    story = normalize_text(row["story"])
    title = normalize_text(row["title"])
    genre = normalize_text(row["genre"])

    reasons = []

    # Missing / empty fields
    if not story:
        reasons.append("empty_story")

    if not title:
        reasons.append("empty_title")

    if not genre:
        reasons.append("empty_genre")

    # Strong contamination patterns
    combined = f"{title}\n{story}"

    for name, pattern in SUSPICIOUS_PATTERNS.items():
        if re.search(pattern, combined, flags=re.IGNORECASE | re.DOTALL):
            reasons.append(name)

    # Extremely short stories are worth inspecting,
    # but are NOT automatically corrupted.
    word_count = len(story.split())

    if word_count < 100:
        reasons.append("very_short_story")

    return {
        "story_id": str(row["id"]),
        "title": title,
        "genre": genre,
        "word_count": word_count,
        "is_suspicious": bool(reasons),
        "reasons": "|".join(reasons),
        "story_preview": story[:500].replace("\n", " "),
    }


def main():
    df = pd.read_parquet(STORIES_PATH)

    print(f"Total stories: {len(df)}")

    results = pd.DataFrame([scan_row(row) for _, row in df.iterrows()])

    suspicious = results[results["is_suspicious"]].copy()

    suspicious.to_csv(OUTPUT_PATH, index=False)

    print("\n=== DATA QUALITY SCAN ===")
    print(f"Total stories:      {len(results)}")
    print(f"Suspicious stories: {len(suspicious)}")
    print(f"Clean stories:      {len(results) - len(suspicious)}")

    print("\n=== REASON COUNTS ===")

    if suspicious.empty:
        print("No suspicious samples found.")
    else:
        reason_counts = suspicious["reasons"].str.split("|").explode().value_counts()

        print(reason_counts.to_string())

        print("\n=== SUSPICIOUS SAMPLES ===")

        display_columns = [
            "story_id",
            "title",
            "genre",
            "word_count",
            "reasons",
        ]

        print(
            suspicious[display_columns]
            .sort_values(["word_count", "story_id"])
            .to_string(index=False)
        )

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
