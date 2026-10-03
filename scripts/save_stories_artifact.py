from pathlib import Path

# pyrefly: ignore [missing-import]
from src.data.loader import load_story_dataset


ARTIFACT_DIR = Path("data/artifacts/retrieval")


def main():
    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading dataset...")

    stories_df = load_story_dataset()

    stories_df.to_parquet(
        ARTIFACT_DIR / "stories.parquet",
        index=False,
    )

    print(f"Loaded {len(stories_df):,} stories.\n")


if __name__ == "__main__":
    main()
