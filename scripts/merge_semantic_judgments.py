import pandas as pd
from pathlib import Path


POOL_PATH = Path("experiments/task1/semantic_judgment_pool.csv")
OLD_PATH = Path("experiments/task1/semantic_judgments_gpt_oss_120b.csv")

OUTPUT_PATH = Path("experiments/task1/semantic_judgments_gpt_oss_120b.csv")
BACKUP_PATH = Path("experiments/task1/semantic_judgments_before_rebuild.csv")


def main():
    pool = pd.read_csv(POOL_PATH)
    old = pd.read_csv(OLD_PATH)

    # Backup previous judgments before overwriting anything.
    old.to_csv(BACKUP_PATH, index=False)

    pool["story_id"] = pool["story_id"].astype(str)
    old["story_id"] = old["story_id"].astype(str)

    # The pool may already contain empty judgment columns.
    # Remove them so the merge does not create _x / _y columns.
    judgment_fields = [
        "relevance",
        "judgment_notes",
        "judge_model",
    ]

    pool = pool.drop(columns=[col for col in judgment_fields if col in pool.columns])

    # Keep only the previous judgments we want to reuse.
    old_judgments = old[
        [
            "query_id",
            "story_id",
            "relevance",
            "judgment_notes",
            "judge_model",
        ]
    ].drop_duplicates(
        subset=["query_id", "story_id"],
        keep="last",
    )

    merged = pool.merge(
        old_judgments,
        on=["query_id", "story_id"],
        how="left",
        validate="one_to_one",
    )

    reused = int(merged["relevance"].notna().sum())
    missing = int(merged["relevance"].isna().sum())

    print("\n=== JUDGMENT REUSE ===")
    print(f"Current pool:      {len(merged)}")
    print(f"Reused judgments: {reused}")
    print(f"Need new judging: {missing}")

    if missing:
        print("\n=== NEW PAIRS ===")

        new_pairs = merged.loc[
            merged["relevance"].isna(),
            [
                "query_id",
                "query",
                "story_id",
                "title",
                "genre",
            ],
        ]

        print(new_pairs.to_string(index=False))

    # Sanity checks
    assert len(merged) == len(pool)
    assert not merged.duplicated(subset=["query_id", "story_id"]).any()

    merged.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved:  {OUTPUT_PATH}")
    print(f"Backup: {BACKUP_PATH}")


if __name__ == "__main__":
    main()
