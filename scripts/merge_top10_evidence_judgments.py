from pathlib import Path

import pandas as pd


OLD_PATH = Path("experiments/task1/evidence_retrieval_judgments.csv")

NEW_PATH = Path("experiments/task3/evidence_top10_new_judgments.csv")

TOP10_PATH = Path("experiments/task1/evidence_retrieval_candidates_top10.csv")

OUTPUT_PATH = Path("experiments/task3/evidence_top10_judgments.csv")


def main():
    old = pd.read_csv(OLD_PATH)
    new = pd.read_csv(NEW_PATH)
    top10 = pd.read_csv(TOP10_PATH)

    keys = [
        "query_id",
        "evidence_story_id",
        "evidence_chunk_id",
    ]

    # Combine all available judgments
    judgments = pd.concat(
        [
            old[keys + ["supported", "judge_reason"]],
            new[keys + ["supported", "judge_reason"]],
        ],
        ignore_index=True,
    )

    judgments = judgments.drop_duplicates(
        subset=keys,
        keep="last",
    )

    # Attach judgments to the canonical Top-10 candidate pool
    final = top10.merge(
        judgments,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    missing = final["supported"].isna().sum()

    print("=== TOP-10 JUDGMENT MERGE ===")
    print(f"Top-10 candidates: {len(top10)}")
    print(f"Available judgments: {len(judgments)}")
    print(f"Final rows: {len(final)}")
    print(f"Missing judgments: {missing}")

    if missing:
        missing_rows = final.loc[
            final["supported"].isna(),
            ["query_id", "chunk_rank", "evidence_chunk_id"],
        ]

        print("\nMissing:")
        print(missing_rows.to_string(index=False))

        raise RuntimeError("Some Top-10 candidates do not have judgments.")

    final["supported"] = final["supported"].astype(bool)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nSupported distribution:")
    print(final["supported"].value_counts().to_string())

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
