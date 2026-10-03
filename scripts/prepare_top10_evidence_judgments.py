from pathlib import Path

import pandas as pd


OLD_JUDGMENTS = Path("experiments/task1/evidence_retrieval_judgments.csv")

TOP10_CANDIDATES = Path("experiments/task1/evidence_retrieval_candidates_top10.csv")

OUTPUT = Path("experiments/task3/evidence_top10_for_judging.csv")


def main():
    old = pd.read_csv(OLD_JUDGMENTS)
    top10 = pd.read_csv(TOP10_CANDIDATES)

    # chunk identity within a query
    keys = [
        "query_id",
        "evidence_story_id",
        "evidence_chunk_id",
    ]

    old_labels = old[keys + ["supported", "judge_reason"]].drop_duplicates(keys)

    merged = top10.merge(
        old_labels,
        on=keys,
        how="left",
    )

    already_judged = merged["supported"].notna()

    new_candidates = merged[~already_judged].copy()

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    new_candidates.to_csv(
        OUTPUT,
        index=False,
    )

    print("=== TOP-10 JUDGMENT PREPARATION ===")
    print(f"Top-10 candidates: {len(merged)}")
    print(f"Reused judgments: {already_judged.sum()}")
    print(f"New candidates:   {len(new_candidates)}")

    print("\nNew candidates by query:")
    print(new_candidates.groupby("query_id").size().to_string())

    print(f"\nSaved to: {OUTPUT}")


if __name__ == "__main__":
    main()
