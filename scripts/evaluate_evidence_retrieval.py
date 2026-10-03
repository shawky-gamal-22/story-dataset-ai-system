from pathlib import Path

import pandas as pd


INPUT_PATH = Path("experiments/task1/evidence_retrieval_judgments.csv")

OUTPUT_PATH = Path("experiments/task1/evidence_retrieval_final_metrics.csv")

# Q042-Q046 are the long-story QA subset in our benchmark.
LONG_QUERY_IDS = {
    "Q042",
    "Q043",
    "Q044",
    "Q045",
    "Q046",
}

KS = [1, 3, 5]


def normalize_supported(series: pd.Series) -> pd.Series:
    """
    Normalize supported values safely whether pandas loads them
    as booleans or strings.
    """
    return series.astype(str).str.strip().str.lower().eq("true")


def evaluate_query(group: pd.DataFrame) -> dict:
    group = group.sort_values("chunk_rank").copy()

    query_id = str(group["query_id"].iloc[0])
    story_id = str(group["story_id"].iloc[0])

    supported_mask = normalize_supported(group["supported"])

    supported_rows = group[supported_mask]

    num_supported = int(supported_mask.sum())

    if supported_rows.empty:
        first_supported_rank = None
        reciprocal_rank = 0.0
    else:
        first_supported_rank = int(supported_rows["chunk_rank"].min())

        reciprocal_rank = 1.0 / first_supported_rank

    result = {
        "query_id": query_id,
        "story_id": story_id,
        "story_group": ("long" if query_id in LONG_QUERY_IDS else "normal"),
        "num_retrieved_chunks": len(group),
        "num_supported_chunks": num_supported,
        "first_supported_rank": first_supported_rank,
        "reciprocal_rank": reciprocal_rank,
    }

    for k in KS:
        result[f"hit@{k}"] = int(
            first_supported_rank is not None and first_supported_rank <= k
        )

    return result


def print_metrics(
    df: pd.DataFrame,
    label: str,
) -> None:
    print(f"\n=== {label} ===")
    print(f"Queries: {len(df)}")

    if df.empty:
        return

    for k in KS:
        print(f"Hit@{k}: " f"{df[f'hit@{k}'].mean():.4f}")

    print(f"MRR: " f"{df['reciprocal_rank'].mean():.4f}")

    no_support = int((df["num_supported_chunks"] == 0).sum())

    print("No supporting evidence in Top-5: " f"{no_support}")


def main():
    df = pd.read_csv(
        INPUT_PATH,
        dtype={
            "query_id": str,
            "story_id": str,
        },
    )

    required_columns = {
        "query_id",
        "story_id",
        "chunk_rank",
        "supported",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError("Missing columns: " f"{sorted(missing_columns)}")

    # Sanity checks
    duplicate_pairs = df.duplicated(
        subset=[
            "query_id",
            "chunk_rank",
        ]
    ).sum()

    if duplicate_pairs:
        raise ValueError("Found duplicate query/rank rows: " f"{duplicate_pairs}")

    results = []

    for _, group in df.groupby(
        "query_id",
        sort=True,
    ):
        results.append(evaluate_query(group))

    results_df = pd.DataFrame(results)

    print("=== EVIDENCE RETRIEVAL EVALUATION ===")

    print_metrics(
        results_df,
        "OVERALL",
    )

    normal_df = results_df[results_df["story_group"] == "normal"]

    long_df = results_df[results_df["story_group"] == "long"]

    print_metrics(
        normal_df,
        "NORMAL STORIES",
    )

    print_metrics(
        long_df,
        "LONG STORIES",
    )

    print("\n=== PER QUERY ===")

    display_columns = [
        "query_id",
        "story_group",
        "num_retrieved_chunks",
        "num_supported_chunks",
        "first_supported_rank",
        "hit@1",
        "hit@3",
        "hit@5",
        "reciprocal_rank",
    ]

    print(results_df[display_columns].to_string(index=False))

    # Show failures separately because these are the
    # queries worth inspecting if retrieval needs improvement.
    failures = results_df[results_df["hit@5"] == 0]

    print("\n=== NO SUPPORTING EVIDENCE " "IN TOP-5 ===")

    if failures.empty:
        print("None.")
    else:
        print(
            failures[
                [
                    "query_id",
                    "story_id",
                    "story_group",
                    "num_retrieved_chunks",
                ]
            ].to_string(index=False)
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
