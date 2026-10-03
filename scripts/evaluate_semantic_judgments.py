from pathlib import Path

import pandas as pd


JUDGMENTS_PATH = Path("experiments/task1/semantic_judgments_gpt_oss_120b.csv")

OUTPUT_PATH = Path("experiments/task1/semantic_retrieval_final_metrics.csv")

KS = [1, 3, 5, 10]


def evaluate_query(group: pd.DataFrame) -> dict:
    query_id = group["query_id"].iloc[0]

    # Conservative relevance definition:
    # only clearly relevant stories count as relevant.
    relevant_ids = set(
        group.loc[
            group["relevance"] == 2,
            "story_id",
        ].astype(str)
    )

    # Only stories actually returned by the retriever have a rank.
    retrieved = (
        group[group["retrieved"] == True]
        .dropna(subset=["retrieval_rank"])
        .sort_values("retrieval_rank")
        .copy()
    )

    retrieved["story_id"] = retrieved["story_id"].astype(str)

    result = {
        "query_id": query_id,
        "num_relevant_in_pool": len(relevant_ids),
    }

    for k in KS:
        top_k = retrieved[retrieved["retrieval_rank"] <= k]

        retrieved_ids = set(top_k["story_id"])

        num_relevant_retrieved = len(relevant_ids & retrieved_ids)

        result[f"hit@{k}"] = int(num_relevant_retrieved > 0)

        result[f"precision@{k}"] = num_relevant_retrieved / k

        result[f"recall@{k}"] = (
            num_relevant_retrieved / len(relevant_ids) if relevant_ids else 0.0
        )

    # Reciprocal rank of first clearly relevant result.
    relevant_ranks = retrieved.loc[
        retrieved["story_id"].isin(relevant_ids),
        "retrieval_rank",
    ]

    if len(relevant_ranks):
        first_relevant_rank = int(relevant_ranks.min())

        result["first_relevant_rank"] = first_relevant_rank

        result["reciprocal_rank"] = 1.0 / first_relevant_rank

    else:
        result["first_relevant_rank"] = None
        result["reciprocal_rank"] = 0.0

    return result


def main():
    df = pd.read_csv(
        JUDGMENTS_PATH,
        dtype={"story_id": str},
    )

    if df["relevance"].isna().any():
        raise ValueError("Missing relevance judgments.")

    if df.duplicated(subset=["query_id", "story_id"]).any():
        raise ValueError("Duplicate query-story pairs found.")

    results = []

    for _, group in df.groupby(
        "query_id",
        sort=True,
    ):
        results.append(evaluate_query(group))

    results_df = pd.DataFrame(results)

    print("=== FINAL SEMANTIC RETRIEVAL EVALUATION ===")
    print(f"Queries: {len(results_df)}")
    print("Relevance definition: relevance == 2")

    for k in KS:
        print(f"\nHit@{k}:       " f"{results_df[f'hit@{k}'].mean():.4f}")

        print(f"Precision@{k}: " f"{results_df[f'precision@{k}'].mean():.4f}")

        print(f"Recall@{k}:    " f"{results_df[f'recall@{k}'].mean():.4f}")

    print(f"\nMRR: " f"{results_df['reciprocal_rank'].mean():.4f}")

    print("\n=== PER QUERY ===")

    display_cols = [
        "query_id",
        "num_relevant_in_pool",
        "first_relevant_rank",
        "hit@1",
        "hit@3",
        "hit@5",
        "hit@10",
        "precision@5",
        "precision@10",
        "recall@10",
    ]

    print(results_df[display_cols].to_string(index=False))

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
