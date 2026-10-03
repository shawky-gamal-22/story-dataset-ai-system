from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from sentence_transformers import CrossEncoder


INPUT_PATH = Path("experiments/task1/evidence_retrieval_judgments.csv")

OUTPUT_PATH = Path("experiments/task3/evidence_reranker_results.csv")

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"


def hit_at_k(
    df: pd.DataFrame,
    rank_col: str,
    k: int,
) -> float:
    hits = []

    for _, group in df.groupby("query_id"):
        top_k = group.nsmallest(k, rank_col)

        hit = top_k["supported"].any()
        hits.append(int(hit))

    return sum(hits) / len(hits)


def mrr(
    df: pd.DataFrame,
    rank_col: str,
) -> float:
    scores = []

    for _, group in df.groupby("query_id"):
        ranked = group.sort_values(rank_col)

        rr = 0.0

        for position, (_, row) in enumerate(
            ranked.iterrows(),
            start=1,
        ):
            if row["supported"]:
                rr = 1.0 / position
                break

        scores.append(rr)

    return sum(scores) / len(scores)


def evaluate_subset(
    df: pd.DataFrame,
    name: str,
):
    print(f"\n{name}")
    print("-" * 40)

    for rank_col, label in [
        ("chunk_rank", "Dense"),
        ("reranker_rank", "Reranked"),
    ]:
        print(
            f"{label:10} "
            f"Hit@1={hit_at_k(df, rank_col, 1):.4f} | "
            f"Hit@3={hit_at_k(df, rank_col, 3):.4f} | "
            f"Hit@5={hit_at_k(df, rank_col, 5):.4f} | "
            f"MRR={mrr(df, rank_col):.4f}"
        )


def main():
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(INPUT_PATH)

    # Make sure supported is a real boolean.
    if df["supported"].dtype != bool:
        df["supported"] = (
            df["supported"]
            .astype(str)
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                }
            )
        )

    if df["supported"].isna().any():
        raise ValueError(
            "Some 'supported' values could not " "be converted to boolean."
        )

    # Restrict experiment to original Top-5 pool.
    df = df[df["chunk_rank"] <= 5].copy()

    num_queries = df["query_id"].nunique()

    print(f"Queries:    {num_queries}")
    print(f"Candidates: {len(df)}")
    print(f"Supported:  {int(df['supported'].sum())}")

    # -----------------------------------------
    # Load reranker
    # -----------------------------------------

    print(f"\nLoading reranker:\n{MODEL_NAME}")

    model = CrossEncoder(
        MODEL_NAME,
        device="cpu",
    )

    pairs = list(
        zip(
            df["query"].astype(str),
            df["evidence_text"].astype(str),
        )
    )

    # -----------------------------------------
    # Warm-up
    # -----------------------------------------

    print("\nWarm-up...")

    model.predict(
        pairs[:1],
        batch_size=1,
        show_progress_bar=False,
    )

    # -----------------------------------------
    # Rerank
    # -----------------------------------------

    print("Reranking...")

    start = time.perf_counter()

    scores = model.predict(
        pairs,
        batch_size=16,
        show_progress_bar=True,
    )

    elapsed = time.perf_counter() - start

    df["reranker_score"] = scores

    df["reranker_rank"] = (
        df.groupby("query_id")["reranker_score"]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    # -----------------------------------------
    # Evaluation
    # -----------------------------------------

    print("\n" + "=" * 70)
    print("EVIDENCE RERANKER EVALUATION")
    print("=" * 70)

    evaluate_subset(
        df,
        "ALL QA (20)",
    )

    normal_df = df[df["query_type"] == "content_qa"]

    evaluate_subset(
        normal_df,
        "CONTENT QA (15)",
    )

    long_df = df[df["query_type"] == "long_story_qa"]

    evaluate_subset(
        long_df,
        "LONG-STORY QA (5)",
    )

    # -----------------------------------------
    # Query-level movement
    # -----------------------------------------

    print("\nAnswer-bearing chunk movement")
    print("-" * 70)

    movements = []

    for query_id, group in df.groupby("query_id"):
        supported = group[group["supported"]]

        if supported.empty:
            dense_best = None
            rerank_best = None
        else:
            dense_best = int(supported["chunk_rank"].min())

            rerank_best = int(supported["reranker_rank"].min())

        movements.append(
            {
                "query_id": query_id,
                "query_type": (group.iloc[0]["query_type"]),
                "best_dense_support_rank": (dense_best),
                "best_reranked_support_rank": (rerank_best),
            }
        )

    movement_df = pd.DataFrame(movements)

    movement_df["changed"] = (
        movement_df["best_dense_support_rank"]
        != movement_df["best_reranked_support_rank"]
    )

    print(movement_df.to_string(index=False))

    # -----------------------------------------
    # Runtime
    # -----------------------------------------

    per_query = elapsed / num_queries
    per_pair = elapsed / len(df)

    print("\nRuntime")
    print("-" * 40)

    print(f"Total reranking: {elapsed:.3f}s")

    print(f"Per query:       " f"{per_query:.3f}s")

    print(f"Per pair:        " f"{per_pair:.4f}s")

    # -----------------------------------------
    # Save
    # -----------------------------------------

    df.sort_values(
        [
            "query_id",
            "reranker_rank",
        ]
    ).to_csv(
        OUTPUT_PATH,
        index=False,
    )

    movement_path = Path("experiments/task3/" "evidence_reranker_movements.csv")

    movement_df.to_csv(
        movement_path,
        index=False,
    )

    print(f"\nSaved results   -> {OUTPUT_PATH}")

    print(f"Saved movements -> {movement_path}")


if __name__ == "__main__":
    main()
