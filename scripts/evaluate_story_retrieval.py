from pathlib import Path

import pandas as pd

# pyrefly: ignore [missing-import]
from src.evaluation.retrieval_evaluator import (
    StoryRetrievalEvaluator,
)
from src.retrieval.retriever import StoryRetriever


BENCHMARK_PATH = Path("benchmark/task1_benchmark_v1.csv")

PLANNER_GOLD_PATH = Path("benchmark/planner_gold.csv")

OUTPUT_DIR = Path("experiments/task1")


def main():
    benchmark_df = pd.read_csv(BENCHMARK_PATH)

    planner_gold_df = pd.read_csv(PLANNER_GOLD_PATH)

    evaluation_df = benchmark_df.merge(
        planner_gold_df,
        on="query_id",
        how="inner",
        validate="one_to_one",
    )

    semantic_df = evaluation_df[evaluation_df["expected_strategy"] == "semantic"].copy()

    if semantic_df.empty:
        raise ValueError("No semantic retrieval queries found.")

    print(
        "Semantic retrieval queries:",
        len(semantic_df),
    )

    retriever = StoryRetriever()

    evaluator = StoryRetrievalEvaluator(retriever)

    results_df, metrics = evaluator.evaluate(
        semantic_df,
        ks=(1, 3, 5, 10),
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = OUTPUT_DIR / "story_retrieval_results.csv"

    results_df.to_csv(
        results_path,
        index=False,
    )

    print("\n=== STORY RETRIEVAL EVALUATION ===")

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\n=== MISSES @ 5 ===")

    misses = results_df[results_df["hit@5"] == 0]

    if misses.empty:
        print("No misses at K=5.")
    else:
        for _, row in misses.iterrows():
            print("\n" + "-" * 70)
            print(f"ID: {row['query_id']}")
            print(f"Query: {row['query']}")
            print(
                "Relevant:",
                row["relevant_story_ids"],
            )
            print(
                "Retrieved:",
                row["retrieved_story_ids"][:5],
            )

    print(f"\nResults saved to: " f"{results_path}")


if __name__ == "__main__":
    main()
