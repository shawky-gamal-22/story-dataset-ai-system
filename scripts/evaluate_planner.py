from pathlib import Path

import pandas as pd

# pyrefly: ignore [missing-import]
from src.evaluation.planner_evaluator import PlannerEvaluator

# pyrefly: ignore [missing-import]
from src.retrieval.planner import QueryPlanner


BENCHMARK_PATH = Path("benchmark/task1_benchmark_v1.csv")

GOLD_PATH = Path("benchmark/planner_gold.csv")

OUTPUT_DIR = Path("experiments/task1")


def main():
    benchmark_df = pd.read_csv(BENCHMARK_PATH)
    gold_df = pd.read_csv(GOLD_PATH)

    evaluation_df = benchmark_df.merge(
        gold_df,
        on="query_id",
        how="inner",
        validate="one_to_one",
    )

    if len(evaluation_df) != len(benchmark_df):
        raise ValueError("Some benchmark queries are missing " "planner gold labels.")

    planner = QueryPlanner()
    evaluator = PlannerEvaluator(planner)

    results_df, metrics = evaluator.evaluate(evaluation_df)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = OUTPUT_DIR / "planner_results.csv"

    results_df.to_csv(
        results_path,
        index=False,
    )

    print("\n=== PLANNER EVALUATION ===")

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    failures = results_df[
        ~(results_df["intent_correct"] & results_df["strategy_correct"])
    ]

    print("\n=== FAILURES ===")

    if failures.empty:
        print("No planner failures.")
    else:
        for _, row in failures.iterrows():
            print("\n" + "-" * 70)
            print(f"ID: {row['query_id']}")
            print(f"Query: {row['query']}")

            print(
                "Intent: "
                f"{row['predicted_intent']} "
                f"(expected {row['expected_intent']})"
            )

            print(
                "Strategy: "
                f"{row['predicted_strategy']} "
                f"(expected {row['expected_strategy']})"
            )

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
