import time
from pathlib import Path

import pandas as pd

from src.chatbot.graph import build_chatbot_graph


BENCHMARK_PATH = Path("benchmark/task1_benchmark_v1.csv")
OUTPUT_PATH = Path("experiments/task1/e2e_outputs.csv")


def main():
    benchmark = pd.read_csv(
        BENCHMARK_PATH,
        dtype={
            "query_id": str,
            "relevant_story_ids": str,
        },
    )

    print(f"Benchmark queries: {len(benchmark)}")

    # Build once — don't reload models/indexes per query.
    graph = build_chatbot_graph()

    results = []

    for i, row in benchmark.iterrows():
        query_id = row["query_id"]
        query = row["query"]

        print("\n" + "=" * 80)
        print(f"[{i + 1}/{len(benchmark)}] {query_id}")
        print(f"Query: {query}")

        start = time.perf_counter()

        try:
            output = graph.invoke(
                {
                    "query": query,
                }
            )

            latency_ms = (time.perf_counter() - start) * 1000

            plan = output.get("plan")

            if hasattr(plan, "model_dump"):
                plan = plan.model_dump()

            plan = plan or {}

            result = {
                **row.to_dict(),
                "predicted_intent": plan.get("intent"),
                "predicted_strategy": plan.get("retrieval_strategy"),
                "answer": output.get("answer"),
                "retrieved_story_ids": str(output.get("story_ids", [])),
                "num_evidence_chunks": len(output.get("evidence", [])),
                "latency_ms": latency_ms,
                "error": None,
            }

            print(
                f"Plan: "
                f"{result['predicted_intent']} / "
                f"{result['predicted_strategy']}"
            )
            print(f"Stories: " f"{result['retrieved_story_ids']}")
            print(f"Evidence chunks: " f"{result['num_evidence_chunks']}")
            print(f"Answer: {result['answer']}")
            print(f"Latency: {latency_ms / 1000:.2f}s")

        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000

            result = {
                **row.to_dict(),
                "predicted_intent": None,
                "predicted_strategy": None,
                "answer": None,
                "retrieved_story_ids": None,
                "num_evidence_chunks": None,
                "latency_ms": latency_ms,
                "error": repr(exc),
            }

            print(f"ERROR: {repr(exc)}")

        results.append(result)

        # Save after every query.
        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        pd.DataFrame(results).to_csv(
            OUTPUT_PATH,
            index=False,
        )

    results_df = pd.DataFrame(results)

    successful = results_df[results_df["error"].isna()]

    print("\n" + "=" * 80)
    print("=== TASK 1 E2E INFERENCE COMPLETE ===")
    print(f"Total queries: {len(results_df)}")
    print(f"Successful: {len(successful)}")
    print(f"Errors: " f"{results_df['error'].notna().sum()}")

    if not successful.empty:
        print(f"Average latency: " f"{successful['latency_ms'].mean() / 1000:.2f}s")
        print(f"P50 latency: " f"{successful['latency_ms'].median() / 1000:.2f}s")
        print(f"P95 latency: " f"{successful['latency_ms'].quantile(0.95) / 1000:.2f}s")

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
