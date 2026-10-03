import json
import statistics
import time
from pathlib import Path

import pandas as pd

from src.chatbot.graph import build_chatbot_graph


BENCHMARK_PATH = Path("benchmark/task1_benchmark_v1.csv")

OUTPUT_JSON = Path("experiments/task3/task1_hybrid_planner.json")

OUTPUT_CSV = Path("experiments/task3/task1_hybrid_planner_predictions.csv")


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    index = int(round((len(values) - 1) * p))
    return values[index]


def main():
    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------
    # Load QA benchmark
    # -------------------------
    df = pd.read_csv(BENCHMARK_PATH)

    qa_df = df[df["query_id"].isin([f"Q{i:03d}" for i in range(27, 47)])].copy()

    print(f"QA benchmark size: {len(qa_df)}")

    # -------------------------
    # Build graph ONCE
    # -------------------------
    print("Building Task 1 graph...")
    graph = build_chatbot_graph()

    # -------------------------
    # Warm-up
    # -------------------------
    print("Running warm-up...")

    graph.invoke({"query": qa_df.iloc[0]["query"]})

    # -------------------------
    # Benchmark
    # -------------------------
    results = []

    for i, (_, row) in enumerate(
        qa_df.iterrows(),
        start=1,
    ):
        print(f"\n[{i}/{len(qa_df)}] " f"{row['query_id']}")

        start = time.perf_counter()

        result = graph.invoke({"query": row["query"]})

        latency = time.perf_counter() - start

        plan = result.get("plan")

        predicted_intent = getattr(plan, "intent", None) if plan is not None else None

        retrieval_strategy = (
            getattr(
                plan,
                "retrieval_strategy",
                None,
            )
            if plan is not None
            else None
        )

        evidence = result.get("evidence", [])

        record = {
            "query_id": row["query_id"],
            "query_type": row["query_type"],
            "query": row["query"],
            "expected_answer": row["expected_answer"],
            "answer": result.get("answer"),
            "predicted_intent": predicted_intent,
            "retrieval_strategy": retrieval_strategy,
            "retrieved_story_ids": result.get(
                "story_ids",
                [],
            ),
            "num_evidence_chunks": len(evidence),
            "context_chars": len(result.get("context", "")),
            "latency_sec": latency,
        }

        results.append(record)

        # Save incrementally
        pd.DataFrame(results).to_csv(
            OUTPUT_CSV,
            index=False,
        )

        print(f"Latency: {latency:.3f}s")

        print(f"Answer: {record['answer']}")

    # -------------------------
    # Summary
    # -------------------------
    latencies = [x["latency_sec"] for x in results]

    normal_latencies = [
        x["latency_sec"] for x in results if x["query_type"] == "content_qa"
    ]

    long_latencies = [
        x["latency_sec"] for x in results if x["query_type"] == "long_story_qa"
    ]

    summary = {
        "model": "Qwen3-4B Q4_K_M",
        "runtime": "llama.cpp",
        "benchmark": "Task1 Q027-Q046",
        "num_queries": len(results),
        "latency": {
            "mean_sec": statistics.mean(latencies),
            "median_sec": statistics.median(latencies),
            "p95_sec": percentile(
                latencies,
                0.95,
            ),
            "min_sec": min(latencies),
            "max_sec": max(latencies),
        },
        "content_qa_mean_sec": (statistics.mean(normal_latencies)),
        "long_story_qa_mean_sec": (statistics.mean(long_latencies)),
        "requests": results,
    }

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 60)
    print("TASK 1 CPU BASELINE")
    print("=" * 60)

    print(f"Queries:       {len(results)}")

    print(f"Mean latency:  " f"{summary['latency']['mean_sec']:.3f}s")

    print(f"Median:        " f"{summary['latency']['median_sec']:.3f}s")

    print(f"P95:           " f"{summary['latency']['p95_sec']:.3f}s")

    print(f"Min:           " f"{summary['latency']['min_sec']:.3f}s")

    print(f"Max:           " f"{summary['latency']['max_sec']:.3f}s")

    print(f"Content QA:    " f"{summary['content_qa_mean_sec']:.3f}s mean")

    print(f"Long QA:       " f"{summary['long_story_qa_mean_sec']:.3f}s mean")

    print(f"\nSaved JSON -> {OUTPUT_JSON}")

    print(f"Saved CSV  -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()


# # from src.chatbot.graph import build_chatbot_graph

# # graph = build_chatbot_graph()

# # result = graph.invoke(
# #     {
# #         "query": 'In "The Chronicles of the Celestial Spoon", what is the name of the spoon that can create horrifying and inedible dishes?'
# #     }
# # )

# # print(result["answer"])

# import pandas as pd

# from src.chatbot.graph import build_chatbot_graph


# BENCHMARK_PATH = "benchmark/task1_benchmark_v1.csv"

# QUERY_IDS = [
#     "Q030",
#     "Q039",
#     "Q043",
# ]


# def main():
#     df = pd.read_csv(BENCHMARK_PATH)

#     selected = df[df["query_id"].isin(QUERY_IDS)].copy()

#     graph = build_chatbot_graph()

#     for _, row in selected.iterrows():

#         print("=" * 80)
#         print(f"Running {row['query_id']}")
#         print(row["query"])
#         print("=" * 80)

#         result = graph.invoke(
#             {
#                 "query": row["query"],
#             },
#             config={
#                 "run_name": f"task3_profile_{row['query_id']}",
#                 "tags": [
#                     "task3",
#                     "cpu-profile",
#                     row["query_id"],
#                 ],
#             },
#         )

#         print(f"\nAnswer:\n{result['answer']}\n")


# if __name__ == "__main__":
#     main()
