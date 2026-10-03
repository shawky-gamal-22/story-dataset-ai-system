from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path

from src.chatbot.graph import build_chatbot_graph
from src.chatbot.generator import AnswerGenerator


BENCHMARK_PATH = Path("benchmark/task1_benchmark_v1.csv")

OUTPUT_JSON = Path("experiments/task3/task1_streaming_top3.json")

OUTPUT_CSV = Path("experiments/task3/task1_streaming_top3_predictions.csv")


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)

    if not values:
        return 0.0

    index = int((len(values) - 1) * p)
    return values[index]


def main():
    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Graph stops after context construction.
    graph = build_chatbot_graph(
        include_generator=False,
    )

    generator = AnswerGenerator()

    with BENCHMARK_PATH.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        rows = list(csv.DictReader(f))

    # Same QA subset used in previous CPU benchmark.
    rows = [
        row
        for row in rows
        if row["query_type"]
        in {
            "content_qa",
            "long_story_qa",
        }
    ]

    print(f"Queries: {len(rows)}")

    # -----------------------------------------
    # Warm-up
    # -----------------------------------------

    print("Running warm-up...")

    warm_query = rows[0]["query"]

    warm_state = graph.invoke(
        {
            "query": warm_query,
        }
    )

    warm_stream = generator.generate_stream(
        query=warm_query,
        context=warm_state["context"],
        intent=warm_state["plan"].intent,
    )

    for _ in warm_stream:
        pass

    print("Warm-up complete.\n")

    results = []

    # -----------------------------------------
    # Benchmark
    # -----------------------------------------

    for index, row in enumerate(
        rows,
        start=1,
    ):
        query = row["query"]

        request_start = time.perf_counter()

        # Planner + retrieval + context builder
        pregen_state = graph.invoke(
            {
                "query": query,
            }
        )

        generation_start = time.perf_counter()

        stream = generator.generate_stream(
            query=query,
            context=pregen_state["context"],
            intent=pregen_state["plan"].intent,
        )

        first_token_time = None
        chunks = []

        for chunk in stream:
            now = time.perf_counter()

            if first_token_time is None:
                first_token_time = now

            chunks.append(chunk)

        request_end = time.perf_counter()

        if first_token_time is None:
            raise RuntimeError(
                f"No streamed content returned for " f"{row['query_id']}"
            )

        answer = "".join(chunks).strip()

        pre_generation_latency = generation_start - request_start

        llm_ttft = first_token_time - generation_start

        end_to_end_ttft = first_token_time - request_start

        generation_latency = request_end - generation_start

        full_response_latency = request_end - request_start

        result = {
            "query_id": row["query_id"],
            "query_type": row["query_type"],
            "query": query,
            "answer": answer,
            "pre_generation_latency_s": (pre_generation_latency),
            "llm_ttft_s": llm_ttft,
            "end_to_end_ttft_s": (end_to_end_ttft),
            "generation_latency_s": (generation_latency),
            "full_response_latency_s": (full_response_latency),
            "context_chars": len(pregen_state["context"]),
        }

        results.append(result)

        print(
            f"[{index:02d}/{len(rows)}] "
            f"{row['query_id']} | "
            f"PreGen={pre_generation_latency:.3f}s | "
            f"LLM TTFT={llm_ttft:.3f}s | "
            f"E2E TTFT={end_to_end_ttft:.3f}s | "
            f"Full={full_response_latency:.3f}s"
        )

        # Save incrementally.
        with OUTPUT_CSV.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=result.keys(),
            )

            writer.writeheader()
            writer.writerows(results)

    # -----------------------------------------
    # Summary
    # -----------------------------------------

    pregen = [r["pre_generation_latency_s"] for r in results]

    llm_ttft = [r["llm_ttft_s"] for r in results]

    e2e_ttft = [r["end_to_end_ttft_s"] for r in results]

    full = [r["full_response_latency_s"] for r in results]

    summary = {
        "num_queries": len(results),
        "pre_generation_mean_s": (statistics.mean(pregen)),
        "llm_ttft_mean_s": (statistics.mean(llm_ttft)),
        "llm_ttft_p50_s": (statistics.median(llm_ttft)),
        "llm_ttft_p95_s": (percentile(llm_ttft, 0.95)),
        "e2e_ttft_mean_s": (statistics.mean(e2e_ttft)),
        "e2e_ttft_p50_s": (statistics.median(e2e_ttft)),
        "e2e_ttft_p95_s": (percentile(e2e_ttft, 0.95)),
        "full_response_mean_s": (statistics.mean(full)),
        "full_response_p50_s": (statistics.median(full)),
        "full_response_p95_s": (percentile(full, 0.95)),
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            {
                "summary": summary,
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("TASK 1 STREAMING BENCHMARK")
    print("=" * 60)

    print(f"Queries:               " f"{summary['num_queries']}")

    print(f"Pre-generation mean:   " f"{summary['pre_generation_mean_s']:.3f}s")

    print()
    print(f"LLM TTFT mean:         " f"{summary['llm_ttft_mean_s']:.3f}s")
    print(f"LLM TTFT P50:          " f"{summary['llm_ttft_p50_s']:.3f}s")
    print(f"LLM TTFT P95:          " f"{summary['llm_ttft_p95_s']:.3f}s")

    print()
    print(f"E2E TTFT mean:         " f"{summary['e2e_ttft_mean_s']:.3f}s")
    print(f"E2E TTFT P50:          " f"{summary['e2e_ttft_p50_s']:.3f}s")
    print(f"E2E TTFT P95:          " f"{summary['e2e_ttft_p95_s']:.3f}s")

    print()
    print(f"Full response mean:    " f"{summary['full_response_mean_s']:.3f}s")
    print(f"Full response P50:     " f"{summary['full_response_p50_s']:.3f}s")
    print(f"Full response P95:     " f"{summary['full_response_p95_s']:.3f}s")

    print(f"\nSaved JSON -> {OUTPUT_JSON}")
    print(f"Saved CSV  -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
