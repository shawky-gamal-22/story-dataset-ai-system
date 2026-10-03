import json
import statistics
import time
from pathlib import Path

import psutil
from openai import OpenAI


BASE_URL = "http://127.0.0.1:8080/v1"
MODEL = "Qwen/Qwen3-4B-GGUF:Q4_K_M"

OUTPUT_PATH = Path("experiments/task3/cpu_llm_baseline.json")

client = OpenAI(
    base_url=BASE_URL,
    api_key="local",
)


TEST_PROMPTS = [
    "Who is the main character in the provided story?",
    "Summarize the provided story in three sentences.",
    "What is the main conflict in the story?",
    "What happens at the end of the story?",
    "What is the main theme of the story?",
]


CONTEXT = """
This is a short benchmark context about Amelia, a scientist
who discovers an abandoned research laboratory. Inside the
laboratory she finds an experimental artificial intelligence
named Echo. Echo reveals that the previous researchers
abandoned the facility after discovering that the system had
developed unexpected reasoning abilities. Amelia decides to
shut the experiment down before it can connect to the outside
network.
""".strip()


def percentile(values, p):
    values = sorted(values)

    index = int(round((len(values) - 1) * p))

    return values[index]


def run_request(query):
    start = time.perf_counter()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using only the provided context. " "Be concise and factual."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{CONTEXT}\n\n" f"User request:\n{query}\n\n" "/no_think"
                ),
            },
        ],
        temperature=0.0,
        max_tokens=128,
    )

    latency = time.perf_counter() - start

    usage = response.usage

    return {
        "query": query,
        "answer": response.choices[0].message.content.strip(),
        "latency_sec": latency,
        "prompt_tokens": (usage.prompt_tokens if usage else None),
        "completion_tokens": (usage.completion_tokens if usage else None),
    }


def main():
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Running warm-up...")

    # Don't include cold-start/warm-up in latency statistics.
    run_request(TEST_PROMPTS[0])

    results = []

    for i, query in enumerate(TEST_PROMPTS, start=1):

        result = run_request(query)
        results.append(result)

        completion_tokens = result["completion_tokens"] or 0

        tokens_per_sec = (
            completion_tokens / result["latency_sec"]
            if result["latency_sec"] > 0
            else 0
        )

        result["effective_tokens_per_sec"] = tokens_per_sec

        print(
            f"[{i}/{len(TEST_PROMPTS)}] "
            f"{result['latency_sec']:.3f}s | "
            f"{completion_tokens} output tokens | "
            f"{tokens_per_sec:.2f} effective tok/s"
        )

    latencies = [x["latency_sec"] for x in results]

    throughput = [x["effective_tokens_per_sec"] for x in results]

    summary = {
        "model": MODEL,
        "quantization": "Q4_K_M",
        "num_requests": len(results),
        "latency": {
            "mean_sec": statistics.mean(latencies),
            "median_sec": statistics.median(latencies),
            "p95_sec": percentile(latencies, 0.95),
            "min_sec": min(latencies),
            "max_sec": max(latencies),
        },
        "effective_generation_rate": {
            "mean_tokens_per_sec": statistics.mean(throughput),
        },
        "client_process_ram_mb": (psutil.Process().memory_info().rss / (1024**2)),
        "requests": results,
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n=== CPU BASELINE ===")
    print(f"Mean latency: " f"{summary['latency']['mean_sec']:.3f}s")
    print(f"Median latency: " f"{summary['latency']['median_sec']:.3f}s")
    print(f"P95 latency: " f"{summary['latency']['p95_sec']:.3f}s")
    print(
        f"Effective generation rate: "
        f"{summary['effective_generation_rate']['mean_tokens_per_sec']:.2f} tok/s"
    )

    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
