import time
from pathlib import Path

import pandas as pd

from src.chatbot.graph import build_chatbot_graph


INPUT_PATH = Path("experiments/task1/e2e_outputs.csv")
QUERY_IDS = {"Q047", "Q048"}


def main():
    df = pd.read_csv(
        INPUT_PATH,
        dtype={
            "query_id": str,
            "relevant_story_ids": str,
        },
    )

    graph = build_chatbot_graph()

    for idx in df.index[df["query_id"].isin(QUERY_IDS)]:
        query_id = df.at[idx, "query_id"]
        query = df.at[idx, "query"]

        print("\n" + "=" * 80)
        print(f"Rerunning {query_id}")
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

            df.at[idx, "predicted_intent"] = plan.get("intent")

            df.at[idx, "predicted_strategy"] = plan.get("retrieval_strategy")

            df.at[idx, "answer"] = output.get("answer")

            df.at[idx, "retrieved_story_ids"] = str(output.get("story_ids", []))

            df.at[idx, "num_evidence_chunks"] = len(output.get("evidence", []))

            df.at[idx, "latency_ms"] = latency_ms
            df.at[idx, "error"] = None

            print(f"Answer: {output.get('answer')}")
            print(f"Latency: {latency_ms / 1000:.2f}s")

        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000

            df.at[idx, "latency_ms"] = latency_ms
            df.at[idx, "error"] = repr(exc)

            print(f"ERROR: {repr(exc)}")

        # Save after each rerun
        df.to_csv(
            INPUT_PATH,
            index=False,
        )

    print("\n" + "=" * 80)
    print("=== FAILED E2E QUERIES RERUN COMPLETE ===")

    subset = df[df["query_id"].isin(QUERY_IDS)]

    print(
        subset[
            [
                "query_id",
                "predicted_intent",
                "predicted_strategy",
                "answer",
                "latency_ms",
                "error",
            ]
        ].to_string(index=False)
    )

    print(
        "\nRemaining errors:",
        df["error"].notna().sum(),
    )

    print(f"Saved: {INPUT_PATH}")


if __name__ == "__main__":
    main()
