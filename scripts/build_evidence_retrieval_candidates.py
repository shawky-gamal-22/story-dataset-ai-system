from pathlib import Path
import ast

import pandas as pd

from src.retrieval.retriever import StoryRetriever


BENCHMARK_PATH = Path("benchmark/task1_benchmark_v1.csv")
OUTPUT_PATH = Path("experiments/task1/evidence_retrieval_candidates_top10.csv")

TOP_K = 10


def parse_story_ids(value) -> list[str]:
    """Parse relevant_story_ids stored in the benchmark CSV."""
    if pd.isna(value):
        return []

    parsed = ast.literal_eval(str(value))
    return [str(story_id) for story_id in parsed]


def main():
    # ---------------------------------------------------------
    # 1. Load benchmark
    # ---------------------------------------------------------
    benchmark = pd.read_csv(BENCHMARK_PATH)

    qa_df = benchmark[
        benchmark["query_type"].isin(["content_qa", "long_story_qa"])
    ].copy()

    print(f"QA queries: {len(qa_df)}")

    # ---------------------------------------------------------
    # 2. Load retriever once
    # ---------------------------------------------------------
    retriever = StoryRetriever()

    print("Evidence metadata columns:")
    print(retriever.evidence_metadata.columns.tolist())

    rows = []

    # ---------------------------------------------------------
    # 3. Retrieve evidence from the GOLD story only
    # ---------------------------------------------------------
    for _, benchmark_row in qa_df.iterrows():

        query_id = benchmark_row["query_id"]
        query = benchmark_row["query"]
        expected_answer = benchmark_row["expected_answer"]

        story_ids = parse_story_ids(benchmark_row["relevant_story_ids"])

        if len(story_ids) != 1:
            print(f"[WARNING] {query_id}: " f"expected one gold story, got {story_ids}")
            continue

        story_id = story_ids[0]

        evidence_df = retriever.retrieve_evidence(
            query=query,
            candidate_story_ids=[story_id],
            top_k=TOP_K,
        )

        # -----------------------------------------------------
        # 4. Save Top-K evidence chunks
        # -----------------------------------------------------
        for rank, (_, evidence_row) in enumerate(
            evidence_df.iterrows(),
            start=1,
        ):
            result = {
                "query_id": query_id,
                "query_type": benchmark_row["query_type"],
                "query": query,
                "story_id": story_id,
                "expected_answer": expected_answer,
                "chunk_rank": rank,
                "score": float(evidence_row["score"]),
            }

            # Keep all evidence metadata.
            for column in evidence_df.columns:
                if column != "score":
                    result[f"evidence_{column}"] = evidence_row[column]

            rows.append(result)

        print(f"{query_id}: " f"story={story_id} " f"retrieved={len(evidence_df)}")

    # ---------------------------------------------------------
    # 5. Save
    # ---------------------------------------------------------
    result_df = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n=== EVIDENCE CANDIDATES BUILT ===")
    print(f"Queries: {result_df['query_id'].nunique()}")
    print(f"Rows: {len(result_df)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
