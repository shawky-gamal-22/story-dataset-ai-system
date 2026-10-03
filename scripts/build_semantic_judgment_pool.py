import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.retrieval.retriever import StoryRetriever


BENCHMARK_PATH = Path("benchmark/task1_benchmark_v1.csv")

PLANNER_GOLD_PATH = Path("benchmark/planner_gold.csv")

OUTPUT_DIR = Path("experiments/task1")

OUTPUT_PATH = OUTPUT_DIR / "semantic_judgment_pool.csv"


def parse_ids(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(x) for x in value]

    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except json.JSONDecodeError:
        pass

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple, set)):
            return [str(x) for x in parsed]
    except (ValueError, SyntaxError):
        pass

    return [value]


def normalize_retrieval_results(
    results: Any,
) -> list[dict]:

    if isinstance(results, pd.DataFrame):
        return results.to_dict(orient="records")

    if isinstance(results, list):
        return results

    raise TypeError("Unsupported retrieval result format: " f"{type(results)}")


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

    print(f"Semantic queries: {len(semantic_df)}")

    retriever = StoryRetriever()

    rows = []

    for _, benchmark_row in semantic_df.iterrows():
        query_id = benchmark_row["query_id"]
        query = benchmark_row["query"]

        silver_ids = set(parse_ids(benchmark_row["relevant_story_ids"]))

        retrieved = retriever.retrieve_stories(
            query=query,
            top_k=10,
        )

        retrieved = normalize_retrieval_results(retrieved)

        retrieved_by_id = {}

        for rank, item in enumerate(
            retrieved,
            start=1,
        ):
            story_id = str(
                item.get(
                    "story_id",
                    item.get("id"),
                )
            )

            retrieved_by_id[story_id] = {
                "rank": rank,
                "score": item.get("score"),
            }

        retrieved_ids = set(retrieved_by_id.keys())

        candidate_ids = silver_ids | retrieved_ids

        print(f"{query_id}: " f"{len(candidate_ids)} candidates")

        for story_id in sorted(candidate_ids):
            story_df = retriever.get_story_by_id(story_id)

            if story_df.empty:
                print(f"WARNING: story " f"{story_id} not found")
                continue

            story = story_df.iloc[0]

            retrieval_info = retrieved_by_id.get(
                story_id,
                {},
            )

            rows.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "story_id": story_id,
                    "title": story["title"],
                    "genre": story["genre"],
                    "story": story["story"],
                    # Where candidate came from
                    "silver_proxy": (story_id in silver_ids),
                    "retrieved": (story_id in retrieved_ids),
                    # Retrieval diagnostics
                    "retrieval_rank": (retrieval_info.get("rank")),
                    "retrieval_score": (retrieval_info.get("score")),
                    # Filled during judging
                    "relevance": None,
                    "judgment_notes": None,
                }
            )

    pool_df = pd.DataFrame(rows)

    pool_df = pool_df.sort_values(
        by=[
            "query_id",
            "retrieval_rank",
            "story_id",
        ],
        na_position="last",
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pool_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n=== POOL SUMMARY ===")
    print(f"Queries: " f"{pool_df['query_id'].nunique()}")
    print(f"Total query-story pairs: " f"{len(pool_df)}")

    print("\nCandidates per query:")
    print(pool_df.groupby("query_id").size().to_string())

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
