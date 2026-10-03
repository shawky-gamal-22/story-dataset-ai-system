import ast
import json
import time
from typing import Any

import pandas as pd

from src.retrieval.retriever import StoryRetriever


class StoryRetrievalEvaluator:
    def __init__(self, retriever: StoryRetriever):
        self.retriever = retriever

    @staticmethod
    def _parse_relevant_ids(value: Any) -> list[str]:
        if value is None:
            return []

        if isinstance(value, list):
            return [str(x) for x in value]

        if pd.isna(value):
            return []

        value = str(value).strip()

        if not value:
            return []

        # JSON list
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except json.JSONDecodeError:
            pass

        # Python-style list
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple, set)):
                return [str(x) for x in parsed]
        except (ValueError, SyntaxError):
            pass

        # Single ID fallback
        return [value]

    @staticmethod
    def _extract_retrieved_ids(
        results: Any,
    ) -> list[str]:

        if isinstance(results, pd.DataFrame):
            if "story_id" in results.columns:
                return results["story_id"].astype(str).tolist()

            if "id" in results.columns:
                return results["id"].astype(str).tolist()

        if isinstance(results, list):
            ids = []

            for item in results:
                if isinstance(item, dict):
                    story_id = item.get(
                        "story_id",
                        item.get("id"),
                    )

                    if story_id is not None:
                        ids.append(str(story_id))

            return ids

        raise TypeError("Unsupported retrieval result format: " f"{type(results)}")

    @staticmethod
    def _reciprocal_rank(
        retrieved_ids: list[str],
        relevant_ids: set[str],
    ) -> float:

        for rank, story_id in enumerate(
            retrieved_ids,
            start=1,
        ):
            if story_id in relevant_ids:
                return 1.0 / rank

        return 0.0

    def evaluate(
        self,
        benchmark_df: pd.DataFrame,
        ks: tuple[int, ...] = (1, 3, 5, 10),
    ) -> tuple[pd.DataFrame, dict]:

        max_k = max(ks)
        rows = []

        for _, row in benchmark_df.iterrows():
            query = row["query"]

            relevant_ids = set(self._parse_relevant_ids(row["relevant_story_ids"]))

            if not relevant_ids:
                continue

            start = time.perf_counter()

            retrieved = self.retriever.retrieve_stories(
                query=query,
                top_k=max_k,
            )

            latency_ms = (time.perf_counter() - start) * 1000

            retrieved_ids = self._extract_retrieved_ids(retrieved)

            result = {
                "query_id": row["query_id"],
                "query": query,
                "relevant_story_ids": sorted(relevant_ids),
                "retrieved_story_ids": retrieved_ids,
                "latency_ms": latency_ms,
            }

            for k in ks:
                top_k_ids = retrieved_ids[:k]

                hits = len(relevant_ids.intersection(top_k_ids))

                result[f"hit@{k}"] = 1 if hits > 0 else 0

                result[f"recall@{k}"] = hits / len(relevant_ids)

            result["reciprocal_rank"] = self._reciprocal_rank(
                retrieved_ids,
                relevant_ids,
            )

            rows.append(result)

        results_df = pd.DataFrame(rows)

        if results_df.empty:
            raise ValueError("No evaluable retrieval queries found.")

        metrics = {
            "num_queries": len(results_df),
        }

        for k in ks:
            metrics[f"hit@{k}"] = results_df[f"hit@{k}"].mean()

            metrics[f"recall@{k}"] = results_df[f"recall@{k}"].mean()

        metrics["mrr"] = results_df["reciprocal_rank"].mean()

        metrics["avg_latency_ms"] = results_df["latency_ms"].mean()

        metrics["p50_latency_ms"] = results_df["latency_ms"].median()

        metrics["p95_latency_ms"] = results_df["latency_ms"].quantile(0.95)

        return results_df, metrics
