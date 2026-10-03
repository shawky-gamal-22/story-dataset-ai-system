import time

import pandas as pd

# pyrefly: ignore [missing-import]
from src.retrieval.planner import QueryPlanner


class PlannerEvaluator:
    def __init__(self, planner: QueryPlanner):
        self.planner = planner

    def evaluate(
        self,
        benchmark_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict]:

        results = []

        for _, row in benchmark_df.iterrows():
            query = row["query"]

            start = time.perf_counter()

            try:
                plan = self.planner.plan(query)
                error = None
            except Exception as exc:
                plan = None
                error = str(exc)

            latency_ms = (time.perf_counter() - start) * 1000

            predicted_intent = plan.intent if plan else None
            predicted_strategy = plan.retrieval_strategy if plan else None

            expected_intent = row["expected_intent"]
            expected_strategy = row["expected_strategy"]

            results.append(
                {
                    "query_id": row["query_id"],
                    "query": query,
                    "query_type": row["query_type"],
                    "expected_intent": expected_intent,
                    "predicted_intent": predicted_intent,
                    "intent_correct": (predicted_intent == expected_intent),
                    "expected_strategy": expected_strategy,
                    "predicted_strategy": predicted_strategy,
                    "strategy_correct": (predicted_strategy == expected_strategy),
                    "latency_ms": latency_ms,
                    "error": error,
                }
            )

        results_df = pd.DataFrame(results)

        metrics = {
            "num_queries": len(results_df),
            "intent_accuracy": results_df["intent_correct"].mean(),
            "strategy_accuracy": results_df["strategy_correct"].mean(),
            "avg_latency_ms": results_df["latency_ms"].mean(),
            "p50_latency_ms": results_df["latency_ms"].median(),
            "p95_latency_ms": results_df["latency_ms"].quantile(0.95),
            "num_errors": results_df["error"].notna().sum(),
        }

        return results_df, metrics
