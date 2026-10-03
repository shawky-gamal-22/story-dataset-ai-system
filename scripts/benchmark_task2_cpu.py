from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from src.classification.prompts import build_classification_messages
from src.data.loader import load_story_dataset


BASE_MODEL = "Qwen/Qwen3-4B"
MAX_LENGTH = 2048
SAFETY_MARGIN = 16

RANDOM_STATE = 42
TEST_SIZE = 0.20

OUTPUT_DIR = Path("experiments/task3")
PREDICTIONS_PATH = OUTPUT_DIR / "task2_cpu_predictions.csv"
METRICS_PATH = OUTPUT_DIR / "task2_cpu_metrics.json"

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed",
)


def build_cpu_prompt(
    tokenizer,
    story: str,
    genre_labels: list[str],
) -> tuple[str, int, bool]:
    """
    Build the Task 2 classification prompt while truncating ONLY the story.

    Important:
    - System instruction is always preserved.
    - Allowed genre list is always preserved.
    - The final "Genre:" instruction is always preserved.
    - The assistant generation boundary is always preserved.

    Returns:
        final_prompt
        final_input_tokens
        story_was_truncated
    """

    # ---------------------------------------------------------
    # 1. Build the complete prompt structure with an empty story.
    #
    # This tells us how many tokens are required for:
    # - system instruction
    # - genre list
    # - chat-template tokens
    # - "Genre:"
    # - assistant generation boundary
    # ---------------------------------------------------------

    empty_messages = build_classification_messages(
        story="",
        genre_labels=genre_labels,
    )

    empty_prompt = tokenizer.apply_chat_template(
        empty_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    empty_ids = tokenizer(
        empty_prompt,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    # ---------------------------------------------------------
    # 2. Tokenize ONLY the story.
    # ---------------------------------------------------------

    story_ids = tokenizer(
        story,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    original_story_tokens = len(story_ids)

    # ---------------------------------------------------------
    # 3. Calculate how much room is available for the story.
    #
    # Leave a small safety margin because tokenization after
    # reconstructing the final prompt can differ slightly around
    # text boundaries.
    # ---------------------------------------------------------

    story_budget = MAX_LENGTH - len(empty_ids) - SAFETY_MARGIN

    if story_budget <= 0:
        raise RuntimeError(
            "The fixed classification prompt is already too large. "
            f"Fixed tokens={len(empty_ids)}, MAX_LENGTH={MAX_LENGTH}"
        )

    story_was_truncated = original_story_tokens > story_budget

    if story_was_truncated:
        story_ids = story_ids[:story_budget]

    truncated_story = tokenizer.decode(
        story_ids,
        skip_special_tokens=True,
    )

    # ---------------------------------------------------------
    # 4. Rebuild the COMPLETE classification prompt.
    #
    # This is the important difference from the old implementation.
    # We truncate the story first, then reconstruct the prompt.
    # ---------------------------------------------------------

    final_messages = build_classification_messages(
        story=truncated_story,
        genre_labels=genre_labels,
    )

    final_prompt = tokenizer.apply_chat_template(
        final_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    final_ids = tokenizer(
        final_prompt,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    # ---------------------------------------------------------
    # 5. Safety verification.
    # ---------------------------------------------------------

    if len(final_ids) > MAX_LENGTH:
        raise RuntimeError(
            "Final prompt still exceeds MAX_LENGTH after "
            "story-only truncation: "
            f"{len(final_ids)} > {MAX_LENGTH}"
        )

    return (
        final_prompt,
        len(final_ids),
        story_was_truncated,
    )


def clean_prediction(raw_output: str) -> str:
    """
    Remove llama/Qwen special tokens while leaving the actual
    model prediction untouched.
    """

    return raw_output.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()


def save_predictions(df: pd.DataFrame) -> None:
    df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )


def calculate_and_save_metrics(
    results_df: pd.DataFrame,
    genre_labels: list[str],
    current_run_wall_time: float,
) -> None:

    y_true = results_df["true_genre"]
    y_pred = results_df["predicted_genre"]

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    latencies = results_df["latency_seconds"].astype(float).to_numpy()

    invalid_count = int((~results_df["is_valid"].astype(bool)).sum())

    truncated_count = int(results_df["truncated"].astype(bool).sum())

    metrics = {
        "runtime": "llama.cpp CPU-only",
        "base_model": "Qwen3-4B Q4_K_M GGUF",
        "adapter": ("checkpoint-100 converted to F16 GGUF LoRA"),
        "num_validation": int(len(results_df)),
        "num_genres": int(len(genre_labels)),
        "max_input_tokens": MAX_LENGTH,
        "truncation_strategy": (
            "story-only truncation with prompt structure preserved"
        ),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "invalid_outputs": invalid_count,
        "invalid_rate": float(invalid_count / len(results_df)),
        "truncated_inputs": truncated_count,
        "truncated_rate": float(truncated_count / len(results_df)),
        "mean_latency_seconds": float(np.mean(latencies)),
        "median_latency_seconds": float(np.median(latencies)),
        "p95_latency_seconds": float(np.percentile(latencies, 95)),
        "min_latency_seconds": float(np.min(latencies)),
        "max_latency_seconds": float(np.max(latencies)),
        "sum_request_latency_seconds": float(np.sum(latencies)),
        "current_run_wall_time_seconds": float(current_run_wall_time),
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 70)
    print("TASK 2 CPU BENCHMARK - CORRECTED")
    print("=" * 70)

    print(f"Validation stories: {len(results_df)}")
    print(f"Genres:             {len(genre_labels)}")

    print()

    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Macro Precision:    {macro_precision:.4f}")
    print(f"Macro Recall:       {macro_recall:.4f}")
    print(f"Macro F1:           {macro_f1:.4f}")
    print(f"Weighted F1:        {weighted_f1:.4f}")

    print()

    print(
        f"Invalid outputs:    "
        f"{invalid_count}/{len(results_df)} "
        f"({invalid_count / len(results_df):.2%})"
    )

    print(
        f"Truncated stories:  "
        f"{truncated_count}/{len(results_df)} "
        f"({truncated_count / len(results_df):.2%})"
    )

    print()

    print(f"Mean latency:       " f"{np.mean(latencies):.3f}s/story")

    print(f"Median latency:     " f"{np.median(latencies):.3f}s/story")

    print(f"P95 latency:        " f"{np.percentile(latencies, 95):.3f}s/story")

    print()

    print(f"Predictions -> {PREDICTIONS_PATH}")
    print(f"Metrics     -> {METRICS_PATH}")


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Loading dataset...")

    df = load_story_dataset().copy()

    genre_labels = sorted(df["genre"].unique().tolist())

    genre_set = set(genre_labels)

    print(f"Stories: {len(df)}")
    print(f"Genres:  {len(genre_labels)}")

    # ---------------------------------------------------------
    # Reproduce the exact same validation split.
    # ---------------------------------------------------------

    train_df, val_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["genre"],
    )

    val_df = val_df.reset_index(drop=True)

    val_df["story_id"] = val_df["id"].astype(str)

    print(f"Train:   {len(train_df)}")
    print(f"Val:     {len(val_df)}")

    # ---------------------------------------------------------
    # Load previous benchmark.
    #
    # IMPORTANT:
    # We KEEP the 171 non-truncated predictions.
    # We RERUN only rows that were truncated by the old buggy
    # preprocessing.
    # ---------------------------------------------------------

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"\n{PREDICTIONS_PATH} does not exist.\n"
            "This corrected script expects the previous "
            "200-story benchmark so it can rerun only the "
            "affected rows."
        )

    old_results = pd.read_csv(
        PREDICTIONS_PATH,
        dtype={"story_id": str},
    )

    # Only rows belonging to the current validation split.
    val_ids = set(val_df["story_id"].astype(str))

    old_results = old_results[old_results["story_id"].astype(str).isin(val_ids)].copy()

    if len(old_results) != len(val_df):
        raise RuntimeError(
            "Existing predictions do not contain the complete "
            "200-story validation split.\n"
            f"Found: {len(old_results)}\n"
            f"Expected: {len(val_df)}"
        )

    # ---------------------------------------------------------
    # Find rows affected by the OLD truncation bug.
    # ---------------------------------------------------------

    old_results["truncated"] = old_results["truncated"].astype(bool)

    affected_ids = set(
        old_results.loc[
            old_results["truncated"],
            "story_id",
        ].astype(str)
    )

    print()
    print(f"Existing predictions: {len(old_results)}")
    print(f"Keeping unchanged:    " f"{len(old_results) - len(affected_ids)}")
    print(f"Rerunning affected:   " f"{len(affected_ids)}")

    if not affected_ids:
        print("\nNo previously truncated rows were found.")
        return

    print("=" * 70)

    # ---------------------------------------------------------
    # Keep the original 171 unaffected rows.
    # ---------------------------------------------------------

    kept_results = old_results[
        ~old_results["story_id"].astype(str).isin(affected_ids)
    ].copy()

    corrected_rows: list[dict] = []

    benchmark_start = time.perf_counter()

    # ---------------------------------------------------------
    # Rerun ONLY affected stories.
    # ---------------------------------------------------------

    affected_val_df = val_df[val_df["story_id"].isin(affected_ids)].copy()

    total_affected = len(affected_val_df)

    for run_idx, (_, row) in enumerate(
        affected_val_df.iterrows(),
        start=1,
    ):
        story_id = str(row["id"])

        prompt, input_tokens, truncated = build_cpu_prompt(
            tokenizer=tokenizer,
            story=row["story"],
            genre_labels=genre_labels,
        )

        start = time.perf_counter()

        response = client.completions.create(
            model="qwen",
            prompt=prompt,
            temperature=0,
            max_tokens=32,
        )

        latency = time.perf_counter() - start

        raw_output = response.choices[0].text.strip()

        prediction = clean_prediction(raw_output)

        is_valid = prediction in genre_set
        is_correct = prediction == row["genre"]

        corrected_row = {
            "story_id": story_id,
            "title": row["title"],
            "true_genre": row["genre"],
            "predicted_genre": prediction,
            "raw_output": raw_output,
            "is_valid": is_valid,
            "is_correct": is_correct,
            "input_tokens": input_tokens,
            # This now means:
            # "story required story-only truncation"
            # rather than:
            # "whole prompt was blindly truncated".
            "truncated": truncated,
            "latency_seconds": latency,
        }

        corrected_rows.append(corrected_row)

        print(
            f"[{run_idx:2d}/{total_affected}] "
            f"{latency:7.2f}s | "
            f"{'VALID' if is_valid else 'INVALID':7s} | "
            f"{'OK' if is_correct else 'WRONG':5s} | "
            f"{row['genre']} -> "
            f"{prediction!r}"
        )

        # -----------------------------------------------------
        # Save progress after every corrected story.
        #
        # If interrupted, the file contains:
        # - the original 171 good rows
        # - all corrected rows completed so far
        # - remaining old affected rows
        #
        # We deliberately preserve all 200 rows.
        # -----------------------------------------------------

        corrected_ids_so_far = {str(r["story_id"]) for r in corrected_rows}

        remaining_old_affected = old_results[
            old_results["story_id"]
            .astype(str)
            .isin(affected_ids - corrected_ids_so_far)
        ].copy()

        progress_df = pd.concat(
            [
                kept_results,
                pd.DataFrame(corrected_rows),
                remaining_old_affected,
            ],
            ignore_index=True,
        )

        save_predictions(progress_df)

    current_run_wall_time = time.perf_counter() - benchmark_start

    # ---------------------------------------------------------
    # Merge:
    #
    # 171 original valid rows
    # +
    # 29 corrected rows
    # =
    # 200 final rows
    # ---------------------------------------------------------

    corrected_df = pd.DataFrame(corrected_rows)

    results_df = pd.concat(
        [
            kept_results,
            corrected_df,
        ],
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # Restore original validation order.
    # ---------------------------------------------------------

    order = {str(story_id): i for i, story_id in enumerate(val_df["story_id"])}

    results_df["_order"] = results_df["story_id"].astype(str).map(order)

    results_df = (
        results_df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    )

    if len(results_df) != len(val_df):
        raise RuntimeError(
            "Final merged prediction count is incorrect: "
            f"{len(results_df)} != {len(val_df)}"
        )

    if results_df["story_id"].duplicated().any():
        duplicates = results_df.loc[
            results_df["story_id"].duplicated(keep=False),
            "story_id",
        ].tolist()

        raise RuntimeError(f"Duplicate story IDs detected: {duplicates}")

    # Save final corrected predictions.
    save_predictions(results_df)

    # ---------------------------------------------------------
    # Final metrics over ALL 200 stories.
    # ---------------------------------------------------------

    calculate_and_save_metrics(
        results_df=results_df,
        genre_labels=genre_labels,
        current_run_wall_time=current_run_wall_time,
    )


if __name__ == "__main__":
    main()
