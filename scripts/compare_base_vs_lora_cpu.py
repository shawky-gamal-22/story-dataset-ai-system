from __future__ import annotations

import pandas as pd
from openai import OpenAI
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from src.classification.prompts import build_classification_messages
from src.data.loader import load_story_dataset


BASE_MODEL = "Qwen/Qwen3-4B"
MAX_LENGTH = 2048
N_SAMPLES = 20

LORA_RESULTS_PATH = "experiments/task3/task2_cpu_predictions.csv"

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed",
)


def build_prompt(tokenizer, story, genre_labels):
    messages = build_classification_messages(
        story=story,
        genre_labels=genre_labels,
    )

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    ids = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    # For this diagnostic, use only examples that fit naturally
    # inside 2048 tokens. This removes truncation as a variable.
    if len(ids) > MAX_LENGTH:
        return None

    return prompt


def clean_prediction(text):
    return text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Loading dataset...")
    df = load_story_dataset().copy()

    genre_labels = sorted(df["genre"].unique().tolist())

    # Same validation split as Task 2.
    _, val_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["genre"],
    )

    val_df = val_df.reset_index(drop=True)
    val_df["story_id"] = val_df["id"].astype(str)

    # Load the already-computed CPU LoRA predictions.
    lora_df = pd.read_csv(
        LORA_RESULTS_PATH,
        dtype={"story_id": str},
    )

    lora_map = dict(
        zip(
            lora_df["story_id"],
            lora_df["predicted_genre"],
        )
    )

    rows = []

    print()
    print("=" * 90)
    print("BASE Q4 vs GGUF LoRA — CPU DIAGNOSTIC")
    print("=" * 90)

    for _, row in val_df.iterrows():

        if len(rows) >= N_SAMPLES:
            break

        story_id = str(row["id"])

        if story_id not in lora_map:
            continue

        prompt = build_prompt(
            tokenizer,
            row["story"],
            genre_labels,
        )

        # Skip long examples so truncation cannot affect this test.
        if prompt is None:
            continue

        response = client.completions.create(
            model="qwen",
            prompt=prompt,
            temperature=0,
            max_tokens=32,
        )

        base_prediction = clean_prediction(response.choices[0].text)

        lora_prediction = str(lora_map[story_id])

        same = base_prediction == lora_prediction

        rows.append(
            {
                "story_id": story_id,
                "true_genre": row["genre"],
                "base_q4": base_prediction,
                "q4_lora": lora_prediction,
                "same": same,
            }
        )

        print(
            f"[{len(rows):02d}/{N_SAMPLES}] "
            f"True={row['genre']!r} | "
            f"Base={base_prediction!r} | "
            f"LoRA={lora_prediction!r} | "
            f"{'SAME' if same else 'DIFFERENT'}"
        )

    result = pd.DataFrame(rows)

    same_count = int(result["same"].sum())
    different_count = len(result) - same_count

    print()
    print("=" * 90)
    print("RESULT")
    print("=" * 90)

    print(f"Compared:   {len(result)}")
    print(f"Same:       {same_count}")
    print(f"Different:  {different_count}")

    if len(result):
        print(f"Agreement:  " f"{same_count / len(result):.2%}")

    output = "experiments/task3/" "task2_base_vs_gguf_lora_diagnostic.csv"

    result.to_csv(
        output,
        index=False,
    )

    print(f"\nSaved -> {output}")


if __name__ == "__main__":
    main()
