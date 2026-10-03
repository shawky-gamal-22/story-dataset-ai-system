import ast
import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


INPUT_PATH = Path("experiments/task1/e2e_outputs.csv")
STORIES_PATH = Path("data/artifacts/retrieval/stories.parquet")
OUTPUT_PATH = Path("experiments/task1/e2e_answer_judgments.csv")

MODEL = "openai/gpt-oss-120b"


JUDGE_PROMPT = """
You are evaluating a story-dataset question answering system.

Judge the SYSTEM ANSWER using ONLY:
1. the USER QUERY,
2. the EXPECTED ANSWER when provided,
3. the REFERENCE STORY TEXT.

Do not use outside knowledge.

Evaluate two dimensions independently.

CORRECTNESS:
2 = fully correct and directly answers the question
1 = partially correct, incomplete, or contains a minor error
0 = incorrect, contradicts the reference, or fails to answer

GROUNDEDNESS:
2 = all substantive claims are supported by the reference story text
1 = mostly grounded, but contains a minor unsupported or overstated claim
0 = contains major unsupported/hallucinated claims

Important:
- The expected answer is a reference, not necessarily the only valid wording.
- A correct paraphrase should receive full credit.
- Extra detail is acceptable only if supported by the reference.
- For discovery/list questions, judge whether the returned stories satisfy the request
  using the supplied reference stories and expected answer.
- For comparison questions, judge the comparison against all supplied stories.
- Be strict about invented plot events, characters, relationships, or themes.

Return valid JSON only:

{
  "correctness": 0,
  "groundedness": 0,
  "reason": "brief explanation"
}
"""


def parse_story_ids(value):
    if pd.isna(value):
        return []

    try:
        parsed = ast.literal_eval(str(value))
    except Exception:
        return []

    if isinstance(parsed, (list, tuple)):
        return [str(x) for x in parsed]

    return [str(parsed)]


def main():
    load_dotenv()

    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )

    df = pd.read_csv(
        INPUT_PATH,
        dtype={
            "query_id": str,
            "relevant_story_ids": str,
        },
    )

    stories = pd.read_parquet(STORIES_PATH)
    stories["id"] = stories["id"].astype(str)

    story_lookup = {row["id"]: row for _, row in stories.iterrows()}

    existing = {}

    if OUTPUT_PATH.exists():
        old = pd.read_csv(
            OUTPUT_PATH,
            dtype={"query_id": str},
        )

        for _, row in old.iterrows():
            if pd.notna(row.get("correctness")) and pd.notna(row.get("groundedness")):
                existing[row["query_id"]] = row.to_dict()

    results = []

    for i, row in df.iterrows():
        query_id = row["query_id"]

        if query_id in existing:
            print(f"[{i+1}/{len(df)}] {query_id} - reused")
            results.append(existing[query_id])
            continue

        story_ids = parse_story_ids(row["relevant_story_ids"])

        reference_parts = []

        for story_id in story_ids:
            story = story_lookup.get(story_id)

            if story is None:
                continue

            reference_parts.append(
                f"TITLE: {story['title']}\n"
                f"GENRE: {story['genre']}\n"
                f"STORY ID: {story_id}\n"
                f"STORY:\n{story['story']}"
            )

        reference_text = "\n\n--- STORY ---\n\n".join(reference_parts)

        user_content = f"""
USER QUERY:
{row['query']}

EXPECTED ANSWER:
{row.get('expected_answer', '')}

SYSTEM ANSWER:
{row.get('answer', '')}

REFERENCE STORY TEXT:
{reference_text}
"""

        print(f"[{i+1}/{len(df)}] Judging {query_id}...")

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": JUDGE_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ],
                temperature=0,
                max_tokens=300,
            )

            content = response.choices[0].message.content.strip()

            # tolerate fenced JSON if the API returns it
            if content.startswith("```"):
                content = content.strip("`")
                if content.startswith("json"):
                    content = content[4:].strip()

            judgment = json.loads(content)

            result = {
                "query_id": query_id,
                "query_type": row["query_type"],
                "correctness": int(judgment["correctness"]),
                "groundedness": int(judgment["groundedness"]),
                "reason": judgment["reason"],
            }

            print(
                f"  correctness={result['correctness']} "
                f"groundedness={result['groundedness']}"
            )

        except Exception as exc:
            result = {
                "query_id": query_id,
                "query_type": row["query_type"],
                "correctness": None,
                "groundedness": None,
                "reason": f"ERROR: {repr(exc)}",
            }

            print(f"  ERROR: {repr(exc)}")

        results.append(result)

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        pd.DataFrame(results).to_csv(
            OUTPUT_PATH,
            index=False,
        )

        # Gentle rate-limit protection.
        time.sleep(0.2)

    print("\n=== ANSWER JUDGING COMPLETE ===")

    result_df = pd.DataFrame(results)

    print(f"Total: {len(result_df)}")
    print(
        "\nCorrectness:\n",
        result_df["correctness"].value_counts(dropna=False).sort_index(),
    )
    print(
        "\nGroundedness:\n",
        result_df["groundedness"].value_counts(dropna=False).sort_index(),
    )

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
