import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


E2E_PATH = Path("experiments/task1/e2e_outputs.csv")
OUTPUT_PATH = Path("experiments/task1/comparison_answer_judgments.csv")

MODEL = "openai/gpt-oss-120b"

COMPARISON_IDS = {
    "Q047",
    "Q048",
    "Q049",
    "Q050",
}


JUDGE_PROMPT = """
You are evaluating answers produced by a story comparison system.

You are given:
- a user question,
- a reference expected answer,
- the system's final answer.

Evaluate CORRECTNESS only.

Score:
2 = The system answer correctly and sufficiently answers the question.
1 = The answer is partially correct, incomplete, or contains a minor error.
0 = The answer is incorrect, fails to answer, or contradicts the reference.

Rules:
- Treat the expected answer as a semantic reference, not required wording.
- Correct paraphrases receive full credit.
- Focus on whether the important facts needed to answer the question are present.
- Do not penalize harmless wording differences.
- Do not use outside knowledge.
- Do not evaluate groundedness because source evidence is not provided.

Return exactly ONE line:

CORRECTNESS=<0|1|2>; REASON=<brief reason>

Do not return JSON.
Do not return Markdown.
Do not add text before or after the line.
Keep the reason under 30 words.
"""


def parse_judgment(content):
    if not content:
        raise ValueError("Judge returned empty output.")

    match = re.search(
        r"CORRECTNESS\s*=\s*([012])\s*;\s*" r"REASON\s*=\s*(.+)",
        content.strip(),
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not match:
        raise ValueError(f"Could not parse judge output: {content!r}")

    return {
        "correctness": int(match.group(1)),
        "reason": match.group(2).strip(),
    }


def main():
    load_dotenv()

    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )

    df = pd.read_csv(
        E2E_PATH,
        dtype={"query_id": str},
    )

    df = df[df["query_id"].isin(COMPARISON_IDS)].copy()

    print(f"Comparison queries: {len(df)}")

    results = []

    for position, (_, row) in enumerate(
        df.iterrows(),
        start=1,
    ):
        query_id = row["query_id"]

        user_content = f"""
USER QUESTION:
{row['query']}

EXPECTED ANSWER:
{row['expected_answer']}

SYSTEM ANSWER:
{row['answer']}
"""

        print(f"[{position}/{len(df)}] " f"Judging {query_id}...")

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
                max_tokens=100,
            )

            content = response.choices[0].message.content

            judgment = parse_judgment(content)

            result = {
                "query_id": query_id,
                "correctness": judgment["correctness"],
                "reason": judgment["reason"],
            }

            print(f"  correctness=" f"{result['correctness']}")

        except Exception as exc:
            result = {
                "query_id": query_id,
                "correctness": None,
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

        time.sleep(0.3)

    result_df = pd.DataFrame(results)

    valid = result_df.dropna(subset=["correctness"])

    print("\n=== COMPARISON ANSWER EVALUATION ===")
    print(f"Queries: {len(result_df)}")
    print(f"Successfully judged: {len(valid)}")

    print("\nCorrectness distribution:")
    print(valid["correctness"].value_counts().sort_index())

    if not valid.empty:
        print(
            "\nFull correctness rate:",
            f"{(valid['correctness'] == 2).mean():.4f}",
        )

        print(
            "At-least-partial correctness rate:",
            f"{(valid['correctness'] >= 1).mean():.4f}",
        )

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
