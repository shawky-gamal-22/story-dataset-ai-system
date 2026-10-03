import ast
import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from src.retrieval.retriever import StoryRetriever


E2E_PATH = Path("experiments/task1/e2e_outputs.csv")
OUTPUT_PATH = Path("experiments/task1/qa_answer_judgments.csv")

MODEL = "openai/gpt-oss-120b"

QA_IDS = {f"Q{i:03d}" for i in range(27, 47)}


JUDGE_PROMPT = """
You are evaluating a retrieval-augmented story question answering system.

You are given:
- a user question,
- a reference expected answer,
- the system's final answer,
- the evidence chunks retrieved by the system.

Evaluate two dimensions independently.

CORRECTNESS
2 = The system answer correctly and sufficiently answers the question.
1 = Partially correct, incomplete, or contains a minor factual error.
0 = Incorrect, fails to answer, or contradicts the reference.

GROUNDEDNESS
2 = All substantive claims in the system answer are supported by the
    retrieved evidence.
1 = The answer is mostly supported, but contains a minor unsupported
    or overstated claim.
0 = Major claims are unsupported by the retrieved evidence.

Rules:
- Use only the supplied information.
- The expected answer is a semantic reference, not a required exact wording.
- Correct paraphrases receive full credit.
- Do not reward an answer merely for matching the expected answer if the
  retrieved evidence does not support it.
- Correctness and groundedness are separate.
- Do not use outside knowledge.

Return exactly ONE line using this format:

CORRECTNESS=<0|1|2>; GROUNDEDNESS=<0|1|2>; REASON=<brief reason>

Do not return JSON.
Do not return Markdown.
Do not add any text before or after the line.
Keep the reason under 30 words.
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


def parse_judgment(content):
    if not content:
        raise ValueError("Judge returned empty output.")

    content = content.strip()

    match = re.search(
        r"CORRECTNESS\s*=\s*([012])\s*;\s*"
        r"GROUNDEDNESS\s*=\s*([012])\s*;\s*"
        r"REASON\s*=\s*(.+)",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not match:
        raise ValueError(f"Could not parse judge output: {content!r}")

    return {
        "correctness": int(match.group(1)),
        "groundedness": int(match.group(2)),
        "reason": match.group(3).strip(),
    }


def main():
    load_dotenv()

    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )

    df = pd.read_csv(
        E2E_PATH,
        dtype={
            "query_id": str,
            "relevant_story_ids": str,
        },
    )

    df = df[df["query_id"].isin(QA_IDS)].copy()

    print(f"QA queries: {len(df)}")

    retriever = StoryRetriever()

    # --------------------------------------------------
    # Load successful previous judgments for resumability
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------

    for position, (_, row) in enumerate(
        df.iterrows(),
        start=1,
    ):
        query_id = row["query_id"]

        if query_id in existing:
            print(f"[{position}/{len(df)}] " f"{query_id} - reused")

            results.append(existing[query_id])
            continue

        story_ids = parse_story_ids(row["relevant_story_ids"])

        # Reconstruct the same evidence retrieval stage.
        evidence = retriever.retrieve_evidence(
            query=row["query"],
            candidate_story_ids=story_ids,
            top_k=5,
        )

        evidence_parts = []

        for rank, (_, chunk) in enumerate(
            evidence.iterrows(),
            start=1,
        ):
            evidence_parts.append(f"[Evidence {rank}]\n" f"{chunk['text']}")

        evidence_text = "\n\n".join(evidence_parts)

        user_content = f"""
USER QUESTION:
{row['query']}

EXPECTED ANSWER:
{row['expected_answer']}

SYSTEM ANSWER:
{row['answer']}

RETRIEVED EVIDENCE:
{evidence_text}
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
                "query_type": row["query_type"],
                "correctness": judgment["correctness"],
                "groundedness": judgment["groundedness"],
                "reason": judgment["reason"],
            }

            print(
                f"  correctness="
                f"{result['correctness']} "
                f"groundedness="
                f"{result['groundedness']}"
            )

        except Exception as exc:
            result = {
                "query_id": query_id,
                "query_type": row["query_type"],
                "correctness": None,
                "groundedness": None,
                "reason": (f"ERROR: {repr(exc)}"),
            }

            print(f"  ERROR: {repr(exc)}")

        results.append(result)

        # Save after every query.
        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        pd.DataFrame(results).to_csv(
            OUTPUT_PATH,
            index=False,
        )

        time.sleep(0.3)

    # --------------------------------------------------
    # Final metrics
    # --------------------------------------------------

    result_df = pd.DataFrame(results)

    valid = result_df.dropna(
        subset=[
            "correctness",
            "groundedness",
        ]
    )

    print("\n=== QA ANSWER EVALUATION ===")

    print(f"Queries: {len(result_df)}")

    print(f"Successfully judged: " f"{len(valid)}")

    print("\nCorrectness distribution:")

    print(valid["correctness"].value_counts().sort_index())

    print("\nGroundedness distribution:")

    print(valid["groundedness"].value_counts().sort_index())

    if not valid.empty:
        full_correctness = (valid["correctness"] == 2).mean()

        partial_correctness = (valid["correctness"] >= 1).mean()

        full_groundedness = (valid["groundedness"] == 2).mean()

        print(
            "\nFull correctness rate:",
            f"{full_correctness:.4f}",
        )

        print(
            "At-least-partial " "correctness rate:",
            f"{partial_correctness:.4f}",
        )

        print(
            "Fully grounded rate:",
            f"{full_groundedness:.4f}",
        )

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
