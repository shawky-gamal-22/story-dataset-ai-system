from pathlib import Path
import time

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field


INPUT_PATH = Path("experiments/task1/evidence_retrieval_candidates.csv")

OUTPUT_PATH = Path("experiments/task1/evidence_retrieval_judgments.csv")

MODEL_NAME = "openai/gpt-oss-120b"

MAX_RETRIES = 5


class EvidenceJudgment(BaseModel):
    supported: bool
    reason: str = Field(description="One short sentence explaining the judgment.")


SYSTEM_PROMPT = """
You are evaluating evidence retrieval for a story question-answering system.

Your task is NOT to answer the question yourself.

Determine whether the provided retrieved chunk contains enough information
to support the expected answer to the question.

Rules:

1. Judge only the provided chunk.
2. Do not use outside knowledge.
3. The wording does not need to exactly match the expected answer.
4. Mark supported=true if the chunk contains enough information to derive
   or verify the expected answer.
5. Mark supported=false if the answer is missing, only weakly implied,
   contradicted, or requires information not present in the chunk.
6. Be strict about names, relationships, causes, actions, and other details.
7. Keep the reason to one short sentence.
8. Return the JSON object immediately without analysis.

Return ONLY valid JSON in exactly this form:

{
  "supported": true,
  "reason": "One short sentence."
}
""".strip()


def judge_chunk(
    client: Groq,
    question: str,
    expected_answer: str,
    chunk_text: str,
) -> EvidenceJudgment:

    user_prompt = f"""
QUESTION:
{question}

EXPECTED ANSWER:
{expected_answer}

RETRIEVED CHUNK:
{chunk_text}
""".strip()

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0,
                max_tokens=512,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
            )

            content = response.choices[0].message.content

            return EvidenceJudgment.model_validate_json(content)

        except Exception as exc:
            last_error = exc

            print(f"  Attempt {attempt}/{MAX_RETRIES} failed: {exc}")

            if attempt < MAX_RETRIES:
                time.sleep(2)

    raise RuntimeError(
        f"Evidence judgment failed after {MAX_RETRIES} attempts."
    ) from last_error


def main():

    load_dotenv()

    client = Groq()

    df = pd.read_csv(INPUT_PATH)

    print(f"Total evidence chunks: {len(df)}")
    print(f"Queries: {df['query_id'].nunique()}")

    # ---------------------------------------------------------
    # Resume support
    # ---------------------------------------------------------
    if OUTPUT_PATH.exists():
        completed_df = pd.read_csv(OUTPUT_PATH)

        completed_keys = set(
            zip(
                completed_df["query_id"].astype(str),
                completed_df["chunk_rank"].astype(int),
            )
        )

        results = completed_df.to_dict("records")

    else:
        completed_keys = set()
        results = []

    print(f"Already completed: {len(completed_keys)}")
    print()

    # ---------------------------------------------------------
    # Judge every retrieved chunk
    # ---------------------------------------------------------
    for _, row in df.iterrows():

        query_id = str(row["query_id"])
        chunk_rank = int(row["chunk_rank"])

        key = (query_id, chunk_rank)

        if key in completed_keys:
            continue

        print(f"Judging {query_id} " f"rank={chunk_rank}...")

        judgment = judge_chunk(
            client=client,
            question=str(row["query"]),
            expected_answer=str(row["expected_answer"]),
            chunk_text=str(row["evidence_text"]),
        )

        result = row.to_dict()

        result["supported"] = judgment.supported
        result["judge_reason"] = judgment.reason

        results.append(result)

        # Incremental save
        output_df = pd.DataFrame(results)

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_df.to_csv(
            OUTPUT_PATH,
            index=False,
        )

        print(f"  supported={judgment.supported}")
        print(f"  reason={judgment.reason}")

    print("\n=== EVIDENCE JUDGING COMPLETE ===")

    final_df = pd.DataFrame(results)

    print(f"Total judgments: {len(final_df)}")

    print("\nSupported distribution:")
    print(final_df["supported"].value_counts(dropna=False).to_string())

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
