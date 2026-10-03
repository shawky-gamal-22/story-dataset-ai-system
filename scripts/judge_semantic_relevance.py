import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field, ValidationError


INPUT_PATH = Path("experiments/task1/semantic_judgment_pool.csv")

OUTPUT_PATH = Path("experiments/task1/semantic_judgments_gpt_oss_120b.csv")

MODEL_NAME = "openai/gpt-oss-120b"
MAX_RETRIES = 5


class RelevanceJudgment(BaseModel):
    relevance: int = Field(ge=0, le=2)
    reason: str


JUDGE_PROMPT = """
You are evaluating semantic story retrieval.

Your task is to judge whether a story is relevant to a user's
story-search query.

Judge relevance ONLY from:
- the user query
- the story title
- the story genre
- the story content

Do not assume relevance merely because of the genre or title.
Use the actual story content as the primary evidence.

Relevance scale:

2 = CLEARLY RELEVANT
The story directly and substantially matches at least one of the
main concepts requested by the query.

1 = PARTIALLY RELEVANT
The story has a real connection to the query, but the connection
is secondary, weak, brief, or only indirectly related.

0 = NOT RELEVANT
The story has no meaningful connection to the requested concepts.

Important rules:

1. If the query uses "or", a story does NOT need to satisfy every
   concept. A substantial match to one requested concept can be
   clearly relevant.

2. If the query lists several related concepts without explicitly
   requiring all of them, judge whether the story meaningfully
   satisfies the overall information need.

3. Do not reward superficial keyword overlap.

4. Do not penalize a story because its genre label differs from
   what might be expected.

5. Base the judgment on semantic meaning and actual story content.

Return ONLY valid JSON in exactly this form:

{
  "relevance": 0,
  "reason": "One short sentence."
}

Do not include markdown or any text outside the JSON.
Return the JSON object immediately without analysis or reasoning.
Keep the reason to one short sentence.

USER QUERY:
__QUERY__

STORY TITLE:
__TITLE__

STORY GENRE:
__GENRE__

STORY:
__STORY__
""".strip()


def build_prompt(row: pd.Series) -> str:
    return (
        JUDGE_PROMPT.replace("__QUERY__", str(row["query"]))
        .replace("__TITLE__", str(row["title"]))
        .replace("__GENRE__", str(row["genre"]))
        .replace("__STORY__", str(row["story"]))
    )


def judge_story(
    client: Groq,
    row: pd.Series,
) -> RelevanceJudgment:
    prompt = build_prompt(row)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content

            if not content:
                raise ValueError("Judge returned empty content.")

            data = json.loads(content)

            return RelevanceJudgment(**data)

        except (
            json.JSONDecodeError,
            ValidationError,
            ValueError,
            Exception,
        ) as exc:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Failed after {MAX_RETRIES} attempts: {exc}"
                ) from exc

            wait_seconds = 2**attempt

            print(f"Retry {attempt}/{MAX_RETRIES} " f"after error: {exc}")

            time.sleep(wait_seconds)

    raise RuntimeError("Unexpected judge failure.")


def load_existing_results() -> pd.DataFrame:
    if not OUTPUT_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(
        OUTPUT_PATH,
        dtype={"story_id": str},
    )


def save_results(df: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )


def main():
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing.")

    client = Groq(api_key=api_key)

    # --------------------------------------------------
    # 1. Load current candidate pool
    # --------------------------------------------------

    pool_df = pd.read_csv(
        INPUT_PATH,
        dtype={"story_id": str},
    )

    pool_df["query_id"] = pool_df["query_id"].astype(str)

    pool_df["story_id"] = pool_df["story_id"].astype(str)

    # --------------------------------------------------
    # 2. Load existing judgments
    # --------------------------------------------------

    existing_df = load_existing_results()

    judgment_columns = [
        "relevance",
        "judgment_notes",
        "judge_model",
    ]

    # The pool may already contain empty judgment
    # columns. Remove them before merging.
    pool_df = pool_df.drop(
        columns=[col for col in judgment_columns if col in pool_df.columns]
    )

    if existing_df.empty:
        results_df = pool_df.copy()

        results_df["relevance"] = pd.NA
        results_df["judgment_notes"] = pd.NA
        results_df["judge_model"] = pd.NA

    else:
        existing_df["query_id"] = existing_df["query_id"].astype(str)

        existing_df["story_id"] = existing_df["story_id"].astype(str)

        # Keep only valid completed judgments.
        completed_judgments = existing_df[existing_df["relevance"].notna()][
            [
                "query_id",
                "story_id",
                "relevance",
                "judgment_notes",
                "judge_model",
            ]
        ].drop_duplicates(
            subset=[
                "query_id",
                "story_id",
            ],
            keep="last",
        )

        # Reconstruct results from the CURRENT pool.
        # This prevents stale rows from old pools
        # remaining in the output.
        results_df = pool_df.merge(
            completed_judgments,
            on=[
                "query_id",
                "story_id",
            ],
            how="left",
            validate="one_to_one",
        )

    # --------------------------------------------------
    # 3. Determine genuinely completed pairs
    # --------------------------------------------------

    completed_mask = results_df["relevance"].notna()

    completed = {
        (
            str(row["query_id"]),
            str(row["story_id"]),
        )
        for _, row in results_df[completed_mask].iterrows()
    }

    total = len(results_df)

    print(f"Total pairs: {total}")
    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {total - len(completed)}")

    # --------------------------------------------------
    # 4. Judge only missing pairs
    # --------------------------------------------------

    for index, row in results_df.iterrows():
        key = (
            str(row["query_id"]),
            str(row["story_id"]),
        )

        if key in completed:
            continue

        current_number = len(completed) + 1

        print(
            f"\n[{current_number}/{total}] "
            f"{row['query_id']} | "
            f"{row['story_id']} | "
            f"{row['title']}"
        )

        try:
            judgment = judge_story(
                client,
                row,
            )

            # IMPORTANT:
            # Update the existing row in-place.
            # Do not append a duplicate row.
            results_df.at[
                index,
                "relevance",
            ] = judgment.relevance

            results_df.at[
                index,
                "judgment_notes",
            ] = judgment.reason

            results_df.at[
                index,
                "judge_model",
            ] = MODEL_NAME

            completed.add(key)

            # Save after EVERY judgment so the script
            # can safely resume after interruption.
            save_results(results_df)

            print(f"Relevance: " f"{judgment.relevance}")

            print(f"Reason: " f"{judgment.reason}")

        except Exception as exc:
            print(f"\nFAILED: {key}")
            print(exc)

            save_results(results_df)

            raise

    # --------------------------------------------------
    # 5. Final validation
    # --------------------------------------------------

    if results_df["relevance"].isna().any():
        missing = int(results_df["relevance"].isna().sum())

        raise RuntimeError(f"Judging finished with " f"{missing} missing judgments.")

    duplicate_count = int(
        results_df.duplicated(
            subset=[
                "query_id",
                "story_id",
            ]
        ).sum()
    )

    if duplicate_count:
        raise RuntimeError(f"Found {duplicate_count} duplicate " f"query-story pairs.")

    if len(results_df) != len(pool_df):
        raise RuntimeError(
            "Final judgment count does not match " "the current candidate pool."
        )

    # Convert back to integer after all NaNs
    # have been filled.
    results_df["relevance"] = results_df["relevance"].astype(int)

    save_results(results_df)

    # --------------------------------------------------
    # 6. Report
    # --------------------------------------------------

    print("\n=== JUDGING COMPLETE ===")

    print(f"Total judgments: {len(results_df)}")

    print(
        "Missing judgments:",
        int(results_df["relevance"].isna().sum()),
    )

    print(
        "Duplicate pairs:",
        int(
            results_df.duplicated(
                subset=[
                    "query_id",
                    "story_id",
                ]
            ).sum()
        ),
    )

    print("\nRelevance distribution:")

    print(results_df["relevance"].value_counts().sort_index().to_string())

    print("\nJudgments by query:")

    print(
        pd.crosstab(
            results_df["query_id"],
            results_df["relevance"],
        ).to_string()
    )

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
