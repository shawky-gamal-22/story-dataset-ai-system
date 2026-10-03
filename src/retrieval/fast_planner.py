import re

from src.retrieval.router import QueryPlan


STORY_ID_PATTERN = re.compile(
    r"\b(?:story\s+)?ID\s+(\d+)\b",
    re.IGNORECASE,
)

QUOTED_TITLE_PATTERN = re.compile(r'"([^"]+)"')


def try_fast_plan(
    user_query: str,
) -> QueryPlan | None:

    query = user_query.strip()
    query_lower = query.lower()

    story_ids = STORY_ID_PATTERN.findall(query)
    titles = QUOTED_TITLE_PATTERN.findall(query)

    # ----------------------------------
    # Comparison
    # ----------------------------------
    if query_lower.startswith("compare ") and len(titles) >= 2:
        return QueryPlan(
            intent="comparison",
            retrieval_strategy="full_story",
            story_titles=titles,
            story_ids=[],
            genre=None,
            semantic_query=None,
        )

    # ----------------------------------
    # Summarization
    # ----------------------------------
    summary_markers = (
        "summarize",
        "summary",
        "what is story",
        "what is the story",
        "what is story id",
    )

    if any(marker in query_lower for marker in summary_markers) and (
        titles or story_ids
    ):
        return QueryPlan(
            intent="summarization",
            retrieval_strategy="full_story",
            story_titles=titles,
            story_ids=story_ids,
            genre=None,
            semantic_query=None,
        )

    # ----------------------------------
    # Metadata lookup
    # ----------------------------------
    metadata_markers = (
        "what genre",
        "what is the genre",
        "which genre",
        "what is the id",
        "what id",
    )

    if any(marker in query_lower for marker in metadata_markers) and (
        titles or story_ids
    ):
        return QueryPlan(
            intent="metadata_lookup",
            retrieval_strategy="metadata",
            story_titles=titles,
            story_ids=story_ids,
            genre=None,
            semantic_query=None,
        )

    # ----------------------------------
    # Specific story QA
    # ----------------------------------
    if (
        "in " in query_lower
        and len(titles) == 1
        and (
            "who" in query_lower
            or "what" in query_lower
            or "where" in query_lower
            or "which" in query_lower
        )
        and query.endswith("?")
    ):
        return QueryPlan(
            intent="story_qa",
            retrieval_strategy="evidence",
            story_titles=titles,
            story_ids=story_ids,
            genre=None,
            semantic_query=None,
        )

    # ----------------------------------
    # No high-confidence rule
    # ----------------------------------
    return None
