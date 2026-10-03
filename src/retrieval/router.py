from typing import Literal

from pydantic import BaseModel, Field


class QueryPlan(BaseModel):
    intent: Literal[
        "story_qa",
        "summarization",
        "story_discovery",
        "metadata_lookup",
        "comparison",
    ]

    retrieval_strategy: Literal[
        "evidence",
        "full_story",
        "semantic",
        "metadata",
        "metadata_semantic",
    ]

    story_titles: list[str] = Field(default_factory=list)
    story_ids: list[str] = Field(default_factory=list)

    genre: str | None = None
    semantic_query: str | None = None
