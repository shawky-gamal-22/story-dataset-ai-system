from typing import Any

from pydantic import BaseModel, Field, ConfigDict

# pyrefly: ignore [missing-import]
from src.retrieval.router import QueryPlan


class ChatbotState(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    # Original user request
    query: str

    # Validated plan produced by QueryPlanner
    plan: QueryPlan | None = None

    # Resolved story IDs
    story_ids: list[str] = Field(default_factory=list)

    # Story records selected during execution
    stories: list[dict[str, Any]] = Field(default_factory=list)

    # Retrieved evidence chunks
    evidence: list[dict[str, Any]] = Field(default_factory=list)

    # Context prepared for generation
    context: str | None = None

    # Final generated response
    answer: str | None = None
