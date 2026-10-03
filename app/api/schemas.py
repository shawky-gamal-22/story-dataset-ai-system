from pydantic import BaseModel, Field


class ClassificationRequest(BaseModel):
    story: str = Field(
        ...,
        min_length=1,
        description="Story text to classify.",
    )


class ClassificationResponse(BaseModel):
    predicted_genre: str


class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Question or request about the story dataset.",
    )
