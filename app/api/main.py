from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    ChatRequest,
    ClassificationRequest,
    ClassificationResponse,
)
from src.chatbot.graph import build_chatbot_graph
from src.chatbot.generator import AnswerGenerator
from src.classification.classifier import CPUGenreClassifier


classifier: CPUGenreClassifier | None = None
chat_graph = None
answer_generator: AnswerGenerator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier
    global chat_graph
    global answer_generator

    print("Loading CPU genre classifier...")
    classifier = CPUGenreClassifier()
    print("CPU genre classifier ready.")

    print("Building Task 1 retrieval graph...")
    chat_graph = build_chatbot_graph(
        include_generator=False,
    )
    answer_generator = AnswerGenerator()
    print("Task 1 graph ready.")

    yield

    classifier = None
    chat_graph = None
    answer_generator = None


app = FastAPI(
    title="Story Dataset AI System",
    description=(
        "CPU-deployable conversational story QA " "and genre classification system."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "classifier_loaded": classifier is not None,
        "chat_graph_loaded": chat_graph is not None,
        "generator_loaded": answer_generator is not None,
    }


@app.post(
    "/api/v1/classify",
    response_model=ClassificationResponse,
)
def classify_story(
    request: ClassificationRequest,
) -> ClassificationResponse:

    if classifier is None:
        raise RuntimeError("Genre classifier is not initialized.")

    result = classifier.classify(request.story)

    return ClassificationResponse(
        predicted_genre=result.predicted_genre,
    )


@app.post("/api/v1/chat")
def chat(request: ChatRequest):

    if chat_graph is None or answer_generator is None:
        raise HTTPException(
            status_code=503,
            detail="Chat system is not initialized.",
        )

    result = chat_graph.invoke(
        {
            "query": request.query,
        }
    )

    context = result.get("context")
    plan = result.get("plan")

    if not context:
        raise HTTPException(
            status_code=404,
            detail="No relevant story context was found.",
        )

    if plan is None:
        raise HTTPException(
            status_code=500,
            detail="Query planning failed.",
        )

    return StreamingResponse(
        answer_generator.generate_stream(
            query=request.query,
            context=context,
            intent=plan.intent,
        ),
        media_type="text/plain",
    )
