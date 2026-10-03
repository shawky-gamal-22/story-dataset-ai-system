from openai import OpenAI
from langsmith.wrappers import wrap_openai

# pyrefly: ignore [missing-import]
from src.chatbot.prompts import (
    QA_SYSTEM_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    DISCOVERY_SYSTEM_PROMPT,
    METADATA_SYSTEM_PROMPT,
)
from dotenv import load_dotenv

load_dotenv()


class AnswerGenerator:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        model: str = "Qwen/Qwen3-4B-GGUF:Q4_K_M",
    ):
        self.client = wrap_openai(
            OpenAI(
                base_url=base_url,
                api_key="local",
            )
        )

        self.model = model

    def generate_stream(
        self,
        query: str,
        context: str,
        intent: str,
    ):

        if not context.strip():
            raise ValueError("Cannot stream an answer without retrieved context.")

        system_prompt = self._select_prompt(intent)

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context}\n\n"
                        f"User request:\n{query}\n\n"
                        "/no_think"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=512,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content

            if delta:
                yield delta

    def generate(self, query: str, context: str, intent: str) -> str:

        if not context.strip():
            return "I could not find enough evidence " "in the retrieved story context."

        system_prompt = self._select_prompt(intent)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context}\n\n"
                        f"User request:\n{query}\n\n"
                        "/no_think"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=512,
        )

        content = response.choices[0].message.content

        if not content:
            raise ValueError("Answer generator returned an empty response.")

        return content.strip()

    def _select_prompt(
        self,
        intent: str,
    ) -> str:

        if intent == "story_qa":
            return QA_SYSTEM_PROMPT

        if intent in {
            "summarization",
            "comparison",
        }:
            return SUMMARY_SYSTEM_PROMPT

        if intent == "story_discovery":
            return DISCOVERY_SYSTEM_PROMPT

        if intent == "metadata_lookup":
            return METADATA_SYSTEM_PROMPT

        raise ValueError(f"Unsupported intent: {intent}")
