from openai import OpenAI
from pydantic import ValidationError

# pyrefly: ignore [missing-import]
from src.retrieval.router import QueryPlan

# pyrefly: ignore [missing-import]
from src.retrieval.planner_prompt import QUERY_PLANNER_PROMPT
from src.retrieval.fast_planner import try_fast_plan


class QueryPlanner:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        model: str = "Qwen/Qwen3-4B-GGUF:Q4_K_M",
    ):
        self.client = OpenAI(
            base_url=base_url,
            api_key="local",
        )

        self.model = model

    def plan(self, user_query: str) -> QueryPlan:
        fast_plan = try_fast_plan(user_query)

        if fast_plan is not None:
            print(f"[FAST PLAN] {fast_plan} -- [USER QUERY]  {user_query}")
            return fast_plan

        print(f"[LLM PLAN] {user_query}")

        prompt = QUERY_PLANNER_PROMPT.replace(
            "__USER_QUERY__",
            user_query,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt + "\n/no_think",
                }
            ],
            temperature=0.0,
            max_tokens=256,
        )

        content = response.choices[0].message.content

        if not content:
            raise ValueError("Query planner returned an empty response.")

        return self._parse_plan(content)

    def _parse_plan(self, content: str) -> QueryPlan:
        content = content.strip()

        if content.startswith("```"):
            lines = content.splitlines()

            if lines[0].startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            content = "\n".join(lines).strip()

        try:
            return QueryPlan.model_validate_json(content)

        except ValidationError as exc:
            raise ValueError(
                f"Planner returned an invalid query plan:\n{content}"
            ) from exc
