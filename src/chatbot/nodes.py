# pyrefly: ignore [missing-import]
from src.chatbot.state import ChatbotState

# pyrefly: ignore [missing-import]
from src.retrieval.planner import QueryPlanner

# pyrefly: ignore [missing-import]
from src.retrieval.retriever import StoryRetriever

# pyrefly: ignore [missing-import]
from src.chatbot.generator import AnswerGenerator


class ChatbotNodes:
    """
    Nodes used by the LangGraph chatbot workflow.
    """

    def __init__(
        self,
        planner: QueryPlanner,
        retriever: StoryRetriever,
        generator: AnswerGenerator,
    ):
        self.planner = planner
        self.retriever = retriever
        self.generator = generator

    def planner_node(
        self,
        state: ChatbotState,
    ) -> dict:
        """
        Analyze the user query and produce a validated QueryPlan.
        """

        plan = self.planner.plan(state.query)

        return {
            "plan": plan,
        }

    def resolve_stories_node(
        self,
        state: ChatbotState,
    ) -> dict:
        """
        Resolve explicit story IDs and titles into canonical story IDs.
        """

        if state.plan is None:
            raise ValueError("Query plan is required before story resolution.")

        resolved_ids: list[str] = []

        # Explicit IDs from the planner
        for story_id in state.plan.story_ids:
            story = self.retriever.get_story_by_id(story_id)

            if not story.empty:
                resolved_ids.append(str(story.iloc[0]["id"]))

        # Explicit titles from the planner
        for title in state.plan.story_titles:
            story = self.retriever.get_story_by_title(title)

            if not story.empty:
                resolved_ids.append(str(story.iloc[0]["id"]))

        # Remove duplicates while preserving order
        resolved_ids = list(dict.fromkeys(resolved_ids))

        return {
            "story_ids": resolved_ids,
        }

    def route_retrieval(
        self,
        state: ChatbotState,
    ) -> str:
        """
        Route execution according to the retrieval strategy
        selected by the query planner.
        """

        if state.plan is None:
            raise ValueError("Query plan is required before routing.")

        return state.plan.retrieval_strategy

    def evidence_retrieval_node(
        self,
        state: ChatbotState,
    ) -> dict:
        """
        Retrieve the most relevant evidence chunks
        from already resolved stories.
        """

        if not state.story_ids:
            return {
                "evidence": [],
            }

        evidence_df = self.retriever.retrieve_evidence(
            query=state.query,
            candidate_story_ids=state.story_ids,
            top_k=5,
        )

        return {
            "evidence": evidence_df.to_dict(orient="records"),
        }

    def full_story_node(
        self,
        state: ChatbotState,
    ) -> dict:
        """
        Retrieve the complete canonical text for
        explicitly resolved stories.
        """

        if not state.story_ids:
            return {
                "stories": [],
            }

        stories: list[dict] = []

        for story_id in state.story_ids:
            story_df = self.retriever.get_story_by_id(story_id)

            if story_df.empty:
                continue

            stories.extend(story_df.to_dict(orient="records"))

        return {
            "stories": stories,
        }

    def semantic_node(
        self,
        state: ChatbotState,
    ) -> dict:
        """
        Discover stories using global semantic search.
        """

        if state.plan is None:
            raise ValueError("Query plan is required before semantic retrieval.")

        semantic_query = state.plan.semantic_query or state.query

        stories_df = self.retriever.retrieve_stories(
            query=semantic_query,
            top_k=5,
        )

        if stories_df.empty:
            return {
                "story_ids": [],
                "stories": [],
            }

        story_ids = stories_df["story_id"].astype(str).tolist()

        return {
            "story_ids": story_ids,
            "stories": stories_df.to_dict(orient="records"),
        }

    def metadata_node(
        self,
        state: ChatbotState,
    ) -> dict:
        """
        Retrieve stories using deterministic structured metadata.
        Supports resolved story references and genre filtering.
        """

        if state.plan is None:
            raise ValueError("Query plan is required before metadata retrieval.")

        # Case 1: explicit story ID/title already resolved
        if state.story_ids:
            stories = []

            for story_id in state.story_ids:
                story_df = self.retriever.get_story_by_id(story_id)

                if story_df.empty:
                    continue

                record = story_df.iloc[0]

                stories.append(
                    {
                        "id": str(record["id"]),
                        "title": record["title"],
                        "genre": record["genre"],
                    }
                )

            return {
                "story_ids": state.story_ids,
                "stories": stories,
            }

        # Case 2: genre filtering
        if state.plan.genre is not None:
            stories_df = self.retriever.get_stories_by_genre(state.plan.genre)

            if stories_df.empty:
                return {
                    "story_ids": [],
                    "stories": [],
                }

            story_ids = stories_df["id"].astype(str).tolist()

            return {
                "story_ids": story_ids,
                "stories": stories_df.to_dict(orient="records"),
            }

        return {
            "story_ids": [],
            "stories": [],
        }

    def metadata_semantic_node(
        self,
        state: ChatbotState,
    ) -> dict:
        """
        Filter stories using structured metadata,
        then rank the filtered candidates semantically.
        """

        if state.plan is None:
            raise ValueError(
                "Query plan is required before " "metadata-semantic retrieval."
            )

        if state.plan.genre is None:
            return {
                "story_ids": [],
                "stories": [],
            }

        # Stage 1: deterministic metadata filtering
        candidates_df = self.retriever.get_stories_by_genre(state.plan.genre)

        if candidates_df.empty:
            return {
                "story_ids": [],
                "stories": [],
            }

        candidate_story_ids = candidates_df["id"].astype(str).tolist()

        # Stage 2: semantic ranking inside
        # the metadata-filtered candidates
        semantic_query = state.plan.semantic_query or state.query

        ranked_df = self.retriever.retrieve_stories(
            query=semantic_query,
            candidate_story_ids=candidate_story_ids,
            top_k=5,
        )

        if ranked_df.empty:
            return {
                "story_ids": [],
                "stories": [],
            }

        story_ids = ranked_df["story_id"].astype(str).tolist()

        return {
            "story_ids": story_ids,
            "stories": ranked_df.to_dict(orient="records"),
        }

    def context_builder_node(
        self,
        state: ChatbotState,
    ) -> dict:
        """
        Convert retrieved data into a clean textual context
        for the answer generation model.
        """

        if state.plan is None:
            raise ValueError("Query plan is required before context building.")

        strategy = state.plan.retrieval_strategy

        # -----------------------------------
        # Evidence-based story QA
        # -----------------------------------
        if strategy == "evidence":
            if not state.evidence:
                return {"context": ""}

            sections = []

            for i, chunk in enumerate(
                state.evidence,
                start=1,
            ):
                sections.append(
                    f"[Evidence {i}]\n"
                    f"Title: {chunk['title']}\n"
                    f"Genre: {chunk['genre']}\n"
                    f"Text: {chunk['text']}"
                )

            return {"context": "\n\n".join(sections)}

        # -----------------------------------
        # Full-story operations
        # -----------------------------------
        if strategy == "full_story":
            if not state.stories:
                return {"context": ""}

            sections = []

            for story in state.stories:
                sections.append(
                    f"Title: {story['title']}\n"
                    f"Genre: {story['genre']}\n"
                    f"Story:\n{story['story']}"
                )

            return {"context": "\n\n".join(sections)}

        # -----------------------------------
        # Story discovery / metadata results
        # -----------------------------------
        if strategy in {
            "semantic",
            "metadata",
            "metadata_semantic",
        }:
            if not state.stories:
                return {"context": ""}

            sections = []

            for i, story in enumerate(
                state.stories,
                start=1,
            ):
                section = (
                    f"[Story {i}]\n"
                    f"ID: "
                    f"{story.get('story_id', story.get('id'))}\n"
                    f"Title: {story['title']}\n"
                    f"Genre: {story['genre']}"
                )

                sections.append(section)

            return {"context": "\n\n".join(sections)}

        raise ValueError(f"Unsupported retrieval strategy: {strategy}")

    def generator_node(
        self,
        state: ChatbotState,
    ) -> dict:
        """
        Generate the final grounded answer.
        """

        if state.context is None:
            raise ValueError("Context must be built before generation.")

        answer = self.generator.generate(
            query=state.query, context=state.context, intent=state.plan.intent
        )

        return {
            "answer": answer,
        }

    def generator_stream(
        self,
        state: ChatbotState,
    ):
        """
        Stream the final grounded answer.
        """

        if state.context is None:
            raise ValueError("Context must be built before generation.")

        yield from self.generator.generate_stream(
            query=state.query,
            context=state.context,
            intent=state.plan.intent,
        )
