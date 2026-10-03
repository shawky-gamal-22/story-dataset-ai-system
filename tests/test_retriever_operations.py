# pyrefly: ignore [missing-import]
from src.retrieval.retriever import StoryRetriever


def main():
    retriever = StoryRetriever()

    # --------------------------------------------------
    # 1. ID lookup
    # --------------------------------------------------

    print("\n=== ID LOOKUP ===")

    story = retriever.get_story_by_id("987253")

    print(story[["id", "title", "genre"]].to_string(index=False))

    # --------------------------------------------------
    # 2. Exact title lookup
    # --------------------------------------------------

    print("\n=== TITLE LOOKUP ===")

    story = retriever.get_story_by_title("The Chronicles of the Celestial Spoon")

    print(story[["id", "title", "genre"]].to_string(index=False))

    # --------------------------------------------------
    # 3. Genre filtering
    # --------------------------------------------------

    print("\n=== GENRE FILTER ===")

    fantasy = retriever.get_stories_by_genre("Fantasy")

    print(f"Fantasy stories: {len(fantasy)}")

    print(fantasy[["id", "title"]].head().to_string(index=False))

    # --------------------------------------------------
    # 4. Metadata + semantic retrieval
    # --------------------------------------------------

    print("\n=== METADATA + SEMANTIC ===")

    ranked = retriever.retrieve_stories(
        query="friendship and adventure",
        candidate_story_ids=(fantasy["id"].astype(str).tolist()),
        top_k=3,
    )

    print(
        ranked[
            [
                "story_id",
                "title",
                "genre",
                "score",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
