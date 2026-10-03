# pyrefly: ignore [missing-import]
from src.retrieval.retriever import StoryRetriever


def main():

    retriever = StoryRetriever()

    # ----------------------------------------
    # Test 1: Semantic story retrieval
    # ----------------------------------------

    query = "a story involving time travel"

    stories = retriever.retrieve_stories(
        query=query,
        top_k=5,
    )

    print("\n=== STORY RETRIEVAL ===")

    print(
        stories[
            [
                "story_id",
                "title",
                "genre",
                "score",
            ]
        ].to_string(index=False)
    )

    # ----------------------------------------
    # Test 2: Evidence retrieval
    # ----------------------------------------

    query = (
        'In "The Chronicles of the Celestial Spoon", '
        "what is the name of the spoon that can create "
        "horrifying and inedible dishes?"
    )

    # For now we're supplying the known story manually.
    # The router will automate this later.
    evidence = retriever.retrieve_evidence(
        query=query,
        candidate_story_ids=["987253"],
        top_k=5,
    )

    print("\n=== EVIDENCE RETRIEVAL ===")

    for _, row in evidence.iterrows():

        print(
            f"\nStory: {row['title']}"
            f"\nChunk: {row['chunk_id']}"
            f"\nScore: {row['score']:.4f}"
        )

        print(row["text"][:500])

        print("-" * 80)


if __name__ == "__main__":
    main()
