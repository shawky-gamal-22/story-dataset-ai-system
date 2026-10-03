# pyrefly: ignore [missing-import]
from src.chatbot.graph import build_chatbot_graph

# pyrefly: ignore [missing-import]
from src.chatbot.state import ChatbotState


def main():
    graph = build_chatbot_graph()

    initial_state = ChatbotState(
        query=('Summarize "The Chronicles ' 'of the Celestial Spoon".')
    )

    result = graph.invoke(initial_state)

    print("\n=== QUERY ===")
    print(result["query"])

    print("\n=== PLAN ===")
    print(result["plan"])

    print("\n=== RESOLVED STORY IDS ===")
    print(result["story_ids"])

    print("\n=== STORIES ===")

    for story in result["stories"]:
        print(f"ID: {story['id']}")
        print(f"Title: {story['title']}")
        print(f"Genre: {story['genre']}")

        story_text = story["story"]

        print(f"Story length: " f"{len(story_text.split()):,} words")

        print("\nPreview:")
        print(story_text[:500])

    print("\n=== FINAL ANSWER ===")
    print(result["answer"])


if __name__ == "__main__":
    main()
