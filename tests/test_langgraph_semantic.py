# pyrefly: ignore [missing-import]
from src.chatbot.graph import build_chatbot_graph

# pyrefly: ignore [missing-import]
from src.chatbot.state import ChatbotState


def main():
    graph = build_chatbot_graph()

    initial_state = ChatbotState(
        query=("Find me stories involving " "friendship and adventure.")
    )

    result = graph.invoke(initial_state)

    print("\n=== QUERY ===")
    print(result["query"])

    print("\n=== PLAN ===")
    print(result["plan"])

    print("\n=== STORY IDS ===")
    print(result["story_ids"])

    print("\n=== SEMANTIC RESULTS ===")

    for i, story in enumerate(
        result["stories"],
        start=1,
    ):
        print(
            f"{i}. {story['title']} "
            f"[{story['genre']}] "
            f"score={story['score']:.4f}"
        )

    print("\n=== FINAL ANSWER ===")
    print(result["answer"])


if __name__ == "__main__":
    main()
