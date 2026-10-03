# pyrefly: ignore [missing-import]
from langchain_core.outputs import chat_result
from langchain_core.outputs import chat_result
from langchain_core.outputs import chat_result
from langchain_core.outputs import chat_result

# pyrefly: ignore [missing-import]
from src.chatbot.graph import build_chatbot_graph

# pyrefly: ignore [missing-import]
from src.chatbot.state import ChatbotState


def main():
    graph = build_chatbot_graph()

    initial_state = ChatbotState(
        query=(
            "In The Chronicles of the Celestial Spoon, "
            "what is the name of the spoon that can create "
            "horrifying and inedible dishes?"
        )
    )

    result = graph.invoke(initial_state)

    print("\n=== QUERY ===")
    print(result["query"])

    print("\n=== PLAN ===")
    print(result["plan"])

    print("\n=== RESOLVED STORY IDS ===")
    print(result["story_ids"])

    print("\n=== EVIDENCE ===")

    for i, chunk in enumerate(result["evidence"], start=1):
        print(f"Chunk {i} (score={chunk['score']:.4f})")
        print(chunk["text"])

    print("\n=== CONTEXT ===")
    print(result["context"])

    print("\n=== FINAL ANSWER ===")
    print(result["answer"])


if __name__ == "__main__":
    main()
