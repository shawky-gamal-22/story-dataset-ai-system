# pyrefly: ignore [missing-import]
from src.chatbot.graph import build_chatbot_graph

# pyrefly: ignore [missing-import]
from src.chatbot.state import ChatbotState


def run_query(graph, query):
    result = graph.invoke(ChatbotState(query=query))

    print("\n" + "=" * 70)
    print("QUERY:")
    print(query)

    print("\nPLAN:")
    print(result["plan"])

    print("\nRESOLVED IDS:")
    print(result["story_ids"])

    print("\nFINAL ANSWER:")
    print(result["answer"])


def main():
    graph = build_chatbot_graph()

    queries = [
        "What genre is story ID 297904?",
        'What is the ID of "The Enigma of Elmwood Manor"?',
        "Show me stories from the Mystery genre.",
    ]

    for query in queries:
        run_query(graph, query)


if __name__ == "__main__":
    main()
