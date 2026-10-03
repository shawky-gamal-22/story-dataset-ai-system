from langgraph.graph import END, START, StateGraph

# pyrefly: ignore [missing-import]
from src.chatbot.nodes import ChatbotNodes

# pyrefly: ignore [missing-import]
from src.chatbot.state import ChatbotState

# pyrefly: ignore [missing-import]
from src.retrieval.planner import QueryPlanner

# pyrefly: ignore [missing-import]
from src.retrieval.retriever import StoryRetriever

# pyrefly: ignore [missing-import]
from src.chatbot.generator import AnswerGenerator


def build_chatbot_graph(
    include_generator: bool = True,
):
    planner = QueryPlanner()
    retriever = StoryRetriever()
    generator = AnswerGenerator()

    nodes = ChatbotNodes(
        planner=planner,
        retriever=retriever,
        generator=generator,
    )

    graph = StateGraph(ChatbotState)

    # -------------------------
    # Nodes
    # -------------------------

    graph.add_node(
        "planner",
        nodes.planner_node,
    )

    graph.add_node(
        "resolve_stories",
        nodes.resolve_stories_node,
    )

    graph.add_node(
        "evidence",
        nodes.evidence_retrieval_node,
    )

    graph.add_node(
        "full_story",
        nodes.full_story_node,
    )

    graph.add_node(
        "semantic",
        nodes.semantic_node,
    )

    graph.add_node(
        "metadata",
        nodes.metadata_node,
    )

    graph.add_node(
        "metadata_semantic",
        nodes.metadata_semantic_node,
    )
    graph.add_node(
        "context_builder",
        nodes.context_builder_node,
    )

    graph.add_node(
        "generator",
        nodes.generator_node,
    )

    # -------------------------
    # Entry
    # -------------------------

    graph.add_edge(
        START,
        "planner",
    )

    # -------------------------
    # Conditional routing
    # -------------------------

    graph.add_conditional_edges(
        "planner",
        nodes.route_retrieval,
        {
            "evidence": "resolve_stories",
            "full_story": "resolve_stories",
            "semantic": "semantic",
            "metadata": "resolve_stories",
            "metadata_semantic": "metadata_semantic",
        },
    )

    # Explicit-story operations need resolution first.
    graph.add_conditional_edges(
        "resolve_stories",
        nodes.route_retrieval,
        {
            "evidence": "evidence",
            "full_story": "full_story",
            "metadata": "metadata",
        },
    )

    graph.add_edge(
        "evidence",
        "context_builder",
    )

    graph.add_edge(
        "full_story",
        "context_builder",
    )

    graph.add_edge(
        "semantic",
        "context_builder",
    )

    graph.add_edge(
        "metadata",
        "context_builder",
    )

    graph.add_edge(
        "metadata_semantic",
        "context_builder",
    )

    if include_generator:
        graph.add_edge(
            "context_builder",
            "generator",
        )

        graph.add_edge(
            "generator",
            END,
        )
    else:
        graph.add_edge(
            "context_builder",
            END,
        )

    return graph.compile()
