from src.retrieval.planner import QueryPlanner


def main():
    planner = QueryPlanner()

    queries = [
        # Specific story QA
        (
            'In "The Chronicles of the Celestial Spoon", '
            "what spoon creates horrifying dishes?"
        ),
        # Summarization
        ('Summarize "The Chronicles of the Celestial Spoon".'),
        # Pure semantic discovery
        ("Find me a story about friendship and adventure."),
        # Metadata + semantic
        ("Find me a Fantasy story involving " "friendship and adventure."),
        # Pure metadata
        ("Show me stories from the Mystery genre."),
        # ID reference
        ("What is story ID 987253 about?"),
        # Comparison
        ('Compare "Story A" and "Story B".'),
        # Dynamic phrasing
        (
            "I want something scary where the main character "
            "is trapped somewhere and tries to escape."
        ),
    ]

    for i, query in enumerate(queries, start=1):
        print("=" * 80)
        print(f"QUERY {i}:")
        print(query)

        try:
            plan = planner.plan(query)

            print("\nPLAN:")
            print(plan)

        except Exception as exc:
            print("\nERROR:")
            print(exc)

        print()


if __name__ == "__main__":
    main()
