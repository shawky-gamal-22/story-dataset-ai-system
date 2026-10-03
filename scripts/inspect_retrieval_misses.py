import pandas as pd

from src.retrieval.retriever import StoryRetriever


MISSED_QUERIES = {
    "Q017": (
        "Find stories about futuristic societies, "
        "advanced technology, or space exploration."
    ),
    "Q020": (
        "Find stories involving investigations, " "hidden clues, and solving a mystery."
    ),
    "Q024": (
        "Find stories set in or strongly connected " "to historical periods and events."
    ),
    "Q026": (
        "Find stories about interstellar conflicts, "
        "galactic missions, or space-faring heroes."
    ),
}


def main():
    retriever = StoryRetriever()

    for query_id, query in MISSED_QUERIES.items():
        print("\n" + "=" * 80)
        print(query_id)
        print(query)

        results = retriever.retrieve_stories(
            query=query,
            top_k=5,
        )

        if isinstance(results, pd.DataFrame):
            print(
                results[
                    [
                        col
                        for col in [
                            "story_id",
                            "id",
                            "title",
                            "genre",
                            "score",
                        ]
                        if col in results.columns
                    ]
                ].to_string(index=False)
            )

        else:
            for rank, result in enumerate(
                results,
                start=1,
            ):
                print(
                    rank,
                    result.get(
                        "story_id",
                        result.get("id"),
                    ),
                    result.get("title"),
                    result.get("genre"),
                    result.get("score"),
                )


if __name__ == "__main__":
    main()
