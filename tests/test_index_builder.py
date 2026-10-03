# pyrefly: ignore [missing-import]
from src.data.loader import load_story_dataset

# pyrefly: ignore [missing-import]
from src.retrieval.index_builder import RetrievalIndexBuilder


def main():

    stories_df = load_story_dataset()

    # Small smoke test only
    sample_df = stories_df.head(5).copy()

    print(f"Testing with {len(sample_df)} stories...\n")

    builder = RetrievalIndexBuilder()

    story_vectors, story_metadata = builder.build_story_representations(sample_df)

    evidence_vectors, evidence_metadata = builder.build_evidence_representations(
        sample_df
    )

    print("\n--- STORY INDEX ---")
    print("Vectors:", story_vectors.shape)
    print("Metadata:", story_metadata.shape)

    print("\n--- EVIDENCE INDEX ---")
    print("Vectors:", evidence_vectors.shape)
    print("Metadata:", evidence_metadata.shape)

    print("\nStory metadata:")
    print(story_metadata.head())

    print("\nEvidence metadata:")
    print(
        evidence_metadata[
            [
                "story_id",
                "chunk_id",
                "start_token",
                "end_token",
            ]
        ].head()
    )


if __name__ == "__main__":
    main()
