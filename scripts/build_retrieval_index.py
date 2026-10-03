import json
from pathlib import Path

import numpy as np

# pyrefly: ignore [missing-import]
from src.data.loader import load_story_dataset

# pyrefly: ignore [missing-import]
from src.retrieval.index_builder import RetrievalIndexBuilder

# pyrefly: ignore [missing-import]
from src.retrieval.vector_store import FaissVectorStore


ARTIFACT_DIR = Path("data/artifacts/retrieval")


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    stories_df = load_story_dataset()

    print(f"Loaded {len(stories_df):,} stories.\n")

    # --------------------------------------------------
    # 0. Canonical story store
    # --------------------------------------------------

    stories_df.to_parquet(
        ARTIFACT_DIR / "stories.parquet",
        index=False,
    )

    print(f"Saved canonical story store: {len(stories_df):,} stories.\n")

    builder = RetrievalIndexBuilder()

    # --------------------------------------------------
    # 1. Story-level representations
    # --------------------------------------------------

    print("Building story representations...")

    story_vectors, story_metadata = builder.build_story_representations(stories_df)

    print(f"Story vectors shape: {story_vectors.shape}")

    # FAISS exact cosine search
    story_store = FaissVectorStore(dimension=story_vectors.shape[1])

    story_store.add(story_vectors)

    story_store.save(ARTIFACT_DIR / "story.index")

    story_metadata.to_parquet(
        ARTIFACT_DIR / "story_metadata.parquet",
        index=False,
    )

    # --------------------------------------------------
    # 2. Evidence representations
    # --------------------------------------------------

    print("\nBuilding evidence representations...")

    evidence_vectors, evidence_metadata = builder.build_evidence_representations(
        stories_df
    )

    print(f"Evidence vectors shape: {evidence_vectors.shape}")

    np.save(
        ARTIFACT_DIR / "evidence_embeddings.npy",
        evidence_vectors,
    )

    evidence_metadata.to_parquet(
        ARTIFACT_DIR / "evidence_metadata.parquet",
        index=False,
    )

    # --------------------------------------------------
    # 3. Configuration
    # --------------------------------------------------

    config = {
        "embedding_model": builder.model_name,
        "embedding_dimension": int(story_vectors.shape[1]),
        "chunk_size": builder.chunk_size,
        "story_overlap": 0,
        "evidence_overlap": builder.evidence_overlap,
        "num_stories": len(story_metadata),
        "num_evidence_chunks": len(evidence_metadata),
        "story_index": "IndexFlatIP",
        "similarity": "cosine",
    }

    with open(
        ARTIFACT_DIR / "index_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    print("\nRetrieval artifacts saved successfully.")

    print("\nArtifacts:")
    for path in ARTIFACT_DIR.iterdir():
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
