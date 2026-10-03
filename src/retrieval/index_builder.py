from typing import Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# pyrefly: ignore [missing-import]
from src.retrieval.chunking import chunk_text_by_tokens


class RetrievalIndexBuilder:
    """
    Build story-level and evidence-level vector representations.

    Story representation:
        non-overlapping chunks -> embeddings -> mean pooling -> normalization

    Evidence representation:
        overlapping chunks -> individual normalized embeddings
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 254,
        evidence_overlap: int = 50,
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.evidence_overlap = evidence_overlap
        self.batch_size = batch_size

        self.model = SentenceTransformer(model_name)
        self.tokenizer = self.model.tokenizer

    def build_story_representations(
        self,
        stories_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Build one vector per story using non-overlapping chunk mean pooling.
        """

        story_vectors = []
        metadata = []

        total = len(stories_df)

        for position, (_, row) in enumerate(stories_df.iterrows(), start=1):

            chunks = chunk_text_by_tokens(
                text=row["story"],
                tokenizer=self.tokenizer,
                chunk_size=self.chunk_size,
                overlap=0,
            )

            chunk_texts = [chunk["text"] for chunk in chunks]

            chunk_embeddings = self.model.encode(
                chunk_texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            # Mean pooling across all chunks belonging to this story
            story_vector = chunk_embeddings.mean(axis=0)

            # Re-normalize after mean pooling
            norm = np.linalg.norm(story_vector)

            if norm > 0:
                story_vector = story_vector / norm

            story_vectors.append(story_vector.astype(np.float32))

            metadata.append(
                {
                    "story_id": str(row["id"]),
                    "title": row["title"],
                    "genre": row["genre"],
                    "num_chunks": len(chunks),
                }
            )

            if position % 100 == 0 or position == total:
                print(f"Story representations: " f"{position}/{total}")

        vectors = np.vstack(story_vectors).astype(np.float32)

        metadata_df = pd.DataFrame(metadata)

        return vectors, metadata_df

    def build_evidence_representations(
        self,
        stories_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Build overlapping evidence chunks and one vector per chunk.
        """

        evidence_records = []

        total = len(stories_df)

        for position, (_, row) in enumerate(stories_df.iterrows(), start=1):

            chunks = chunk_text_by_tokens(
                text=row["story"],
                tokenizer=self.tokenizer,
                chunk_size=self.chunk_size,
                overlap=self.evidence_overlap,
            )

            for chunk in chunks:
                evidence_records.append(
                    {
                        "story_id": str(row["id"]),
                        "title": row["title"],
                        "genre": row["genre"],
                        "chunk_id": chunk["chunk_id"],
                        "start_token": chunk["start_token"],
                        "end_token": chunk["end_token"],
                        "text": chunk["text"],
                    }
                )

            if position % 100 == 0 or position == total:
                print(f"Evidence chunking: " f"{position}/{total}")

        evidence_df = pd.DataFrame(evidence_records)

        print(f"\nEncoding {len(evidence_df):,} evidence chunks...")

        evidence_vectors = self.model.encode(
            evidence_df["text"].tolist(),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        evidence_vectors = np.asarray(
            evidence_vectors,
            dtype=np.float32,
        )

        return evidence_vectors, evidence_df
