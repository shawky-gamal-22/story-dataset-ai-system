from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# pyrefly: ignore [missing-import]
from src.retrieval.vector_store import FaissVectorStore


class StoryRetriever:
    """
    Hierarchical retriever for the story QA system.

    Stage 1:
        Retrieve candidate stories using the story-level FAISS index.

    Stage 2:
        Retrieve evidence chunks only from the selected candidate stories.
    """

    def __init__(
        self,
        artifact_dir: str | Path = "data/artifacts/retrieval",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.artifact_dir = Path(artifact_dir)

        # Embedding model
        self.model = SentenceTransformer(model_name)

        # Story-level FAISS index
        self.story_store = FaissVectorStore.load(self.artifact_dir / "story.index")

        # Story metadata
        self.story_metadata = pd.read_parquet(
            self.artifact_dir / "story_metadata.parquet"
        )

        self.stories = pd.read_parquet(self.artifact_dir / "stories.parquet")

        self.stories["id"] = self.stories["id"].astype(str)

        # Evidence embeddings
        self.evidence_embeddings = np.load(
            self.artifact_dir / "evidence_embeddings.npy"
        ).astype(np.float32)

        # Evidence metadata
        self.evidence_metadata = pd.read_parquet(
            self.artifact_dir / "evidence_metadata.parquet"
        )

        self._validate_artifacts()

    def _validate_artifacts(self) -> None:
        """
        Ensure stored vectors and metadata are aligned.
        """

        if self.story_store.size != len(self.story_metadata):
            raise ValueError("Story index and story metadata are not aligned.")

        if len(self.evidence_embeddings) != len(self.evidence_metadata):
            raise ValueError("Evidence embeddings and metadata are not aligned.")

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:
        """
        Encode and normalize a query.
        """

        vector = self.model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.asarray(
            vector,
            dtype=np.float32,
        )

    def retrieve_stories(
        self,
        query: str,
        top_k: int = 3,
        candidate_story_ids: list[str] | None = None,
    ) -> pd.DataFrame:

        query_vector = self.embed_query(query)

        # Global semantic retrieval
        if candidate_story_ids is None:
            scores, indices = self.story_store.search(
                query_vector,
                top_k=top_k,
            )

            results = self.story_metadata.iloc[indices].copy().reset_index(drop=True)

            results["score"] = scores

            return results

        # Candidate-restricted semantic retrieval
        candidate_ids = {str(story_id) for story_id in candidate_story_ids}

        mask = (
            self.story_metadata["story_id"].astype(str).isin(candidate_ids).to_numpy()
        )

        candidate_indices = np.flatnonzero(mask)

        if len(candidate_indices) == 0:
            return pd.DataFrame()

        # Reconstruct story vectors from FAISS.
        candidate_vectors = np.vstack(
            [
                self.story_store.index.reconstruct(int(index))
                for index in candidate_indices
            ]
        ).astype(np.float32)

        scores = candidate_vectors @ query_vector

        top_k = min(
            top_k,
            len(candidate_indices),
        )

        local_indices = np.argsort(scores)[::-1][:top_k]

        global_indices = candidate_indices[local_indices]

        results = self.story_metadata.iloc[global_indices].copy().reset_index(drop=True)

        results["score"] = scores[local_indices]

        return results

    def retrieve_evidence(
        self,
        query: str,
        candidate_story_ids: list[str],
        top_k: int = 5,
    ) -> pd.DataFrame:
        """
        Retrieve evidence chunks restricted to candidate stories.
        """

        candidate_story_ids = {str(story_id) for story_id in candidate_story_ids}

        mask = (
            self.evidence_metadata["story_id"]
            .astype(str)
            .isin(candidate_story_ids)
            .to_numpy()
        )

        candidate_indices = np.flatnonzero(mask)

        if len(candidate_indices) == 0:
            return pd.DataFrame()

        candidate_vectors = self.evidence_embeddings[candidate_indices]

        query_vector = self.embed_query(query)

        # Both query and evidence vectors are normalized.
        # Therefore dot product = cosine similarity.
        scores = candidate_vectors @ query_vector

        top_k = min(
            top_k,
            len(candidate_indices),
        )

        local_top_indices = np.argsort(scores)[::-1][:top_k]

        global_indices = candidate_indices[local_top_indices]

        results = (
            self.evidence_metadata.iloc[global_indices].copy().reset_index(drop=True)
        )

        results["score"] = scores[local_top_indices]

        return results

    def get_story_by_id(
        self,
        story_id: str,
    ) -> pd.DataFrame:
        story_id = str(story_id)

        return (
            self.stories[self.stories["id"] == story_id].copy().reset_index(drop=True)
        )

    def get_story_by_title(
        self,
        title: str,
    ) -> pd.DataFrame:
        title_normalized = title.strip().casefold()

        mask = self.stories["title"].str.strip().str.casefold() == title_normalized

        return self.stories[mask].copy().reset_index(drop=True)

    def get_stories_by_genre(
        self,
        genre: str,
    ) -> pd.DataFrame:
        genre_normalized = genre.strip().casefold()

        mask = self.stories["genre"].str.strip().str.casefold() == genre_normalized

        return self.stories[mask].copy().reset_index(drop=True)
