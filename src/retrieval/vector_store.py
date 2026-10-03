from pathlib import Path

import faiss
import numpy as np


class FaissVectorStore:
    """
    Lightweight FAISS vector store using inner-product similarity.

    Vectors are expected to be L2-normalized, so inner product
    is equivalent to cosine similarity.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)

    def add(self, vectors: np.ndarray) -> None:
        vectors = self._prepare_vectors(vectors)
        self.index.add(vectors)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:

        query_vector = self._prepare_vectors(query_vector)

        if query_vector.shape[0] != 1:
            raise ValueError("search() expects a single query vector.")

        top_k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(
            query_vector,
            top_k,
        )

        return scores[0], indices[0]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(path),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "FaissVectorStore":

        index = faiss.read_index(str(path))

        store = cls(index.d)
        store.index = index

        return store

    @property
    def size(self) -> int:
        return self.index.ntotal

    def _prepare_vectors(
        self,
        vectors: np.ndarray,
    ) -> np.ndarray:

        vectors = np.asarray(
            vectors,
            dtype=np.float32,
        )

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected vectors with dimension "
                f"{self.dimension}, "
                f"got {vectors.shape[1]}."
            )

        return np.ascontiguousarray(vectors)
