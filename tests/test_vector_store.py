import numpy as np

# pyrefly: ignore [missing-import]
from src.retrieval.vector_store import FaissVectorStore


def main():

    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.8, 0.2, 0.0],
        ],
        dtype=np.float32,
    )

    # Normalize test vectors
    vectors /= np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True,
    )

    store = FaissVectorStore(dimension=3)

    store.add(vectors)

    query = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float32,
    )

    scores, indices = store.search(
        query,
        top_k=3,
    )

    print("Index size:", store.size)
    print("Indices:", indices)
    print("Scores:", scores)


if __name__ == "__main__":
    main()
