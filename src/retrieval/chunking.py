from typing import List, Dict
from transformers import PreTrainedTokenizerBase


def chunk_text_by_tokens(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int = 254,
    overlap: int = 0,
) -> List[Dict]:
    """
    Split text into token-based chunks.

    Parameters
    ----------
    text : str
        Input story text.

    tokenizer : PreTrainedTokenizerBase
        Tokenizer used by the embedding model.

    chunk_size : int
        Maximum number of content tokens per chunk.

    overlap : int
        Number of tokens shared between consecutive chunks.

    Returns
    -------
    List[Dict]
        Each chunk contains:
        - chunk_id
        - start_token
        - end_token
        - text
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False,
        verbose=False,
    )

    stride = chunk_size - overlap

    chunks = []

    for chunk_id, start in enumerate(range(0, len(token_ids), stride)):
        end = min(start + chunk_size, len(token_ids))

        chunk_token_ids = token_ids[start:end]

        chunk_text = tokenizer.decode(
            chunk_token_ids,
            skip_special_tokens=True,
        )

        chunks.append(
            {
                "chunk_id": chunk_id,
                "start_token": start,
                "end_token": end,
                "text": chunk_text,
            }
        )

        if end >= len(token_ids):
            break

    return chunks
