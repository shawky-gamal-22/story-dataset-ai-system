from sentence_transformers import SentenceTransformer

from src.retrieval.chunking import chunk_text_by_tokens


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

tokenizer = model.tokenizer


sample_text = "This is a simple test story. " * 200


# Story-level chunks
story_chunks = chunk_text_by_tokens(
    text=sample_text,
    tokenizer=tokenizer,
    chunk_size=254,
    overlap=0,
)


# Evidence chunks
evidence_chunks = chunk_text_by_tokens(
    text=sample_text,
    tokenizer=tokenizer,
    chunk_size=254,
    overlap=50,
)


print("Non-overlap chunks:", len(story_chunks))
print("Evidence chunks:", len(evidence_chunks))

print("\nFirst non-overlap chunk:")
print(
    story_chunks[0]["start_token"],
    story_chunks[0]["end_token"],
)

print("\nFirst two evidence chunks:")

for chunk in evidence_chunks[:2]:
    print(
        chunk["chunk_id"],
        chunk["start_token"],
        chunk["end_token"],
    )
