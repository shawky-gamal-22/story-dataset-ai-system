# Task 1 Retrieval Benchmark — v1

## Purpose

This benchmark is the first fixed evaluation set for Task 1 of the Story Dataset AI System.

It is designed to compare retrieval strategies before finalizing the architecture:

1. Full-story embedding retrieval
2. Summary embedding retrieval
3. Summary → candidate stories → chunk retrieval
4. Summary → chunk retrieval → reranking
5. Exact/metadata routing

The benchmark is intentionally small enough to inspect manually but diverse enough to expose differences between retrieval strategies.

---

## Dataset

- Dataset: `FareedKhan/1k_stories_100_genre`
- Stories: 1,000
- Fields: `id`, `title`, `story`, `genre`

---

## Query Distribution

| Query type | Count |
|---|---:|
| Exact lookup | 3 |
| Exact title lookup | 5 |
| Metadata / genre | 8 |
| Semantic discovery | 10 |
| Content QA | 15 |
| Long-story QA | 5 |
| Cross-story | 4 |
| **Total** | **50** |

---

## Ground Truth

### Gold

Gold queries have explicitly defined ground truth from the dataset or from manual inspection of the story text.

They include:

- ID/title lookup
- Genre/count lookup
- Content QA
- Long-story QA
- Cross-story queries

### Silver — Genre Proxy

The semantic discovery queries currently use the dataset's genre labels as a reproducible proxy for semantic relevance.

For example:

> "Find stories involving magical worlds, enchanted places, or supernatural quests."

uses all stories labeled `Fantasy` as the initial relevant set.

This is useful for the first retrieval experiment, but it is **not treated as perfect semantic ground truth**. Some stories may be semantically relevant outside their assigned genre.

We should later add a smaller manually judged semantic benchmark if the first experiments show that this distinction matters.

---

## How We Will Use It

### Experiment A — Full Story Embeddings

```text
Query
  ↓
Query Embedding
  ↓
Full Story Vector Search
  ↓
Top-K
```

Measure:

- Recall@K
- Hit Rate@K
- MRR
- Retrieval latency

### Experiment B — Summary Embeddings

```text
Query
  ↓
Query Embedding
  ↓
Summary Vector Search
  ↓
Top-K Stories
```

Measure the same metrics.

### Experiment C — Hierarchical Retrieval

```text
Query
  ↓
Summary Retrieval
  ↓
Candidate Stories
  ↓
Chunk Retrieval
  ↓
Optional Reranker
```

For content QA, evaluate whether the retrieved chunks contain the evidence needed to answer the question.

---

## Important Evaluation Rule

Do **not** change the benchmark queries between experiments.

The same benchmark should be used for:

- embedding model comparison
- story vs summary representation
- chunking experiments
- reranker experiments
- K selection
- final retrieval evaluation

This makes the comparisons meaningful.

---

## Suggested Metrics

### Story-level retrieval

For every query:

- Recall@1
- Recall@3
- Recall@5
- Recall@10
- Hit Rate@K
- MRR

### Evidence retrieval

For content/long-story QA:

- Evidence Hit@K
- Context Recall
- Reranker Recall
- Relevant-chunk rank

### Final answer

After the retrieval pipeline is connected to the LLM:

- Answer correctness
- Faithfulness / grounding
- Context relevance
- End-to-end latency

---

## Next Experiment

The immediate next step is:

> **Build the simplest full-story embedding baseline and run it against this benchmark.**

Do not add summaries, rerankers, or complex routing yet.

We first need a baseline number.

Then we can test whether each additional component actually improves retrieval quality.
