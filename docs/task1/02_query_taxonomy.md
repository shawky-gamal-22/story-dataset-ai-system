# Task 1 — Query Taxonomy & Retrieval Strategy

## 1. Objective

The goal of Task 1 is to build a conversational chatbot that can answer questions about any story in the dataset, including:

* Story identification
* Plot and events
* Characters and relationships
* Settings
* Themes
* Dialogue
* Cross-story questions
* Genre and metadata queries

The retrieval system should provide relevant and grounded context to the LLM while keeping latency and resource usage low enough for the deployment constraints.

---

## 2. Why Query Taxonomy?

Not all user queries require the same retrieval strategy.

For example:

* `What is the genre of story 123?` is a structured lookup problem.
* `Tell me about stories involving time travel.` is a semantic discovery problem.
* `Who betrayed the main character?` requires finding relevant evidence inside a story.
* `Give me a short summary of story X.` requires retrieving the complete story.

Therefore, the system should route different query types through appropriate retrieval paths instead of applying semantic vector search to every query.

---

# 3. Query Taxonomy

## 3.1 Metadata / Structured Queries

Examples:

```text
What is the genre of story 123?
How many stories are in the Horror genre?
Show me stories from the Science Fiction genre.
```

### Retrieval Strategy

Use deterministic filtering / database lookup.

```text
Query
  ↓
Query Router
  ↓
Metadata / ID lookup
  ↓
Result
```

Semantic retrieval is unnecessary for these queries.

---

## 3.2 Exact Story / Title Lookup

Examples:

```text
Tell me about story 245.
What is the story "The Last Journey" about?
Summarize "The Last Journey".
```

### Retrieval Strategy

1. Try exact ID/title matching.
2. If exact matching fails, use semantic retrieval as a fallback.

```text
Query
  ↓
Exact ID / Title Match
  │
  ├── Found → Retrieve Story
  │
  └── Not Found → Semantic Search
```

---

# 4. Semantic Story Discovery

Examples:

```text
Find stories about time travel.
Which stories involve artificial intelligence?
Show me stories about friendship and betrayal.
```

These queries are primarily asking:

> Which stories are semantically relevant to this query?

This suggests a story-level semantic retrieval stage.

---

# 5. Hierarchical Retrieval Strategy

A potential improvement is to represent each story using more than its raw text.

For every story, we can maintain:

```text
Story
│
├── Metadata
│   ├── ID
│   ├── Title
│   └── Genre
│
├── Summary
│   └── Summary Embedding
│
├── Original Story
│
└── Chunks
    └── Chunk Embeddings
```

The summary and original story/chunks share the same `story_id`.

## 5.1 Story-Level Discovery

Instead of embedding only the complete story, we can generate an offline summary for each story and create an embedding from the summary.

```text
User Query
    ↓
Query Embedding
    ↓
Summary Vector Search
    ↓
Top N Candidate Stories
```

The purpose of the summary embedding is **story discovery**, not necessarily answering the user's question.

### Why?

A long story may contain many unrelated events, characters, and details. Its full-text embedding may represent a mixture of information.

A summary provides a more compact semantic representation of the main story.

---

# 6. Evidence Retrieval Inside Candidate Stories

Once relevant stories have been identified, the system can retrieve the specific evidence needed to answer the question.

For short stories:

```text
Candidate Story
      ↓
Full Story
      ↓
LLM
```

For long stories:

```text
Candidate Story
      ↓
Story Chunks
      ↓
Vector Retrieval
      ↓
Reranker
      ↓
Top Relevant Chunks
      ↓
LLM
```

This creates a two-level retrieval process:

> **Summary retrieval finds the relevant story.
> Chunk retrieval finds the relevant evidence.**

---

# 7. Adaptive Retrieval

We should not necessarily use the expensive chunk retrieval + reranking pipeline for every story.

A potential strategy is to adapt the retrieval path based on story length.

```text
Query
  ↓
Summary Retrieval
  ↓
Candidate Stories
  ↓
Check Story Length
  │
  ├── Short Story
  │      ↓
  │   Full Story → LLM
  │
  └── Long Story
         ↓
      Chunk Retrieval
         ↓
       Reranker
         ↓
    Relevant Chunks
         ↓
         LLM
```

This can reduce unnecessary retrieval and reranking computation for shorter stories.

The exact length threshold should be determined experimentally based on token counts and the selected LLM context window rather than being fixed at this stage.

---

# 8. Why Not Use Summary Retrieval Alone?

A summary may identify the correct story without containing the specific information required to answer a question.

For example:

```text
Question:
Who killed the victim?

Summary:
A detective investigates a mysterious murder and eventually
uncovers a shocking truth.
```

The summary can help identify the relevant story, but the answer may only appear in the original story.

Therefore:

> Summary embeddings are considered a **candidate story retrieval mechanism**, not the final evidence source.

---

# 9. Initial Architecture

The current candidate architecture is:

```text
                         User Query
                              │
                              ▼
                       Query Router
                              │
              ┌───────────────┼────────────────┐
              │               │                │
              ▼               ▼                ▼
          Metadata       Exact Lookup      Semantic Search
              │               │                │
              │               │                ▼
              │               │        Summary Embedding
              │               │                │
              │               │          Candidate Stories
              │               │                │
              │               │        ┌───────┴───────┐
              │               │        │               │
              │               │    Short Story    Long Story
              │               │        │               │
              │               │    Full Story    Chunk Retrieval
              │               │        │               │
              │               │        │           Reranker
              │               │        │               │
              └───────────────┴────────┴───────┬───────┘
                                               │
                                               ▼
                                              LLM
                                               │
                                               ▼
                                             Answer
```

This is the **current hypothesis**, not the final architecture.

---

# 10. Initial Architectural Decisions

## Decision 1 — Do not use semantic retrieval for everything

Structured queries should use deterministic retrieval whenever possible.

This improves:

* Accuracy
* Latency
* Simplicity
* Predictability

---

## Decision 2 — Separate Story Discovery from Evidence Retrieval

The retrieval problem can be decomposed into:

```text
Stage 1:
Which story is relevant?

Stage 2:
Which part of that story answers the question?
```

This allows us to optimize each stage independently.

---

## Decision 3 — Start with the simplest baseline

Before introducing summaries, rerankers, or complex routing, we should establish a baseline.

### Baseline

```text
Query
  ↓
Story Embedding Search
  ↓
Top-K Stories
  ↓
LLM
```

Then compare it against:

### Summary-based retrieval

```text
Query
  ↓
Summary Embedding Search
  ↓
Top-K Stories
  ↓
LLM
```

And eventually:

### Hierarchical retrieval

```text
Query
  ↓
Summary Retrieval
  ↓
Candidate Stories
  ↓
Chunk Retrieval
  ↓
Reranker
  ↓
LLM
```

This lets us justify each additional component using measured results.

---

# 11. Reranking Strategy

A reranker should not automatically be introduced from the beginning.

Potential pipeline:

```text
Vector Search
     ↓
Top 10–20 Candidates
     ↓
Reranker
     ↓
Top 3–5
```

The reranker should be evaluated based on whether it provides a meaningful improvement in retrieval quality relative to its additional latency and CPU/GPU cost.

---

# 12. Evaluation Strategy

Before finalizing the retrieval architecture, we need a fixed evaluation set.

Initial target:

**~50 Task 1 queries**

Suggested distribution:

| Query Type         | Target |
| ------------------ | -----: |
| ID / Exact Lookup  |     10 |
| Genre / Metadata   |     10 |
| Semantic Discovery |     10 |
| Story Content QA   |     15 |
| Cross-Story        |      5 |
| **Total**          | **50** |

Each query should have a manually verified expected result.

For retrieval experiments, we should record:

* Relevant story ID(s)
* Relevant chunk(s), where applicable
* Expected answer / key facts for QA queries

---

# 13. Retrieval Metrics

For story retrieval:

* Recall@K
* Precision@K
* Hit Rate@K
* MRR

For final QA:

* Answer correctness
* Context relevance
* Faithfulness / grounding
* Response relevance

For system performance:

* Retrieval latency
* Reranking latency
* End-to-end latency
* Memory usage

---

# 14. Current Open Questions

The following decisions are intentionally not finalized yet:

### Embeddings

* Which embedding model?
* What embedding dimension?
* Summary embeddings vs full-story embeddings?
* Do we need separate embeddings for summaries and chunks?

### Retrieval

* Story-level vs chunk-level retrieval?
* How many candidate stories (`K`)?
* How many chunks?
* Should we use hybrid retrieval?

### Summarization

* Which model should generate summaries?
* How long should summaries be?
* Does summary retrieval actually improve Recall@K?

### Reranking

* Which reranker?
* At what stage should it be used?
* Does its accuracy improvement justify its latency?

### Routing

* Rule-based router?
* Lightweight classifier?
* LLM-based router?

### LLM

* Which LLM provides the best accuracy/resource tradeoff?
* Can it fit together with the classification model within the 24GB VRAM constraint?

### Deployment

* How will the selected models run on CPU?
* What quantization strategy is appropriate?
* What is the resulting latency?

---

# 15. Next Steps

We will proceed in the following order:

### Step 1 — Build the Evaluation Set

Create the ~50 representative Task 1 queries and label the expected relevant stories / evidence.

**Output:**

```text
experiments/task1/evaluation_set.csv
```

---

### Step 2 — Select Candidate Embedding Models

Choose a small number of candidate embedding models based on:

* Semantic retrieval quality
* English story/query support
* Embedding dimension
* Model size
* CPU/GPU inference cost
* Availability for local deployment

Do not optimize prematurely.

---

### Step 3 — Build the First Retrieval Baseline

Implement:

```text
Query
 ↓
Story Embedding
 ↓
Vector Search
 ↓
Top-K Stories
```

Measure Recall@K, Hit Rate@K, MRR, and latency.

---

### Step 4 — Evaluate Summary-Based Retrieval

Generate summaries offline and compare:

```text
Full Story Embedding
        vs
Summary Embedding
```

using the same evaluation set.

---

### Step 5 — Evaluate Hierarchical Retrieval

If summary retrieval is useful:

```text
Summary Retrieval
      ↓
Candidate Stories
      ↓
Chunk Retrieval
      ↓
Reranker
```

Measure whether each additional stage improves retrieval quality enough to justify its cost.

---

### Step 6 — Select the LLM

After the retrieval strategy is reasonably understood, evaluate candidate LLMs while considering:

* QA quality
* Context handling
* VRAM
* Quantization
* CPU inference feasibility
* Latency

---

### Step 7 — Build the Final Task 1 Pipeline

Combine:

* Query routing
* Metadata lookup
* Exact lookup
* Semantic retrieval
* Hierarchical retrieval where needed
* Reranking where justified
* LLM generation
* Grounding controls

---

# 16. Current Status

### Completed

* Dataset integrity analysis
* Genre distribution analysis
* Story length analysis
* Query taxonomy
* Initial retrieval architecture
* Hierarchical retrieval hypothesis

### Current Hypothesis

The most promising architecture to investigate is:

```text
                    Query
                      │
                      ▼
                 Query Router
                      │
          ┌───────────┴───────────┐
          │                       │
     Structured              Semantic
     Retrieval               Retrieval
                                  │
                          Summary Embeddings
                                  │
                           Candidate Stories
                                  │
                     ┌────────────┴────────────┐
                     │                         │
                 Short Story              Long Story
                     │                         │
                 Full Story              Chunk Retrieval
                     │                         │
                     │                     Reranker
                     │                         │
                     └────────────┬────────────┘
                                  │
                                  ▼
                                 LLM
                                  │
                                  ▼
                               Answer
```

### Important

This architecture is **not yet finalized**.

The next step is not to add more components.

The next step is to **build the evaluation set and run experiments** that tell us which components are actually useful.
