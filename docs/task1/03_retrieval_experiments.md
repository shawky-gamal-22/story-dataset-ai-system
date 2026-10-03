# Task 1 — Retrieval Experiments Report

## 1. Objective

The objective of the retrieval experiments was to design an efficient retrieval
pipeline for question answering and story discovery over the Story Dataset.

The dataset contains 1,000 stories. Initial analysis showed that most stories are
substantially longer than the input context supported by the selected embedding
model. Therefore, directly embedding the full raw story required careful
evaluation.

The retrieval experiments focused on answering the following questions:

1. Can truncated full-story embeddings provide an acceptable baseline?
2. Does chunking improve story-level retrieval?
3. Should overlapping chunks be used?
4. Should the same representation be used for story discovery and evidence retrieval?
5. How many evidence chunks should be passed to the downstream LLM?
6. Would retrieval-oriented story summaries provide a useful alternative representation?

The experiments ultimately led to a hierarchical retrieval architecture that
separates:

- deterministic metadata retrieval,
- story-level semantic retrieval,
- and fine-grained evidence retrieval.


---

# 2. Evaluation Benchmark

A fixed benchmark of 50 queries was constructed to evaluate the retrieval system.

The benchmark contains multiple query categories:

| Query Type | Count | Purpose |
|---|---:|---|
| Exact Lookup | 3 | Retrieve stories by explicit ID |
| Exact Title | 5 | Retrieve stories using their titles |
| Metadata | 8 | Retrieve/filter stories using structured metadata |
| Semantic Discovery | 10 | Find stories matching semantic concepts |
| Content QA | 15 | Retrieve stories needed to answer detailed questions |
| Long Story QA | 5 | Evaluate retrieval from longer stories |
| Cross-Story | 4 | Retrieve evidence involving multiple stories |
| **Total** | **50** | |

The benchmark was kept fixed across experiments to make comparisons reproducible.

For semantic discovery queries, dataset genre labels were used as a reproducible
silver relevance proxy where appropriate. Therefore, semantic discovery metrics
should not be interpreted as perfect human semantic relevance judgments.


---

# 3. Evaluation Metrics

Story-level retrieval was evaluated using:

- Recall@1
- Recall@3
- Recall@5
- Recall@10
- Hit@1
- Hit@5
- Hit@10
- Mean Reciprocal Rank (MRR)

These metrics capture different retrieval properties.

**Recall@K** measures how many relevant stories were recovered within the first K
results.

**Hit@K** measures whether at least one relevant story appeared in the first K
results.

**MRR** rewards systems that place the first relevant story near the top of the
ranking.

Latency was also measured because retrieval efficiency is important for the final
interactive chatbot.


---

# 4. Embedding Model

The initial retrieval experiments used:

`sentence-transformers/all-MiniLM-L6-v2`

The model produces:

- 384-dimensional embeddings
- normalized embeddings for cosine/dot-product retrieval
- maximum sequence length of 256 tokens

The model was selected as a lightweight retrieval baseline because the final system
must eventually support efficient deployment.


---

# 5. Token-Length Analysis

Before designing the retrieval strategy, all stories were tokenized using the
MiniLM tokenizer.

The resulting token distribution was:

| Statistic | Tokens |
|---|---:|
| Mean | 1203 |
| Median | 1193 |
| P90 | 1736.5 |
| P95 | 2002.25 |
| P99 | 2720.69 |
| Maximum | 3670 |

The embedding model accepts only 256 tokens.

Out of 1,000 stories:

**902 stories (90.2%) exceeded the embedding model context window.**

Therefore, directly encoding the story using the default model behavior would
truncate most of the dataset.

For a median-length story:

`256 / 1193 ≈ 21.5%`

Only roughly the first fifth of the story would be represented.

This observation motivated the first retrieval experiment.


---

# 6. Experiment 0 — Truncated Story Embeddings

## 6.1 Method

The simplest baseline embedded each story directly using MiniLM.

Because the stories frequently exceeded the 256-token model context, the input was
automatically truncated.

The effective representation was therefore approximately:

Story
→ first 256 tokens
→ MiniLM
→ 384-dimensional story embedding

This experiment intentionally provided a naive baseline against which more complete
representations could be compared.


## 6.2 Results

| Metric | Truncated Story |
|---|---:|
| Recall@1 | 0.214 |
| Recall@3 | 0.294 |
| Recall@5 | 0.382 |
| Recall@10 | 0.414 |
| Hit@1 | 0.340 |
| Hit@5 | 0.580 |
| Hit@10 | 0.580 |
| MRR | 0.418 |

Average retrieval latency was approximately:

**20.5 ms/query**


## 6.3 Query-Type Observations

Content QA performed moderately:

- Hit@5: 0.733
- Recall@10: 0.733
- MRR: 0.547

Long-story QA:

- Hit@5: 0.800
- Recall@10: 0.800
- MRR: 0.667

Cross-story retrieval failed completely in this baseline.

Semantic discovery also remained relatively weak.


## 6.4 Conclusion

The experiment established an important baseline but was rejected as the final
retrieval representation.

The main issue was not simply retrieval accuracy.

The representation systematically discarded most of the story for 90.2% of the
dataset.

Important events near the middle or end of a story could therefore never influence
the embedding.

This motivated a chunk-based representation.


---

# 7. Experiment 1 — Chunk Embeddings with Mean Pooling

## 7.1 Hypothesis

Instead of embedding only the beginning of each story, the complete story could be
divided into chunks.

Each chunk could be embedded independently and the embeddings aggregated into one
story representation.

The initial architecture was:

Story
→ token chunks
→ MiniLM embedding per chunk
→ mean pooling
→ L2 normalization
→ one story embedding


## 7.2 Results

The chunk-mean representation substantially improved overall retrieval.

| Metric | Truncated | Chunk Mean |
|---|---:|---:|
| Recall@1 | 0.214 | 0.390 |
| Recall@3 | 0.294 | 0.444 |
| Recall@5 | 0.382 | 0.460 |
| Recall@10 | 0.414 | 0.530 |
| Hit@1 | 0.340 | 0.480 |
| Hit@5 | 0.580 | 0.660 |
| Hit@10 | 0.580 | 0.760 |
| MRR | 0.418 | 0.567 |


## 7.3 Content QA

One of the strongest improvements occurred for content-oriented questions.

The chunk-based representation achieved approximately:

- Recall@1: 0.933
- Hit@1: 0.933
- MRR: 0.933

Compared with the truncated representation, this showed that representing the full
story was particularly important for questions referring to events that may occur
outside the beginning of the story.


## 7.4 Cross-Story Retrieval

The truncated representation had failed on cross-story queries.

Chunk-based representation recovered some relevant stories, demonstrating that
full-story coverage was important for this query type as well.


## 7.5 Semantic Discovery

Semantic discovery did not improve as strongly as content QA.

This suggested an important limitation of mean pooling.

A story may contain many different events and concepts. Averaging all chunk
embeddings can dilute its strongest global semantic concepts.

Therefore:

**Chunk mean pooling was useful for representing the complete story, but it was not
necessarily the ideal semantic representation for every retrieval task.**


---

# 8. Experiment 2 — Retrieval-Oriented Story Summaries

## 8.1 Motivation

A second approach was investigated for global story representation.

Instead of averaging many local chunk embeddings, an LLM could generate a concise
retrieval-oriented summary containing:

- main characters,
- relationships,
- setting,
- primary conflict,
- important events,
- distinctive entities,
- major discoveries,
- and important ending events.

The intended architecture was:

Story
→ LLM retrieval summary
→ embedding
→ semantic story retrieval

The original story and chunks would still be retained for evidence retrieval.


## 8.2 Initial Pilot

A stratified pilot containing 20 stories was created.

The pilot included:

- 5 short stories
- 8 medium stories
- 7 long stories

The stories were also selected from stories referenced by the retrieval benchmark
where possible.


## 8.3 First Summary Generation Attempt

The initial summaries had the following MiniLM token statistics:

| Statistic | Summary Tokens |
|---|---:|
| Mean | 214.4 |
| Median | 222.5 |
| Maximum | 311 |

Seven out of twenty summaries exceeded the MiniLM 256-token context limit.

Therefore:

**35% of the initial summaries would still be truncated by the embedding model.**

This defeated one of the main objectives of summary-based representation.


## 8.4 Token-Aware Summary Pipeline

The summarization pipeline was revised.

The updated pipeline became:

Story
→ retrieval-oriented summary
→ MiniLM token validation
→ if <= 256 tokens: accept
→ otherwise: LLM compression
→ validate again

The prompt was also changed to prioritize concise, distinctive retrieval information
instead of enforcing a strict word-count target.


## 8.5 Second Pilot

After the revision:

- 20/20 summaries were generated successfully.
- 19/20 summaries fit directly within the embedding context.
- 1/20 required fallback compression.
- 0/20 exceeded the final MiniLM token limit.

Therefore, the token-aware summarization pipeline was technically successful.


## 8.6 Decision

Summary-based retrieval remained a promising global representation.

However, generating summaries for all 1,000 stories introduces additional:

- LLM preprocessing cost,
- preprocessing latency,
- external inference dependency,
- and pipeline complexity.

Given the assessment time constraints and the strong results already achieved using
chunk-based representations, summary generation was retained as an investigated
extension rather than made a dependency of the final retrieval pipeline.

A full-corpus summary retrieval comparison would require generating summaries for
all 1,000 stories so that it could be evaluated fairly against the other
representations.


---

# 9. Experiment 3 — Overlapping Chunks

## 9.1 Motivation

Non-overlapping chunks can split a semantic unit across chunk boundaries.

For example:

Chunk A:
"John discovered a hidden door and decided to..."

Chunk B:
"...open it. Inside, he discovered the missing documents."

A question about what John discovered behind the door may not match either chunk as
strongly as it would if the complete event appeared in one chunk.

Therefore, overlapping chunks were evaluated for evidence retrieval.


## 9.2 Configuration

The embedding model has a 256-token context.

After reserving space for special tokens, the chunk configuration used:

- content chunk size: 254 tokens
- overlap: 50 tokens
- stride: 204 tokens

The overlap is approximately 20% of the chunk size.


## 9.3 Generated Index

Processing all 1,000 stories generated:

**6,131 overlapping chunks**

Each chunk retained:

- story ID
- title
- genre
- chunk ID
- token start
- token end
- original chunk text
- 384-dimensional embedding


---

# 10. Overlap vs. Non-Overlap Story Representation

The overlapping chunk embeddings were also mean-pooled to test whether overlap
improved story-level retrieval.

| Metric | Truncated | No Overlap | Overlap 50 |
|---|---:|---:|---:|
| Recall@1 | 0.214 | 0.390 | **0.410** |
| Recall@3 | 0.294 | **0.444** | 0.426 |
| Recall@5 | 0.382 | **0.460** | 0.444 |
| Recall@10 | 0.414 | **0.530** | 0.510 |
| Hit@1 | 0.340 | 0.480 | **0.500** |
| Hit@5 | 0.580 | **0.660** | 0.640 |
| Hit@10 | 0.580 | **0.760** | 0.720 |
| MRR | 0.418 | **0.567** | 0.555 |

Overlap slightly improved the first-position metrics:

- Recall@1: 0.390 → 0.410
- Hit@1: 0.480 → 0.500

However, non-overlapping chunks performed better on most broader retrieval metrics:

- Recall@10: 0.530 vs. 0.510
- Hit@10: 0.760 vs. 0.720
- MRR: 0.567 vs. 0.555

This behavior is reasonable because overlapping text is represented multiple times
when the vectors are averaged, which can introduce weighting bias into the global
story representation.


---

# 11. Query-Type Results with Overlapping Representation

| Query Type | Recall@1 | Recall@5 | Recall@10 | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Content QA | 0.933 | 0.933 | 0.933 | 0.933 | 0.933 | 0.933 | 0.933 |
| Cross-Story | 0.000 | 0.125 | 0.250 | 0.000 | 0.250 | 0.500 | 0.167 |
| Long Story QA | 0.800 | 0.800 | 0.800 | 0.800 | 0.800 | 0.800 | 0.800 |
| Semantic Discovery | 0.020 | 0.080 | 0.180 | 0.200 | 0.600 | 0.700 | 0.330 |

Content QA remained particularly strong.

Semantic discovery remained more difficult, reinforcing the observation that
mean-pooled local embeddings are better at preserving story coverage than capturing
a single global semantic representation.


---

# 12. Evidence Retrieval Experiment

Story retrieval alone is insufficient for detailed question answering.

Once a candidate story has been identified, the system must locate the specific
passage that contains the answer.

Therefore, individual overlapping chunk embeddings were retained as an evidence
index.

The evidence retrieval architecture became:

Query
→ retrieve candidate story
→ restrict search to chunks belonging to candidate story/stories
→ compare query against chunk embeddings
→ retrieve highest-scoring evidence chunks


---

# 13. Qualitative Evidence Retrieval Case Study

The following benchmark query was inspected:

> In "The Chronicles of the Celestial Spoon", what is the name of the spoon that
> can create horrifying and inedible dishes?

The correct story was:

`The Chronicles of the Celestial Spoon`

with story ID:

`987253`

Story-level retrieval ranked it first with similarity:

`0.7249`

The second result had similarity:

`0.3483`

showing a strong separation for this example.


## 13.1 Chunk Retrieval

Evidence retrieval within the correct story produced:

| Rank | Chunk | Score |
|---:|---:|---:|
| 1 | 1 | 0.6874 |
| 2 | 0 | 0.6811 |
| 3 | 3 | 0.6522 |
| 4 | 4 | 0.6288 |
| 5 | 2 | 0.6253 |

The actual answer-bearing evidence occurred in chunk 2 at rank 5.

The passage states that the Celestial Spoon had a twin called the:

**Shadow Spoon**

which had the power to create horrifying and inedible dishes.


## 13.2 Important Observation

The highest-ranked chunks contained many references to the "Celestial Spoon",
because the query itself also contained the story title and repeated the concept of
a spoon.

The answer-bearing chunk contained the more specific concept "Shadow Spoon", but it
was ranked fifth.

This demonstrated that dense retrieval similarity does not guarantee that the
answer-bearing passage will always be ranked first.

It also demonstrated why using only the Top-1 or Top-3 chunks could remove necessary
evidence before generation.


---

# 14. Evidence Context Decision

Based on this qualitative analysis, the downstream QA pipeline uses up to:

**Top-5 evidence chunks**

Each chunk contains at most approximately 254 content tokens.

Therefore, the maximum raw evidence budget is approximately:

`5 × 254 ≈ 1,270 tokens`

before prompt overhead.

This remains a compact context for the downstream language model while providing
more protection against evidence-ranking errors.

Top-5 is treated as a practical engineering choice rather than a globally optimized
hyperparameter. A larger evidence benchmark would be required to prove that five is
the optimal value.


---

# 15. Deterministic Retrieval vs. Semantic Retrieval

The experiments also demonstrated that not every query should be sent through the
embedding retriever.

For example:

- story ID lookup,
- exact title lookup,
- and genre filtering

already have structured solutions.

Embedding retrieval introduces unnecessary uncertainty for these queries.

Therefore, the final architecture routes them separately.


## Deterministic Queries

Examples:

`Tell me about story 987253`

→ ID lookup

`Summarize "The Chronicles of the Celestial Spoon"`

→ exact title lookup

`Show me horror stories`

→ genre metadata filtering


## Semantic Queries

Queries without explicit structured identifiers use semantic retrieval.

Example:

`Find stories involving time travel`

→ semantic story retrieval

One qualitative test for this query returned stories including:

- The Time Travelers Chronicles
- The Timeless Clock
- The Chronicles of the Time-Shifted
- The Time Travel Chronicles of the Lost Timekeeper
- The Time Travelers Quest

indicating that the semantic representation captured the intended concept.


---

# 16. Final Retrieval Architecture

The experiments led to a hierarchical retrieval architecture.

```text
                           User Query
                               |
                               v
                         Query Router
                               |
          +--------------------+--------------------+
          |                    |                    |
       Story ID            Exact Title           Genre
          |                    |                    |
          +--------- Deterministic Retrieval ------+
                               |
                               v
                           Story IDs


        Queries without deterministic identifiers
                               |
                               v
                    Semantic Story Retrieval
                               |
                Non-overlapping Chunks
                               |
                        Mean Pooling
                               |
                     Story-Level Vectors
                               |
                               v
                       Candidate Stories


                       Candidate Story IDs
                               |
                               v
                     Evidence Retrieval
                               |
                 254-token Overlapping Chunks
                       50-token overlap
                               |
                               v
                       Top-5 Evidence
                               |
                               v
                      Downstream LLM
                               |
                               v
                       Grounded Answer