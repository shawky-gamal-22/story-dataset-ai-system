# Task 1 --- Conversational Story Question Answering System

## 1. Executive Summary

Task 1 required a conversational chatbot that can answer questions about
any story in `FareedKhan/1k_stories_100_genre`, including exact lookup,
metadata queries, semantic story discovery, detailed story QA, and
cross-story comparison.

The final solution is a **query-aware hierarchical RAG system**. It
routes each query to the most appropriate path instead of applying
global vector search to every request:

-   deterministic lookup for IDs, exact titles, and genres;
-   story-level semantic retrieval for discovery;
-   candidate-restricted evidence retrieval for story-specific QA;
-   canonical full-story context for operations requiring broad context.

The system uses **Qwen3-4B Q4_K_M via llama.cpp** for planning and
answer generation, **all-MiniLM-L6-v2** for 384-dimensional embeddings,
**FAISS IndexFlatIP** for normalized story-vector search, overlapping
evidence chunks for detailed QA, and **LangGraph** for orchestration.

### Key Results

  Component                                                             Result
  ---------------------------------------- -----------------------------------
  Planner intent accuracy                                         100% (50/50)
  Planner strategy accuracy                                       100% (50/50)
  Semantic retrieval Hit@5                                                100%
  Semantic retrieval MRR                                                  0.95
  Evidence retrieval Hit@5                                                 95%
  Normal-story evidence Hit@5                                             100%
  Long-story evidence Hit@5                                                80%
  QA fully correct                           88.89% of 18 automatically judged
  QA at least partially correct                100% of 18 automatically judged
  QA fully grounded                            100% of 18 automatically judged
  Comparison fully correct                                           75% (3/4)
  Initial successful E2E average latency                               28.47 s
  E2E P50 / P95                                              25.04 s / 46.08 s

The evaluation exposed four main improvement areas: long-story evidence
ranking, occasional answer incompleteness, expensive full-story
comparisons, and CPU LLM latency.

------------------------------------------------------------------------

## 2. Problem Decomposition

The chatbot must support different information needs:

1.  **Exact lookup** --- story ID or title.
2.  **Metadata lookup** --- genre-based queries.
3.  **Semantic discovery** --- e.g. AI, ghosts, time travel, historical
    settings.
4.  **Story QA** --- characters, conflicts, events, relationships,
    settings, themes.
5.  **Summarization** --- broad context from a known story.
6.  **Comparison** --- reason across two or more stories.

These are not the same retrieval problem. Exact identifiers should not
depend on embeddings, while detailed QA should not retrieve arbitrary
chunks from the entire corpus. This motivated a routed architecture.

------------------------------------------------------------------------

## 3. Dataset Analysis and Quality

The raw dataset contained 1,000 rows with no nulls and no duplicate IDs,
titles, or story texts.

Despite the dataset name advertising 100 genres, inspection found **99
unique labels**: 98 genres contain 10 stories and `Historical Adventure`
contains 20.

### Story Length

  Statistic       Words
  ----------- ---------
  Mean          985.861
  Std           430.611
  Min                83
  Median          978.5
  P95            1669.3
  P99            2268.5
  Max              3093

81.3% of stories are below 1,500 words, 98.3% are below 2,000, and 17
exceed 2,000 words.

Using the MiniLM tokenizer, stories average about 1,203 tokens, with
median 1,193, P95 2,002.25 and maximum 3,670. **902/1,000 stories
(90.2%) exceed MiniLM's 256-token maximum sequence length.** This made
naïve full-story truncation unsuitable as the final representation.

### Malformed Record

A quality scan flagged 17 suspicious records. Manual review showed that
almost all were valid short stories or false positives. One record was
genuinely malformed:

-   ID `308506`
-   title
    `The Chronicles of Eldareth: The Quest for the Lost Orb<|im_end|>`

Its story field contained generation instructions/chat-template
artifacts rather than a story.

A conservative filter removes a record only when it simultaneously
contains chat-template markers, a story-generation instruction, an
assistant marker, and fewer than 150 words. Raw data is preserved; only
the processed corpus is filtered.

Final retrieval corpus:

-   **999 canonical stories**
-   **999 × 384 story vectors**
-   **6,130 evidence chunks**
-   **384-dimensional evidence vectors**

------------------------------------------------------------------------

## 4. Architecture

``` text
User Query
    |
    v
Query Planner
    |
    +--> metadata_lookup ------> deterministic metadata
    |
    +--> story_discovery ------> story semantic retrieval
    |
    +--> story_qa -------------> resolve story --> evidence retrieval
    |
    +--> summarization --------> resolve story --> full canonical story
    |
    +--> comparison -----------> resolve stories --> broad context
    |
    +--> metadata_semantic ----> filter first --> semantic ranking
                                      |
                                      v
                                Context Builder
                                      |
                                      v
                                  Generator
                                      |
                                      v
                                 Final Answer
```

The graph is implemented with LangGraph. Planning, retrieval, context
construction, and generation are separate components so each can be
evaluated and optimized independently.

------------------------------------------------------------------------

## 5. Query Planner

The planner produces a structured `QueryPlan` containing:

-   intent: `story_qa`, `summarization`, `story_discovery`,
    `metadata_lookup`, or `comparison`;
-   strategy: `evidence`, `full_story`, `semantic`, `metadata`, or
    `metadata_semantic`;
-   story titles;
-   story IDs;
-   optional genre;
-   optional semantic query.

The LLM identifies requested entities but does **not** invent database
facts. Python performs actual title/ID resolution.

### Planner Evaluation

  Metric                   Result
  ------------------- -----------
  Queries                      50
  Intent accuracy          1.0000
  Strategy accuracy        1.0000
  Errors                        0
  Avg latency           8598.8 ms
  P50                   8659.4 ms
  P95                   9669.2 ms

Routing quality is excellent, but \~8.6 s CPU latency is a clear Task 3
optimization target.

------------------------------------------------------------------------

## 6. Retrieval Strategy

### 6.1 Story-Level Retrieval

Used for broad discovery: *Which stories match this concept?*

Each story receives one vector created from the complete story.

### 6.2 Evidence-Level Retrieval

Used for detailed QA: *Which passages inside the known story support the
answer?*

Search is restricted to candidate story IDs, preventing semantically
similar chunks from unrelated stories from contaminating context.

### 6.3 Story Representation Experiments

**Naïve truncation baseline**

  Metric                              Result
  --------------------- --------------------
  Recall@1 / @5 / @10     .214 / .382 / .414
  Hit@1 / @5 / @10        .340 / .580 / .580
  MRR                                   .418
  Avg latency                       20.52 ms

Because most stories exceed 256 tokens, truncation discards most story
content.

**Non-overlapping chunk embeddings + mean pooling**

  Metric                              Result
  --------------------- --------------------
  Recall@1 / @5 / @10     .390 / .460 / .530
  Hit@1 / @5 / @10        .480 / .660 / .760
  MRR                                  .5667

This substantial improvement justified chunk-and-mean as the final story
representation.

A summary-embedding pilot was also feasible, but was not selected
because it adds LLM preprocessing cost, reproducibility complexity, and
potential summary information loss.

### 6.4 Evidence Chunking

Evidence chunks use:

-   chunk size: 254 tokens;
-   overlap: 50;
-   stride: 204.

Overlap protects facts crossing chunk boundaries.

### 6.5 Vector Search

Story vectors are normalized and stored in **FAISS `IndexFlatIP`**,
making inner product equivalent to cosine similarity.

Exact search was selected instead of HNSW because the corpus has only
999 story vectors and 6,130 evidence chunks. Approximate search would
add tuning and potential recall loss without meaningful benefit at this
scale.

Evidence embeddings are searched only inside candidate stories.

### 6.6 Deterministic Paths

ID, exact title, and genre queries use structured lookup rather than
embeddings. For combined structured + semantic requests, the policy is:

**filter first → semantic rank second**

------------------------------------------------------------------------

## 7. Generation Model and Grounding

The planner and generator use **Qwen3-4B Q4_K_M** through `llama.cpp`.

Development hardware:

-   16 GB RAM;
-   4 GB GPU VRAM.

The model was chosen as a practical balance between
instruction-following capability and resource cost, while providing a
direct GGUF/llama.cpp path toward CPU deployment.

Generation prompts are intent-specific:

-   QA: supplied context only, no outside knowledge, concise answer,
    insufficient-evidence fallback.
-   Summary: supplied canonical story only.
-   Discovery: present retrieved stories without inventing plot facts or
    exposing similarity scores.
-   Metadata: structured results only.

------------------------------------------------------------------------

## 8. Benchmark

A 50-query benchmark covers:

  Type                        Count
  ------------------------ --------
  Exact lookup                    3
  Exact title                     5
  Metadata                        8
  Semantic discovery             10
  Content QA                     15
  Long-story QA                   5
  Cross-story comparison          4
  **Total**                  **50**

Routing gold:

-   Q001--Q016 → metadata;
-   Q017--Q026 → semantic discovery;
-   Q027--Q046 → evidence QA;
-   Q047--Q050 → comparison/full story.

The current benchmark contains no dedicated summarization query, so no
standalone summarization accuracy is claimed.

------------------------------------------------------------------------

## 9. Semantic Story Retrieval Evaluation

The initial semantic benchmark used genre membership as a silver
relevance proxy.

Silver results included Hit@5 = 0.60 and MRR = 0.3693. Manual inspection
showed many retrieved stories were semantically relevant despite genre
mismatch. Therefore, genre membership was not a valid semantic ground
truth.

A content-based judgment pool of 178 candidates was created:

-   156 clearly relevant;
-   10 partially relevant;
-   12 irrelevant.

Using only clearly relevant (`2`) as positive:

  Metric           Result
  -------------- --------
  Hit@1             .9000
  Precision@1       .9000
  Hit@3            1.0000
  Precision@3       .9333
  Hit@5            1.0000
  Precision@5       .8400
  Hit@10           1.0000
  Precision@10      .8400
  MRR               .9500

Recall@1/3/5/10 was .0608/.1875/.2736/.5362. These are **pooled Recall@K
values over the judged candidate pool**, not exhaustive corpus recall.

The gap between silver Hit@5 (.60) and content-judged Hit@5 (1.00)
demonstrates that evaluation design materially affects the apparent
quality of semantic retrieval.

------------------------------------------------------------------------

## 10. Evidence Retrieval Evaluation

Q027--Q046 were evaluated by supplying the correct story ID, retrieving
Top-5 chunks within that story, and judging whether each chunk supported
the expected answer.

97 chunk judgments were collected: 37 supporting and 60 non-supporting.

  Metric     Overall   Normal (15)   Long (5)
  -------- --------- ------------- ----------
  Hit@1        .5000         .5333      .4000
  Hit@3        .8000         .9333      .4000
  Hit@5        .9500        1.0000      .8000
  MRR          .6600         .7167      .4900

Only one query had no supporting evidence in Top-5.

### Q043 Failure

For Q043, the correct answer existed in the source, but its
answer-bearing chunk fell just beyond Top-5. This is a **ranking/depth
failure**, not missing information or destructive chunking.

This supports a future retrieve-wider + rerank experiment. A reranker
was intentionally deferred because dense Hit@5 is already 95% overall
and 100% on normal stories; adding latency and complexity for one
observed miss was not justified yet.

------------------------------------------------------------------------

## 11. End-to-End Results

With Qwen configured at a 4,096-token context:

  Metric           Result
  ------------- ---------
  Total                50
  Successful           48
  Errors                2
  Avg latency     28.47 s
  P50             25.04 s
  P95             46.08 s

Q047 and Q048 failed because concatenated full-story comparison prompts
exceeded 4,096 tokens (\~4,747 and \~4,661 tokens respectively).

They were rerun at an 8,192-token context only to complete quality
evaluation:

-   Q047: 225.19 s
-   Q048: 197.54 s

This is not considered the desired production design. It demonstrates
that full-story concatenation does not scale on CPU; comparison should
eventually retrieve targeted evidence from each story.

------------------------------------------------------------------------

## 12. QA Answer Evaluation

Q027--Q046 were evaluated for correctness and groundedness using
retrieved Top-5 evidence rather than full stories.

Two judge requests repeatedly returned empty output, so 18/20 were
automatically judged.

  Metric                                 Result
  ---------------------------- ----------------
  Fully correct                  16/18 = 88.89%
  Partially correct               2/18 = 11.11%
  Incorrect                                0/18
  At least partially correct               100%
  Fully grounded                           100%

The groundedness figure applies only to the 18 successfully judged
queries.

### Q030

The answer correctly identified the US--Soviet Cold War but omitted the
nuclear-war threat.

**Classification:** grounded generation-completeness failure.

### Q039

The answer correctly referenced Elmwood Manor but omitted the expected
specificity about its history and Sir Reginald Blackwood.

**Classification:** grounded generation-completeness/specificity
failure.

Q043 and Q046 are excluded from the automatic aggregate because the
external judge repeatedly returned empty output.

------------------------------------------------------------------------

## 13. Comparison Evaluation

The four comparison queries were manually adjudicated after the external
judge repeatedly returned empty output.

  ------------------------------------------------------------------------
  Query                                        Score Finding
  --------------------- ---------------------------- ---------------------
  Q047                                           2/2 Correct
                                                     AI/common-theme
                                                     comparison

  Q048                                           2/2 Correct technological
                                                     conflicts

  Q049                                           1/2 Both stories qualify,
                                                     but answer
                                                     unnecessarily
                                                     privileged one

  Q050                                           2/2 Correctly compared
                                                     both endings
  ------------------------------------------------------------------------

Results:

-   fully correct: **3/4 = 75%**
-   partially correct: **1/4 = 25%**
-   incorrect: **0%**
-   at least partially correct: **100%**

Q049 is a **comparison reasoning / completeness failure**.

No comparison groundedness percentage is claimed because the manual
review used semantic reference answers rather than independently judged
evidence contexts.

------------------------------------------------------------------------

## 14. Failure Taxonomy

The evaluation identified distinct failure classes:

1.  **Data quality:** malformed record 308506 → conservative
    processed-data filter.
2.  **Embedding truncation:** 90.2% exceed MiniLM limit → chunk-and-mean
    story vectors.
3.  **Evidence ranking:** Q043 answer-bearing evidence just outside
    Top-5 → future wider retrieval/reranking.
4.  **Generation completeness:** Q030/Q039 → grounded but incomplete
    answers.
5.  **Comparison reasoning:** Q049 → model mentions both but incorrectly
    privileges one.
6.  **Context-window scalability:** Q047/Q048 → full-story comparison
    exceeds 4096 and becomes extremely slow at 8192.
7.  **CPU inference latency:** retrieval is milliseconds while
    planning/generation are seconds.

------------------------------------------------------------------------

## 15. Engineering Decisions and Trade-offs

### Deterministic lookup over embeddings

Exact identifiers and metadata use deterministic operations because
semantic similarity adds uncertainty without benefit.

### Chunk-and-mean over truncation

Measured story lengths showed truncation loses most content. Chunk
aggregation improved retrieval substantially.

### Separate story and evidence representations

Story discovery and passage QA optimize different objectives; separate
representations improve both.

### Exact FAISS over HNSW

At 999 stories, exact search is simpler and avoids approximate-recall
trade-offs.

### No reranker yet

Evidence Hit@5 is already 95% overall. Reranking is evidence-backed
future work, not complexity added without demonstrated need.

### Quantized 4B LLM

Qwen3-4B Q4_K_M balances capability and local resource requirements and
aligns with Task 3 CPU deployment.

### Full-story comparison exposed a design limit

Complete context maximizes information coverage but caused context
overflow and extreme latency. Targeted comparison evidence is the
preferred future design.

------------------------------------------------------------------------

## 16. Resource and Latency Analysis

Retrieval artifacts are small: 999 story vectors and 6,130 evidence
vectors at 384 dimensions. Exact retrieval operates in milliseconds.

The dominant latency comes from the LLM:

-   planner average: \~8.6 s;
-   E2E average: \~28.47 s;
-   large comparison contexts: \~198--225 s.

Therefore, Task 3 optimization should prioritize:

-   eliminating LLM planning for obvious deterministic queries;
-   smaller/distilled routing;
-   prompt/context compression;
-   targeted comparison evidence;
-   llama.cpp CPU tuning and quantization benchmarking.

Aggressively optimizing the already-small FAISS index would have limited
impact.

------------------------------------------------------------------------

## 17. Reproducibility

Core structure:

``` text
src/
├── data/loader.py
├── retrieval/
│   ├── chunking.py
│   ├── index_builder.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── router.py
│   ├── planner_prompt.py
│   └── planner.py
├── chatbot/
│   ├── state.py
│   ├── nodes.py
│   ├── graph.py
│   ├── prompts.py
│   └── generator.py
└── evaluation/
    ├── task1_evaluator.py
    └── retrieval_evaluator.py
```

Retrieval artifacts:

``` text
data/artifacts/retrieval/
├── story.index
├── story_metadata.parquet
├── evidence_embeddings.npy
├── evidence_metadata.parquet
├── index_config.json
└── stories.parquet
```

Important evaluation artifacts:

``` text
benchmark/task1_benchmark_v1.csv
benchmark/planner_gold.csv
experiments/task1/planner_results.csv
experiments/task1/semantic_retrieval_final_metrics.csv
experiments/task1/e2e_outputs.csv
experiments/task1/qa_answer_judgments.csv
experiments/task1/comparison_answer_judgments.csv
```

Graph entry point:

``` python
from src.chatbot.graph import build_chatbot_graph

graph = build_chatbot_graph()
result = graph.invoke({"query": query})
```

------------------------------------------------------------------------

## 18. Evaluation Lessons

### Semantic relevance is not genre membership

The initial silver benchmark materially understated retrieval quality.

### Retrieval and generation require separate metrics

Component evaluation distinguishes wrong-story retrieval, evidence
ranking, generation omission, hallucination, and comparison reasoning.

### Correctness and groundedness are independent

Q030 and Q039 are examples of answers that are grounded but incomplete.

### Judge infrastructure is not ground truth

Failed external-judge requests were excluded explicitly rather than
silently counted. Comparison was manually reviewed when automated
judging repeatedly failed.

### One global "Task 1 accuracy" would be misleading

Planner, retrieval, QA, and comparison solve different problems and are
reported with their appropriate metrics.

------------------------------------------------------------------------

## 19. Limitations

1.  Long-story evidence Hit@5 is 80% on the five long-story benchmark
    cases.
2.  The generator occasionally omits requested details.
3.  Full-story comparison is computationally expensive.
4.  CPU latency remains high before Task 3 optimization.
5.  Planner quality is excellent but \~8.6 s routing latency is
    expensive.
6.  Semantic Recall@K is pool-based, not exhaustive corpus recall.
7.  Automatic QA judging covered 18/20 queries.
8.  No dedicated summarization query exists in the current benchmark.
9.  Comparison groundedness was not independently scored.

------------------------------------------------------------------------

## 20. Future Improvements

### Retrieve wider + rerank

Retrieve Top-10/20 evidence candidates and apply a lightweight
cross-encoder, especially for long stories. Adoption should depend on
measured Hit@K versus latency.

### Dedicated comparison retrieval

Retrieve relevant evidence independently from each story, build
structured per-story context, then synthesize the comparison. This
directly addresses Q047/Q048.

### Deterministic fast paths before the LLM planner

Exact ID/title/genre patterns can bypass Qwen entirely.

### Smaller router

Planning is a constrained classification/structured-output problem and
may not require a 4B generator.

### Prompt/context compression

Reducing prefill size should improve CPU latency.

### Completeness-aware generation

Prompt the generator to explicitly cover every requested component,
targeting Q030, Q039, and Q049.

------------------------------------------------------------------------

## 21. What I Would Do Differently

If rebuilding Task 1:

1.  perform tokenizer-length analysis before choosing a representation;
2.  create content-based semantic relevance judgments earlier instead of
    relying on genre proxies;
3.  design comparison as its own evidence-retrieval problem from the
    start;
4.  persist actual retrieved evidence with every E2E result for exact
    later groundedness reproduction;
5.  separate quality optimization from latency optimization earlier,
    because measurements show LLM inference---not vector search---is the
    primary latency bottleneck.

------------------------------------------------------------------------

## 22. Conclusion

Task 1 produced a modular conversational story QA system built around
**query-aware hierarchical retrieval** rather than a generic
one-size-fits-all RAG pipeline.

The system combines deterministic retrieval for structured requests,
story-level semantic retrieval for discovery, candidate-restricted
evidence retrieval for detailed QA, and grounded generation through a
quantized local LLM.

Measured results show perfect benchmark routing, 100% content-judged
semantic Hit@5, 95% evidence Hit@5, and strong grounded QA quality among
successfully judged cases. Equally important, the evaluation exposes
concrete limitations rather than hiding them behind a single aggregate
accuracy number.

The resulting Task 1 system is therefore both a strong functional
baseline and a measured foundation for the CPU latency and deployment
optimization required in Task 3.
