# Story Dataset AI System

End-to-end solution for the **Story Dataset AI System technical assessment**, covering:

1. Conversational story QA and retrieval
2. 99-class story genre classification
3. CPU-oriented optimization and deployment

The final application exposes both tasks through FastAPI. Task 1 uses hierarchical retrieval plus Qwen3-4B Q4_K_M served by `llama.cpp`; Task 2 uses TF-IDF + LinearSVC for the final CPU deployment.

## Final System

```text
FastAPI
├── /api/v1/chat
│   └── Hybrid Planner
│       └── Hierarchical Retrieval
│           └── Dense Top-5 Evidence
│               └── Context Builder
│                   └── Qwen3-4B Q4_K_M / llama.cpp
│                       └── Streaming Answer
└── /api/v1/classify
    └── TF-IDF
        └── LinearSVC
            └── 99 Genres
```

### Results Snapshot

| Component | Final result |
|---|---:|
| Clean corpus | 999 stories |
| Actual genre classes | 99 |
| Planner benchmark | 100% intent + strategy accuracy / 50 queries |
| Evidence retrieval | Hit@1 0.50 / Hit@3 0.80 / Hit@5 0.95 |
| Task 1 CPU mean latency | 37.47 s → **15.69 s** |
| Task 1 CPU speedup | **2.39×** |
| Streaming mean E2E TTFT, Top-5 | **6.74 s** |
| Best Task 2 model | Qwen3-4B QLoRA |
| QLoRA validation | **65.00% Accuracy / 61.97% Macro-F1** |
| Final Task 2 CPU model | TF-IDF + LinearSVC |
| CPU classifier validation | **57.50% Accuracy / 52.02% Macro-F1** |
| CPU classifier model inference | **~0.151 ms/story** |
| Peak llama-server Working Set | **~8.32 GiB** |
| Final API smoke test | **8/8 paths passed** |

> The best classification model and the final CPU deployment model are intentionally different. Qwen QLoRA produced the best validation quality; TF-IDF + LinearSVC was selected for CPU deployment after the GGUF/LoRA CPU path showed severe quality and latency regression.

---

## Dataset and Data Quality

Dataset: `FareedKhan/1k_stories_100_genre`

Expected fields: `id`, `title`, `genre`, `story`.

The raw dataset contains 1,000 rows and is advertised as having 100 genres. Audit results:

- 1,000 raw stories
- no nulls or duplicate IDs/titles/story texts
- **99 actual unique genres**
- most genres contain 10 stories
- `Historical Adventure` contains 20 stories
- one confirmed malformed generation-prompt row: ID `308506`

`src/data/loader.py` applies a conservative malformed-row rule while leaving the raw source unchanged.

Final processed corpus:

```text
999 stories
99 genres
```

---

# Task 1 — Conversational Story QA

Supported request types include exact ID/title lookup, metadata questions, semantic story discovery, content QA, long-story QA, summarization, and cross-story comparison.

## Architecture

```text
User Query
   ↓
Hybrid Query Planner
   ├── metadata / exact lookup
   ├── semantic discovery
   ├── story QA / evidence
   ├── full story
   └── metadata + semantic
   ↓
Hierarchical Retrieval
   ├── Story level: "Which story?"
   └── Evidence level: "Which part?"
   ↓
Context Builder
   ↓
Qwen3-4B Q4_K_M
   ↓
Streaming Answer
```

The final planner combines deterministic fast paths with a Qwen fallback. On `benchmark/planner_gold.csv`, intent and strategy accuracy were both 100% across 50 queries.

Retrieval uses `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) and FAISS `IndexFlatIP`. Evidence chunks use 254 tokens, 50-token overlap, and stride 204.

Final production retrieval is **Dense Top-5**.

### Evidence Retrieval

| Set | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---:|---:|---:|---:|
| All QA (20) | 0.50 | 0.80 | **0.95** | 0.66 |
| Content QA (15) | 0.533 | 0.933 | **1.00** | 0.717 |
| Long-story QA (5) | 0.40 | 0.40 | 0.80 | 0.49 |

Content-judged semantic discovery reached Hit@1 0.90, Hit@3/5/10 1.00, and MRR 0.95.

For answer evaluation, 18/20 QA answers received usable external judgments: 16/18 were fully correct, all 18 were at least partially correct, and all 18 judged answers were grounded. Two judge calls did not return usable judgments, so no 100%-of-20 grounding claim is made.

Cross-story comparison manual evaluation: 3/4 fully correct and 4/4 at least partially correct.

---

# Task 2 — Genre Classification

After cleaning: 999 stories and 99 genre classes.

### Same-split model comparison

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| TF-IDF + LinearSVC | 57.50% | 52.02% |
| DistilBERT 512 | 56.00% | 48.78% |
| Hierarchical DistilBERT | 55.00% | 49.49% |
| **Qwen3-4B QLoRA** | **65.00%** | **61.97%** |

The TF-IDF baseline also achieved 60.26% mean accuracy and 56.31% mean Macro-F1 in 5-fold stratified CV, with ~0.162 ms/story model inference.

The QLoRA configuration used rank 16, alpha 32, dropout 0.05, LR 1e-4, effective batch size 8, 2048 max sequence length, and NF4 4-bit base quantization. A preprocessing bug where right truncation could remove supervised target tokens was fixed by reserving target/prompt space and truncating only story content.

The reproducible checkpoint-100/epoch-1 QLoRA evaluation achieved 65.00% accuracy and 61.97% Macro-F1 with 5/200 invalid free-generation labels.

A targeted Task 1 regression test compared the base model and genre LoRA on the same 20 full-story QA questions. No factual regression was observed on that tested subset; this is a scoped regression check, not a universal catastrophic-forgetting claim.

---

# Task 3 — CPU Optimization

Local benchmark environment:

```text
CPU: AMD Ryzen 5 4600H
6 physical / 12 logical cores
RAM: ~15.37 GiB
LLM runtime: llama.cpp CPU build
Model: Qwen3-4B Q4_K_M
Context: 8192
```

`llama-server --list-devices` reported no accelerator devices. Peak observed llama-server Working Set was ~8.32 GiB.

### Task 1

Before planner optimization:

```text
Mean 37.472 s | Median 41.324 s | P95 46.993 s
```

After the hybrid fast planner:

```text
Mean 15.688 s | Median 14.603 s | P95 27.931 s
```

That is a **58.1% mean-latency reduction / 2.39× speedup**, while the separate planner benchmark retained 100% intent and strategy accuracy.

Production Dense Top-5 streaming:

```text
Pre-generation mean: 0.980 s
LLM TTFT mean:       5.763 s
E2E TTFT mean:       6.743 s
Full response mean: 11.511 s
```

Top-3 was faster but caused retrieval/answer regressions and was rejected.

A `cross-encoder/ms-marco-MiniLM-L6-v2` reranker improved ordinary content QA but harmed long-story retrieval, so global reranking was also rejected.

### Task 2 CPU Decision

The genre LoRA was converted to GGUF and loaded for CPU inference, but the resulting benchmark dropped to 16.00% accuracy / 13.34% Macro-F1 with ~25.43 s mean latency per story and severe class collapse.

The final Task 2 CPU deployment therefore uses:

```text
Story → TF-IDF → LinearSVC → Genre
```

This intentionally trades some quality for dramatically better CPU latency and reliable behavior.

---

# Production API

API code: `app/api/`.

### `GET /health`

Reports initialization status for the classifier, chat graph, and generator.

### `POST /api/v1/chat`

```json
{
  "query": "In \"The House of Shadows\", what happens in the story?"
}
```

The API invokes the LangGraph pipeline with `include_generator=False`, retrieves/builds context, then streams the answer through `AnswerGenerator.generate_stream()`.

### `POST /api/v1/classify`

```json
{
  "story": "A detective investigates a mysterious murder and follows hidden clues."
}
```

Example:

```json
{
  "predicted_genre": "Crime"
}
```

---

# Installation and Running

The project targets Python `>=3.13` and uses `uv`.

```powershell
uv sync
```

The serialized CPU classifier artifacts were created with `scikit-learn==1.6.1`, which is pinned for artifact compatibility.

## 1. Start the Task 1 LLM server

Example Windows command used during CPU evaluation:

```powershell
D:\tools\llama-b11146-bin-win-cpu-x64\llama-server.exe `
  -hf Qwen/Qwen3-4B-GGUF:Q4_K_M `
  --host 127.0.0.1 `
  --port 8080 `
  -c 8192
```

Expected OpenAI-compatible endpoint: `http://127.0.0.1:8080/v1`.

## 2. Start FastAPI

```powershell
uv run uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

Development:

```powershell
uv run uvicorn app.api.main:app --reload
```

Swagger is available at `/docs`.

## 3. Run the integration smoke test

```powershell
uv run python scripts/smoke_test_api.py
```

Final run:

```text
Health                  PASS
Exact ID lookup         PASS
Title metadata          PASS
Semantic discovery      PASS
Story QA                PASS
Story summary           PASS
Story comparison        PASS
Genre classification    PASS

8/8 passed
```

This is an integration test, not an accuracy benchmark.

---

# Repository Structure and File Paths

```text
cyshield_Task/
├── app/                    # FastAPI production layer
├── benchmark/              # Fixed evaluation queries/gold labels
├── data/                   # Raw, processed, analysis and persistent artifacts
├── docs/                   # Architecture, decisions and task reports
├── experiments/            # Task-specific experiment outputs
├── models/                 # Serialized model artifacts/checkpoints
├── notebooks/              # EDA, retrieval and model-training experiments
├── scripts/                # Executable evaluation/benchmark jobs
├── src/                    # Reusable production implementation
├── tests/                  # Component and LangGraph tests
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── README.md
```

## `app/`

- `app/api/main.py` — FastAPI app, lifespan initialization, `/health`, `/api/v1/chat`, `/api/v1/classify`.
- `app/api/schemas.py` — Pydantic API request/response models.
- `app/api/__init__.py`, `app/__init__.py` — package markers.

## `benchmark/`

- `benchmark/task1_benchmark_v1.csv` — 50-query Task 1 benchmark: exact lookup/title, metadata, semantic discovery, content QA, long-story QA, comparison.
- `benchmark/planner_gold.csv` — gold planner intent/retrieval-strategy labels.

## `data/`

- `data/raw/` — raw source data; kept unchanged.
- `data/processed/` — cleaned/processed representations.
- `data/analysis/` — analysis outputs.
- `data/artifacts/retrieval/story.index` — FAISS story-level index.
- `data/artifacts/retrieval/story_metadata.parquet` — story ID/title/genre/chunk metadata.
- `data/artifacts/retrieval/evidence_embeddings.npy` — dense evidence vectors.
- `data/artifacts/retrieval/evidence_metadata.parquet` — evidence text and token/chunk boundaries.
- `data/artifacts/retrieval/index_config.json` — retrieval index configuration.
- `data/artifacts/retrieval/stories.parquet` — canonical cleaned full-story store.
- `data/artifacts/classification/` — predictions, metrics, confusion matrices, invalid outputs, and classification comparisons.

## `src/data/`

- `src/data/loader.py` — canonical dataset loader and conservative malformed-row filtering.

## `src/retrieval/`

- `src/retrieval/chunking.py` — chunking utilities.
- `src/retrieval/index_builder.py` — persistent story/evidence index construction.
- `src/retrieval/vector_store.py` — vector index loading/search abstraction.
- `src/retrieval/retriever.py` — high-level story/evidence retrieval.
- `src/retrieval/router.py` — retrieval routing.
- `src/retrieval/planner_prompt.py` — LLM planner prompt.
- `src/retrieval/planner.py` — structured hybrid query planner.
- `src/retrieval/fast_planner.py` — low-latency deterministic planner path.

## `src/chatbot/`

- `src/chatbot/state.py` — LangGraph state.
- `src/chatbot/nodes.py` — planner, resolution, retrieval, context and generation nodes.
- `src/chatbot/graph.py` — Task 1 LangGraph construction.
- `src/chatbot/prompts.py` — answer-generation prompts.
- `src/chatbot/generator.py` — OpenAI-compatible llama.cpp client; normal and streaming generation.
- `src/chatbot/pipeline.py` — higher-level chatbot pipeline utilities.

## `src/classification/`

- `src/classification/classifier.py` — Qwen/PEFT experimental classifier plus final `CPUGenreClassifier`.
- `src/classification/labels.py` — genre label utilities.
- `src/classification/prompts.py` — Qwen classification prompts.
- `src/classification/__init__.py` — package marker.

## `src/evaluation/`

- `src/evaluation/task1_evaluator.py` — Task 1 evaluation utilities.
- `src/evaluation/retrieval_evaluator.py` — retrieval metrics/evaluation utilities.

## `src/deployment/`

Reserved for reusable deployment/optimization utilities rather than one-off experiment scripts.

## `models/`

- `models/genre_lora/` — Task 2 QLoRA checkpoint artifacts when retained locally.
- `models/tfidf_model/tfidf_vectorizer.joblib` — production TF-IDF vectorizer.
- `models/tfidf_model/linear_svc.joblib` — production LinearSVC classifier.
- `models/tfidf_model/label_encoder.joblib` — mapping from classifier outputs to genre names.

## `notebooks/`

- `notebooks/01_dataset_analysis.ipynb` — EDA, lengths, genres and data quality.
- `notebooks/02_retrieval_experiments.ipynb` — retrieval baselines/chunking experiments.
- `notebooks/03_classification_baseline.ipynb` — classical classification baseline.
- `notebooks/04_transformer_classification.ipynb` — transformer classification experiments.

## `scripts/`

### CPU and deployment benchmarks

- `scripts/benchmark_cpu_llm.py` — raw CPU LLM latency/generation rate.
- `scripts/benchmark_task1_cpu.py` — real Task 1 CPU E2E latency.
- `scripts/benchmark_task1_streaming.py` — pre-generation, TTFT and full-response latency.
- `scripts/benchmark_task2_cpu.py` — Task 2 CPU benchmark.
- `scripts/compare_base_vs_lora_cpu.py` — base-vs-LoRA CPU diagnostic.

### Retrieval construction/evaluation

- `scripts/build_retrieval_index.py` — builds retrieval artifacts.
- `scripts/build_evidence_retrieval_candidates.py` — builds evidence candidates for judging.
- `scripts/build_semantic_judgment_pool.py` — semantic relevance candidate pool.
- `scripts/evaluate_planner.py` — planner benchmark.
- `scripts/evaluate_story_retrieval.py` — story retrieval evaluation.
- `scripts/evaluate_evidence_retrieval.py` — dense evidence retrieval metrics.
- `scripts/evaluate_semantic_judgments.py` — semantic discovery metrics.
- `scripts/evaluate_evidence_reranker.py` — reranker evaluation.
- `scripts/evaluate_evidence_reranker_top10.py` — Top-10 reranker experiment.

### Judging and inspection

- `scripts/judge_semantic_relevance.py` — content-based semantic relevance judgments.
- `scripts/judge_evidence_retrieval.py` — evidence relevance judging.
- `scripts/judge_evidence_top10_new.py` — Top-10 evidence judging.
- `scripts/judge_task1_answers.py` — Task 1 answer judging.
- `scripts/judge_task1_qa_answers.py` — QA correctness/grounding judging.
- `scripts/judge_task1_comparisons.py` — comparison evaluation.
- `scripts/merge_semantic_judgments.py` — merges semantic judgments.
- `scripts/merge_top10_evidence_judgments.py` — merges Top-10 evidence judgments.
- `scripts/prepare_top10_evidence_judgments.py` — prepares Top-10 judging input.
- `scripts/inspect_retrieval_misses.py` — retrieval failure inspection.
- `scripts/inspect_reranker_failures.py` — reranker failure inspection.
- `scripts/rerun_failed_e2e.py` — reruns failed E2E cases.

### Integration

- `scripts/smoke_test_api.py` — verifies all major production API paths.

## `tests/`

- `tests/test_chunking.py` — chunking behavior.
- `tests/test_exact_metadata.py` — exact metadata lookup.
- `tests/test_index_builder.py` — index construction.
- `tests/test_langgraph_full_story.py` — full-story graph path.
- `tests/test_langgraph_metadata_semantic.py` — metadata+semantic graph path.
- `tests/test_langgraph_metadata.py` — metadata graph path.
- `tests/test_langgraph_planner.py` — planner integration.
- `tests/test_langgraph_semantic.py` — semantic graph path.
- `tests/test_planner.py` — planner behavior.
- `tests/test_query_plan.py` — query-plan behavior.
- `tests/test_retriever_operations.py` — retriever operations.
- `tests/test_retriever.py` — high-level retriever behavior.
- `tests/test_vector_store.py` — vector-store behavior.

## `docs/`

- `docs/task1/` — Task 1 report/material.
- `docs/task2/` — Task 2 report/material.
- `docs/task3/` — Task 3 CPU optimization/deployment report/material.

## `experiments/`

- `experiments/task1/` — Task 1 experimental outputs.
- `experiments/task2/` — Task 2 experimental outputs.
- `experiments/task3/` — Task 3 experimental outputs.

---

# Design Decisions and Trade-offs

**Hierarchical retrieval instead of one global search:** story selection and evidence selection are different problems. Exact IDs/titles can also bypass unnecessary semantic search.

**Dense Top-5 instead of Top-3:** Top-3 was faster but produced measurable retrieval/answer regressions.

**No global reranker:** the tested cross-encoder improved normal content QA but degraded long-story retrieval.

**QLoRA as best-quality classifier, LinearSVC as CPU classifier:** QLoRA won on validation quality, but its tested GGUF CPU path did not preserve that quality and was orders of magnitude slower.

**Streaming generation:** does not reduce total model computation, but improves perceived responsiveness by returning text before completion.

---

# Known Limitations

- Only 999 valid stories remain after cleaning, spread over 99 classes.
- Most validation genres have roughly two examples, so per-class metrics are noisy.
- Several genres are semantically close.
- The Qwen classification experiment can produce out-of-taxonomy labels without constrained decoding.
- CPU LLM generation remains the dominant Task 1 latency cost.
- Long-story evidence retrieval remains harder than ordinary content QA.
- The tested generic reranker is not consistently beneficial.
- The final CPU classifier prioritizes latency/reliability over the highest observed classification quality.
- API smoke-test success proves integration health, not semantic correctness.

---

# Reproducibility

The repository separates responsibilities deliberately:

```text
notebooks/    exploration and training
src/          reusable implementation
scripts/      executable experiments/evaluations
tests/        focused component tests
data/         processed data and artifacts
models/       serialized model artifacts
docs/         reports and decisions
app/          production API
```

Task 1 expects its local OpenAI-compatible LLM server at `http://127.0.0.1:8080/v1`; FastAPI runs at `http://127.0.0.1:8000`.

Final integration command:

```powershell
uv run python scripts/smoke_test_api.py
```

The final architecture reflects the measured trade-off between retrieval quality, classification quality, CPU latency, memory use, and deployment reliability.
