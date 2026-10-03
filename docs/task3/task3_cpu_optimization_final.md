# Task 3 --- CPU Inference Optimization and Deployment

> **Status: Complete**
>
> This document reports the final Task 3 experiments, deployment
> decisions, failed optimizations, and measured quality/latency/resource
> trade-offs for the Story Dataset AI System.

------------------------------------------------------------------------

## 1. Objective

Task 3 requires the Story Dataset AI System to run on CPU with low
latency while preserving as much of the quality achieved in Tasks 1 and
2 as practical.

The optimization process followed a measurement-driven workflow:

1.  Establish CPU baselines.
2.  Profile the end-to-end pipeline.
3.  Identify dominant bottlenecks.
4.  Apply one optimization at a time.
5.  Re-measure latency and resource usage.
6.  Perform quality regression checks.
7.  Keep or reject each optimization based on measured quality/latency
    trade-offs.

The final system therefore does **not** automatically adopt the fastest
configuration. Optimizations that caused unacceptable quality
regressions were retained as ablations and rejected from the production
configuration.

------------------------------------------------------------------------

## 2. CPU Deployment Environment

### 2.1 Hardware and Runtime

  Component             Configuration
  --------------------- -----------------------------------------
  CPU                   AMD Ryzen 5 4600H with Radeon Graphics
  Physical CPU cores    6
  Logical processors    12
  System RAM            15.37 GiB (\~16 GB)
  Runtime               llama.cpp b11146, Windows x64 CPU build
  Accelerator devices   None detected by llama.cpp
  LLM                   Qwen3-4B Q4_K_M GGUF
  Context window        8192 tokens
  API                   OpenAI-compatible llama-server HTTP API

CPU-only execution was explicitly verified with:

``` text
llama-server --list-devices
```

which reported:

``` text
Available devices:
  (none)
```

Therefore, no GPU/accelerator device was available to llama.cpp for
model, KV-cache, or tensor offloading during the reported CPU deployment
experiments.

------------------------------------------------------------------------

## 3. Final Task 1 Runtime Architecture

The Task 1 conversational system uses hierarchical retrieval rather than
sending the complete corpus to the language model.

``` text
User Query
    ↓
Hybrid Query Planner
    ├── Fast deterministic rules
    └── Qwen fallback when routing is ambiguous
    ↓
Story Resolution / Retrieval Strategy
    ├── Exact ID/title lookup
    ├── Metadata lookup
    ├── Story-level semantic discovery
    ├── Evidence retrieval for QA
    └── Full-story retrieval for summary/comparison
    ↓
Context Construction
    ↓
Qwen3-4B Q4_K_M via llama.cpp
    ↓
Streaming Answer
```

Retrieval components:

-   Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
-   Embedding dimensionality: 384
-   Vector search: FAISS inner-product search
-   Story-level representation for story discovery
-   Evidence-level chunks for story QA
-   Evidence chunk size: approximately 254 tokens
-   Evidence overlap: 50 tokens
-   Conservative production evidence configuration: Dense Top-5

The embedding model and graph are initialized before request timing.
Startup/download time is not counted as per-request latency.

------------------------------------------------------------------------

## 4. Initial CPU LLM Smoke Baseline

A five-prompt synthetic benchmark was first used to characterize the
local Qwen runtime.

  Metric                                Result
  --------------------------- ----------------
  Mean latency                         3.393 s
  Median latency                       2.944 s
  P95 latency                          5.274 s
  Effective generation rate     10.82 tokens/s

This was only a runtime smoke test. It excluded planning, retrieval,
context construction, and the full Task 1 graph.

------------------------------------------------------------------------

## 5. Original End-to-End Task 1 CPU Baseline

The full Task 1 pipeline was benchmarked on 20 QA queries:

-   15 `content_qa`
-   5 `long_story_qa`

  Metric                 Original Baseline
  -------------------- -------------------
  Mean                            37.472 s
  Median                          41.324 s
  P95                             46.993 s
  Minimum                         10.120 s
  Maximum                         50.866 s
  Content QA mean                 35.044 s
  Long-story QA mean              44.754 s

This established the original end-to-end CPU baseline before
optimization.

------------------------------------------------------------------------

## 6. Pipeline Profiling

LangSmith traces were used to profile representative requests.

Typical steady-state timing:

``` text
Planner             ~8–9 s
Story resolution    ~0.01 s
Evidence retrieval  ~0.02–0.03 s
Generation          ~6–15 s
```

The first retrieval call could be slower because of initialization, but
warmed retrieval itself was inexpensive.

The two dominant optimization targets were therefore:

1.  LLM-based query planning.
2.  Qwen prompt processing and generation.

This prevented unnecessary optimization of FAISS retrieval, which was
not the dominant latency source.

------------------------------------------------------------------------

## 7. Hybrid Deterministic/LLM Query Planner

The original system invoked Qwen for every query plan. A hybrid planner
was introduced:

``` text
Query
  ↓
Fast deterministic rules
  ↓
Can route safely?
  ├── Yes → deterministic QueryPlan
  └── No  → Qwen planner fallback
```

The rules were intentionally conservative. A broader rule was tested and
rejected because it caused metadata and comparison routing errors.

### 7.1 Final Planner Benchmark

Evaluated on all 50 planner benchmark queries:

  Metric                           Result
  ----------------------------- ---------
  Intent accuracy                    100%
  Retrieval-strategy accuracy        100%
  Average planner latency         4.270 s
  P50                             6.858 s
  P95                             9.024 s
  Planner errors                        0

The hybrid design therefore reduced unnecessary LLM planning while
preserving planner behavior on the benchmark.

------------------------------------------------------------------------

## 8. Task 1 End-to-End Result After Hybrid Planning

The same 20 QA queries were re-benchmarked.

  Metric                 Original   Hybrid Planner
  -------------------- ---------- ----------------
  Mean                   37.472 s     **15.688 s**
  Median                 41.324 s     **14.603 s**
  P95                    46.993 s     **27.931 s**
  Minimum                10.120 s      **2.680 s**
  Maximum                50.866 s     **34.896 s**
  Content QA mean        35.044 s     **16.278 s**
  Long-story QA mean     44.754 s     **13.916 s**

Measured improvement:

-   Mean latency reduction: **58.1%**
-   Mean speedup: **2.39×**
-   Median latency reduction: **64.7%**
-   P95 latency reduction: **40.6%**
-   Content-QA mean reduction: approximately **53.5%**
-   Long-story-QA mean reduction: approximately **68.9%**

The planner benchmark remained 100% correct for intent and retrieval
strategy. This establishes planner-routing preservation; final answer
quality is evaluated separately where context-changing optimizations are
tested.

------------------------------------------------------------------------

## 9. Streaming Inference and TTFT

Streaming was introduced to improve perceived responsiveness.

Three latency quantities were separated:

-   **Pre-generation latency:** planning + resolution + retrieval +
    context construction.
-   **LLM TTFT:** time from sending the generation request to receiving
    the first non-empty generated token.
-   **End-to-end TTFT:** time from user request entry until the first
    generated token is visible.

This avoids incorrectly reporting retrieval/planning time as model TTFT.

### 9.1 Dense Top-5 Streaming Baseline

Across the 20 QA queries:

  Metric                     Result
  ---------------------- ----------
  Pre-generation mean       0.980 s
  LLM TTFT mean             5.763 s
  LLM TTFT P50              5.355 s
  LLM TTFT P95             12.823 s
  End-to-end TTFT mean      6.743 s
  End-to-end TTFT P50       6.049 s
  End-to-end TTFT P95      13.101 s
  Full-response mean       11.511 s
  Full-response P50        11.414 s
  Full-response P95        19.499 s

Approximate mean-latency composition showed that pre-generation was
relatively small while model prefill/TTFT and remaining generation
dominated.

Streaming improves perceived responsiveness, but it does not remove the
underlying model computation.

------------------------------------------------------------------------

## 10. Context Reduction: Dense Top-5 → Top-3

Because prompt processing was expensive, evidence context was reduced
from five chunks to three.

  Metric                      Top-5      Top-3       Change
  ---------------------- ---------- ---------- ------------
  Pre-generation mean       0.980 s    0.946 s        -3.5%
  LLM TTFT mean             5.763 s    2.884 s   **-50.0%**
  LLM TTFT P50              5.355 s    1.982 s       -63.0%
  LLM TTFT P95             12.823 s    5.458 s       -57.4%
  End-to-end TTFT mean      6.743 s    3.831 s   **-43.2%**
  Full-response mean       11.511 s    6.908 s   **-40.0%**
  Full-response P50        11.414 s    3.825 s       -66.5%
  Full-response P95        19.499 s   19.058 s        -2.3%

The latency improvement was substantial, but the configuration was
**rejected** because quality regressed.

### 10.1 Top-3 Quality Regression

Observed failures included:

-   Q027 --- required evidence was lost.
-   Q036 --- required evidence was outside Top-3.
-   Q039 --- answer became incomplete.
-   Q043 --- incorrect characterization of Archer's relationship to the
    Red Rose.
-   Q046 --- incorrect description of Project Echo's objective.

Therefore, fixed Dense Top-3 was retained only as a latency/quality
ablation and was **not** selected as the production default.

------------------------------------------------------------------------

## 11. Evidence Retrieval and CrossEncoder Reranking

A lightweight reranker was evaluated:

``` text
cross-encoder/ms-marco-MiniLM-L6-v2
```

The objective was to retrieve broadly, rerank on CPU, and send a smaller
evidence subset to Qwen.

### 11.1 Top-5 Reranking

#### All QA

  Method           Hit@1       Hit@3   Hit@5         MRR
  ---------- ----------- ----------- ------- -----------
  Dense            0.500       0.800   0.950       0.660
  Reranked     **0.650**   **0.850**   0.950   **0.762**

#### Content QA

  Method           Hit@1       Hit@3   Hit@5         MRR
  ---------- ----------- ----------- ------- -----------
  Dense            0.533       0.933   1.000       0.717
  Reranked     **0.800**   **1.000**   1.000   **0.889**

Examples:

-   Q027: support rank 3 → 1
-   Q029: rank 2 → 1
-   Q032: rank 3 → 1
-   Q036: rank 4 → 3

#### Long-story QA

  Method       Hit@1   Hit@3   Hit@5     MRR
  ---------- ------- ------- ------- -------
  Dense        0.400   0.400   0.800   0.490
  Reranked     0.200   0.400   0.800   0.380

The generic CrossEncoder improved ordinary content QA but did not solve
long-story evidence selection.

------------------------------------------------------------------------

## 12. Top-10 Candidate Retrieval

Q043 demonstrated that the correct evidence was not always present
inside the first five dense candidates. The candidate pool was expanded
to up to ten chunks.

Across the 20 QA queries:

-   Candidate rows: **145**
-   Supporting chunks: **46**
-   Newly discovered supporting chunks beyond the earlier pool: **9**
-   Dense candidate **Hit@10 = 100%**

This showed that broad dense retrieval could achieve complete candidate
coverage on the evaluated benchmark.

------------------------------------------------------------------------

## 13. Top-10 CrossEncoder Experiment

### 13.1 All QA

  Method           Hit@1   Hit@3   Hit@5      Hit@10         MRR
  ---------- ----------- ------- ------- ----------- -----------
  Dense            0.500   0.800   0.950   **1.000**       0.666
  Reranked     **0.650**   0.800   0.850   **1.000**   **0.758**

### 13.2 Content QA

  Method           Hit@1   Hit@3   Hit@5   Hit@10         MRR
  ---------- ----------- ------- ------- -------- -----------
  Dense            0.533   0.933   1.000    1.000       0.717
  Reranked     **0.800**   0.933   1.000    1.000   **0.883**

### 13.3 Long-story QA

  Method           Hit@1   Hit@3       Hit@5   Hit@10         MRR
  ---------- ----------- ------- ----------- -------- -----------
  Dense        **0.400**   0.400   **0.800**    1.000   **0.515**
  Reranked         0.200   0.400       0.400    1.000       0.381

### 13.4 CPU Reranker Cost

  Metric              Result
  --------------- ----------
  Total runtime      5.527 s
  Mean/query         0.276 s
  Mean/pair         0.0381 s

The computational overhead was small compared with Qwen inference, but
quality remained the deciding factor.

------------------------------------------------------------------------

## 14. Reranker Failure Analysis

### Q043 --- Relationship Question

The answer-bearing passage explicitly describes Archer as a master spy
who spent years fighting the Red Rose, hunting its operatives,
dismantling operations, and uncovering secrets.

-   Dense rank: 8
-   CrossEncoder rank: 6

Higher-ranked passages mentioned Archer and the Red Rose but did not
contain the relationship required to answer the question.

### Q044 --- Character Occupation

The required opening evidence states that Amir works as an accountant at
a prestigious law firm.

-   Dense rank: 1
-   CrossEncoder rank: 8

The reranker demoted the exact answer-bearing passage.

### Q046 --- Project Objective

The required evidence describes Project Echo as creating an artificial
intelligence capable of surpassing human intelligence.

-   Dense rank: 5
-   CrossEncoder rank: 9

Again, topically related passages were preferred over the exact
answer-bearing evidence.

### Conclusion

``` text
High semantic relevance
        ≠
Answer-bearing evidence
```

The universal strategy `Dense Top-10 → CrossEncoder → Top-3 → Qwen` was
therefore rejected.

------------------------------------------------------------------------

## 15. Final Task 1 Retrieval Decision

  --------------------------------------------------------------------------
  Configuration     Benefit           Limitation          Decision
  ----------------- ----------------- ------------------- ------------------
  Dense Top-5       Hit@5 = 95%;      More Qwen context   **Production
                    conservative                          baseline**
                    evidence coverage                     

  Dense Top-3       Much lower TTFT   QA regressions      Rejected as
                    and response                          default
                    latency                               

  Top-5 +           Strong content-QA Weak/inconsistent   Not global default
  CrossEncoder      ordering          long-story behavior 

  Dense Top-10      Hit@10 = 100% on  Too much context if Candidate-recall
                    benchmark         sent directly       result

  Top-10 +          Small reranking   Can demote          Rejected
  CrossEncoder →    cost              answer-bearing      
  Top-3                               long-story evidence 
  --------------------------------------------------------------------------

Final Task 1 evidence path:

``` text
Dense retrieval
    ↓
Top-5 evidence
    ↓
Qwen3-4B Q4_K_M
    ↓
Streaming answer
```

This is a deliberate quality-first choice rather than the lowest-latency
configuration.

------------------------------------------------------------------------

## 16. CPU Thread Tuning

Thread count was empirically tested.

  ---------------------------------------------------------------------------
  Configuration     Mean Latency         Median            P95      Effective
                                                                   Generation
                                                                         Rate
  --------------- -------------- -------------- -------------- --------------
  Automatic          **3.393 s**    **2.944 s**    **5.274 s**        **10.82
  (`-1`)                                                              tok/s**

  6 threads              3.636 s        3.247 s        5.656 s    10.17 tok/s

  12 threads             4.130 s        3.599 s        6.428 s     8.92 tok/s
  ---------------------------------------------------------------------------

Explicitly increasing thread count did not improve performance.
llama.cpp automatic thread configuration was retained.

------------------------------------------------------------------------

## 17. Task 1 Runtime Memory Footprint

Memory was measured directly from the `llama-server` process.

  Metric               Idle After Model Load   Peak During Task 1 Workload
  ---------------- ------------------------- -----------------------------
  Working Set        5291.70 MB (\~5.17 GiB)   **8521.26 MB (\~8.32 GiB)**
  Private Memory     3060.93 MB (\~2.99 GiB)   **6919.19 MB (\~6.76 GiB)**

Peak memory was measured while executing the 20-query Task 1 QA workload
with Qwen3-4B Q4_K_M and an 8192-token context window.

The process remained within the available \~15.37 GiB system RAM.

A separate RAM-stress run produced different latency measurements and
was used only for memory profiling; it does not replace the controlled
latency benchmark reported above.

------------------------------------------------------------------------

# Part II --- Task 2 CPU Classification Deployment

## 18. Task 2 Reference Classification Results

Before CPU deployment optimization, several classification approaches
had already been evaluated.

### 18.1 TF-IDF + LinearSVC --- 5-Fold Cross-Validation

  Metric                          Result
  ------------------- ------------------
  Accuracy               0.6026 ± 0.0353
  Macro Precision        0.5880 ± 0.0461
  Macro Recall           0.5990 ± 0.0373
  Macro F1               0.5631 ± 0.0408
  Weighted F1            0.5637 ± 0.0397
  OOF Accuracy                    0.6026
  OOF Macro F1                    0.5760
  Inference latency     \~0.162 ms/story
  Training time           \~2.395 s/fold

### 18.2 Same 799/200 Split --- Model Comparison

  Model                              Accuracy     Macro F1
  ------------------------------ ------------ ------------
  TF-IDF + LinearSVC                   57.50%       52.02%
  DistilBERT, first 512 tokens         56.00%       48.78%
  Hierarchical DistilBERT              55.00%       49.49%
  **Qwen3-4B QLoRA, epoch 1**      **65.00%**   **61.97%**

For Qwen QLoRA epoch 1:

  Metric                        Result
  -------------------- ---------------
  Accuracy                  **65.00%**
  Macro Precision               65.98%
  Macro Recall                  65.40%
  Macro F1                  **61.97%**
  Weighted F1                   61.75%
  Invalid outputs        5/200 = 2.50%
  T4 average latency     2.727 s/story

Qwen QLoRA therefore provided the strongest classification quality
before CPU deployment conversion.

------------------------------------------------------------------------

## 19. QLoRA Training and Deployment Artifact

The classifier used Qwen3-4B with 4-bit NF4 loading and a PEFT LoRA
adapter.

LoRA configuration:

-   Rank `r = 16`
-   Alpha `32`
-   Dropout `0.05`
-   Target modules:
    -   `q_proj`
    -   `k_proj`
    -   `v_proj`
    -   `o_proj`
    -   `gate_proj`
    -   `up_proj`
    -   `down_proj`
-   Trainable parameters: approximately 33.0M
-   Trainable fraction: approximately 0.8145%
-   Training: 3 epochs configured, but the retained reproducible
    checkpoint is **checkpoint-100 / epoch 1**
-   Learning rate: `1e-4`
-   Gradient accumulation: `8`
-   Gradient checkpointing enabled
-   Assistant-only classification loss
-   Maximum sequence length: 2048 tokens

An important training preprocessing bug was previously detected: naïvely
truncating the complete sequence could remove the supervised answer
tokens. Training preprocessing was corrected to reserve the target and
truncate the story instead.

The reproducible checkpoint used for final evaluation is checkpoint-100.
A later interrupted training attempt reached step 196 but was not saved
and is not used in any reported result.

------------------------------------------------------------------------

## 20. Catastrophic-Forgetting / Task 1 Regression Check

Because Task 2 reused the Task 1 Qwen base with a genre LoRA, Task 1
behavior was compared with LoRA disabled vs enabled on the 20 QA
questions Q027--Q046.

The comparison used:

-   the same quantized Qwen base,
-   the same question,
-   the same full-story context,
-   the same tokenizer/prompting,
-   deterministic generation.

Result:

-   **No factual QA regression was observed across the tested
    20-question subset.**
-   Manual semantic review judged the expected answer preserved on
    20/20.
-   In some cases the LoRA answer was more concise.
-   Q039 was more complete with the LoRA because it included Sir
    Reginald Blackwood and the Heart of the Abyss.

This result is intentionally scoped to the evaluated 20-question QA
subset; it is not a claim that catastrophic forgetting is impossible on
all inputs.

------------------------------------------------------------------------

## 21. PEFT LoRA → GGUF Conversion

For llama.cpp CPU deployment, the PEFT adapter was converted to a GGUF
LoRA adapter using llama.cpp's `convert_lora_to_gguf.py`.

Source checkpoint:

``` text
models/genre_lora/checkpoint-100
```

Output:

``` text
models/genre_lora/genre_lora.gguf
```

Conversion:

-   Output type: F16
-   Exported tensors: 504
-   GGUF LoRA size: approximately **66.1 MB**
-   Original PEFT adapter safetensors: approximately **132.2 MB**

F16 was deliberately used for the adapter so that LoRA quantization
would not introduce another quality variable.

The converted adapter loaded successfully in llama.cpp together with the
Q4_K_M base. llama.cpp emitted `CPU_REPACK` fallback warnings for LoRA
tensors, but the server loaded successfully and accepted inference
requests.

------------------------------------------------------------------------

## 22. Initial CPU Classification Benchmark and Truncation Failure

The first 200-story CPU benchmark produced:

  Metric               Initial CPU Run
  ------------------ -----------------
  Accuracy                      12.50%
  Macro Precision               10.70%
  Macro Recall                   9.77%
  Macro F1                       8.61%
  Weighted F1                   11.02%
  Invalid outputs      29/200 = 14.50%
  Truncated inputs     29/200 = 14.50%
  Mean latency          25.503 s/story
  Median latency        25.625 s/story
  P95 latency           40.796 s/story

Inspection showed a perfect relationship:

-   All 171 non-truncated prompts produced valid labels.
-   All 29 truncated prompts produced invalid outputs.

The invalid generations frequently continued the story text instead of
producing a genre.

### Root Cause

The initial deployment preprocessing used:

``` python
truncated_ids = full_ids[:2048]
```

This truncated the **complete chat prompt**. For long stories, it could
remove:

-   the final `Genre:` instruction,
-   the end of the user message,
-   the assistant generation boundary.

The model therefore received a prompt ending inside the story and
behaved like a continuation model.

### Fix

The CPU preprocessing was changed to:

1.  Build the fixed classification structure.
2.  Calculate its token cost.
3.  Reserve the complete system instruction, allowed-genre list,
    `Genre:` instruction, and assistant boundary.
4.  Truncate **only the story** to the remaining token budget.
5.  Rebuild the complete chat prompt.

Only the 29 affected stories were rerun; the 171 unaffected predictions
were preserved.

This initial 12.5% result is retained as a debugging result and is
**not** the final CPU classification metric.

------------------------------------------------------------------------

## 23. Corrected Qwen GGUF + LoRA CPU Benchmark

After story-only truncation:

  Metric                     Corrected CPU Result
  ------------------------ ----------------------
  Validation stories                          200
  Genres                                       99
  Accuracy                             **16.00%**
  Macro Precision                          15.43%
  Macro Recall                             16.16%
  Macro F1                             **13.34%**
  Weighted F1                              13.21%
  Invalid outputs               **0/200 = 0.00%**
  Story-truncated inputs          29/200 = 14.50%
  Mean latency                 **25.428 s/story**
  Median latency                   25.625 s/story
  P95 latency                      41.397 s/story

The preprocessing correction completely eliminated invalid outputs,
confirming that the truncation diagnosis was correct.

However, classification quality remained far below the original HF/PEFT
QLoRA result.

------------------------------------------------------------------------

## 24. CPU Class-Collapse Analysis

After the preprocessing fix, predictions remained heavily concentrated
in a small number of labels.

Top predicted classes included:

  Predicted Genre       Count   Share of 200
  ------------------- ------- --------------
  Fantasy                  84          42.0%
  Science Fiction          36          18.0%
  Alternate History        12           6.0%
  Dystopian                11           5.5%
  Mystery                   6           3.0%
  Adventure                 5           2.5%
  Apocalyptic               5           2.5%

`Fantasy + Science Fiction` alone accounted for:

**120 / 200 = 60% of all predictions.**

This is inconsistent with the approximately balanced 99-class target
distribution and indicates severe prediction collapse in the llama.cpp
deployment behavior.

------------------------------------------------------------------------

## 25. Base Q4 vs GGUF LoRA Diagnostic

To determine whether the converted LoRA visibly changed the Q4_K_M
classifier behavior, 20 validation examples were evaluated with:

1.  Q4_K_M base without LoRA.
2.  Previously saved Q4_K_M + GGUF LoRA predictions.

Result:

  Metric                      Result
  ----------------------- ----------
  Examples compared               20
  Identical predictions       **20**
  Different predictions        **0**
  Agreement                 **100%**

A representative example also showed:

``` text
Expected genre:          Steampunk Fantasy
HF NF4 + PEFT LoRA:      Steampunk Fantasy
Q4_K_M base:             Fantasy
Q4_K_M + GGUF LoRA:      Fantasy
```

The adapter was successfully loaded by llama.cpp, so the evidence does
**not** justify claiming that it was technically absent. The defensible
conclusion is narrower:

> On the evaluated diagnostic sample, the converted GGUF LoRA produced
> no observable change in classification predictions relative to the
> Q4_K_M base, and the deployed behavior did not preserve the classifier
> quality measured with the original HF/PEFT setup.

Potential causes could include differences in base quantization, adapter
conversion/application behavior, runtime implementation, or their
interaction. Further low-level investigation was not justified for this
assessment because the deployment already failed both the quality and
latency objectives.

------------------------------------------------------------------------

## 26. Final Task 2 CPU Deployment Decision

### 26.1 Quality and Latency Comparison

  ------------------------------------------------------------------------
  Configuration             Accuracy           Macro F1  Approx. Inference
                                                                   Latency
  --------------- ------------------ ------------------ ------------------
  **Qwen3-4B              **65.00%**         **61.97%**      2.727 s/story
  QLoRA --- HF/T4                                       
  reference**                                           

  TF-IDF +                    57.50%             52.02%          **\~0.151
  LinearSVC ---                                                 ms/story**
  CPU                                                   

  DistilBERT                  56.00%             48.78%                ---

  Hierarchical                55.00%             49.49%                ---
  DistilBERT                                            

  Qwen Q4_K_M +               16.00%             13.34% **25.428 s/story**
  GGUF LoRA ---                                         
  corrected CPU                                         
  ------------------------------------------------------------------------

Compared with the best Qwen QLoRA model, TF-IDF sacrifices:

-   **7.5 percentage points of accuracy**
-   **9.95 percentage points of Macro F1**

but provides extremely low CPU inference latency.

The Qwen GGUF + LoRA path, in contrast, is both much slower and
substantially less accurate than TF-IDF in the measured deployment.

### 26.2 Final Decision

For **Task 2 CPU deployment**, the selected classifier is:

``` text
Story
  ↓
TF-IDF (1–2 grams, max_features=50,000)
  ↓
LinearSVC
  ↓
One of 99 genres
```

The Qwen QLoRA model remains the **best Task 2 classification model by
quality**, while TF-IDF + LinearSVC is the **final CPU deployment
choice** because it provides the best measured deployment trade-off.

The failed GGUF-LoRA experiment is retained as an important Task 3
result demonstrating why post-conversion quality validation is required.

------------------------------------------------------------------------

# Part III --- Final Deployment Conclusions

## 27. Final CPU Architecture

The final deployment uses different CPU-efficient strategies for the two
tasks.

### Task 1 --- Conversational Story QA

``` text
User Query
    ↓
Hybrid deterministic/LLM planner
    ↓
Hierarchical story/evidence retrieval
    ↓
Dense Top-5 evidence
    ↓
Qwen3-4B Q4_K_M / llama.cpp
    ↓
Streaming response
```

### Task 2 --- Genre Classification

``` text
Story
    ↓
TF-IDF
    ↓
LinearSVC
    ↓
99-class genre prediction
```

This differs from the initial idea of using one Qwen base for both
tasks. The shared-base approach was investigated, but the converted CPU
LoRA failed to preserve classification quality. The final design
therefore favors measured production behavior over architectural
uniformity.

------------------------------------------------------------------------

## 28. Final Optimization Decision Matrix

  --------------------------------------------------------------------------
  Experiment        Latency Effect    Quality Effect    Final Decision
  ----------------- ----------------- ----------------- --------------------
  Qwen Q4_K_M via   Enables CPU LLM   Task 1 behavior   **Keep for Task 1**
  llama.cpp         deployment        usable            

  Hybrid planner    Large improvement Planner benchmark **Keep**
                                      preserved         

  Streaming         Better perceived  No quality change **Keep**
                    latency           expected from     
                                      streaming itself  

  Dense Top-3       \~40% lower mean  Multiple QA       **Reject as
                    full-response     regressions       default**
                    latency                             

  Top-5             Small CPU cost    Helps content QA, Do not deploy
  CrossEncoder                        not long QA       globally

  Top-10 dense      Hit@10 = 100%     Too much direct   Keep as
  candidates                          LLM context       analysis/candidate
                                                        strategy

  Top-10 → reranker Low reranking     Long-story        **Reject globally**
  → Top-3           overhead          evidence demotion 

  Explicit 6/12     No speed          N/A               **Reject; retain
  threads           improvement                         auto**

  Qwen GGUF genre   Very slow         Severe            **Reject for Task 2
  LoRA                                classification    CPU**
                                      degradation       

  TF-IDF +          \~0.151 ms/story  57.5% Accuracy /  **Select for Task 2
  LinearSVC                           52.02% Macro F1   CPU**
  --------------------------------------------------------------------------

------------------------------------------------------------------------

## 29. Key Final Results

### Task 1 CPU

-   CPU-only runtime explicitly verified.
-   Hardware: Ryzen 5 4600H, 6 physical / 12 logical cores, \~16 GB RAM.
-   Qwen runtime: Qwen3-4B Q4_K_M via llama.cpp.
-   Original end-to-end mean: **37.472 s**
-   Hybrid-planner mean: **15.688 s**
-   Mean reduction: **58.1%**
-   Speedup: **2.39×**
-   Streaming Dense Top-5 mean end-to-end TTFT: **6.743 s**
-   Streaming Dense Top-5 full-response mean: **11.511 s**
-   Peak llama-server Working Set: **\~8.32 GiB**
-   Final evidence configuration: **Dense Top-5**
-   Top-3 was faster but rejected because of QA regressions.
-   Generic CrossEncoder reranking was rejected as a universal
    compression strategy.

### Task 2 Quality

Best model before CPU conversion:

-   Qwen3-4B QLoRA checkpoint-100
-   Accuracy: **65.00%**
-   Macro F1: **61.97%**
-   Invalid output rate: **2.50%**

Task 1 regression check with genre LoRA:

-   No factual QA regression observed on the evaluated 20-question
    subset.

### Task 2 CPU Deployment

Qwen Q4_K_M + converted F16 GGUF LoRA:

-   Corrected Accuracy: **16.00%**
-   Corrected Macro F1: **13.34%**
-   Invalid output rate after preprocessing fix: **0%**
-   Mean CPU latency: **25.428 s/story**
-   20-example base-vs-LoRA diagnostic: **100% prediction agreement**
-   Deployment path rejected.

Selected CPU classifier:

-   TF-IDF + LinearSVC
-   Accuracy on same 799/200 split: **57.50%**
-   Macro F1: **52.02%**
-   Inference latency: **\~0.151 ms/story**

------------------------------------------------------------------------

## 30. Engineering Lessons

### 30.1 Measure before optimizing

Profiling showed that LLM planning and inference dominated Task 1
latency; warmed vector retrieval was already inexpensive.

### 30.2 Streaming and compute optimization are different

Streaming reduces perceived latency but does not eliminate generation
cost.

### 30.3 Context size strongly affects CPU LLM latency

Reducing Top-5 to Top-3 evidence cut mean LLM TTFT by approximately 50%
and mean full-response latency by approximately 40%.

### 30.4 Faster is not automatically better

Top-3 was rejected because the latency improvement came with measurable
answer-quality regressions.

### 30.5 Candidate recall and final evidence selection are different problems

Dense Top-10 achieved 100% Hit@10 on the 20-query evidence benchmark,
but safely compressing those candidates into a smaller answer-bearing
subset remained difficult.

### 30.6 Generic semantic relevance is not the same as answer-bearing relevance

The CrossEncoder could rank passages about the correct entities highly
while demoting the passage containing the exact fact required by the
question.

### 30.7 Quantized deployment must be re-evaluated

The HF/PEFT QLoRA classifier achieved 65% accuracy, but the llama.cpp
Q4_K_M + converted GGUF LoRA deployment achieved only 16% after
preprocessing was corrected. Training-time metrics cannot simply be
assumed to survive conversion and runtime changes.

### 30.8 Preprocessing bugs can look like model failures

The first CPU classification run had 29 invalid outputs. All 29
corresponded exactly to prompts whose complete sequence had been
truncated. Story-only truncation reduced invalid outputs from 14.5% to
0%.

### 30.9 Failed optimizations are useful results

Both the reranker experiment and GGUF-LoRA deployment were rejected.
They are retained because they provide measured evidence for the final
architecture and prevent unjustified production choices.

### 30.10 Deployment architecture should follow measured requirements

Although sharing one Qwen base across QA and classification was
architecturally attractive, the measured CPU classification behavior did
not justify it. The final system therefore uses Qwen for Task 1 and a
lightweight discriminative classifier for Task 2.

------------------------------------------------------------------------

## 31. Final Status

### Task 3 Checklist

-   [x] Initial CPU inference smoke benchmark
-   [x] End-to-end Task 1 CPU baseline
-   [x] Pipeline profiling
-   [x] Hybrid deterministic/LLM planner
-   [x] Planner regression benchmark
-   [x] End-to-end re-benchmark after planner optimization
-   [x] Streaming generation
-   [x] TTFT instrumentation
-   [x] Dense Top-5 → Top-3 latency experiment
-   [x] Top-3 answer-quality regression analysis
-   [x] Lightweight CrossEncoder reranking
-   [x] Top-10 candidate-recall experiment
-   [x] Reranker failure analysis
-   [x] CPU-only runtime verification
-   [x] CPU hardware/runtime profiling
-   [x] CPU thread tuning
-   [x] Peak llama-server memory measurement
-   [x] PEFT LoRA → GGUF conversion
-   [x] llama.cpp LoRA loading validation
-   [x] Task 2 CPU classification benchmark
-   [x] Classification truncation bug diagnosis and correction
-   [x] Base-Q4 vs GGUF-LoRA behavior diagnostic
-   [x] Final quality/latency/resource trade-off analysis
-   [x] Final Task 1 production retrieval decision
-   [x] Final Task 2 CPU deployment decision

**Task 3 is complete.**

------------------------------------------------------------------------

## 32. Final Conclusion

The final CPU deployment was selected through empirical measurement
rather than by assuming that every optimization would preserve model
quality.

For Task 1, the strongest accepted optimizations were hybrid
deterministic/LLM planning, Q4_K_M llama.cpp inference, streaming, and
conservative Dense Top-5 evidence retrieval. The hybrid planner reduced
the controlled mean end-to-end CPU latency from 37.472 seconds to 15.688
seconds while preserving 100% planner intent and retrieval-strategy
accuracy on the 50-query planner benchmark. Streaming further reduced
perceived response delay, with a mean end-to-end TTFT of 6.743 seconds
in the Dense Top-5 configuration. Peak llama-server Working Set remained
approximately 8.32 GiB on a \~16 GB laptop.

More aggressive context reduction produced better latency but was
rejected because answer quality regressed. Generic CrossEncoder
reranking improved ordinary content-QA evidence ordering but was
unreliable for long narrative questions, where topical relevance did not
always correspond to answer-bearing evidence.

For Task 2, Qwen3-4B QLoRA remained the highest-quality classifier,
reaching 65.00% accuracy and 61.97% Macro F1. However, the attempted
Q4_K_M + GGUF LoRA CPU deployment did not preserve that behavior. After
correcting a prompt-truncation bug, it achieved only 16.00% accuracy and
13.34% Macro F1 at 25.428 seconds per story. A 20-example diagnostic
also showed identical predictions between the Q4 base and Q4 + GGUF LoRA
on every tested example. Consequently, that deployment path was
rejected.

The final CPU classifier is TF-IDF + LinearSVC, which achieves 57.50%
accuracy and 52.02% Macro F1 on the same validation split while
requiring only approximately 0.151 ms per story. This sacrifices some
classification quality relative to Qwen QLoRA but provides a
dramatically better CPU deployment trade-off.

The final system therefore uses **Qwen3-4B Q4_K_M for conversational
QA** and **TF-IDF + LinearSVC for CPU genre classification**. This
architecture is not the most uniform design, but it is the design best
supported by the measured quality, latency, memory, and reliability
results.
