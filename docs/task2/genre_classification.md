# Task 2 — Story Genre Classification

## 1. Objective

The objective of Task 2 is to classify each story into one of the genre labels provided by the dataset.

The original dataset is advertised as containing 1,000 stories across 100 genres. However, dataset inspection identified two important issues:

1. The dataset contains 99 unique genre labels rather than 100.
2. One sample (`id=308506`) is malformed and contains a generation prompt rather than an actual story.

The malformed sample was excluded using the same conservative preprocessing policy used by the retrieval system in Task 1.

The resulting classification corpus contains:

- 999 valid stories
- 99 genre classes
- 97 genres with 10 stories
- 1 genre (`Fantasy`) with 9 stories
- 1 genre (`Historical Adventure`) with 20 stories

Because the dataset is very small relative to the number of classes, the classification problem is challenging: most classes provide only approximately 8 training examples after the train/validation split.

The main evaluation metric is **Macro F1**, because it assigns equal importance to every genre rather than allowing larger classes to dominate the aggregate score.


---

## 2. Experimental Goals

Rather than selecting a single architecture immediately, several progressively more complex approaches were evaluated.

The experiments were designed to answer the following questions:

1. How strong is a lightweight classical text-classification baseline?
2. Does a pretrained Transformer encoder improve classification quality?
3. Is truncation of long stories a major classification bottleneck?
4. Can the same LLM family used by the Task 1 chatbot be adapted for genre classification?
5. What quality/latency/resource trade-offs will matter for the CPU deployment requirement in Task 3?

The experiment sequence was:

1. TF-IDF + LinearSVC
2. DistilBERT with 512-token truncation
3. Hierarchical DistilBERT using the full story
4. Qwen3-4B with QLoRA adaptation


---

## 3. Dataset Preprocessing

The same cleaned dataset used by the Task 1 system was used for classification to ensure consistency between components.

The malformed-story filter is intentionally conservative. A story is considered malformed only when multiple generation-template indicators occur together, including:

- `<|im_start|>`
- a story-generation instruction
- `<|im_start|> assistant`
- abnormally short story content

This avoids deleting legitimate short stories based only on length.

After preprocessing:

| Property | Value |
|---|---:|
| Valid stories | 999 |
| Genre classes | 99 |
| Minimum samples/class | 9 |
| Maximum samples/class | 20 |
| Typical samples/class | 10 |


---

## 4. Evaluation Protocol

For neural experiments, a fixed stratified train/validation split was created:

- Training samples: 799
- Validation samples: 200
- Random seed: 42
- Stratification: genre label

The same split was reused across TF-IDF, DistilBERT, hierarchical DistilBERT, and Qwen experiments whenever a direct held-out comparison was required.

This prevents improvements from being attributed to differences in the validation samples.

The reported metrics are:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Weighted F1

Macro F1 is treated as the primary classification metric because the task contains 99 classes and the class distribution is not perfectly uniform.


---

# 5. Experiment 1 — TF-IDF + LinearSVC

## 5.1 Motivation

A classical text-classification baseline was implemented before training neural models.

This serves two purposes:

1. Establish a strong low-cost baseline.
2. Provide an important CPU deployment reference for Task 3.

The pipeline uses:

- TF-IDF features
- word unigrams and bigrams
- Linear Support Vector Classification


## 5.2 Configuration

```python
TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=50_000,
    sublinear_tf=True,
)

LinearSVC(C=1.0)
```

A 5-fold stratified cross-validation experiment was first performed.


## 5.3 Cross-Validation Results

| Metric | Mean ± Std |
|---|---:|
| Accuracy | 60.26% ± 3.53% |
| Macro Precision | 58.80% ± 4.61% |
| Macro Recall | 59.90% ± 3.73% |
| Macro F1 | 56.31% ± 4.08% |
| Weighted F1 | 56.37% ± 3.97% |

Additional measurements:

- Average training time: 2.395 seconds/fold
- Average inference latency: 0.162 ms/story

The out-of-fold evaluation produced:

- Accuracy: 60.26%
- Macro F1: 57.60%


## 5.4 Fixed-Split Results

For direct comparison with the neural models, the same 799/200 train/validation split was also used.

| Metric | Score |
|---|---:|
| Accuracy | 57.50% |
| Macro Precision | 52.86% |
| Macro Recall | 57.07% |
| Macro F1 | 52.02% |
| Weighted F1 | 52.17% |

Performance:

- Training time: 9.103 seconds
- Inference latency: 0.151 ms/story


## 5.5 Observation

TF-IDF + LinearSVC provides a surprisingly competitive baseline.

It is also extremely attractive from a deployment perspective because inference requires only a fraction of a millisecond on CPU.

Therefore, neural approaches must provide a meaningful quality improvement to justify their significantly higher computational cost.


---

# 6. Experiment 2 — DistilBERT with Truncation

## 6.1 Motivation

The first neural baseline uses:

`distilbert-base-uncased`

The initial implementation uses the standard 512-token maximum sequence length.

However, token-length analysis showed that most stories are significantly longer than the encoder context window.

Using the DistilBERT tokenizer:

| Statistic | Tokens |
|---|---:|
| Mean | 1204.1 |
| Median | 1193 |
| P90 | 1737 |
| P95 | 2002.5 |
| Maximum | 3670 |

Stories exceeding 512 tokens:

**897 / 999 = 89.79%**

Therefore, the truncation baseline intentionally measures whether using only the beginning of each story is sufficient for genre classification.


## 6.2 Training Configuration

- Model: `distilbert-base-uncased`
- Maximum sequence length: 512
- Full encoder fine-tuning
- Epochs: 10
- Learning rate: `2e-5`
- Train batch size: 8
- Evaluation batch size: 16
- Best checkpoint selected by Macro F1


## 6.3 Results

The best validation checkpoint occurred at epoch 9.

| Metric | Score |
|---|---:|
| Accuracy | 56.00% |
| Macro Precision | 49.03% |
| Macro Recall | 55.56% |
| Macro F1 | 48.78% |
| Weighted F1 | 49.02% |

Total runtime was approximately 9 minutes 31 seconds.


## 6.4 Observation

Despite using a pretrained Transformer, the model did not outperform the TF-IDF baseline.

One possible explanation was the severe truncation of the stories.

This motivated the next experiment.


---

# 7. Experiment 3 — Hierarchical Full-Story DistilBERT

## 7.1 Motivation

Approximately 89.8% of stories exceed DistilBERT's 512-token context window.

To test whether truncation was the primary performance bottleneck, a hierarchical full-story architecture was implemented.


## 7.2 Architecture

Each story is divided into chunks of approximately 510 content tokens.

Every chunk is independently encoded using the same DistilBERT encoder.

The CLS representation of each chunk is extracted:

```text
Story
  |
  +--> Chunk 1 --> DistilBERT --> CLS_1
  +--> Chunk 2 --> DistilBERT --> CLS_2
  +--> Chunk 3 --> DistilBERT --> CLS_3
  ...
```

The chunk representations are aggregated using masked mean pooling:

```text
Story representation
        =
mean(valid chunk representations)
```

The resulting story representation is passed to a 99-class classification head.

The encoder weights are shared across all chunks and are fine-tuned during training.


## 7.3 Configuration

- Chunk size: 510 content tokens
- Chunk overlap: none
- Shared DistilBERT encoder
- Chunk representation: CLS
- Story aggregation: masked mean pooling
- Train batch size: 2
- Gradient accumulation: 4
- Epochs: 10


## 7.4 Results

The best checkpoint occurred at epoch 8.

| Metric | Score |
|---|---:|
| Accuracy | 55.00% |
| Macro Precision | 49.66% |
| Macro Recall | 54.55% |
| Macro F1 | 49.49% |
| Weighted F1 | 49.57% |

Training runtime was approximately 31 minutes 39 seconds.


## 7.5 Truncation Ablation

Comparing the two DistilBERT approaches:

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| DistilBERT — first 512 tokens | 56.00% | 48.78% |
| Hierarchical DistilBERT — full story | 55.00% | 49.49% |

Using the full story improved Macro F1 by only approximately **0.71 percentage points**, while Accuracy decreased by approximately 1 percentage point.

The hierarchical model was also approximately 3.3× slower to train.


## 7.6 Conclusion

This experiment shows that story truncation was not the primary bottleneck.

Although most stories exceed 512 tokens, providing the full story to the encoder produced only a marginal Macro F1 improvement.

The likely challenge is instead the combination of:

- 99 semantically overlapping classes
- approximately 8 training examples per class
- subtle distinctions between related genres
- very limited supervised data

Therefore, further complexity in the DistilBERT architecture was not pursued.


---

# 8. Experiment 4 — Qwen3-4B + QLoRA

## 8.1 Motivation

Task 1 already uses Qwen3-4B as the language-model family for conversational question answering.

The assessment allows Task 1 and Task 2 to use either the same model or separate models.

Therefore, an additional experiment investigates whether Qwen3-4B can be efficiently adapted to genre classification using parameter-efficient fine-tuning.

This experiment also allows evaluation of an important system-design question:

> Can a single LLM family support both story QA and genre classification while remaining within the available training-memory budget?


---

## 8.2 Classification Formulation

Genre classification is formulated as closed-set generative classification.

The model receives:

1. A system instruction defining the task.
2. The list of the 99 valid genres.
3. The story text.

The expected assistant response contains only the genre name.

Conceptually:

```text
System:
You are a story genre classifier.
Choose exactly one genre from the allowed genre list.
Output only the genre.

User:
Allowed genres:
[...]

Story:
[...]

Assistant:
Steampunk Fantasy
```

Qwen's chat template is used with thinking disabled:

```python
enable_thinking=False
```


---

# 9. QLoRA Configuration

Full fine-tuning of a 4B-parameter model would be unnecessarily expensive for this small dataset.

Instead, QLoRA is used.

The base model is loaded in 4-bit precision using BitsAndBytes.

Configuration:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

This uses:

- NF4 quantization
- double quantization
- FP16 computation
- 4-bit frozen base weights


## 9.1 LoRA Configuration

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

Trainable parameters:

- Trainable: 33,030,144
- Total model parameters: 4,055,498,240
- Trainable fraction: 0.8145%

Therefore, less than 1% of the model parameters are updated.


---

# 10. Context-Length Analysis

Before selecting a training sequence length, Qwen tokenizer lengths were measured on the training prompts.

| Statistic | Tokens |
|---|---:|
| Mean | 1593.9 |
| Median | 1592 |
| P75 | 1823.5 |
| P90 | 2125.4 |
| P95 | 2388.8 |
| P99 | 3178.7 |
| Maximum | 4056 |

Coverage analysis:

| Threshold | Samples exceeding threshold |
|---|---:|
| 1024 | 89.36% |
| 1536 | 58.45% |
| 2048 | 12.64% |
| 2560 | 3.63% |
| 3072 | 1.38% |
| 4096 | 0% |

A maximum sequence length of **2048 tokens** was selected.

This fully covers approximately **87.36%** of training examples while keeping memory and training time practical on a Tesla T4.

This represents an explicit resource/coverage trade-off rather than an arbitrary context-length choice.


---

# 11. Story-Only Truncation

A critical issue was discovered during dataset construction.

Initially, the complete formatted sequence was truncated to 2048 tokens.

Because the assistant genre label appears at the end of the sequence, this caused the supervised target itself to be removed for long examples.

The initial implementation produced:

**100 / 799 training examples with zero supervised tokens.**

This would cause those samples to contribute no useful classification supervision.

The preprocessing pipeline was therefore changed to:

1. Calculate the fixed prompt overhead.
2. Reserve tokens for the assistant response.
3. Truncate only the story.
4. Append the genre answer after truncation.

After the correction:

- Minimum supervised target tokens: 3
- Median supervised target tokens: 5
- Maximum supervised target tokens: 8
- Samples with zero supervised tokens: **0**

This check is performed before training.


---

# 12. Assistant-Only Loss

The model should learn to generate the correct genre, not reproduce the long system prompt, genre list, or story.

Therefore, loss is calculated only over the assistant genre response.

Prompt tokens receive:

```python
label = -100
```

and are ignored by cross-entropy loss.

Conceptually:

```text
System prompt     → -100
Genre list        → -100
Story             → -100
Assistant genre   → supervised tokens
```

This focuses the optimization objective directly on genre classification.


---

# 13. Training Configuration

The QLoRA training configuration is:

| Hyperparameter | Value |
|---|---|
| Base model | Qwen3-4B |
| Quantization | 4-bit NF4 |
| Compute dtype | FP16 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Max sequence length | 2048 |
| Train batch size | 1 |
| Gradient accumulation | 8 |
| Effective batch size | 8 |
| Learning rate | 1e-4 |
| Scheduler | Cosine |
| Warmup steps | 15 |
| Weight decay | 0.01 |
| Gradient checkpointing | Enabled |
| Epoch target | Up to 3 |
| Seed | 42 |

Gradient checkpointing is enabled to reduce activation memory at the cost of additional computation.


---

# 14. Hardware and Resource Usage

QLoRA training was performed using a Google Colab Tesla T4 GPU.

Approximate hardware:

- GPU: NVIDIA Tesla T4
- GPU memory: ~15 GB
- FP16 supported
- BF16 not used

Observed GPU memory during setup:

```text
Initial runtime                 ~1.1 GB
Qwen3-4B 4-bit base loaded     ~3.6 GB
QLoRA training setup           ~5.2 GB before active training
```

The experiment therefore remains comfortably below the assessment's 24 GB VRAM constraint.


---
# 15. Final Qwen3-4B + QLoRA Evaluation

The first-epoch QLoRA checkpoint (`checkpoint-100`) was selected as the final evaluated
classification checkpoint.

Although additional optimization steps were explored, no later complete checkpoint with
a comparable retained validation artifact was available. Therefore, later training progress
is not reported as a validated improvement.

The final checkpoint was independently re-evaluated on Kaggle using the same deterministic
799/200 stratified split (`random_state=42`) used for the other fixed-split experiments.

This rerun is treated as the authoritative final evaluation because the complete prediction
and evaluation artifacts were retained.

Generation was deterministic:

```python
do_sample=False
```

No post-hoc mapping of invalid generated labels to valid genres was performed.

## 15.1 Final Validation Results

| Metric | Score |
|---|---:|
| Accuracy | **65.00%** |
| Macro Precision | **65.98%** |
| Macro Recall | **65.40%** |
| Macro F1 | **61.97%** |
| Weighted F1 | **61.75%** |
| Invalid output rate | **2.50%** |

Evaluation environment:

- GPU: NVIDIA Tesla T4 (~15 GB VRAM)
- Validation stories: 200
- Average generation latency: **2.727 s/story**
- Total evaluation runtime: **9.10 minutes**

The final Kaggle evaluation differs slightly from an earlier Colab evaluation of the same
checkpoint. The retained Kaggle predictions and artifacts are therefore used consistently
throughout the final report.

---

# 16. Invalid Genre Generation

Five of the 200 validation samples generated labels outside the fixed 99-class taxonomy.

Invalid output rate:

**5 / 200 = 2.5%**

The five invalid predictions were:

| Ground Truth | Generated Output |
|---|---|
| Supernatural Drama | Mystic Fiction |
| Humor | Culinary Comedy |
| Slice of Life | Community Drama |
| Workplace Drama | Corporate Drama |
| Experimental | Colorful Fantasy |

These outputs are semantically plausible genre names but are not valid classes in the
dataset taxonomy.

They were counted as incorrect predictions.

No nearest-label mapping or manual correction was applied because such post-processing
would artificially improve the measured classification result.

This demonstrates a failure mode specific to generative classification:

> Asking an LLM to select from a closed label set does not guarantee that generation will
> remain inside that set.

A fixed classification head such as LinearSVC cannot produce an out-of-taxonomy class.

For production deployment, this can be addressed using:

- constrained decoding over valid genre outputs,
- deterministic output validation,
- or a fallback/rejection mechanism.

Constrained output generation is considered as part of the Task 3 deployment optimization.

---

# 17. Confusion Analysis

A complete 99-class confusion matrix was generated from the retained validation predictions.

Because the validation set contains only 200 samples across 99 classes, most genres have
approximately two validation examples. Per-class metrics and individual confusion counts
should therefore be interpreted cautiously.

The most frequent observed confusion pairs include:

| Ground Truth | Predicted | Count |
|---|---|---:|
| Fantasy | Magical Realism | 2 |
| Gothic | Horror | 2 |
| Historical Adventure | Steampunk | 2 |
| Mythic Fiction | Mythology | 2 |
| Romantic Fantasy | Magical Realism | 2 |
| Adventure | Mystery Comedy | 1 |
| Alternate Reality | Alternate History | 1 |
| Alternate Reality | Soft Science Fiction | 1 |
| Animal Fiction | Animal Fantasy | 1 |
| Apocalyptic | Post-Apocalyptic | 1 |
| Cyberpunk | Techno-Thriller | 1 |
| Dreamlike | Magical Realism | 1 |
| Dystopian | Post-Apocalyptic | 1 |
| Evolutionary Fiction | Environmental | 1 |

Several mistakes occur between semantically related genres, including:

- Gothic vs. Horror
- Mythic Fiction vs. Mythology
- Fantasy / Romantic Fantasy vs. Magical Realism
- Dystopian / Apocalyptic vs. Post-Apocalyptic
- Cyberpunk vs. Techno-Thriller

This supports the earlier hypothesis that the classification difficulty is driven not only
by the small amount of training data, but also by fine-grained semantic overlap between
many of the 99 labels.

The complete confusion matrix, classification report, predictions, and confusion analysis
are stored under:

```text
data/artifacts/classification/
```

---

# 18. Final Model Comparison

All results below use the same fixed 799/200 stratified split.

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| TF-IDF + LinearSVC | 57.50% | 52.02% |
| DistilBERT — first 512 tokens | 56.00% | 48.78% |
| Hierarchical DistilBERT — full story | 55.00% | 49.49% |
| **Qwen3-4B + QLoRA — checkpoint-100** | **65.00%** | **61.97%** |

Compared with the same-split TF-IDF baseline, Qwen3-4B + QLoRA improves:

- Accuracy by **7.50 percentage points**
- Macro F1 by **9.95 percentage points**

It is therefore the strongest evaluated classification model in terms of held-out
classification quality.

The result should not be interpreted as proving that Qwen is automatically the best
deployment architecture, because Task 3 introduces CPU latency and resource constraints.

---

# 19. Quality vs. Efficiency Trade-Off

The classification experiments reveal a significant quality/efficiency trade-off.

| Model | Macro F1 | Observed Inference Latency |
|---|---:|---:|
| TF-IDF + LinearSVC | 52.02% | ~0.151 ms/story |
| Qwen3-4B + QLoRA | **61.97%** | ~2.727 s/story on T4 GPU |

These latency values were measured in different execution environments and therefore
should not be interpreted as a hardware-normalized comparison.

However, the difference illustrates why Task 3 must explicitly benchmark CPU deployment.

TF-IDF remains an important deployment reference because it provides extremely low latency,
while Qwen provides substantially higher classification quality.

Task 3 will determine the practical quality/speed/resource trade-off after deployment
optimization.

---

# 20. Why QLoRA Instead of Full Fine-Tuning?

QLoRA was selected because:

1. The dataset contains only 799 training samples.
2. Full fine-tuning of approximately 4B parameters would be unnecessarily expensive.
3. Only approximately **0.8145%** of the model parameters are trainable.
4. The frozen quantized base substantially reduces training-memory requirements.
5. The experiment remains below the assessment's 24 GB VRAM constraint.
6. The resulting adapter can be independently enabled or disabled.
7. The same Qwen model family is already used by the Task 1 chatbot.

However, freezing the base weights does not by itself prove that Task 1 behavior remains
unchanged.

When the adapter is active, LoRA updates alter intermediate model computations and may
change generation behavior.

For this reason, Task 1 regression was evaluated empirically.

---

# 21. Task 1 QA Regression / Catastrophic Forgetting Check

Because Qwen3-4B is also used by the Task 1 chatbot, the classification adapter was tested
for regression on the existing Task 1 QA benchmark.

The experiment compared:

```text
Qwen3-4B base
       vs.
Qwen3-4B + genre-classification LoRA
```

## 21.1 Controlled Evaluation Setup

Twenty grounded Task 1 QA questions were selected:

- 15 standard content-QA questions
- 5 long-story QA questions

The benchmark corresponds to Task 1 queries `Q027–Q046`.

For every question, both model configurations received:

- the identical question,
- the identical full-story context,
- the identical tokenizer,
- the identical quantized base model,
- the identical QA prompt,
- and deterministic generation.

The only experimental variable was whether the genre LoRA adapter was enabled.

Using fixed full-story context intentionally removes retrieval quality as a confounding
variable. The experiment therefore tests whether classification adaptation changes the
underlying story-QA generation behavior.

Prompt lengths ranged up to **3,772 tokens**, and none of the 20 QA prompts required
truncation.

## 21.2 Results

No factual correctness regression was observed across the 20 tested QA questions.

The adapter-enabled model preserved the expected answer on all evaluated cases.

In many cases, the adapted model generated a shorter answer while retaining the required
fact.

For example:

```text
Expected:
Shadow Spoon

Base:
The name of the spoon that can create horrifying and inedible dishes is the Shadow Spoon.

LoRA:
The Shadow Spoon.
```

Another useful case was the Elmwood Manor question.

The expected answer referenced both Elmwood Manor's mysterious history and its former
owner, Sir Reginald Blackwood.

The base response mentioned the manor and its mysterious past, while the adapter-enabled
response additionally named Sir Reginald Blackwood.

This experiment therefore found:

| Regression Metric | Result |
|---|---:|
| QA questions tested | 20 |
| QA prompts truncated | 0 |
| Observed factual regressions | **0** |
| Expected answer preserved with LoRA | **20 / 20** |

These results provide evidence that the genre adapter did not degrade factual story-QA
behavior on this benchmark subset.

This result is deliberately scoped to the tested benchmark. It should not be interpreted
as proof that catastrophic forgetting is impossible or that every Task 1 behavior is
unchanged.

The retained regression outputs are stored in:

```text
data/artifacts/classification/task1_base_vs_genre_lora.csv
```

---

# 22. Checkpoint Selection

The retained `checkpoint-100`, corresponding to the first complete training epoch, is used
as the final evaluated QLoRA checkpoint.

Additional training was attempted beyond this checkpoint, and the optimization loss
continued to decrease. However, no later complete checkpoint with retained validation
predictions was available for a controlled comparison.

Therefore, no claim is made that additional epochs improve validation performance.

This is particularly important because the dataset is extremely small and lower training
loss does not necessarily imply better generalization.

The final reported checkpoint is therefore selected based on reproducible retained
evaluation evidence rather than assumed improvement from additional training.

Further epochs remain a possible future experiment using validation-based checkpoint
selection.

---

# 23. Reusable Classification Implementation

The final classification pipeline is exposed through reusable code under:

```text
src/classification/
├── __init__.py
├── classifier.py
├── labels.py
└── prompts.py
```

The implementation separates:

- taxonomy construction,
- prompt construction,
- model/adapter loading,
- generation,
- and output validation.

The reference Qwen classifier loads:

- Qwen3-4B,
- a 4-bit NF4 base,
- the trained PEFT LoRA adapter,
- deterministic generation,
- and validation against the 99 allowed genre labels.

This implementation represents the evaluated classification architecture.

CPU-specific inference and model-format conversion are intentionally handled separately
under Task 3 because they change the deployment runtime rather than the classification
methodology itself.

---

# 24. Final Findings

## 24.1 Classical baselines remain valuable

TF-IDF + LinearSVC achieved **52.02% Macro F1** on the fixed validation split while requiring
approximately **0.151 ms/story** for inference in the measured environment.

It remains a strong CPU deployment reference.

## 24.2 Long-context coverage was not the primary DistilBERT bottleneck

Although approximately 89.8% of stories exceed the standard 512-token DistilBERT context,
hierarchical full-story processing improved Macro F1 by only approximately **0.71 percentage
points**.

The substantial additional computation was therefore not justified by the observed quality
gain.

## 24.3 The task is strongly data-limited

The cleaned dataset contains 99 classes but only 999 stories.

Most classes therefore contribute approximately eight training examples after the split,
while several genre categories are semantically close to one another.

This creates a difficult fine-grained few-shot classification problem.

## 24.4 Qwen3-4B + QLoRA produced the strongest classification result

The final retained evaluation achieved:

- **65.00% Accuracy**
- **65.98% Macro Precision**
- **65.40% Macro Recall**
- **61.97% Macro F1**
- **61.75% Weighted F1**

This improves Macro F1 by approximately **9.95 percentage points** over the same-split
TF-IDF baseline.

## 24.5 Generative classification introduces an output-validity problem

The model produced five plausible but invalid genre labels, corresponding to a **2.5%**
invalid-output rate.

This motivates constrained generation or strict output validation during deployment.

## 24.6 Task 1 QA behavior was preserved on the regression subset

With the genre LoRA enabled, no factual regression was observed across the 20 tested
Task 1 QA questions.

This supports the feasibility of sharing the Qwen model family between the two tasks,
although the result is limited to the evaluated benchmark subset.

## 24.7 Production selection remains a Task 3 question

Qwen3-4B + QLoRA is the strongest **evaluated quality model**, but Task 3 requires
CPU-only, low-latency deployment.

Therefore, final production architecture selection must additionally consider:

- CPU latency,
- peak RAM,
- model size,
- quantization effects,
- output constraints,
- Task 1 latency,
- classification-quality retention,
- and operational complexity.

---

# 25. Task 2 Conclusion

Task 2 evaluated four progressively more capable classification approaches:

1. TF-IDF + LinearSVC
2. DistilBERT with 512-token truncation
3. Hierarchical full-story DistilBERT
4. Qwen3-4B with QLoRA

The strongest held-out classification result was obtained by **Qwen3-4B + QLoRA
checkpoint-100**, achieving:

**65.00% Accuracy and 61.97% Macro F1.**

This represents improvements of:

- **+7.50 percentage points Accuracy**
- **+9.95 percentage points Macro F1**

over the same-split TF-IDF baseline.

The classification adapter was additionally tested on 20 grounded Task 1 QA questions.
No factual correctness regression was observed on this evaluation subset when the adapter
was enabled.

The final evaluation artifacts include:

```text
data/artifacts/classification/
├── qwen_e1_metrics.json
├── qwen_e1_predictions.csv
├── qwen_e1_classification_report.csv
├── qwen_e1_confusion_matrix.csv
├── qwen_e1_confusion_matrix.png
├── qwen_e1_top_confusions.csv
├── qwen_e1_invalid_outputs.csv
└── task1_base_vs_genre_lora.csv
```

Task 2 is therefore complete from the modeling and evaluation perspective.

The next stage is **Task 3**, where the selected candidates will be optimized and benchmarked
for CPU-only low-latency deployment.

