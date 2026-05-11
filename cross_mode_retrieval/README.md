# Cross-Modal Retrieval on MemeCap

This repository benchmarks three retrieval setups on [MemeCap](https://github.com/eujhwang/meme-cap):

1. Zero-shot CLIP (`ViT-B/32`, OpenAI weights)
2. Custom dual encoder (`ResNet-50` + `DistilBERT`) trained with symmetric InfoNCE
3. CLIP + LoRA fine-tuning

The task is: given a query (image-only or image+title), retrieve the matching meme caption from a candidate pool.

## Scope and Inputs

Two query modes are evaluated:

- `Type 1`: image only
- `Type 2`: image + title (fused embedding)

Outputs are retrieval metrics:

- `R@1`, `R@5`, `R@10`
- `MedR` (median rank, lower is better)
- `MRR`

## Repository Layout

```text
cross_mode_retrieval/
├── config.py
├── main.py
├── trainer.py
├── evaluation.py
├── models/
│   ├── custom_model.py
│   ├── clip_zeroshot.py
│   ├── clip_lora.py
│   └── fusion.py
├── data/
│   ├── download_data.py
│   ├── dataset.py
│   ├── memes-train.json
│   ├── memes-test.json
│   └── images/
└── outputs/
    ├── history/
    └── results.json
```

## Setup

```bash
pip install -r requirements.txt
python -m data.download_data
```

## Run Commands

```bash
# Run all tasks with all fusion strategies sequentially (default)
python main.py --device cuda:0

# Run a specific task with a specific fusion strategy
python main.py --task lora --fusion_strategy weighted_sum --device cuda:0

# Available fusion strategies: concat_project | cross_attention | add | weighted_sum | gated
# Available tasks: all | zeroshot | custom | lora
# Available input types: 0=both | 1=image-only | 2=image+title
```

## Method Summary

### Custom Dual Encoder

- Image branch: `ResNet-50` backbone + projection head
- Text branch: `DistilBERT` + projection head
- Training objective: symmetric InfoNCE
- Optimizer: `AdamW`
- Scheduler: `ReduceLROnPlateau`
- Early stopping: validation `R@5`

### CLIP + LoRA

- Base model: OpenCLIP `ViT-B/32`
- CLIP base frozen, LoRA adapters injected into transformer linear blocks
- Same symmetric InfoNCE objective
- Significantly fewer trainable parameters than full fine-tuning

## Experimental Results

Source: [`outputs/results.json`](./outputs/results.json)

Results are split into two sections per model:
- **Type 1 — Image Only (No Fusion):** the fusion module is never called; the query is the raw image embedding.
- **Type 2 — Image + Title (With Fusion):** image and title embeddings are combined via the listed fusion strategy.

Bold = best per column within each section.

---

### Zero-shot CLIP

#### Type 1 — Image Only (No Fusion)

Single result — zero-shot CLIP is not trained, so there is one Type 1 evaluation.

| R@1 | R@5 | R@10 | MedR | MRR |
|---:|---:|---:|---:|---:|
| **45.26%** | **64.76%** | **68.87%** | **2** | **54.23%** |

#### Type 2 — Image + Title (With Fusion)

CLIP backbone is frozen; the fusion module has random (untrained) weights.

| Fusion Strategy | R@1 | R@5 | R@10 | MedR | MRR |
|---|---:|---:|---:|---:|---:|
| concat_project | 0.36% | 1.25% | 2.15% | 290 | 1.47% |
| cross_attention | 6.62% | 11.27% | 13.24% | 164 | 9.54% |
| gated | 15.56% | 27.19% | 32.92% | 40 | 21.56% |
| add | 19.68% | 30.77% | 35.42% | 31 | 25.77% |
| **weighted_sum** | **36.85%** | **56.89%** | **61.00%** | **3** | **45.66%** |

---

### Custom Dual Encoder

#### Type 1 — Image Only (No Fusion)

Best result across 5 training runs (fusion module never called for Type 1).

| R@1 | R@5 | R@10 | MedR | MRR |
|---:|---:|---:|---:|---:|
| **1.43%** | **3.76%** | **6.44%** | **170** | **3.73%** |

#### Type 2 — Image + Title (With Fusion)

| Fusion Strategy | R@1 | R@5 | R@10 | MedR | MRR |
|---|---:|---:|---:|---:|---:|
| concat_project | 2.68% | 6.80% | 11.63% | 125 | 6.04% |
| weighted_sum | 9.66% | 16.28% | 21.82% | 103 | 14.19% |
| cross_attention | 11.09% | 19.32% | 23.79% | 92 | 15.66% |
| add | 11.63% | 19.50% | 25.04% | 76 | 16.20% |
| **gated** | **12.34%** | **19.50%** | **24.87%** | **66** | **16.88%** |

---

### CLIP + LoRA

#### Type 1 — Image Only (No Fusion)

Best result across 5 training runs (fusion module never called for Type 1).

| R@1 | R@5 | R@10 | MedR | MRR |
|---:|---:|---:|---:|---:|
| **52.95%** | **69.23%** | **73.88%** | **1** | **60.64%** |

#### Type 2 — Image + Title (With Fusion)

| Fusion Strategy | R@1 | R@5 | R@10 | MedR | MRR |
|---|---:|---:|---:|---:|---:|
| concat_project | 23.26% | 47.76% | 57.96% | 7 | 34.60% |
| cross_attention | 27.37% | 46.15% | 54.38% | 7 | 36.46% |
| add | 53.31% | 70.30% | 76.21% | 1 | 60.70% |
| gated | 56.17% | 72.45% | 77.82% | 1 | 63.80% |
| **weighted_sum** | **58.14%** | **74.06%** | **77.28%** | **1** | **65.46%** |

## Key Insights

1. **`weighted_sum` is the best fusion strategy for CLIP-based models (Type 2).**
   - Zero-shot Type 2: `weighted_sum` R@1 36.85% vs `concat_project` 0.36% — a complete collapse with naive concatenation.
   - LoRA Type 2: `weighted_sum` achieves the best overall result (R@1 58.14%, MRR 65.46%).

2. **`gated` fusion is the best for the custom dual encoder (Type 2).**
   - Gated achieves the best MedR (66) and MRR (16.88%) among all Type 2 custom runs.
   - The gap between strategies is smaller because the model is generally weak regardless — a representation quality issue.

3. **Type 1 uses no fusion — results reflect pure image representation quality.**
   - The fusion module is never called for Type 1 queries (see `custom_model.py:165`, `trainer.py:103`).
   - For Custom/LoRA, different "model variant" rows represent separately trained models with different (unused) fusion modules; minor differences come from random initialization.

4. **CLIP pretraining is the dominant factor.**
   - LoRA (Type 1) R@1 ~50–53% vs Custom (Type 1) R@1 ~0.7–1.4% — over 40 percentage points difference.
   - No fusion strategy closes this gap.

5. **Practical recommendations:**
   - Best overall: `CLIP + LoRA`, Type 2, `weighted_sum` (R@1 58.14%, MedR 1).
   - No-training baseline: Zero-shot Type 1 (R@1 45.26%).
   - Avoid `concat_project` for zero-shot Type 2 — it collapses entirely (R@1 0.36%).

## Reproducibility Artifacts

- Final comparison: [`outputs/results.json`](./outputs/results.json)
- Per-epoch logs: `outputs/history/{custom|lora}_{fusion_strategy}_type{1|2}_history.json`
  - e.g. [`outputs/history/lora_weighted_sum_type2_history.json`](./outputs/history/lora_weighted_sum_type2_history.json)
