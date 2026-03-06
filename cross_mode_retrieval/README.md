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
    ├── checkpoints/
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
# Zero-shot
python main.py --task zeroshot --input_type 1 --device cuda:0
python main.py --task zeroshot --input_type 2 --device cuda:0

# Custom dual encoder
python main.py --task custom --input_type 1 --device cuda:0
python main.py --task custom --input_type 2 --device cuda:0

# CLIP + LoRA
python main.py --task lora --input_type 1 --device cuda:0
python main.py --task lora --input_type 2 --device cuda:0
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

| Experiment | R@1 | R@5 | R@10 | MedR | MRR |
|---|---:|---:|---:|---:|---:|
| Zero-shot CLIP (Type 1) | 44.19% | 63.51% | 67.62% | 2 | 53.06% |
| Zero-shot CLIP (Type 2) | 0.36% | 1.25% | 2.15% | 288 | 1.47% |
| Custom Dual Encoder (Type 1) | 1.43% | 3.76% | 7.87% | 175 | 3.75% |
| Custom Dual Encoder (Type 2) | 3.04% | 9.66% | 15.03% | 97 | 7.43% |
| CLIP + LoRA (Type 1) | **52.24%** | **69.05%** | **74.78%** | **1** | **60.06%** |
| CLIP + LoRA (Type 2) | 26.12% | 50.27% | 60.29% | 5 | 37.28% |

## Training Dynamics (from `outputs/history`)

| Run | Best Epoch (val R@5) | Epochs Logged | Best val R@5 |
|---|---:|---:|---:|
| `custom_type1` | 7 | 11 | 4.81% |
| `custom_type2` | 8 | 12 | 11.17% |
| `lora_type1` | 3 | 7 | 65.81% |
| `lora_type2` | 6 | 8 | 48.28% |

## Key Insights

1. `LoRA + Type 1` wins because it starts from a strong aligned image-text space (CLIP pretraining) and then adapts that space to MemeCap.
   - This explains the jump from zero-shot Type 1 to LoRA Type 1 (`R@1: 44.19% -> 52.24%`).
   - History files also show quick convergence (best epochs around `3-6`), consistent with parameter-efficient adaptation rather than training from scratch.
2. The custom dual encoder underperforms mainly due to data/optimization burden.
   - It must learn cross-modal alignment almost from zero with limited task data.
   - Even with decreasing train loss, retrieval quality remains much lower than CLIP-based variants, which is typical when pretraining prior is absent.
3. `Type 2` hurts zero-shot CLIP because the fusion module is not pretrained jointly with CLIP’s retrieval objective.
   - In Type 2, the query becomes a transformed representation (image + title) that can drift away from the caption embedding space CLIP expects.
   - Without supervised adaptation, this distribution shift can dominate, causing the severe drop (`R@1: 44.19% -> 0.36%`).
4. `Type 2` helps the custom model because title provides extra supervised signal it otherwise lacks.
   - For the custom model, title acts as auxiliary context and partially compensates weak image-text alignment (`R@5: 3.76% -> 9.66%`).
5. Data characteristics likely amplify the above behavior.
   - In train split, titles are short (about `5.44` words on average), while meme captions are longer (about `17.71` words).
   - Short/noisy title text can be helpful after adaptation (custom/LoRA), but is risky for zero-shot fusion.

In short: pretrained alignment + light adaptation explains LoRA success, while unadapted fusion explains zero-shot Type 2 collapse.

Observed behavior:

- LoRA converges faster and reaches substantially better retrieval than the custom model.
- Type 2 helps the custom model (`Type 2 > Type 1`) but hurts zero-shot CLIP in this setup.
- Best overall configuration is LoRA with Type 1 query.

## Reproducibility Artifacts

- Final comparison: [`outputs/results.json`](./outputs/results.json)
- Per-epoch logs:
  - [`outputs/history/custom_type1_history.json`](./outputs/history/custom_type1_history.json)
  - [`outputs/history/custom_type2_history.json`](./outputs/history/custom_type2_history.json)
  - [`outputs/history/lora_type1_history.json`](./outputs/history/lora_type1_history.json)
  - [`outputs/history/lora_type2_history.json`](./outputs/history/lora_type2_history.json)
