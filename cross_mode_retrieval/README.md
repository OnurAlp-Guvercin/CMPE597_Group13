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

Latest training/evaluation run (`weighted_sum`):

| Experiment | R@1 | R@5 | R@10 | MedR | MRR |
|---|---:|---:|---:|---:|---:|
| Zero-shot CLIP (Type 1) | 44.19% | 63.51% | 67.62% | 2 | 53.06% |
| Zero-shot CLIP (Type 2) | 36.14% | 55.81% | 59.93% | 4 | 44.83% |
| Custom Dual Encoder (Type 1) | 1.43% | 5.19% | 8.59% | 178 | 4.23% |
| Custom Dual Encoder (Type 2) | 7.69% | 17.17% | 23.61% | 96 | 13.03% |
| CLIP + LoRA (Type 1) | 52.59% | 68.16% | 72.81% | 1 | 59.71% |
| CLIP + LoRA (Type 2) | **57.25%** | **73.35%** | **77.82%** | **1** | **64.37%** |

Previous run (`concat_project`):

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

1. The new `weighted_sum` run improves all Type 2 settings, especially CLIP-based ones.
   - Zero-shot CLIP Type 2 rises from `R@1 0.36%` (`concat_project`) to `36.14%`.
   - LoRA Type 2 rises from `R@1 26.12%` to `57.25%` and becomes the best overall setup in the current report.
2. CLIP pretraining remains the strongest prior.
   - Both in old and new runs, LoRA consistently outperforms the custom dual encoder by a large margin.
   - Type 1 is still strong (`zeroshot 44.19%`, `lora 52.59%` R@1), showing robust image-text alignment from CLIP initialization.
3. Type 2 behavior is fusion-dependent, not inherently bad.
   - With `concat_project`, zero-shot Type 2 collapsed.
   - With `weighted_sum`, Type 2 becomes competitive/even dominant (for LoRA).
4. The custom model benefits from extra title context but still lags behind CLIP-based models.
   - In the latest run, custom Type 2 (`R@1 7.69%`) is much better than custom Type 1 (`1.43%`), but absolute performance is still low.
5. Practical takeaway from current experiments:
   - If compute budget allows training, prefer `CLIP + LoRA` with Type 2 + `weighted_sum`.
   - For no-training evaluation, zero-shot Type 1 remains a strong baseline, while zero-shot Type 2 is now usable with `weighted_sum`.
