# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a **CMPE 597 Special Topics: Deep Learning** (Spring 2026, Bogazici University) term project. Three tasks on meme understanding using the MemeCap dataset, all worth 30 pts each:

1. **`cross_mode_retrieval/`** — Task 2.1: Cross-modal retrieval (image → meme caption)
2. **`literal_vs_metaphorical/`** — Task 2.2: Binary classification of literal vs. metaphorical caption pairs
3. **`meme_sentiment_classification/`** — Task 2.3: Meme sentiment classification (multiclass)

---

## Dataset

**MemeCap**: 6,384 memes (5,823 train / 559 test), scraped from Reddit.

Each JSON entry has:
- `img_fname` — image filename
- `title` — Reddit post title (used as meme title)
- `meme_captions` — metaphorical interpretation of the meme
- `img_captions` — literal/objective description of the image
- `metaphors` — list of `{metaphor, meaning}` mappings

The label construction for Task 2.2 derives from the dataset itself: `(image, meme_caption)` = positive (metaphorical), `(image, img_caption)` = negative (literal).

---

## Cross-Modal Retrieval (Task 2.1)

### Setup & Run
```bash
cd cross_mode_retrieval
pip install -r requirements.txt
python -m data.download_data

# Run all tasks and all fusion strategies
python main.py --device cuda:0

# Run a specific task and fusion strategy
python main.py --task lora --fusion_strategy weighted_sum --device cuda:0
```

Available `--task` values: `zeroshot`, `custom`, `lora`  
Available `--fusion_strategy` values: `concat_project`, `cross_attention`, `add`, `weighted_sum`, `gated`

### Architecture

Three retrieval systems compared across two input types:
- **Zero-shot CLIP** (ViT-B/32) — no training baseline
- **Custom Dual Encoder** — ResNet-50 (image) + DistilBERT (text), symmetric InfoNCE loss
- **CLIP + LoRA** — CLIP ViT-B/32 with low-rank adapters (r=8, alpha=16)

Input types:
- **Type 1** — image embedding only
- **Type 2** — image + meme title fused into one embedding (via `models/fusion.py`)

Five fusion strategies in `models/fusion.py`: `concat_project`, `cross_attention`, `add`, `weighted_sum`, `gated`.

### Key Config (`config.py`)
All configuration uses dataclasses:
- `TrainConfig`: batch_size=512, lr=1e-4, epochs=15, temperature=0.07, embed_dim=256, fp16=True
- `LoRAFinetuneConfig`: targets `["attn.out_proj", "mlp.c_fc", "mlp.c_proj"]`
- `DataConfig`: val_split=0.1

### Evaluation
Required metrics: **R@1, R@5**. Also computed: R@10, Median Rank, MRR (in `evaluation.py`).

---

## Literal vs. Metaphorical (Task 2.2)

### Setup & Run
```bash
cd literal_vs_metaphorical
pip install -r requirements.txt
python download_images.py
python train.py
```

### Architecture

- **Base:** CLIP ViT-B/32 (frozen encoders)
- **Fusion strategies** (`model.py`): `concat`, `multiply`, `bilinear`, `attention`
- **Head:** 2-layer MLP classifier with BCEWithLogitsLoss

### Key Config (`config.py`)
Flat module-level constants: `MODEL_NAME`, `BATCH_SIZE=32`, `LR=1e-4`, `EPOCHS=10`, `FUSION="bilinear"`, `HIDDEN=512`, `DROPOUT=0.3`.

---

## Meme Sentiment Classification (Task 2.3)

**`meme_sentiment_classification/`** — multiclass classification: given a meme image (and optionally its caption), predict its emotion.

### Setup & Run
```bash
cd meme_sentiment_classification
pip install -r requirements.txt
python -m data.download_data

# 2.3(a) — label generation
python annotate_sentiment.py

# 2.3(b) — MLP probes per modality (not yet implemented)
# python train_mlp_probes.py

# 2.3(c) — custom fused classifier (not yet implemented)
# python train.py
```

### Task 2.3(a) — Sentiment annotation (`annotate_sentiment.py`)

Goal: map each meme's `meme_captions[0]` to an emotion label using a pretrained model.

- **Model:** `j-hartmann/emotion-english-distilroberta-base` — 7 classes: `anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`
- **Output files:** `outputs/annotated-trainval.json` / `outputs/annotated-test.json` — original JSON entries extended with `sentiment_label`, `sentiment_score`, `sentiment_caption_used`
- **Class imbalance report:** per-class counts, percentages, ASCII bar chart, imbalance ratio (max/min), Shannon entropy; prints warning if ratio ≥ 5x
- **Manual noise check:** prints `MANUAL_REVIEW_N=50` randomly sampled `(caption, label, confidence)` tuples for human inspection of label quality

### Task 2.3(b) — MLP probes per modality (not yet implemented)

Goal: investigate how much sentiment information each modality carries independently.

- Extract frozen CLIP ViT-B/32 embeddings for meme images and meme captions separately
- Train two independent MLP classifiers:
  - **Image-only MLP** — input: CLIP image embedding
  - **Caption-only MLP** — input: CLIP meme-caption embedding
- Labels come from the annotated JSON produced in 2.3(a)
- Report and compare classification metrics (accuracy, per-class F1) for both probes

### Task 2.3(c) — Custom fused classifier (not yet implemented)

Goal: design a multiclass classifier that outperforms the unimodal baselines from 2.3(b).

- Input options to explore: image only, caption only, image + meme caption, image + img_caption
- Fuse CLIP embeddings (concat / attention / bilinear) → MLP classification head
- Output classes are the emotion labels from 2.3(a)
- Report performance and compare against the 2.3(b) probes; run ablation study over fusion strategies and input combinations

---

## Project Milestones (Spring 2026)

| Date | Milestone |
|------|-----------|
| 10 March | Task 2.1a,b — zero-shot evaluation |
| 24 March | Task 2.1c,d — custom model + LoRA finetuning |
| 7 April | Task 2.2a,b,c — literal vs. metaphorical |
| 28 April | Task 2.3a — sentiment label generation |
| 12 May | Final oral evaluation |
