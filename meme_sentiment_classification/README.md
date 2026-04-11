# Task 2.3(a) — Sentiment Annotation Report

## Approach

To assign emotion labels to the dataset, we used the pretrained model `j-hartmann/emotion-english-distilroberta-base`. For each meme, we fed the first `meme_captions` entry into the model and obtained one of 7 emotion labels: `anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`. We chose this model because it covers the most common emotional tones out of the box and requires no additional fine-tuning.

---

## Class Distribution

### Train split (5,823 samples)

| Emotion  | Count |    % |
|----------|------:|-----:|
| neutral  | 3,200 | 55.0% |
| disgust  | 1,091 | 18.7% |
| sadness  |   437 |  7.5% |
| anger    |   374 |  6.4% |
| joy      |   314 |  5.4% |
| fear     |   219 |  3.8% |
| surprise |   188 |  3.2% |

- Imbalance ratio (max / min): **17.02×** (neutral vs. surprise)
- Shannon entropy: **2.027 / 2.807** (uniform = 2.807)

### Test split (559 samples)

| Emotion  | Count |    % |
|----------|------:|-----:|
| neutral  |   293 | 52.4% |
| disgust  |   104 | 18.6% |
| sadness  |    47 |  8.4% |
| anger    |    45 |  8.1% |
| joy      |    37 |  6.6% |
| surprise |    17 |  3.0% |
| fear     |    16 |  2.9% |

- Imbalance ratio (max / min): **18.31×** (neutral vs. fear)
- Shannon entropy: **2.092 / 2.807** (uniform = 2.807)

---

## Class Imbalance

Both splits show the same pattern: `neutral` dominates at roughly 52–55%, while `fear` and `surprise` together account for only about 6%. The imbalance ratios (~17–18×) are well above the 5× warning threshold, and the Shannon entropy sits around 2.03–2.09 out of a theoretical maximum of 2.81 — meaning the distribution is noticeably far from uniform.

This is expected for meme data. Meme captions are often indirect and ironic, which makes it hard for the model to commit to a specific emotion, so it defaults to `neutral`. The practical risk is that a classifier trained on these labels could achieve ~52% accuracy just by always predicting `neutral`. To counter this, we will need either class-weighted loss or resampling of minority classes (fear, surprise, joy) before training in tasks 2.3(b) and 2.3(c).

---

## Manual Label Check

We randomly sampled 50 examples from the training set (seed=42) and inspected whether the assigned labels were plausible given the caption text. Most high-confidence predictions (≥ 0.7) looked reasonable. The problematic cases were concentrated among low-confidence predictions, as expected.

**Clear mislabels or questionable assignments:**

- **[12]** `disgust` (conf **0.383**): *"Meme poster is trying to form a person's name from the representations"* — the caption describes a word puzzle; `neutral` is the obvious correct label. The model had no basis for `disgust` here.
- **[15]** `sadness` (conf **0.308**): *"Meme poster will respond to emails saying not to reply and sowing chaos."* — this reads as mischievous or playful, not sad. Lowest confidence in the sample; clearly a guess.
- **[22]** `sadness` (conf **0.357**): *"Meme poster is tired of the world screwing them over and gives up."* — `sadness` is arguably defensible, but the frustration/resignation tone could equally be `anger`. Low confidence reflects the model's uncertainty.
- **[10]** `anger` (conf **0.511**): *"The meme poster promotes Snickers that it could satisfy hunger and get rid of being anger by hunger."* — this is a positive product endorsement, not an expression of anger. `neutral` or `joy` would fit better.
- **[19]** `anger` (conf **0.513**): *"Meme poster is trying to convey that the World Cup winning of western Europe are united now but had centuries of war."* — a historical/political observation; `neutral` is more appropriate.
- **[03]** `neutral` (conf **0.417**): *"Meme poster feels old because their McDonald's toy from four years ago is now a meme."* — the caption explicitly describes feeling old, which leans toward `sadness`. Low confidence here is a sign the model was uncertain between the two.

**General observations:**

- Samples with confidence below ~0.50 were consistently unreliable — roughly 6 of the 50 reviewed samples fell in this range, and most of them were wrong or debatable.
- High-confidence predictions (≥ 0.85) were almost always plausible, even when the caption was indirect.
- Sarcasm and irony in meme captions are a systematic source of error: the model reads the surface phrasing rather than the intended tone.

**Estimated mislabel rate from this sample:** roughly 8–12%.

One option to explore before training is dropping samples with `sentiment_score < 0.5`, which would remove the noisiest labels at the cost of losing a small fraction of training data.

---

# Task 2.3(b) — MLP Probes per Modality

## Approach

We extracted frozen CLIP ViT-B/32 embeddings (512-d) for each modality independently and trained two separate MLP classifiers:

- **Image-only probe** — input: CLIP image embedding
- **Caption-only probe** — input: CLIP meme-caption embedding

Both probes share the same architecture and training setup. Labels come from the annotated JSON produced in 2.3(a). Only samples with valid images are used for a fair comparison (train: 5,161 / test: 504 out of original 5,823 / 559).

---

## Training Details

**Split:** 90% train / 10% validation (of the image-valid subset)  
**Optimizer:** Adam, early stopping on validation Macro F1 (patience=5)

### Image-only probe
Stopped at epoch 13 (best checkpoint at epoch 8, Macro F1 = 0.2079). The probe's validation accuracy peaked at ~52% early on but that was a degenerate solution collapsing to `neutral`; Macro F1 kept climbing only slowly, revealing the model struggled to learn minority classes.

### Caption-only probe
Trained for the full 20 epochs (best at epoch 19, Macro F1 = 0.4196). Convergence was more stable — loss dropped steadily from 1.94 → 1.08, and Macro F1 improved consistently across all classes.

---

## Test Results

### Image-only probe

| Metric       | Value  |
|--------------|-------:|
| Accuracy     | 32.54% |
| Macro F1     | 0.1819 |
| Weighted F1  | 0.3305 |

Per-class F1:

| Emotion  |    F1 | Support |
|----------|------:|--------:|
| anger    | 0.086 |      41 |
| disgust  | 0.079 |      95 |
| fear     | 0.151 |      14 |
| joy      | 0.110 |      34 |
| neutral  | 0.526 |     263 |
| sadness  | 0.245 |      42 |
| surprise | 0.077 |      15 |

### Caption-only probe

| Metric       | Value  |
|--------------|-------:|
| Accuracy     | 43.25% |
| Macro F1     | 0.3860 |
| Weighted F1  | 0.4490 |

Per-class F1:

| Emotion  |    F1 | Support |
|----------|------:|--------:|
| anger    | 0.296 |      41 |
| disgust  | 0.287 |      95 |
| fear     | 0.391 |      14 |
| joy      | 0.514 |      34 |
| neutral  | 0.547 |     263 |
| sadness  | 0.375 |      42 |
| surprise | 0.292 |      15 |

---

## Modality Comparison

| Probe         | Accuracy | Macro F1 | Weighted F1 |
|---------------|:--------:|:--------:|:-----------:|
| Image-only    |  32.54%  |  0.1819  |    0.3305   |
| Caption-only  |  43.25%  |  0.3860  |    0.4490   |

Per-class F1 delta (caption − image):

| Emotion  | Image | Caption |     Δ |
|----------|------:|--------:|------:|
| anger    | 0.086 |   0.296 | +0.210 |
| disgust  | 0.079 |   0.287 | +0.208 |
| fear     | 0.151 |   0.391 | +0.240 |
| joy      | 0.110 |   0.514 | +0.405 |
| neutral  | 0.526 |   0.547 | +0.021 |
| sadness  | 0.245 |   0.375 | +0.130 |
| surprise | 0.077 |   0.292 | +0.215 |

---

## Analysis

Caption text carries significantly more sentiment signal than the image alone. The caption probe outperforms the image probe on every single class, with the largest gaps on `joy` (+0.40) and `fear` (+0.24). This is expected: CLIP image embeddings encode visual semantics (objects, scene type) rather than emotional tone, whereas the caption directly expresses the meme poster's intent in language that the emotion annotator also operates on.

The image probe's accuracy (32.54%) barely exceeds the majority-class baseline (~52% for `neutral` on raw accuracy) only because the class distribution in the image-valid subset is slightly different — in practice the image probe's Macro F1 of 0.18 shows it is near-chance for minority classes.

Both probes suffer from the heavy `neutral` dominance identified in 2.3(a). The per-class F1 for `surprise` (0.077 / 0.292) and `disgust` (0.079 / 0.287) remain low even for the caption probe, pointing to label noise and under-representation as the main bottlenecks for 2.3(c).

**Implications for 2.3(c):** A fused model (image + caption) should improve over the caption-only baseline, but the gain from images may be modest. Class-weighted loss or oversampling of minority emotions (`fear`, `surprise`, `joy`) will likely matter more than fusion strategy.
