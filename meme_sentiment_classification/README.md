# Task 2.3 — Meme Sentiment Classification

---

## Task 2.3(a) — Sentiment Annotation

### Approach

To assign emotion labels to the dataset, we used the pretrained model `j-hartmann/emotion-english-distilroberta-base`. For each meme, we fed the first `meme_captions` entry into the model and obtained one of 7 emotion labels: `anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`. We chose this model because it covers the most common emotional tones out of the box and requires no additional fine-tuning.

---

### Class Distribution

#### Train split (5,823 samples)

| Emotion  | Count |     % |
| -------- | ----: | ----: |
| neutral  | 2,153 | 37.0% |
| anger    |   959 | 16.5% |
| sadness  |   860 | 14.8% |
| joy      |   621 | 10.7% |
| surprise |   495 |  8.5% |
| disgust  |   393 |  6.7% |
| fear     |   342 |  5.9% |

- Imbalance ratio (max / min): **6.30×** (neutral vs. fear)
- Shannon entropy: **2.516 / 2.807** (uniform = 2.807)

#### Test split (559 samples)

| Emotion  | Count |     % |
| -------- | ----: | ----: |
| neutral  |   230 | 41.1% |
| anger    |   100 | 17.9% |
| sadness  |    74 | 13.2% |
| joy      |    55 |  9.8% |
| surprise |    45 |  8.1% |
| disgust  |    30 |  5.4% |
| fear     |    25 |  4.5% |

- Imbalance ratio (max / min): **9.20×** (neutral vs. fear)
- Shannon entropy: **2.406 / 2.807** (uniform = 2.807)

---

### Class Imbalance

Both splits show the same rank ordering — `neutral` dominates at 37–41%, while `fear` and `disgust` together account for only about 10–12%. The imbalance ratios (6.3× train, 9.2× test) exceed the 5× warning threshold, though they are considerably milder than typical meme sentiment datasets (which can reach 17–18×). The Shannon entropy of 2.52 (train) and 2.41 (test) out of a theoretical maximum of 2.81 indicates a moderately skewed but not extreme distribution.

Compared to an earlier annotation pass on the same task, the current distribution is substantially more balanced: `neutral` dropped from ~55% to 37%, and previously underrepresented classes like `anger` (6.4% → 16.5%) and `sadness` (7.5% → 14.8%) are now well-populated. This makes the learning problem more tractable.

The practical risk remains that a classifier trained naively could achieve ~41% accuracy by always predicting `neutral`. To counter this, we apply class-weighted loss and oversampling of minority classes in tasks 2.3(b) and 2.3(c).

---

### Manual Label Check

We randomly sampled 50 examples from the training set (seed=42) and inspected whether the assigned labels were plausible given the caption text. Most high-confidence predictions (≥ 0.7) looked reasonable. Problematic cases were concentrated among low-confidence predictions.

**Clear mislabels or questionable assignments:**

- **[12]** `disgust` (conf **0.383**): _"Meme poster is trying to form a person's name from the representations"_ — the caption describes a word puzzle; `neutral` is the obvious correct label. The model had no basis for `disgust` here.
- **[15]** `sadness` (conf **0.308**): _"Meme poster will respond to emails saying not to reply and sowing chaos."_ — this reads as mischievous or playful, not sad. Lowest confidence in the sample; clearly a guess.
- **[22]** `sadness` (conf **0.357**): _"Meme poster is tired of the world screwing them over and gives up."_ — `sadness` is arguably defensible, but the frustration/resignation tone could equally be `anger`. Low confidence reflects the model's uncertainty.
- **[10]** `anger` (conf **0.511**): _"The meme poster promotes Snickers that it could satisfy hunger and get rid of being anger by hunger."_ — this is a positive product endorsement, not an expression of anger. `neutral` or `joy` would fit better.
- **[19]** `anger` (conf **0.513**): _"Meme poster is trying to convey that the World Cup winning of western Europe are united now but had centuries of war."_ — a historical/political observation; `neutral` is more appropriate.
- **[03]** `neutral` (conf **0.417**): _"Meme poster feels old because their McDonald's toy from four years ago is now a meme."_ — the caption explicitly describes feeling old, which leans toward `sadness`. Low confidence here is a sign the model was uncertain between the two.

**General observations:**

- Samples with confidence below ~0.50 were consistently unreliable — roughly 6 of the 50 reviewed samples fell in this range, and most of them were wrong or debatable.
- High-confidence predictions (≥ 0.85) were almost always plausible, even when the caption was indirect.
- Sarcasm and irony in meme captions are a systematic source of error: the model reads the surface phrasing rather than the intended tone.

**Estimated mislabel rate from this sample:** roughly 8–12%.

One option to explore before training is dropping samples with `sentiment_score < 0.5`, which would remove the noisiest labels at the cost of losing a small fraction of training data.

---

## Task 2.3(b) — MLP Probes per Modality

### Approach

We extracted frozen CLIP ViT-B/32 embeddings (512-d) for each modality independently and trained two separate MLP classifiers:

- **Image-only probe** — input: CLIP image embedding
- **Caption-only probe** — input: CLIP meme-caption embedding

Both probes share the same architecture and training setup. Labels come from the annotated JSON produced in 2.3(a). Only samples with valid images are used for a fair comparison (train: 5,160 / test: 504 out of original 5,823 / 559).

---

### Training Details

**Split:** 90% train / 10% validation (of the image-valid subset)  
**Optimizer:** AdamW, early stopping on validation Macro F1 (patience=10)

#### Image-only probe

Stopped at epoch 20 (best checkpoint at epoch 11, Macro F1 = 0.2253). Validation accuracy was highly unstable throughout — swinging between 14% and 41% across consecutive epochs — indicating the image embeddings carry weak and inconsistent sentiment signal. The best Macro F1 of 0.2253 on validation did not hold on the test set (0.1983).

#### Caption-only probe

Ran for the full 20 epochs (best at epoch 20, Macro F1 = 0.3730). Convergence was more stable — loss dropped steadily from 1.95 → 1.28 — though val accuracy still oscillated (0.38–0.44 range), reflecting the inherent noise in the annotation labels. The best validation Macro F1 of 0.3730 translated to 0.3533 on the test set.

---

### Test Results

#### Image-only probe

| Metric      |  Value |
| ----------- | -----: |
| Accuracy    | 23.81% |
| Macro F1    | 0.1983 |
| Weighted F1 | 0.2584 |

Per-class F1 (test support in parentheses):

| Emotion  |    F1 | Support |
| -------- | ----: | ------: |
| anger    | 0.238 |      90 |
| disgust  | 0.076 |      25 |
| fear     | 0.151 |      22 |
| joy      | 0.239 |      47 |
| neutral  | 0.333 |     213 |
| sadness  | 0.278 |      66 |
| surprise | 0.073 |      41 |

#### Caption-only probe

| Metric      |  Value |
| ----------- | -----: |
| Accuracy    | 36.51% |
| Macro F1    | 0.3533 |
| Weighted F1 | 0.3634 |

Per-class F1:

| Emotion  |    F1 | Support |
| -------- | ----: | ------: |
| anger    | 0.402 |      90 |
| disgust  | 0.176 |      25 |
| fear     | 0.386 |      22 |
| joy      | 0.428 |      47 |
| neutral  | 0.335 |     213 |
| sadness  | 0.485 |      66 |
| surprise | 0.262 |      41 |

---

### Modality Comparison

| Probe        | Accuracy | Macro F1 | Weighted F1 |
| ------------ | :------: | :------: | :---------: |
| Image-only   |  23.81%  |  0.1983  |   0.2584    |
| Caption-only |  36.51%  |  0.3533  |   0.3634    |

Per-class F1 delta (caption − image):

| Emotion  | Image | Caption |      Δ |
| -------- | ----: | ------: | -----: |
| anger    | 0.238 |   0.402 | +0.164 |
| disgust  | 0.076 |   0.176 | +0.100 |
| fear     | 0.151 |   0.386 | +0.235 |
| joy      | 0.239 |   0.428 | +0.189 |
| neutral  | 0.333 |   0.335 | +0.002 |
| sadness  | 0.278 |   0.485 | +0.208 |
| surprise | 0.073 |   0.262 | +0.188 |

---

### Analysis

Caption text carries substantially more sentiment signal than the image alone. The caption probe outperforms the image probe on every class, with the largest gaps on `fear` (+0.235), `sadness` (+0.208), and `anger` (+0.164). This is expected: CLIP image embeddings encode visual semantics (objects, scene composition) rather than emotional tone, whereas the caption directly expresses the meme poster's intent in language the sentiment annotator also operates on.

The most striking finding is the near-zero gap on `neutral` (+0.002). Both probes perform similarly on the majority class, suggesting that CLIP visual features are moderately useful for detecting absence of strong emotion — but add nothing for discriminating between specific emotions.

The image probe's Macro F1 of 0.198 is roughly consistent with near-chance performance on 7 classes (random = 0.143), adjusted for the moderate imbalance. It shows a moderate ability to distinguish `neutral`, `anger`, `joy`, and `sadness` — classes that may have more distinct visual signatures — but essentially fails on `disgust` (0.076) and `surprise` (0.073).

**Implications for 2.3(c):** The caption-only probe sets the baseline to beat. A fused model should improve over this baseline primarily for emotionally expressive classes (`anger`, `sadness`, `joy`) where visual cues can reinforce textual signal. The near-zero image contribution to `neutral` suggests the fusion model will need gating to avoid the image modality degrading text-only predictions. Class-weighted loss and minority-class oversampling remain important given the 9.2× imbalance in the test set.

---

## Task 2.3(c) — Custom Architecture: CrossModal Attention Fusion Network (CAFN)

### Architecture Design

Rather than simply concatenating CLIP image and caption embeddings (late fusion, which is what parallel MLPs implicitly do), the CAFN learns _cross-modal interactions_ before classification. The full pipeline:

1. **CLIP image embedding** (512-d) → Image projection head (Linear + LayerNorm + GELU) → 256-d
2. **CLIP caption embedding** (512-d) → Text projection head (Linear + LayerNorm + GELU) → 256-d
3. **Bidirectional cross-attention**: image embedding queries text keys/values; text embedding queries image keys/values (multi-head, 4 heads each)
4. **Gated fusion**: a learned 2-way softmax gate predicts per-sample modality weights; the attended representations are combined as `g_i · i_att + g_t · t_att`
5. **Final representation**: gated fusion output concatenated with the cross-attended image features → 512-d → LayerNorm + GELU + Dropout
6. **Classification head**: 512-d → 256-d → GELU → Dropout → 7 classes

The cross-attention mechanism allows the model to learn _which textual concepts are relevant when processing the image_ and vice versa — something a simple concatenation of embeddings cannot do. The gated fusion allows the model to adaptively suppress the weaker modality (image) when it is uninformative for a given sample.

### Additional Training Design Choices

| Technique                            | Motivation                                                                                                                      |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| **Label smoothing** (ε=0.1)          | ~8–12% estimated mislabel rate from 2.3(a) manual check; prevents overconfidence on noisy labels                                |
| **Mixup** on embeddings (α=0.2)      | Generates synthetic training examples in embedding space; improves minority-class generalisation without requiring new data     |
| **Capped weighted sampler**          | Inverse-frequency oversampling, capped at 2× median weight to boost minorities without completely starving the majority class   |
| **Modality dropout** (p=0.15)        | Randomly zeros one modality for 15% of training samples; keeps the model robust when images are missing at inference            |
| **Cosine LR with linear warm-up**    | 3-epoch warm-up then cosine decay; avoids early trajectory instability                                                          |
| **Gradient clipping** (max norm 1.0) | Stabilises training with attention layers                                                                                       |
| **Smoothed early stopping**          | Patience counted on 3-epoch rolling average of val Macro F1, not raw per-epoch value; prevents stopping on a single noisy epoch |

### Results

_(To be filled in after final run)_

| Metric      | Image-only MLP | Caption-only MLP | **CAFN** |
| ----------- | :------------: | :--------------: | :------: |
| Accuracy    |     23.81%     |      36.51%      |    —     |
| Macro F1    |     0.1983     |      0.3533      |    —     |
| Weighted F1 |     0.2584     |      0.3634      |    —     |

### Discussion

The core motivation for CAFN over the part (b) baselines is that cross-attention can leverage complementary information: the caption tells the model _what emotion is expressed_, while the image provides _visual context_ about the scene and facial expressions. For emotionally expressive classes (`anger`, `sadness`, `joy`), image features can reinforce or disambiguate textual signal.

The gated fusion is particularly important given the findings of 2.3(b): the image modality is near-useless for `neutral` and `disgust`, but moderately informative for `anger` and `joy`. Without gating, a naive fusion would degrade text-only performance by injecting image noise into classes where images carry no signal. The learned gates allow the model to effectively become a caption-only classifier for those classes, while using image features for the classes that benefit.

The capped weighted sampler addresses the 9.2× imbalance in the test set without the neutral collapse observed in earlier uncapped runs, where the model stopped predicting neutral almost entirely.
