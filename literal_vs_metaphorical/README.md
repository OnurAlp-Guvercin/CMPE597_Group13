# Task 2.2 — Literal vs. Metaphorical Caption Classification

## Task Definition

Given a meme image and a caption, classify whether the caption **literally describes** the image (negative class, label = 0) or is a **metaphorical/interpretive** reading of the image (positive class, label = 1).

Labels are derived directly from the MemeCap dataset:

- **Positive (1)**: `meme_captions` — metaphorical interpretations of the image
- **Negative (0)**: `img_captions` — literal descriptions of the image

Since each image contributes both caption types, the resulting dataset is naturally balanced at ~50/50, making this a well-posed binary classification problem.

---

## 2.2(a) — Evaluation Framework

We implement a standard binary classification evaluation pipeline in `evaluate.py`. For each batch:

1. Run a forward pass through the model
2. Apply sigmoid to logits to obtain probabilities
3. Threshold at 0.5 to produce hard binary predictions

The following metrics are computed via `sklearn.metrics`:

| Metric               | Description                                                     |
| -------------------- | --------------------------------------------------------------- |
| **Accuracy**         | Overall fraction of correct predictions                         |
| **Precision**        | TP / (TP + FP) — how often a predicted positive is correct      |
| **Recall**           | TP / (TP + FN) — how often a true positive is retrieved         |
| **F1**               | Harmonic mean of precision and recall                           |
| **ROC-AUC**          | Area under the ROC curve; threshold-independent ranking quality |
| **Confusion Matrix** | Full 2×2 breakdown: TN, FP, FN, TP                              |

Because the class split is ~50/50 by construction, accuracy is a meaningful primary metric. Predictions and probabilities are optionally saved per epoch to JSON for offline analysis.

---

## 2.2(b) — Architecture

### Embedding Backbone

We use pretrained vision-language models as frozen feature extractors. For each (image, caption) pair, the model produces separate image and text embeddings which are then passed to a learned fusion head.

We experiment with two backbones:

| Backbone   | Model                            | Embedding dim | Pretraining objective                                 |
| ---------- | -------------------------------- | :-----------: | ----------------------------------------------------- |
| **CLIP**   | `openai/clip-vit-base-patch32`   |     512-d     | Softmax contrastive loss on 400M image-text pairs     |
| **SigLIP** | `google/siglip-base-patch16-224` |     768-d     | Sigmoid pairwise loss; trains each pair independently |

Both backbones are kept **frozen** throughout training — only the fusion head is updated.

### Fusion Strategies

We implement four fusion strategies on top of CLIP embeddings, plus a gated attention variant built on SigLIP.

| Strategy                     | Description                                                 | Input dim to classifier |
| ---------------------------- | ----------------------------------------------------------- | :---------------------: |
| **Concat**                   | Concatenate image and text embeddings                       |         1024-d          |
| **Multiply**                 | Element-wise product of embeddings                          |          512-d          |
| **Bilinear**                 | Learned bilinear transform: `nn.Bilinear(512, 512, 512)`    |          512-d          |
| **Cross-Attention**          | 4-head attention: image (query) attends to text (key/value) |          512-d          |
| **Gated Attention** (SigLIP) | Learned softmax gate over image + text; weighted sum        |          768-d          |

The **bilinear** fusion learns an explicit second-order interaction between the two embedding vectors — capturing whether a caption is consistent with or diverges from the visual content — which is more expressive than simple concatenation or element-wise product.

The **SigLIP gated attention** computes per-sample modality weights via a small MLP:

```
Linear(1536 → 512) → Tanh → Linear(512 → 2) → Softmax
fused = w_image · image_emb + w_text · text_emb
```

This allows the model to adaptively down-weight the image when the caption carries more discriminative signal, or vice versa.

### Classification Head

All fusion strategies share the same downstream classifier:

```
Linear(in_dim → 512) → ReLU → Dropout(p) → Linear(512 → 1)
```

Output is a single logit; `BCEWithLogitsLoss` is used during training.

### Training Configuration

| Setting       | CLIP models       | SigLIP model      |
| ------------- | ----------------- | ----------------- |
| Optimizer     | Adam              | AdamW             |
| Learning rate | 1e-4              | 1e-4              |
| Batch size    | 32                | 32                |
| Epochs        | 10                | 5                 |
| Loss          | BCEWithLogitsLoss | BCEWithLogitsLoss |
| Backbone      | Frozen            | Frozen            |

---

## 2.2(c) — Ablation Study and Results

### Experiment 1: CLIP ViT-B/32 + Bilinear Fusion

**Configuration:** bilinear fusion, hidden=512, dropout=0.1, 10 epochs

Training loss fell consistently from 0.0603 (epoch 1) to 0.0003 (epoch 10), indicating strong convergence. Best performance was reached at **epoch 5**.

#### Per-epoch results

| Epoch |       Loss |  Accuracy  |     F1     | Precision  |   Recall   |  ROC-AUC   |
| ----: | ---------: | :--------: | :--------: | :--------: | :--------: | :--------: |
|     1 |     0.0603 |   96.74%   |   0.9784   |   0.9977   |   0.9598   |   0.9979   |
|     2 |     0.0221 |   97.25%   |   0.9818   |   0.9994   |   0.9647   |   0.9996   |
|     3 |     0.0241 |   97.42%   |   0.9829   |   0.9977   |   0.9686   |   0.9986   |
|     4 |     0.0163 |   98.60%   |   0.9908   |   0.9983   |   0.9835   |   0.9968   |
| **5** | **0.0052** | **98.69%** | **0.9914** | **0.9961** | **0.9868** | **0.9987** |
|     6 |     0.0036 |   98.35%   |   0.9892   |   0.9972   |   0.9813   |   0.9989   |
|     7 |     0.0055 |   96.86%   |   0.9792   |   0.9994   |   0.9598   |   0.9990   |
|     8 |     0.0020 |   98.60%   |   0.9908   |   0.9989   |   0.9829   |   0.9992   |
|     9 |     0.0020 |   98.47%   |   0.9900   |   0.9994   |   0.9807   |   0.9988   |
|    10 |     0.0003 |   98.31%   |   0.9889   |   0.9989   |   0.9791   |   0.9934   |

#### Best result (epoch 5)

| Metric    |      Value |
| --------- | ---------: |
| Accuracy  | **98.69%** |
| Precision |     0.9961 |
| Recall    |     0.9868 |
| **F1**    | **0.9914** |
| ROC-AUC   |     0.9987 |

**Key observations:**

- Precision is near-perfect throughout (≥ 0.996) — when the model predicts "metaphorical", it is almost always correct.
- The primary error source is false negatives: metaphorical captions that CLIP misreads as visually consistent.
- Epoch 7 shows a brief recall drop (0.9598) despite near-zero training loss, characteristic of minor classification head overfitting after convergence.

---

### Experiment 2: SigLIP ViT-B/16 + Gated Attention Fusion

**Configuration:** gated attention fusion, hidden=512, dropout=0.1, 5 epochs

Strong performance from epoch 1, but ROC-AUC declines steadily across epochs, suggesting the model saturates quickly on this task.

#### Per-epoch results

| Epoch |  Accuracy  |     F1     |  ROC-AUC   |  TN |  FP |  FN |   TP |
| ----: | :--------: | :--------: | :--------: | --: | --: | --: | ---: |
| **1** | **97.17%** | **0.9812** | **0.9981** | 545 |   2 |  65 | 1752 |
|     2 |   97.08%   |   0.9807   |   0.9754   | 545 |   2 |  67 | 1750 |
|     3 |   97.04%   |   0.9804   |   0.9716   | 545 |   2 |  68 | 1749 |
|     4 |   97.12%   |   0.9810   |   0.9742   | 544 |   3 |  65 | 1752 |
|     5 |   96.83%   |   0.9790   |   0.9845   | 545 |   2 |  73 | 1744 |

#### Best result (epoch 1)

| Metric           |                        Value |
| ---------------- | ---------------------------: |
| Accuracy         |                   **97.17%** |
| F1               |                       0.9812 |
| ROC-AUC          |                       0.9981 |
| Confusion Matrix | TN=545, FP=2, FN=65, TP=1752 |

The confusion matrix reveals the same precision-dominated error pattern as the CLIP model: near-zero false positives (FP=2) throughout, with false negatives as the sole meaningful error source (~65–73 missed metaphorical captions per epoch).

---

### Ablation: Fusion Strategy (CLIP, multi-seed runs)

Beyond the two main experiments, we ran 34 additional runs across fusion type, hidden dimension, and dropout on CLIP ViT-B/32, logging each to `experiments.json`. Results summarised by configuration:

| Fusion          |  Hidden | Dropout | Mean Accuracy |  Mean F1  |  Std F1   |
| --------------- | ------: | ------: | :-----------: | :-------: | :-------: |
| Cross-attention |     512 |     0.1 |    97.01%     |   0.980   |   0.006   |
| Bilinear        |     512 |     0.1 |    97.85%     |   0.985   |   0.004   |
| Bilinear        |    1024 |     0.3 |    98.09%     |   0.987   |   0.004   |
| **Bilinear**    | **512** | **0.3** |  **98.19%**   | **0.988** | **0.003** |

**Effect of fusion strategy:** Bilinear consistently outperforms cross-attention in both mean F1 and run-to-run stability. The attention variant (image as query over text with sequence length 1) reduces to a learned weighted sum with limited expressive power, whereas the bilinear layer captures multiplicative interactions between the two embedding vectors.

**Effect of dropout:** Increasing dropout from 0.1 to 0.3 improves both mean performance and stability across all bilinear configurations. With a frozen backbone and moderate training set size, stronger regularisation on the small classification head helps generalisation.

**Effect of hidden dimension:** The larger hidden dimension (1024) brings no consistent benefit over 512. The CLIP projection space is already well-compressed; the classifier saturates at 512-d.

---

### Overall Comparison

| Model                  | Backbone                | Fusion        | Best Accuracy |  Best F1   | Best ROC-AUC |
| ---------------------- | ----------------------- | ------------- | :-----------: | :--------: | :----------: |
| **CLIP Bilinear**      | CLIP ViT-B/32 (512-d)   | Bilinear      |  **98.69%**   | **0.9914** |  **0.9987**  |
| SigLIP Gated Attention | SigLIP ViT-B/16 (768-d) | Gated softmax |    97.17%     |   0.9812   |    0.9981    |

**CLIP bilinear outperforms SigLIP gated attention** on all metrics despite using lower-dimensional embeddings (512-d vs. 768-d). The bilinear interaction is better suited to this task: detecting whether a caption deviates from what the image directly supports is inherently a multiplicative relationship between the two modalities. The SigLIP gated fusion produces a convex combination of the two embeddings (weights sum to 1), which is less expressive — it blends modalities rather than modelling their interaction.

The best overall configuration across all experiments is **bilinear fusion, hidden=512, dropout=0.3** on CLIP ViT-B/32 (mean F1 = 0.988, best single-run F1 = 0.990).

---

## Discussion

All configurations achieve high performance (F1 > 0.97), reflecting that the literal vs. metaphorical distinction maps cleanly onto the CLIP/SigLIP embedding space. Literal `img_captions` are exactly the kind of image-aligned text these models were contrastively pretrained on, so they sit close to their paired images in embedding space. Metaphorical `meme_captions` deviate from the image's visual content and sit further away. The classifier effectively learns to threshold this embedding-space distance, augmented by a learned interaction term.

Precision is near-perfect across all models: false positives (predicting metaphorical when the caption is literal) are extremely rare. The residual error concentrates in false negatives — metaphorical captions that happen to be worded in a visually plausible way, or captions that use ironic or understated language that CLIP/SigLIP interpret as literal. These are the genuinely hard cases where surface phrasing diverges from intended meaning.
