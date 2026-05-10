"""
Task 2.3(c) — Custom Multimodal Fusion Architecture for Meme Sentiment Classification
======================================================================================

Architecture: CrossModal Attention Fusion Network (CAFN)
---------------------------------------------------------
Rather than simply concatenating CLIP image and caption embeddings (late fusion),
this model learns *cross-modal interactions* via a lightweight cross-attention block
before classification. The pipeline:

    1. CLIP image embedding  (512-d)  →  Image projection head  →  256-d
    2. CLIP caption embedding (512-d) →  Text projection head   →  256-d
    3. Cross-attention: image queries attend to text keys/values (and vice versa)
    4. Gated fusion: learned scalar gates weight each modality's attended output
    5. Concatenate gated outputs  →  512-d
    6. Classification head with residual connection  →  7 classes

Additional design choices vs. part (b):
  • Label smoothing (ε=0.1) instead of hard cross-entropy, to reduce overconfidence
    on noisy labels identified in 2.3(a).
  • Mixup augmentation on embeddings to improve minority-class generalisation.
  • Cosine-annealing LR schedule with warm-up.
  • Optional modality dropout during training (randomly zero one modality per sample)
    so the model stays robust when an image is missing.
"""

from __future__ import annotations

import json
import math
import os
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# Adjust this dir before running
IMAGE_DIR           = "/Users/coskunomer/CMPE597_Group13/literal_vs_metaphorical/images"
ANNOTATED_TRAIN_JSON = "outputs/annotated-trainval.json"
ANNOTATED_TEST_JSON  = "outputs/annotated-test.json"
OUTPUT_DIR          = "outputs"
CLIP_MODEL          = "openai/clip-vit-base-patch32"
SEED                = 42

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

LR         = 3e-4
BATCH_SIZE = 128
EPOCHS     = 100
DROPOUT    = 0.30
USE_MIXUP           = True   
USE_WEIGHTED_SAMPLER = True 

LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
LABEL2IDX: Dict[str, int] = {lbl: i for i, lbl in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)
CLIP_DIM = 512
PROJ_DIM = 256       # projection dimension before cross-attention
NUM_HEADS = 4        # attention heads in cross-attention
LABEL_SMOOTH = 0.10  # label smoothing epsilon
MIXUP_ALPHA = 0.2    # mixup beta distribution parameter (0 = disabled)
MOD_DROP_P = 0.15    # probability of zeroing one modality during training
VAL_RATIO = 0.10
PATIENCE = 20   
WARMUP_EPOCHS = 3 

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MultimodalDataset(Dataset):
    def __init__(
        self,
        img_embs: torch.Tensor,
        txt_embs: torch.Tensor,
        labels: torch.Tensor,
    ):
        assert len(img_embs) == len(txt_embs) == len(labels)
        self.img = img_embs
        self.txt = txt_embs
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.img[idx], self.txt[idx], self.labels[idx]


def load_annotated(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_embeddings(
    items: List[dict],
    processor: CLIPProcessor,
    model: CLIPModel,
    device: str,
    batch_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns img_embeds, text_embeds, labels, valid_mask (all length N)."""
    img_dir = IMAGE_DIR
    N = len(items)
    img_embs  = torch.zeros(N, CLIP_DIM)
    txt_embs  = torch.zeros(N, CLIP_DIM)
    label_ids = torch.zeros(N, dtype=torch.long)
    valid_mask = torch.zeros(N, dtype=torch.bool)

    model.eval()
    for start in tqdm(range(0, N, batch_size), desc="  Extracting"):
        batch   = items[start : start + batch_size]
        indices = list(range(start, min(start + batch_size, N)))

        # text
        captions = [
            item.get("sentiment_caption_used") or
            (item["meme_captions"][0] if item.get("meme_captions") else "")
            for item in batch
        ]
        text_in = processor(
            text=captions, return_tensors="pt",
            padding=True, truncation=True, max_length=77
        ).to(device)
        with torch.no_grad():
            t = model.text_model(**text_in)
            t = model.text_projection(t.pooler_output)
            t = t / t.norm(dim=-1, keepdim=True)
        txt_embs[indices] = t.cpu()

        # image
        img_idx, images = [], []
        for li, item in enumerate(batch):
            fname = item.get("img_fname", "")
            fpath = os.path.join(img_dir, fname)
            if fname and os.path.exists(fpath):
                try:
                    images.append(Image.open(fpath).convert("RGB"))
                    img_idx.append(indices[li])
                except Exception:
                    pass
        if images:
            img_in = processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                v = model.vision_model(**img_in)
                v = model.visual_projection(v.pooler_output)
                v = v / v.norm(dim=-1, keepdim=True)
            for k, gi in enumerate(img_idx):
                img_embs[gi] = v[k].cpu()
                valid_mask[gi] = True

        for li, (item, gi) in enumerate(zip(batch, indices)):
            label_ids[gi] = LABEL2IDX.get(item.get("sentiment_label", "neutral"), 4)

    return img_embs, txt_embs, label_ids, valid_mask

class ModalityProjector(nn.Module):
    """Projects a CLIP embedding to PROJ_DIM with layer-norm."""
    def __init__(self, in_dim: int = CLIP_DIM, out_dim: int = PROJ_DIM):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class CrossModalAttention(nn.Module):
    """
    Single-layer multi-head cross-attention.
    Query comes from modality A, keys/values from modality B.
    Returns the attended representation for modality A.
    """
    def __init__(self, dim: int = PROJ_DIM, num_heads: int = NUM_HEADS, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = query.unsqueeze(1)    # (B, 1, D)
        kv = context.unsqueeze(1) # (B, 1, D)
        attended, _ = self.attn(q, kv, kv)
        attended = attended.squeeze(1)  # (B, D)
        return self.norm(attended + query)  # residual connection


class GatedFusion(nn.Module):
    """
    Learns per-modality scalar gates from both attended representations,
    then combines them: out = gate_i * img_att + gate_t * txt_att
    """
    def __init__(self, dim: int = PROJ_DIM):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, 2),
            nn.Softmax(dim=-1),
        )

    def forward(
        self, img_att: torch.Tensor, txt_att: torch.Tensor
    ) -> torch.Tensor:
        combined = torch.cat([img_att, txt_att], dim=-1)  # (B, 2D)
        gates = self.gate(combined)                        # (B, 2)
        g_i, g_t = gates[:, 0:1], gates[:, 1:2]
        fused = g_i * img_att + g_t * txt_att             # (B, D)
        return fused


class CrossModalFusionNet(nn.Module):
    """
    Full CAFN classifier.

    Forward pass:
        img (B,512), txt (B,512), mod_drop_p (float, training only)
    Returns:
        logits (B, NUM_CLASSES)
    """
    def __init__(
        self,
        clip_dim: int = CLIP_DIM,
        proj_dim: int = PROJ_DIM,
        num_heads: int = NUM_HEADS,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.30,
    ):
        super().__init__()

        # 1. Projection heads
        self.img_proj = ModalityProjector(clip_dim, proj_dim)
        self.txt_proj = ModalityProjector(clip_dim, proj_dim)

        # 2. Cross-attention (bidirectional)
        self.img_attn_txt = CrossModalAttention(proj_dim, num_heads)  # img queries text
        self.txt_attn_img = CrossModalAttention(proj_dim, num_heads)  # text queries image

        # 3. Gated fusion of attended representations
        self.gate_fusion = GatedFusion(proj_dim)

        # 4. Self-attended image + text
        # We also keep a direct path of attended features for richer rep.
        self.final_proj = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim * 2),
            nn.LayerNorm(proj_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 5. Classification head
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes),
        )

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        mod_drop_p: float = 0.0,
    ) -> torch.Tensor:

        if self.training and mod_drop_p > 0:
            mask = torch.rand(img.size(0), device=img.device)
            zero_img = mask < mod_drop_p / 2
            zero_txt = (mask >= mod_drop_p / 2) & (mask < mod_drop_p)
            img = img.clone()
            txt = txt.clone()
            img[zero_img] = 0.0
            txt[zero_txt] = 0.0

        # Project
        i = self.img_proj(img)  # (B, proj_dim)
        t = self.txt_proj(txt)  # (B, proj_dim)

        # Bidirectional cross-attention
        i_att = self.img_attn_txt(i, t)   # image attends to text
        t_att = self.txt_attn_img(t, i)   # text attends to image

        # Gated fusion → single (B, proj_dim)
        fused = self.gate_fusion(i_att, t_att)

        combined = torch.cat([fused, i_att], dim=-1)  # (B, proj_dim * 2)
        combined = self.final_proj(combined)

        return self.classifier(combined)

def mixup_batch(
    img: torch.Tensor,
    txt: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = MIXUP_ALPHA,
    num_classes: int = NUM_CLASSES,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns mixed img, txt, and soft one-hot labels."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    B = img.size(0)
    idx = torch.randperm(B, device=img.device)
    img_mix = lam * img + (1 - lam) * img[idx]
    txt_mix = lam * txt + (1 - lam) * txt[idx]
    y = F.one_hot(labels, num_classes).float()
    y_mix = lam * y + (1 - lam) * y[idx]
    return img_mix, txt_mix, y_mix

class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing: float = LABEL_SMOOTH, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        targets: (B,) long labels  OR  (B, C) soft labels (from mixup).
        """
        log_probs = F.log_softmax(logits, dim=-1)
        if targets.dim() == 1:
            # Convert to soft targets with label smoothing
            y = torch.zeros_like(log_probs)
            y.fill_(self.smoothing / (self.num_classes - 1))
            y.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        else:
            # Already soft (from mixup); apply smoothing on top
            y = (1 - self.smoothing) * targets + self.smoothing / self.num_classes
        return -(y * log_probs).sum(dim=-1).mean()


def build_weighted_sampler(labels: torch.Tensor) -> WeightedRandomSampler:
    labels = labels.long()
    counts = torch.bincount(labels, minlength=NUM_CLASSES).float().clamp(min=1)
    weights_per_class = 1.0 / counts
    sample_weights = weights_per_class[labels]
    return WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(labels),
        replacement=True,
    )

def train_val_split(dataset: Dataset, val_ratio: float, seed: int = 42):
    n_val = max(1, int(len(dataset) * val_ratio))
    n_train = len(dataset) - n_val
    return torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

def get_train_labels(full_ds: MultimodalDataset, train_subset) -> torch.Tensor:
    """
    Get labels for the training subset directly from the base dataset's label tensor.
    Works by converting subset indices (range or list) to a LongTensor and indexing.
    Avoids any chain-walking that can fail on older PyTorch versions.
    """
    idx = torch.tensor(list(train_subset.indices), dtype=torch.long)
    return full_ds.labels[idx].long()

def get_labels_from_subset(subset) -> torch.Tensor:
    """
    Extract labels from a Subset by iterating it directly.
    Avoids any assumption about how .indices is stored across PyTorch versions.
    """
    return torch.stack([subset[i][2] for i in range(len(subset))]).long()


def train_cafn(
    train_ds: Dataset,
    val_ds: Dataset,
    device: str,
    epochs: int = EPOCHS,
    lr: float = 3e-4,
    batch_size: int = 128,
    dropout: float = 0.30,
    use_mixup: bool = True,
    use_weighted_sampler: bool = True,
    train_labels: torch.Tensor = None,
) -> CrossModalFusionNet:

    model = CrossModalFusionNet(dropout=dropout).to(device)
    criterion = SoftCrossEntropyLoss(smoothing=LABEL_SMOOTH)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(1, epochs - WARMUP_EPOCHS)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if use_weighted_sampler:
        if train_labels is None:
            train_labels = get_labels_from_subset(train_ds)
        sampler = build_weighted_sampler(train_labels)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_macro_f1 = -1.0
    best_state = None
    bad_epochs = 0

    print(f"\n  Training CAFN  ({len(train_ds)} train / {len(val_ds)} val)")
    print(f"  {'Epoch':>5}  {'Loss':>8}  {'Val Acc':>8}  {'Macro F1':>9}  {'LR':>10}")
    print(f"  {'─'*52}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for img_emb, txt_emb, lbl in train_loader:
            img_emb = img_emb.to(device)
            txt_emb = txt_emb.to(device)
            lbl     = lbl.to(device)

            if use_mixup and MIXUP_ALPHA > 0:
                img_emb, txt_emb, soft_lbl = mixup_batch(img_emb, txt_emb, lbl)
                logits = model(img_emb, txt_emb, mod_drop_p=MOD_DROP_P)
                loss = criterion(logits, soft_lbl)
            else:
                logits = model(img_emb, txt_emb, mod_drop_p=MOD_DROP_P)
                loss = criterion(logits, lbl)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Validation
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for img_emb, txt_emb, lbl in val_loader:
                preds = model(img_emb.to(device), txt_emb.to(device)).argmax(dim=1).cpu()
                all_preds.extend(preds.tolist())
                all_true.extend(lbl.tolist())

        acc      = accuracy_score(all_true, all_preds)
        macro_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
        avg_loss = total_loss / len(train_loader)

        print(f"  {epoch:>5}  {avg_loss:>8.4f}  {acc:>7.3f}  {macro_f1:>9.4f}  {current_lr:>10.2e}", end="")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
            print("  ✓")
        else:
            bad_epochs += 1
            print(f"  ({bad_epochs}/{PATIENCE})")
            if bad_epochs >= PATIENCE:
                print("  Early stopping.")
                break

    if best_state:
        model.load_state_dict(best_state)

    os.makedirs(os.path.join(OUTPUT_DIR, "probes"), exist_ok=True)
    ckpt = os.path.join(OUTPUT_DIR, "probes", "cafn_best.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"  Best checkpoint → {ckpt}  (val Macro F1 = {best_macro_f1:.4f})")

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_cafn(
    model: CrossModalFusionNet,
    test_ds: MultimodalDataset,
    device: str,
    batch_size: int = 128,
) -> Dict:
    model.eval()
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    all_preds, all_true = [], []
    for img_emb, txt_emb, lbl in loader:
        preds = model(img_emb.to(device), txt_emb.to(device)).argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_true.extend(lbl.tolist())

    acc         = accuracy_score(all_true, all_preds)
    macro_f1    = f1_score(all_true, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_true, all_preds, average="weighted", zero_division=0)
    per_class   = f1_score(
        all_true, all_preds, average=None,
        labels=list(range(NUM_CLASSES)), zero_division=0
    )

    print(f"\n{'='*65}")
    print("  Test Results — CrossModal Attention Fusion Network (CAFN)")
    print(f"{'='*65}")
    print(f"  Accuracy   : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Macro F1   : {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    print(f"\n  Per-class F1:")
    print(f"  {'Label':<12} {'F1':>8}  {'Support':>8}")
    print(f"  {'─'*32}")
    support = Counter(all_true)
    for i, lbl in enumerate(LABELS):
        print(f"  {lbl:<12} {per_class[i]:>8.4f}  {support[i]:>8}")
    print(f"{'='*65}")

    print("\n  Full classification report:")
    print(classification_report(all_true, all_preds, target_names=LABELS, zero_division=0))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": {lbl: float(per_class[i]) for i, lbl in enumerate(LABELS)},
    }

PART_B_RESULTS = {
    "Image-only MLP": {
        "accuracy": 0.2381, "macro_f1": 0.1983, "weighted_f1": 0.2584,
        "per_class_f1": {
            "anger": 0.2384, "disgust": 0.0759, "fear": 0.1505,
            "joy": 0.2390, "neutral": 0.3333, "sadness": 0.2778, "surprise": 0.0732,
        },
    },
    "Caption-only MLP": {
        "accuracy": 0.3651, "macro_f1": 0.3533, "weighted_f1": 0.3634,
        "per_class_f1": {
            "anger": 0.4019, "disgust": 0.1758, "fear": 0.3860,
            "joy": 0.4276, "neutral": 0.3345, "sadness": 0.4853, "surprise": 0.2619,
        },
    },
}



def print_full_comparison(cafn_results: Dict) -> None:
    all_results = {**PART_B_RESULTS, "CAFN (ours)": cafn_results}

    print(f"\n{'═'*72}")
    print("  FULL COMPARISON — Part (b) vs Part (c)")
    print(f"{'═'*72}")
    print(f"  {'Model':<22} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12}")
    print(f"  {'─'*58}")
    for name, r in all_results.items():
        print(
            f"  {name:<22} {r['accuracy']:>9.4f}  "
            f"{r['macro_f1']:>9.4f}  {r['weighted_f1']:>11.4f}"
        )

    print(f"\n  Per-class F1 breakdown:")
    header = f"  {'Label':<12}" + "".join(f"{n:>18}" for n in all_results)
    print(header)
    print(f"  {'─'*72}")
    for lbl in LABELS:
        row = f"  {lbl:<12}"
        for r in all_results.values():
            row += f"{r['per_class_f1'][lbl]:>18.4f}"
        print(row)
    print(f"{'═'*72}\n")

def main() -> None:
    device = DEVICE

    seed_everything(SEED)
    print("=" * 65)
    print("  Task 2.3(c) — CrossModal Attention Fusion Network (CAFN)")
    print("=" * 65)
    print(f"  Device  : {device}")
    print(f"  CLIP    : {CLIP_MODEL}")
    print(f"  Epochs  : {EPOCHS}  |  LR: {LR}  |  BS: {BATCH_SIZE}")
    print(f"  Mixup   : {USE_MIXUP}  |  Weighted sampler: {USE_WEIGHTED_SAMPLER}")

    # ---- Load CLIP ----
    print("\nLoading CLIP model …")
    processor  = CLIPProcessor.from_pretrained(CLIP_MODEL)
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
    clip_model.eval()

    # ---- Extract embeddings ----
    print("\n--- Train split ---")
    train_items = load_annotated(ANNOTATED_TRAIN_JSON)
    tr_img, tr_txt, tr_lbl, tr_mask = extract_embeddings(
        train_items, processor, clip_model, device
    )

    print("\n--- Test split ---")
    test_items = load_annotated(ANNOTATED_TEST_JSON)
    te_img, te_txt, te_lbl, te_mask = extract_embeddings(
        test_items, processor, clip_model, device
    )

    tr_idx = tr_mask.nonzero(as_tuple=True)[0]
    te_idx = te_mask.nonzero(as_tuple=True)[0]
    print(f"\n  Image-valid: train={len(tr_idx)}, test={len(te_idx)}")

    full_train_ds = MultimodalDataset(
        tr_img[tr_idx], tr_txt[tr_idx], tr_lbl[tr_idx]
    )
    test_ds = MultimodalDataset(
        te_img[te_idx], te_txt[te_idx], te_lbl[te_idx]
    )

    train_sub, val_sub = train_val_split(full_train_ds, VAL_RATIO, SEED)

    # ---- Train CAFN ----
    print("\n" + "═" * 65)
    print("  Training CrossModal Attention Fusion Network")
    print("═" * 65)
    train_labels = get_train_labels(full_train_ds, train_sub)
    print(f"  Train labels extracted: {len(train_labels)} samples")

    model = train_cafn(
        train_ds=train_sub,
        val_ds=val_sub,
        device=device,
        epochs=EPOCHS,
        lr=LR,
        batch_size=BATCH_SIZE,
        dropout=DROPOUT,
        use_mixup=USE_MIXUP,
        use_weighted_sampler=USE_WEIGHTED_SAMPLER,
        train_labels=train_labels,
    )

    # ---- Evaluate ----
    cafn_results = evaluate_cafn(model, test_ds, device, BATCH_SIZE)

    # ---- Full comparison ----
    print_full_comparison(cafn_results)

    # ---- Save results ----
    out_path = os.path.join(OUTPUT_DIR, "cafn_results.json")
    with open(out_path, "w") as f:
        json.dump(cafn_results, f, indent=2)
    print(f"  Results saved → {out_path}")


if __name__ == "__main__":
    main()