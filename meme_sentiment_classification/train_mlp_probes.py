import argparse
import json
import os
import random
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
LABEL2IDX: Dict[str, int] = {lbl: i for i, lbl in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)
CLIP_DIM = 512          # ViT-B/32 joint embedding dimension
VAL_RATIO = 0.1
PATIENCE = 5            # early stopping on val macro-F1


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EmbeddingDataset(Dataset):
    """Wraps pre-computed (embedding, label_idx) pairs."""

    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor):
        assert len(embeddings) == len(labels)
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# CLIP embedding extraction
# ---------------------------------------------------------------------------

def load_annotated(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _valid_image(img_dir: str, fname: str) -> bool:
    return bool(fname) and os.path.exists(os.path.join(img_dir, fname))


def extract_embeddings(
    items: List[dict],
    processor: CLIPProcessor,
    model: CLIPModel,
    device: str,
    batch_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        img_embeds   — (N, 512) float32, None rows = missing image (filled with zeros)
        text_embeds  — (N, 512) float32
        labels       — (N,) int64
        valid_mask   — (N,) bool, True where image was available
    """
    img_dir = config.IMAGE_DIR
    N = len(items)

    img_embeds = torch.zeros(N, CLIP_DIM)
    text_embeds = torch.zeros(N, CLIP_DIM)
    label_ids = torch.zeros(N, dtype=torch.long)
    valid_mask = torch.zeros(N, dtype=torch.bool)

    model.eval()

    for start in tqdm(range(0, N, batch_size), desc="  Extracting embeddings"):
        batch = items[start : start + batch_size]
        indices = list(range(start, min(start + batch_size, N)))

        # ---- text ----
        captions = [
            item.get("sentiment_caption_used") or
            (item["meme_captions"][0] if item.get("meme_captions") else "")
            for item in batch
        ]
        text_inputs = processor(
            text=captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(device)

        with torch.no_grad():
            t_raw = model.text_model(**text_inputs)
            t_out = model.text_projection(t_raw.pooler_output)
            t_out = t_out / t_out.norm(dim=-1, keepdim=True)

        text_embeds[indices] = t_out.cpu()

        # ---- image (skip missing files) ----
        img_indices, images = [], []
        for local_i, item in enumerate(batch):
            fname = item.get("img_fname", "")
            fpath = os.path.join(img_dir, fname)
            if _valid_image(img_dir, fname):
                try:
                    img = Image.open(fpath).convert("RGB")
                    images.append(img)
                    img_indices.append(indices[local_i])
                except Exception:
                    pass  # corrupt file — skip

        if images:
            img_inputs = processor(
                images=images, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                i_raw = model.vision_model(**img_inputs)
                i_out = model.visual_projection(i_raw.pooler_output)
                i_out = i_out / i_out.norm(dim=-1, keepdim=True)
            for k, global_i in enumerate(img_indices):
                img_embeds[global_i] = i_out[k].cpu()
                valid_mask[global_i] = True

        # ---- labels ----
        for local_i, (item, global_i) in enumerate(zip(batch, indices)):
            label_ids[global_i] = LABEL2IDX.get(item.get("sentiment_label", "neutral"), 4)

    return img_embeds, text_embeds, label_ids, valid_mask


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLPProbe(nn.Module):
    def __init__(
        self,
        in_dim: int = CLIP_DIM,
        hidden: int = 512,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Class weights (inverse frequency)
# ---------------------------------------------------------------------------

def compute_class_weights(labels: torch.Tensor, num_classes: int, device: str) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float()
    counts = counts.clamp(min=1)
    weights = counts.sum() / (num_classes * counts)
    return weights.to(device)


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train_val_split(dataset: Dataset, val_ratio: float, seed: int = 42):
    n_val = max(1, int(len(dataset) * val_ratio))
    n_train = len(dataset) - n_val
    return torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )


def train_probe(
    name: str,
    train_ds: Dataset,
    val_ds: Dataset,
    device: str,
) -> MLPProbe:
    train_labels = torch.stack([train_ds[i][1] for i in range(len(train_ds))])
    class_weights = compute_class_weights(train_labels, NUM_CLASSES, device)

    model = MLPProbe(
        in_dim=CLIP_DIM,
        hidden=config.HIDDEN,
        num_classes=NUM_CLASSES,
        dropout=config.DROPOUT,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=config.LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    best_macro_f1 = -1.0
    best_state = None
    bad_epochs = 0

    print(f"\n  Training {name} probe  ({len(train_ds)} train / {len(val_ds)} val)")
    print(f"  {'Epoch':>5}  {'Loss':>8}  {'Val Acc':>8}  {'Macro F1':>9}")
    print(f"  {'─'*40}")

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for emb, lbl in train_loader:
            emb, lbl = emb.to(device), lbl.to(device)
            optimizer.zero_grad()
            loss = criterion(model(emb), lbl)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # validation
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for emb, lbl in val_loader:
                preds = model(emb.to(device)).argmax(dim=1).cpu()
                all_preds.extend(preds.tolist())
                all_true.extend(lbl.tolist())

        acc = accuracy_score(all_true, all_preds)
        macro_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
        avg_loss = total_loss / len(train_loader)

        print(f"  {epoch:>5}  {avg_loss:>8.4f}  {acc:>7.3f}  {macro_f1:>9.4f}", end="")

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

    if best_state is not None:
        model.load_state_dict(best_state)

    # save checkpoint
    os.makedirs(os.path.join(config.OUTPUT_DIR, "probes"), exist_ok=True)
    ckpt = os.path.join(config.OUTPUT_DIR, "probes", f"{name}_best.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"  Best checkpoint → {ckpt}")

    return model


@torch.no_grad()
def evaluate_probe(
    name: str,
    model: MLPProbe,
    test_ds: Dataset,
    device: str,
) -> Dict[str, float]:
    model.eval()
    loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    all_preds, all_true = [], []
    for emb, lbl in loader:
        preds = model(emb.to(device)).argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_true.extend(lbl.tolist())

    acc = accuracy_score(all_true, all_preds)
    macro_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_true, all_preds, average="weighted", zero_division=0)
    per_class_f1 = f1_score(all_true, all_preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)

    print(f"\n{'='*60}")
    print(f"  Test Results — {name} probe")
    print(f"{'='*60}")
    print(f"  Accuracy   : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Macro F1   : {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    print(f"\n  Per-class F1:")
    print(f"  {'Label':<12} {'F1':>8}  {'Support':>8}")
    print(f"  {'─'*32}")
    support = Counter(all_true)
    for i, lbl in enumerate(LABELS):
        print(f"  {lbl:<12} {per_class_f1[i]:>8.4f}  {support[i]:>8}")
    print(f"{'='*60}")

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": {lbl: float(per_class_f1[i]) for i, lbl in enumerate(LABELS)},
    }


# ---------------------------------------------------------------------------
# Summary comparison
# ---------------------------------------------------------------------------

def print_comparison(results: Dict[str, Dict]) -> None:
    print(f"\n{'═'*60}")
    print("  SUMMARY — Modality Comparison")
    print(f"{'═'*60}")
    print(f"  {'Probe':<18} {'Accuracy':>9} {'Macro F1':>9} {'Wtd F1':>9}")
    print(f"  {'─'*50}")
    for name, r in results.items():
        print(
            f"  {name:<18} {r['accuracy']:>8.4f}  "
            f"{r['macro_f1']:>8.4f}  {r['weighted_f1']:>8.4f}"
        )
    print()

    if len(results) == 2:
        names = list(results.keys())
        r0, r1 = results[names[0]], results[names[1]]
        print(f"  Per-class F1 comparison:")
        print(f"  {'Label':<12} {names[0]:>14} {names[1]:>14}  {'Δ (img−cap)':>12}")
        print(f"  {'─'*58}")
        for lbl in LABELS:
            f0 = r0["per_class_f1"][lbl]
            f1 = r1["per_class_f1"][lbl]
            delta = f0 - f1
            print(f"  {lbl:<12} {f0:>14.4f} {f1:>14.4f}  {delta:>+12.4f}")
    print(f"{'═'*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Task 2.3(b) MLP probes")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or config.DEVICE
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size

    seed_everything(config.SEED)

    print(f"Device : {device}")
    print(f"Model  : {config.CLIP_MODEL}")

    # ---- load CLIP ----
    print("\nLoading CLIP model …")
    processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL)
    clip_model = CLIPModel.from_pretrained(config.CLIP_MODEL).to(device)
    clip_model.eval()

    # ---- extract embeddings ----
    print("\n--- Train split ---")
    train_items = load_annotated(config.ANNOTATED_TRAIN_JSON)
    train_img_emb, train_txt_emb, train_labels, train_valid = extract_embeddings(
        train_items, processor, clip_model, device
    )

    print("\n--- Test split ---")
    test_items = load_annotated(config.ANNOTATED_TEST_JSON)
    test_img_emb, test_txt_emb, test_labels, test_valid = extract_embeddings(
        test_items, processor, clip_model, device
    )

    # ---- filter to samples with valid images (fair comparison) ----
    train_idx = train_valid.nonzero(as_tuple=True)[0]
    test_idx = test_valid.nonzero(as_tuple=True)[0]

    print(f"\n  Image-valid samples — train: {len(train_idx)}, test: {len(test_idx)}")

    # Build datasets (same indices for both probes)
    img_train_ds = EmbeddingDataset(train_img_emb[train_idx], train_labels[train_idx])
    txt_train_ds = EmbeddingDataset(train_txt_emb[train_idx], train_labels[train_idx])
    img_test_ds  = EmbeddingDataset(test_img_emb[test_idx],   test_labels[test_idx])
    txt_test_ds  = EmbeddingDataset(test_txt_emb[test_idx],   test_labels[test_idx])

    img_train_sub, img_val_sub = train_val_split(img_train_ds, VAL_RATIO, config.SEED)
    txt_train_sub, txt_val_sub = train_val_split(txt_train_ds, VAL_RATIO, config.SEED)

    # ---- train probes ----
    print("\n" + "═"*60)
    print("  Training: Image-only MLP probe")
    print("═"*60)
    img_probe = train_probe("image", img_train_sub, img_val_sub, device)

    print("\n" + "═"*60)
    print("  Training: Caption-only MLP probe")
    print("═"*60)
    cap_probe = train_probe("caption", txt_train_sub, txt_val_sub, device)

    # ---- evaluate ----
    img_results = evaluate_probe("Image-only", img_probe, img_test_ds, device)
    cap_results = evaluate_probe("Caption-only", cap_probe, txt_test_ds, device)

    # ---- summary ----
    print_comparison({"Image-only": img_results, "Caption-only": cap_results})

    # ---- save results ----
    out = {
        "image_probe": {k: v for k, v in img_results.items() if k != "per_class_f1"},
        "image_probe_per_class_f1": img_results["per_class_f1"],
        "caption_probe": {k: v for k, v in cap_results.items() if k != "per_class_f1"},
        "caption_probe_per_class_f1": cap_results["per_class_f1"],
    }
    out_path = os.path.join(config.OUTPUT_DIR, "probe_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Results saved → {out_path}")


if __name__ == "__main__":
    main()
