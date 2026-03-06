import json
import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional, Any, Protocol, runtime_checkable

from evaluation import evaluate_retrieval, print_metrics
from config import CHECKPOINT_DIR, HISTORY_DIR


# ------------------------------------------------------------------------------
#  Protocol describing the interface all retrieval models must satisfy
# ------------------------------------------------------------------------------
@runtime_checkable
class RetrievalModel(Protocol):
    """Structural type shared by CLIPZeroShot, CustomDualEncoder, CLIPLoRA."""

    def get_query_type1(self, images: torch.Tensor) -> torch.Tensor: ...
    def get_query_type2(self, images: torch.Tensor, titles: List[str]) -> torch.Tensor: ...
    def get_candidates(self, captions: List[str], device: Any = ...) -> torch.Tensor: ...
    def eval(self) -> Any: ...
    def train(self, mode: bool = True) -> Any: ...
    def parameters(self) -> Any: ...
    def state_dict(self) -> Any: ...
    def load_state_dict(self, state_dict: Any) -> Any: ...


# ------------------------------------------------------------------------------
#  Collect all embeddings from a DataLoader
# ------------------------------------------------------------------------------
@torch.no_grad()
def collect_embeddings(
    model: RetrievalModel,
    loader: DataLoader,  # type: ignore[type-arg]
    input_type: int = 1,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Run the model over the entire loader and return stacked embeddings.

    Returns dict with keys:
        "query"      – query embeddings      [N, D]
        "candidate"  – candidate embeddings  [N, D]
    """
    model.eval()
    all_query: list[torch.Tensor] = []
    all_cand: list[torch.Tensor] = []

    for batch in tqdm(loader, desc="Collecting embeddings", leave=False):
        images = batch["image"].to(device)
        titles: List[str] = batch["title"]
        captions: List[str] = batch["meme_caption"]

        # Build query
        if input_type == 1:
            q = model.get_query_type1(images)
        else:  # Type 2
            q = model.get_query_type2(images, titles)

        # Build candidates (caption embeddings)
        c = model.get_candidates(captions, device)

        all_query.append(q.cpu())
        all_cand.append(c.cpu())

    return {
        "query": torch.cat(all_query, dim=0),
        "candidate": torch.cat(all_cand, dim=0),
    }


# ------------------------------------------------------------------------------
#  Evaluate a model
# ------------------------------------------------------------------------------
def evaluate_model(
    model: RetrievalModel,
    loader: DataLoader,  # type: ignore[type-arg]
    input_type: int = 1,
    device: str = "cuda",
    header: str = "",
) -> Dict[str, float]:
    """Evaluate retrieval performance and print results."""
    embs = collect_embeddings(model, loader, input_type, device) #torch.Size([559, 512]), torch.Size([559, 512])
    metrics = evaluate_retrieval(embs["query"], embs["candidate"])
    if header:
        print_metrics(metrics, header)
    return metrics


def save_training_history(
    history_path: str,
    run_name: str,
    input_type: int,
    ckpt_path: str,
    history: List[Dict[str, Any]],
    best_metrics: Dict[str, float],
) -> None:
    """Persist epoch-by-epoch training history for later inspection."""
    payload = {
        "run_name": run_name,
        "input_type": int(input_type),
        "best_checkpoint": ckpt_path,
        "best_metrics": {k: float(v) for k, v in best_metrics.items()},
        "epochs": history,
    }
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ------------------------------------------------------------------------------
#  Single training epoch
# ------------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,  # type: ignore[type-arg]
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    input_type: int = 1,
    device: str = "cuda",
    use_fp16: bool = True,
) -> float:
    """Returns average loss for the epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    # Cast to Any so Pylance accepts the custom __call__ signature
    model_fn: Any = model

    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch["image"].to(device)
        captions: List[str] = batch["meme_caption"]
        titles: Optional[List[str]] = batch["title"] if input_type == 2 else None

        optimizer.zero_grad()

        if use_fp16 and scaler is not None:
            with autocast():
                loss: torch.Tensor = model_fn(images, captions, titles=titles,
                                              input_type=input_type)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = model_fn(images, captions, titles=titles,
                            input_type=input_type)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ------------------------------------------------------------------------------
#  Full training loop
# ------------------------------------------------------------------------------
def train_model(
    model: nn.Module,
    train_loader: DataLoader,  # type: ignore[type-arg]
    val_loader: DataLoader,  # type: ignore[type-arg]
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    epochs: int = 20,
    patience: int = 5,
    input_type: int = 1,
    device: str = "cuda",
    use_fp16: bool = True,
    run_name: str = "model",
) -> Dict[str, float]:
    """
    Train with early stopping on validation R@5.

    Returns the best validation metrics dict.
    """
    scaler = GradScaler() if use_fp16 else None
    best_r5 = -1.0
    bad_epochs = 0
    best_metrics: Dict[str, float] = {}
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{run_name}_best.pt")
    history_path = os.path.join(HISTORY_DIR, f"{run_name}_history.json")
    history: List[Dict[str, Any]] = []

    print(f"\n{'─'*60}")
    print(f"  Training: {run_name} | Type {input_type} | {epochs} epochs")
    print(f"  History file: {history_path}")
    print(f"{'─'*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, scaler,
            input_type, device, use_fp16,
        )
        dt = time.time() - t0

        # Validation
        retrieval_model: RetrievalModel = model  # type: ignore[assignment]
        val_metrics = evaluate_model(
            retrieval_model, val_loader, input_type, device,
            header=f"Epoch {epoch}/{epochs}  (loss={avg_loss:.4f}, {dt:.1f}s)",
        )

        if scheduler is not None:
            scheduler.step(val_metrics["R@5"])

        # Early stopping on R@5
        improved = val_metrics["R@5"] > best_r5
        if improved:
            best_r5 = val_metrics["R@5"]
            bad_epochs = 0
            best_metrics = val_metrics
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ New best R@5 = {best_r5:.4f}  →  saved {ckpt_path}")
        else:
            bad_epochs += 1
            print(f"  No improvement ({bad_epochs}/{patience})")

        history.append({
            "epoch": int(epoch),
            "train_loss": float(avg_loss),
            "epoch_seconds": float(dt),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "val_R@1": float(val_metrics.get("R@1", 0.0)),
            "val_R@5": float(val_metrics.get("R@5", 0.0)),
            "val_R@10": float(val_metrics.get("R@10", 0.0)),
            "val_MedR": float(val_metrics.get("MedR", 0.0)),
            "val_MRR": float(val_metrics.get("MRR", 0.0)),
            "is_best": improved,
            "best_r5_so_far": float(best_r5),
        })
        save_training_history(
            history_path, run_name, input_type, ckpt_path, history, best_metrics
        )

        if not improved and bad_epochs >= patience:
            print("  Early stopping triggered.")
            break

    # Restore best checkpoint
    if os.path.exists(ckpt_path):
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
        print(f"  Restored best checkpoint from {ckpt_path}")
        print(f"  Training history saved to {history_path}")

    return best_metrics
