import argparse
import json
import os
import random
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import (
    DataConfig, TrainConfig, CLIPZeroShotConfig,
    CustomModelConfig, LoRAFinetuneConfig, FusionConfig,
    OUTPUT_DIR,
)
from data.dataset import (
    MemecapDataset, train_val_split, collate_fn, default_transform,
)
from torch.utils.data import DataLoader
from trainer import evaluate_model, train_model


# Reproducibility
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# Build data loaders
def build_loaders(data_cfg: DataConfig, transform=None,
                  use_img_caption: bool = False, batch_size: int = 64,
                  num_workers: int = 4):
    """Create train / val / test DataLoaders."""
    full_train = MemecapDataset(
        data_cfg.train_json, data_cfg.image_dir,
        transform=transform, image_size=data_cfg.image_size,
        use_img_caption=use_img_caption,
    )
    test_ds = MemecapDataset(
        data_cfg.test_json, data_cfg.image_dir,
        transform=transform, image_size=data_cfg.image_size,
        use_img_caption=use_img_caption,
    )
    train_sub, val_sub = train_val_split(
        full_train, data_cfg.val_ratio, data_cfg.seed,
    )
    loader_kwargs = {
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    return (
        DataLoader(train_sub, batch_size=batch_size, shuffle=True,
                   **loader_kwargs),
        DataLoader(val_sub, batch_size=batch_size, shuffle=False,
                   **loader_kwargs),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                   **loader_kwargs),
    )


# ------------------------------------------------------------------------------
#  TASK (b): Zero-shot CLIP evaluation
# ------------------------------------------------------------------------------
def run_zeroshot(data_cfg, clip_cfg, fusion_cfg, device, input_types,
                 use_img_caption=False, num_workers: int = 4):
    from models.clip_zeroshot import CLIPZeroShot

    print("\n" + "═" * 60)
    print("  TASK (b): Zero-shot CLIP Evaluation")
    print("═" * 60)

    model = CLIPZeroShot(
        model_name=clip_cfg.clip_model_name,
        pretrained=clip_cfg.clip_pretrained,
        fusion_strategy=fusion_cfg.strategy,
        fusion_hidden=fusion_cfg.hidden_dim,
        device=device,
    )

    # Use CLIP's own preprocessing
    _, _, test_loader = build_loaders(
        data_cfg, transform=model.get_preprocess(),
        use_img_caption=use_img_caption,
        batch_size=64,
        num_workers=num_workers,
    )

    results = {}
    for it in input_types:
        header = f"Zero-shot CLIP | Type {it}"
        metrics = evaluate_model(
            model, test_loader, input_type=it, device=device,
            header=header,
        )
        results[f"zeroshot_type{it}"] = metrics

    return results


# ------------------------------------------------------------------------------
#  TASK (c): Custom dual-encoder trained from scratch
# ------------------------------------------------------------------------------
def run_custom(data_cfg, train_cfg, model_cfg, fusion_cfg, device,
               input_types, use_img_caption=False):
    from models.custom_model import CustomDualEncoder

    print("\n" + "═" * 60)
    print("  TASK (c): Custom Dual-Encoder (trained from scratch)")
    print("═" * 60)

    results = {}
    for it in input_types:
        seed_everything(train_cfg.seed)

        model = CustomDualEncoder(
            embed_dim=model_cfg.embed_dim,
            image_backbone=model_cfg.image_backbone,
            text_model=model_cfg.text_model,
            dropout=model_cfg.dropout,
            temperature=train_cfg.temperature,
            fusion_strategy=fusion_cfg.strategy,
            fusion_hidden=fusion_cfg.hidden_dim,
            freeze_image=model_cfg.freeze_image_backbone,
            freeze_text=model_cfg.freeze_text_backbone,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable params: {n_params:,}")

        train_loader, val_loader, test_loader = build_loaders(
            data_cfg, transform=default_transform(data_cfg.image_size),
            use_img_caption=use_img_caption,
            batch_size=train_cfg.batch_size,
            num_workers=train_cfg.num_workers,
        )

        optimizer = AdamW(model.parameters(),
                          lr=train_cfg.learning_rate,
                          weight_decay=train_cfg.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                       patience=2)

        train_model(
            model, train_loader, val_loader, optimizer, scheduler,
            epochs=train_cfg.epochs, patience=train_cfg.patience,
            input_type=it, device=device, use_fp16=train_cfg.fp16,
            run_name=f"custom_type{it}",
        )

        # Final test evaluation
        test_metrics = evaluate_model(
            model, test_loader, input_type=it, device=device,
            header=f"Custom Model TEST | Type {it}",
        )
        results[f"custom_type{it}"] = test_metrics

    return results


# ------------------------------------------------------------------------------
#  TASK (d): CLIP + LoRA fine-tuning
# ------------------------------------------------------------------------------
def run_lora(data_cfg, train_cfg, lora_cfg, fusion_cfg, device,
             input_types, use_img_caption=False):
    from models.clip_lora import CLIPLoRA

    print("\n" + "═" * 60)
    print("  TASK (d): CLIP + LoRA Fine-Tuning")
    print("═" * 60)

    results = {}
    for it in input_types:
        seed_everything(train_cfg.seed)

        model = CLIPLoRA(
            model_name=lora_cfg.clip_model_name,
            pretrained=lora_cfg.clip_pretrained,
            lora_r=lora_cfg.lora_r,
            lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout,
            lora_targets=lora_cfg.lora_target_modules,
            fusion_strategy=fusion_cfg.strategy,
            fusion_hidden=fusion_cfg.hidden_dim,
            device=device,
        )

        ntrain = model.trainable_param_count()
        ntotal = model.total_param_count()
        print(f"  Trainable: {ntrain:,} / {ntotal:,} "
              f"({100*ntrain/ntotal:.2f}%)")

        # Use CLIP's own preprocessing
        train_loader, val_loader, test_loader = build_loaders(
            data_cfg, transform=model.get_preprocess(),
            use_img_caption=use_img_caption,
            batch_size=train_cfg.batch_size,
            num_workers=train_cfg.num_workers,
        )

        optimizer = AdamW(model.trainable_params(),
                          lr=lora_cfg.learning_rate,
                          weight_decay=train_cfg.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                       patience=2)

        train_model(
            model, train_loader, val_loader, optimizer, scheduler,
            epochs=lora_cfg.epochs, patience=train_cfg.patience,
            input_type=it, device=device, use_fp16=train_cfg.fp16,
            run_name=f"lora_type{it}",
        )

        # Final test evaluation
        test_metrics = evaluate_model(
            model, test_loader, input_type=it, device=device,
            header=f"CLIP+LoRA TEST | Type {it}",
        )
        results[f"lora_type{it}"] = test_metrics

    return results


# ------------------------------------------------------------------------------
#  Summary table
# ------------------------------------------------------------------------------
def print_summary(all_results: dict):
    """Print a neat comparison table."""
    print("\n" + "═" * 72)
    print("  SUMMARY – Cross-Modal Retrieval Results")
    print("═" * 72)
    header = f"  {'Experiment':<30s} {'R@1':>8s} {'R@5':>8s} {'R@10':>8s} {'MedR':>8s} {'MRR':>8s}"
    print(header)
    print("  " + "─" * 68)
    for name, metrics in all_results.items():
        r1 = metrics.get("R@1", 0) * 100
        r5 = metrics.get("R@5", 0) * 100
        r10 = metrics.get("R@10", 0) * 100
        medr = metrics.get("MedR", 0)
        mrr = metrics.get("MRR", 0) * 100
        print(f"  {name:<30s} {r1:>7.2f}% {r5:>7.2f}% {r10:>7.2f}% {medr:>7.1f} {mrr:>7.2f}%")
    print()


# ------------------------------------------------------------------------------
#  CLI argument parsing
# ------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="MemeCap Cross-Modal Retrieval")
    p.add_argument("--task", type=str, default="all",
                   choices=["all", "zeroshot", "custom", "lora"],
                   help="Which experiment to run.")
    p.add_argument("--input_type", type=int, default=0,
                   choices=[0, 1, 2],
                   help="0=both, 1=image-only, 2=image+title")
    p.add_argument("--use_img_caption", action="store_true",
                   help="Incorporate literal image caption into title text.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


# ------------------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------------------
def main():
    args = parse_args()

    # Configs
    data_cfg = DataConfig()
    train_cfg = TrainConfig()
    clip_cfg = CLIPZeroShotConfig()
    model_cfg = CustomModelConfig()
    lora_cfg = LoRAFinetuneConfig()
    fusion_cfg = FusionConfig()

    # CLI overrides
    if args.epochs is not None:
        train_cfg.epochs = args.epochs
        lora_cfg.epochs = args.epochs
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.lr is not None:
        train_cfg.learning_rate = args.lr
        lora_cfg.learning_rate = args.lr
    if args.num_workers is not None:
        train_cfg.num_workers = args.num_workers

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_cfg.device = device
    print(f"Device: {device}")

    seed_everything(train_cfg.seed)

    # Input types
    if args.input_type == 0:
        input_types = [1, 2]
    else:
        input_types = [args.input_type]

    # Run experiments
    all_results = {}

    if args.task in ("all", "zeroshot"):
        res = run_zeroshot(data_cfg, clip_cfg, fusion_cfg, device,
                           input_types, args.use_img_caption,
                           num_workers=train_cfg.num_workers)
        all_results.update(res)

    if args.task in ("all", "custom"):
        res = run_custom(data_cfg, train_cfg, model_cfg, fusion_cfg,
                         device, input_types, args.use_img_caption)
        all_results.update(res)

    if args.task in ("all", "lora"):
        res = run_lora(data_cfg, train_cfg, lora_cfg, fusion_cfg,
                       device, input_types, args.use_img_caption)
        all_results.update(res)

    # Summary
    if all_results:
        print_summary(all_results)

        # Save results JSON
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "results.json")
        serialisable = {k: {mk: float(mv) for mk, mv in v.items()}
                        for k, v in all_results.items()}
        with open(out_path, "w") as f:
            json.dump(serialisable, f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
