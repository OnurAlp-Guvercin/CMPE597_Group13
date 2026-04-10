"""
Task 2.3(a) — Sentiment Annotation
====================================
Uses a pretrained emotion classifier to assign one of 7 emotion labels
(anger, disgust, fear, joy, neutral, sadness, surprise) to each meme's
meme_caption.  Outputs:

  outputs/annotated-trainval.json  — train split with added fields:
                                      sentiment_label, sentiment_score,
                                      sentiment_caption_used
  outputs/annotated-test.json      — test split (same schema)

Also prints:
  • Class distribution + imbalance statistics for both splits
  • A random subset of MANUAL_REVIEW_N samples for label-noise inspection

Usage:
  python annotate_sentiment.py
"""

import json
import os
import random
import math
from collections import Counter

import torch
from transformers import pipeline
from tqdm import tqdm

import config


# ── helpers ──────────────────────────────────────────────────────────────────

def load_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(data)} items → {path}")


def get_meme_caption(item: dict) -> str:
    """Return the first meme_caption string, or '' if absent."""
    caps = item.get("meme_captions", [])
    if isinstance(caps, list):
        return caps[0].strip() if caps else ""
    if isinstance(caps, str):
        return caps.strip()
    return ""


# ── annotation ───────────────────────────────────────────────────────────────

def annotate(items: list, classifier, batch_size: int) -> list:
    """
    Run the emotion classifier over all items in batches.
    Adds three fields to each item:
      sentiment_label        — predicted emotion string
      sentiment_score        — model confidence (0–1)
      sentiment_caption_used — the meme_caption text that was classified
    """
    captions = [get_meme_caption(item) for item in items]

    # Replace empty captions with a neutral placeholder so the model
    # still produces a (low-confidence) prediction rather than crashing.
    inputs = [c if c else "[no caption]" for c in captions]

    results = []
    for i in tqdm(range(0, len(inputs), batch_size), desc="  Classifying"):
        batch = inputs[i : i + batch_size]
        batch_results = classifier(batch, truncation=True, max_length=512)
        results.extend(batch_results)

    annotated = []
    for item, caption, result in zip(items, captions, results):
        entry = dict(item)
        entry["sentiment_label"] = result["label"]
        entry["sentiment_score"] = round(result["score"], 4)
        entry["sentiment_caption_used"] = caption
        annotated.append(entry)

    return annotated


# ── reporting ─────────────────────────────────────────────────────────────────

def report_distribution(items: list, split_name: str) -> Counter:
    """Print class counts, percentages, ASCII bar chart, and imbalance stats."""
    labels = [item["sentiment_label"] for item in items]
    counts = Counter(labels)
    total = len(labels)

    # Compute Shannon entropy as an additional imbalance measure
    entropy = -sum(
        (c / total) * math.log2(c / total)
        for c in counts.values() if c > 0
    )
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0

    print(f"\n{'=' * 60}")
    print(f"  Class Distribution — {split_name}  ({total} samples)")
    print(f"{'=' * 60}")
    print(f"  {'Emotion':<12}  {'Count':>6}  {'Pct':>6}  Bar")
    print(f"  {'-' * 55}")

    for label, count in counts.most_common():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:<12}  {count:>6}  {pct:>5.1f}%  {bar}")

    most_common_label, most_common_count = counts.most_common(1)[0]
    least_common_label, least_common_count = counts.most_common()[-1]
    imbalance_ratio = most_common_count / least_common_count

    print(f"\n  Most common : {most_common_label} ({most_common_count})")
    print(f"  Least common: {least_common_label} ({least_common_count})")
    print(f"  Imbalance ratio (max / min) : {imbalance_ratio:.2f}x")
    print(f"  Shannon entropy             : {entropy:.3f} / {max_entropy:.3f} "
          f"(uniform = {max_entropy:.3f})")

    if imbalance_ratio >= 5:
        print(f"\n  [WARNING] Severe class imbalance detected "
              f"(ratio {imbalance_ratio:.1f}x >= 5x).")
        print("  Consider over-sampling minority classes or using weighted loss.")
    elif imbalance_ratio >= 2:
        print(f"\n  [NOTE] Moderate class imbalance (ratio {imbalance_ratio:.1f}x).")

    print(f"{'=' * 60}")
    return counts


def print_manual_review(items: list, n: int, seed: int) -> None:
    """
    Print a random subset for manual label-noise inspection.
    Reviewers should check whether the predicted emotion label makes sense
    given the meme caption text.
    """
    rng = random.Random(seed)
    sample = rng.sample(items, min(n, len(items)))

    print(f"\n{'=' * 60}")
    print(f"  Manual Label-Noise Review — {len(sample)} random samples")
    print(f"  Inspect whether the emotion label fits the meme caption.")
    print(f"{'=' * 60}")

    for i, item in enumerate(sample, 1):
        label = item["sentiment_label"]
        score = item["sentiment_score"]
        caption = item["sentiment_caption_used"]
        # Truncate long captions for readability
        display_caption = (caption[:110] + "…") if len(caption) > 110 else caption
        print(f"\n  [{i:02d}]  {label:<10}  (conf {score:.3f})")
        print(f"        \"{display_caption}\"")

    print(f"\n{'=' * 60}")
    print("  End of manual review sample.")
    print(f"{'=' * 60}\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    device = 0 if torch.cuda.is_available() else -1
    device_name = f"cuda:{device}" if device >= 0 else "cpu"
    print(f"Sentiment model : {config.SENTIMENT_MODEL}")
    print(f"Device          : {device_name}")
    print(f"Batch size      : {config.SENTIMENT_BATCH_SIZE}")

    classifier = pipeline(
        "text-classification",
        model=config.SENTIMENT_MODEL,
        device=device,
    )

    splits = [
        (config.TRAIN_JSON, config.ANNOTATED_TRAIN_JSON, "Train (trainval)"),
        (config.TEST_JSON,  config.ANNOTATED_TEST_JSON,  "Test"),
    ]

    annotated_splits = {}
    for json_path, out_path, split_name in splits:
        if not os.path.exists(json_path):
            print(f"\n[WARN] {json_path} not found — run download_data.py first.")
            continue

        print(f"\n─── {split_name} ─────────────────────────────────────")
        items = load_json(json_path)
        print(f"  Loaded {len(items)} items from {os.path.basename(json_path)}")

        annotated = annotate(items, classifier, config.SENTIMENT_BATCH_SIZE)
        save_json(annotated, out_path)
        annotated_splits[split_name] = annotated

    # ── Class distribution report ─────────────────────────────────────────────
    for split_name, annotated in annotated_splits.items():
        report_distribution(annotated, split_name)

    # ── Manual label-noise inspection (train split only) ─────────────────────
    train_key = "Train (trainval)"
    if train_key in annotated_splits:
        print_manual_review(
            annotated_splits[train_key],
            n=config.MANUAL_REVIEW_N,
            seed=config.SEED,
        )
    elif annotated_splits:
        # Fallback: use whichever split was annotated
        first_key = next(iter(annotated_splits))
        print_manual_review(
            annotated_splits[first_key],
            n=config.MANUAL_REVIEW_N,
            seed=config.SEED,
        )


if __name__ == "__main__":
    main()
