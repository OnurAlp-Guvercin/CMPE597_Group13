from __future__ import annotations

import json
import os
import random
import math
from collections import Counter

import torch
from transformers import pipeline
from tqdm import tqdm

import config
from sanitize_captions import sanitize_caption


# ------------------------- helpers -------------------------

def load_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(data)} items → {path}")


def get_meme_captions(item: dict) -> list:
    """Return all non-empty caption strings for an item."""
    caps = item.get("meme_captions", [])
    if isinstance(caps, str):
        caps = [caps]
    return [c.strip() for c in caps if isinstance(c, str) and c.strip()]


def _majority_vote(labels: list[str], exclude: str = "neutral") -> str:
    """
    Filter out `exclude` labels, then return the most common remainder.
    Falls back to `exclude` if nothing survives (all were excluded or list empty).
    On a tie, returns the label that appears first in most_common() ordering
    (i.e. whichever tied label the Counter encountered first).
    """
    filtered = [l for l in labels if l != exclude]
    if not filtered:
        return exclude
    return Counter(filtered).most_common(1)[0][0]


def annotate(items: list, classifier, batch_size: int) -> list:
    """
    Classify every caption for every item, strip neutrals, majority-vote.

    Fields added to each item:
      sentiment_label          - final predicted emotion string
      sentiment_score          - mean confidence across captions that voted
                                 for the winning label (or mean of all if fallback)
      sentiment_captions_used  - list of captions that were classified
      sentiment_votes          - {label: count} tally before neutral removal
    """
    pairs: list[tuple[int, str]] = []
    for idx, item in enumerate(items):
        captions = get_meme_captions(item)
        if captions:
            for cap in captions:
                pairs.append((idx, cap))
        else:
            pairs.append((idx, "[no caption]"))

    inputs = [sanitize_caption(cap) for _, cap in pairs]

    raw_results: list[dict] = []
    for i in tqdm(range(0, len(inputs), batch_size), desc="  Classifying"):
        batch = inputs[i : i + batch_size]
        raw_results.extend(classifier(batch, truncation=True, max_length=512))

    per_item: dict[int, list[tuple[str, float]]] = {i: [] for i in range(len(items))}
    for (idx, _), result in zip(pairs, raw_results):
        per_item[idx].append((result["label"], result["score"]))

    annotated = []
    for idx, item in enumerate(items):
        predictions = per_item[idx]            # [(label, score), ...]
        all_labels  = [p[0] for p in predictions]
        votes = Counter(l for l in all_labels if l != "neutral")

        winner = _majority_vote(all_labels)

        winner_scores = [s for l, s in predictions if l == winner]
        mean_score = round(sum(winner_scores) / len(winner_scores), 4)

        entry = dict(item)
        entry["sentiment_label"]         = winner
        entry["sentiment_score"]         = mean_score
        entry["sentiment_captions_used"] = (
            [sanitize_caption(c) for c in get_meme_captions(item)]
            or ["[no caption]"]
        )
        entry["sentiment_votes"]         = dict(votes)
        annotated.append(entry)

    return annotated


# ------------------------- reporting -------------------------

def report_distribution(items: list, split_name: str) -> Counter:
    """
    Print class counts, percentages, ASCII bar chart, and imbalance stats.
    Also computes Shannon entropy as an additional measure of class imbalance.

    Shannon Entropy (H) is calculated as:
    H = -Σ (p_i * log2(p_i)) for each class i, where p_i is the proportion of samples in class i.

    Shannon entropy measures the uncertainty in the class distribution. 
    A higher entropy (up to log2(num_classes)) indicates a more balanced distribution, while a lower entropy indicates more imbalance. 
    This complements the imbalance ratio by accounting for the overall distribution shape, not just the extremes.
    """

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
    rng = random.Random(seed)
    sample = rng.sample(items, min(n, len(items)))

    print(f"\n{'=' * 60}")
    print(f"  Manual Label-Noise Review — {len(sample)} random samples")
    print(f"  Inspect whether the emotion label fits the meme caption.")
    print(f"{'=' * 60}")

    for i, item in enumerate(sample, 1):
        label = item["sentiment_label"]
        score = item["sentiment_score"]
        votes = item.get("sentiment_votes", {})
        captions = item.get("sentiment_captions_used", [])      

        print(f"\n  [{i:02d}]  {label:<10}  (conf {score:.3f})  votes: {votes}")
        for cap in captions:
            display = (cap[:110] + "…") if len(cap) > 110 else cap
            print(f"        \"{display}\"")

    print(f"\n{'=' * 60}")
    print("  End of manual review sample.")
    print(f"{'=' * 60}\n")

# ------------------------- main -------------------------

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

        print(f"\n--- {split_name} -------------------------------------------")
        items = load_json(json_path)
        print(f"  Loaded {len(items)} items from {os.path.basename(json_path)}")

        annotated = annotate(items, classifier, config.SENTIMENT_BATCH_SIZE)
        save_json(annotated, out_path)
        annotated_splits[split_name] = annotated

    # Class distribution report
    for split_name, annotated in annotated_splits.items():
        report_distribution(annotated, split_name)

    # Manual label-noise inspection (train split only)
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
