import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)
import json


def evaluate(model, loader, device, save_path=None):
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            logits = model(inputs)
            probs = torch.sigmoid(logits)

            preds = (probs > 0.5).long()

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ---- metrics ----
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    roc = roc_auc_score(all_labels, all_probs)

    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("ROC-AUC:", roc)
    print("Confusion Matrix:\n", cm)

    if save_path:
        with open(save_path, "w") as f:
            json.dump(
                {
                    "labels": [int(x) for x in all_labels],
                    "predictions": [int(x) for x in all_preds],
                    "probabilities": [float(x) for x in all_probs]
                },
                f,
                indent=2
            )

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc),
        "confusion_matrix": cm.tolist()
    }