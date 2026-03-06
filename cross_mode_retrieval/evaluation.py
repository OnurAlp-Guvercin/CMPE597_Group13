import torch
from typing import Dict


@torch.no_grad()
def compute_similarity_matrix(
    query_embeds: torch.Tensor,
    candidate_embeds: torch.Tensor,
) -> torch.Tensor:
    """
    Cosine-similarity matrix [N_query × N_candidate].
    Both inputs are assumed L2-normalised (or we normalise here).
    """
    q = torch.nn.functional.normalize(query_embeds, dim=-1)
    c = torch.nn.functional.normalize(candidate_embeds, dim=-1)
    return q @ c.T  # [N_q, N_c]


def recall_at_k(sim_matrix: torch.Tensor, k: int) -> float:
    """
    Recall@K: fraction of queries whose ground-truth is in the top-K results.
    Ground truth for query i is candidate i (diagonal).
    """
    N = sim_matrix.size(0)
    # Indices of top-k candidates for every query
    _, topk_indices = sim_matrix.topk(k, dim=1)          # [N, k]
    gt = torch.arange(N, device=sim_matrix.device).unsqueeze(1)  # [N, 1] since we know diagonal is correct match
    correct = (topk_indices == gt).any(dim=1).float()     # [N]
    return correct.mean().item()


def median_rank(sim_matrix: torch.Tensor) -> float:
    """Median rank of the correct match (1-indexed).
    Measures in which position the correct candidate appears when sorting by similarity."""
    N = sim_matrix.size(0)
    # Rank: number of items with higher similarity + 1
    gt_sims = sim_matrix.diag().unsqueeze(1)              # [N, 1]
    ranks = (sim_matrix >= gt_sims).sum(dim=1).float()    # [N]
    return ranks.median().item()


def mean_reciprocal_rank(sim_matrix: torch.Tensor) -> float:
    """Mean Reciprocal Rank (MRR).
    Measures the average of 1/rank for the correct match across all queries."""
    N = sim_matrix.size(0)
    gt_sims = sim_matrix.diag().unsqueeze(1)
    ranks = (sim_matrix >= gt_sims).sum(dim=1).float()
    return (1.0 / ranks).mean().item()


def evaluate_retrieval(
    query_embeds: torch.Tensor,
    candidate_embeds: torch.Tensor,
    ks: tuple = (1, 5, 10),
) -> Dict[str, float]:
    """
    Full evaluation suite.

    Args:
        query_embeds:     [N, D]  (e.g., image or fused image+title embeddings)
        candidate_embeds: [N, D]  (e.g., meme caption embeddings)
        ks:               which K values to report Recall@K for

    Returns:
        dict with keys like "R@1", "R@5", "R@10", "MedR", "MRR"
    """
    sim = compute_similarity_matrix(query_embeds, candidate_embeds)
    metrics: Dict[str, float] = {}
    for k in ks:
        metrics[f"R@{k}"] = recall_at_k(sim, k)
    metrics["MedR"] = median_rank(sim)
    metrics["MRR"] = mean_reciprocal_rank(sim)
    return metrics


def print_metrics(metrics: Dict[str, float], header: str = "") -> None:
    """Pretty-print a metrics dictionary."""
    if header:
        print(f"\n{'='*60}")
        print(f"  {header}")
        print(f"{'='*60}")
    for k, v in metrics.items():
        if k == "MedR":
            print(f"  {k:>6s}: {v:.1f}")
        else:
            print(f"  {k:>6s}: {v:.4f}  ({v*100:.2f}%)" if "R@" in k or k == "MRR"
                  else f"  {k:>6s}: {v:.4f}")
    print()
