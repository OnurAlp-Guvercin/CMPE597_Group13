from typing import Dict, Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_similarity_matrix(
    query_embeds: torch.Tensor,
    candidate_embeds: torch.Tensor,
) -> torch.Tensor:
    q = F.normalize(query_embeds, dim=-1)
    c = F.normalize(candidate_embeds, dim=-1)
    return q @ c.T


def recall_at_k(sim_matrix: torch.Tensor, k: int) -> float:
    n = sim_matrix.size(0)
    _, topk_indices = sim_matrix.topk(k, dim=1)
    gt = torch.arange(n, device=sim_matrix.device).unsqueeze(1)
    return (topk_indices == gt).any(dim=1).float().mean().item()


def median_rank(sim_matrix: torch.Tensor) -> float:
    gt_sims = sim_matrix.diag().unsqueeze(1)
    ranks = (sim_matrix >= gt_sims).sum(dim=1).float()
    return ranks.median().item()


def mean_reciprocal_rank(sim_matrix: torch.Tensor) -> float:
    gt_sims = sim_matrix.diag().unsqueeze(1)
    ranks = (sim_matrix >= gt_sims).sum(dim=1).float()
    return (1.0 / ranks).mean().item()


def evaluate_retrieval(
    query_embeds: torch.Tensor,
    candidate_embeds: torch.Tensor,
    ks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    sim = compute_similarity_matrix(query_embeds, candidate_embeds)
    metrics = {f"R@{k}": recall_at_k(sim, k) for k in ks}
    metrics["MedR"] = median_rank(sim)
    metrics["MRR"] = mean_reciprocal_rank(sim)
    return metrics


def print_metrics(metrics: Dict[str, float], header: str = "") -> None:
    if header:
        print(f"\n{'='*60}")
        print(f"  {header}")
        print(f"{'='*60}")

    for key, value in metrics.items():
        if key == "MedR":
            print(f"  {key:>6s}: {value:.1f}")
        elif "R@" in key or key == "MRR":
            print(f"  {key:>6s}: {value:.4f}  ({value*100:.2f}%)")
        else:
            print(f"  {key:>6s}: {value:.4f}")
    print()
