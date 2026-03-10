import torch
import torch.nn as nn


class ConcatProjectFusion(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2 * in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, img_emb: torch.Tensor, title_emb: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([img_emb, title_emb], dim=-1))


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, img_emb: torch.Tensor, title_emb: torch.Tensor) -> torch.Tensor:
        img_seq = img_emb.unsqueeze(1)
        title_seq = title_emb.unsqueeze(1)
        out, _ = self.attn(title_seq, img_seq, img_seq)
        return self.norm(out.squeeze(1) + title_emb)


class AdditiveFusion(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, img_emb: torch.Tensor, title_emb: torch.Tensor) -> torch.Tensor:
        return self.norm(img_emb + title_emb)


class WeightedSumFusion(nn.Module):
    def __init__(self, embed_dim: int, init_alpha: float = 0.7):
        super().__init__()
        if not 0.0 < init_alpha < 1.0:
            raise ValueError("init_alpha must be in (0, 1)")
        logit = torch.logit(torch.tensor(init_alpha, dtype=torch.float32))
        self.logit_alpha = nn.Parameter(logit)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, img_emb: torch.Tensor, title_emb: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.logit_alpha)
        return self.norm(alpha * img_emb + (1.0 - alpha) * title_emb)


class GatedFusion(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, img_emb: torch.Tensor, title_emb: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([img_emb, title_emb], dim=-1))
        return self.norm(gate * img_emb + (1 - gate) * title_emb)


def build_fusion(
    strategy: str,
    in_dim: int,
    out_dim: int,
    hidden_dim: int = 512,
) -> nn.Module:
    if strategy == "concat_project":
        return ConcatProjectFusion(in_dim, out_dim, hidden_dim)
    if strategy == "cross_attention":
        return CrossAttentionFusion(in_dim)
    if strategy == "add":
        return AdditiveFusion(in_dim)
    if strategy == "weighted_sum":
        return WeightedSumFusion(in_dim)
    if strategy == "gated":
        return GatedFusion(in_dim)
    raise ValueError(f"Unknown fusion strategy: {strategy}")
