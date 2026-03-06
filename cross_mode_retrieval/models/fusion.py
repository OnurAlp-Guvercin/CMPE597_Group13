import torch
import torch.nn as nn


class ConcatProjectFusion(nn.Module):
    """
    Concatenate image & title embeddings, then project back to `out_dim`.

    [img ; title]  →  Linear(2·in_dim, hidden) → GELU → Linear(hidden, out_dim)
    """
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
    """
    Use title as query, image as key/value in a single cross-attention layer,
    then pool to a single vector.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, img_emb: torch.Tensor, title_emb: torch.Tensor) -> torch.Tensor:
        # Treat each embedding as a single-token sequence
        img_seq = img_emb.unsqueeze(1)      # [B, 1, D]
        title_seq = title_emb.unsqueeze(1)   # [B, 1, D]
        out, _ = self.attn(title_seq, img_seq, img_seq)  # [B, 1, D]
        out = self.norm(out.squeeze(1) + title_emb)       # residual + norm
        return out


class AdditiveFusion(nn.Module):
    """Simple element-wise addition followed by LayerNorm."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, img_emb: torch.Tensor, title_emb: torch.Tensor) -> torch.Tensor:
        return self.norm(img_emb + title_emb)


class GatedFusion(nn.Module):
    """
    Learnable gate that decides how much of each modality to keep.
    gate = σ(W·[img ; title] + b)
    fused = gate * img + (1 - gate) * title
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, img_emb: torch.Tensor, title_emb: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([img_emb, title_emb], dim=-1))
        return self.norm(g * img_emb + (1 - g) * title_emb)


def build_fusion(strategy: str, in_dim: int, out_dim: int, hidden_dim: int = 512) -> nn.Module:
    """Factory that returns the requested fusion module."""
    if strategy == "concat_project":
        return ConcatProjectFusion(in_dim, out_dim, hidden_dim)
    elif strategy == "cross_attention":
        return CrossAttentionFusion(in_dim)
    elif strategy == "add":
        return AdditiveFusion(in_dim)
    elif strategy == "gated":
        return GatedFusion(in_dim)
    else:
        raise ValueError(f"Unknown fusion strategy: {strategy}")
