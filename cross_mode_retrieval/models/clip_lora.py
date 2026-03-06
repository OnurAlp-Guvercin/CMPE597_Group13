import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from typing import Any, List, Optional

from models.fusion import build_fusion


# ------------------------------------------------------------------------------
#  Lightweight inline LoRA layer
# ------------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that adds a low-rank residual.

        y = W_frozen · x + (B · A) · x · (α / r)

    Only A and B are trainable (rank r << min(in, out)).
    """

    def __init__(self, original_linear: nn.Linear, r: int = 8,
                 alpha: int = 16, dropout: float = 0.1):
        super().__init__()
        self.original = original_linear
        in_f, out_f = original_linear.in_features, original_linear.out_features
        self.in_features = in_f
        self.out_features = out_f
        self.r = r
        self.scale = alpha / r

        # Low-rank factors
        self.lora_A = nn.Parameter(torch.randn(in_f, r) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(r, out_f))
        self.lora_dropout = nn.Dropout(dropout)

        # Freeze the original weight
        for p in self.original.parameters():
            p.requires_grad = False

    @property
    def weight(self) -> torch.Tensor:
        # Some modules (e.g. MultiheadAttention) read out_proj.weight directly.
        delta_w = (self.lora_A @ self.lora_B).T * self.scale  # [out, in]
        return self.original.weight + delta_w

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.original.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use F.linear so modules that consume weight/bias stay consistent.
        return F.linear(x, self.weight, self.bias)


# ------------------------------------------------------------------------------
#  Inject LoRA into an existing model
# ------------------------------------------------------------------------------
def inject_lora(
    model: nn.Module,
    target_name_keywords: List[str],
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
) -> int:
    """
    Walk `model` and replace matching nn.Linear layers with LoRALinear.

    Args:
        target_name_keywords: list of substrings to match against parameter
            names (e.g. ["q_proj", "v_proj", "in_proj"]).
        r, alpha, dropout: LoRA hyper-parameters.

    Returns:
        Number of layers replaced.
    """
    replaced = 0
    for name, module in list(model.named_modules()):
        # Check if any keyword matches
        if not any(kw in name for kw in target_name_keywords):
            continue
        if not isinstance(module, nn.Linear):
            continue

        # Navigate to the parent module to perform replacement
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], LoRALinear(module, r, alpha, dropout))
        replaced += 1

    return replaced


# ------------------------------------------------------------------------------
#  Full CLIP+LoRA model
# ------------------------------------------------------------------------------
class CLIPLoRA(nn.Module):
    """
    CLIP with LoRA adapters for cross-modal retrieval fine-tuning.

    Trainable components:
        1. LoRA adapters in visual & text transformers
        2. Fusion module  (Type 2)
        3. Learnable logit scale
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_targets: Optional[List[str]] = None,
        fusion_strategy: str = "concat_project",
        fusion_hidden: int = 512,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        default_targets = ["attn.out_proj", "mlp.c_fc", "mlp.c_proj"]

        # Load the pre-trained CLIP
        clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device="cpu",
        )
        self.clip_model: Any = clip_model
        self.tokenizer: Any = open_clip.get_tokenizer(model_name)
        self.embed_dim = int(self.clip_model.visual.output_dim)

        # Freeze all CLIP weights first
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # Inject LoRA adapters
        targets = lora_targets if lora_targets is not None else default_targets
        n_replaced = inject_lora(
            self.clip_model, targets, lora_r, lora_alpha, lora_dropout,
        )
        if n_replaced == 0 and lora_targets is not None:
            # open_clip attention blocks do not expose q_proj/v_proj module names.
            # Fallback to module names that exist in ViT-B/32 style CLIP blocks.
            print("[CLIPLoRA] Warning: no LoRA target matched. "
                  f"Retrying with defaults: {default_targets}")
            targets = default_targets
            n_replaced = inject_lora(
                self.clip_model, targets, lora_r, lora_alpha, lora_dropout,
            )
        print(f"[CLIPLoRA] Injected LoRA into {n_replaced} layers "
              f"(r={lora_r}, α={lora_alpha})")

        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.tensor(2.6593))  # ln(1/0.07)

        # Fusion head
        self.fusion = build_fusion(
            fusion_strategy, self.embed_dim, self.embed_dim, fusion_hidden,
        )

        self.to(device)

    # encoding
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(
            self.clip_model.encode_image(images.to(self.device)), dim=-1,
        )

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts).to(self.device)
        return F.normalize(self.clip_model.encode_text(tokens), dim=-1)

    # query builders
    def get_query_type1(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_images(images)

    def get_query_type2(
        self, images: torch.Tensor, titles: List[str],
    ) -> torch.Tensor:
        img_emb = self.encode_images(images)
        title_emb = self.encode_texts(titles)
        return F.normalize(self.fusion(img_emb, title_emb), dim=-1)

    def get_candidates(self, captions: List[str], device: str = "") -> torch.Tensor:
        return self.encode_texts(captions)

    # loss
    def compute_loss(
        self,
        query_embeds: torch.Tensor,
        caption_embeds: torch.Tensor,
    ) -> torch.Tensor:
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits = logit_scale * (query_embeds @ caption_embeds.T)
        labels = torch.arange(logits.size(0), device=logits.device)
        return (F.cross_entropy(logits, labels) +
                F.cross_entropy(logits.T, labels)) / 2.0

    # forward (training)
    def forward(
        self,
        images: torch.Tensor,
        meme_captions: List[str],
        titles: Optional[List[str]] = None,
        input_type: int = 1,
    ) -> torch.Tensor:
        if input_type == 1:
            query = self.get_query_type1(images)
        else:
            assert titles is not None
            query = self.get_query_type2(images, titles)
        caption_emb = self.encode_texts(meme_captions)
        return self.compute_loss(query, caption_emb)

    # parameter groups (for optimizer)
    def trainable_params(self):
        """Yield only parameters that require grad."""
        return (p for p in self.parameters() if p.requires_grad)

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_preprocess(self):
        return self.preprocess
