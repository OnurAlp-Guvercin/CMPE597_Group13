from typing import Any, List, Optional

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fusion import build_fusion


class LoRALinear(nn.Module):
    def __init__(
        self,
        original_linear: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.original = original_linear
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        self.in_features = in_features
        self.out_features = out_features

        self.r = r
        self.scale = alpha / r
        self.lora_A = nn.Parameter(torch.randn(in_features, r) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.lora_dropout = nn.Dropout(dropout)

        for param in self.original.parameters():
            param.requires_grad = False

    @property
    def weight(self) -> torch.Tensor:
        delta_w = (self.lora_A @ self.lora_B).T * self.scale
        return self.original.weight + delta_w

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.original.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


def inject_lora(
    model: nn.Module,
    target_name_keywords: List[str],
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
) -> int:
    replaced = 0
    for name, module in list(model.named_modules()):
        if not any(keyword in name for keyword in target_name_keywords):
            continue
        if not isinstance(module, nn.Linear):
            continue

        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], LoRALinear(module, r, alpha, dropout))
        replaced += 1

    return replaced


class CLIPLoRA(nn.Module):
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

        clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device="cpu",
        )
        self.clip_model: Any = clip_model
        self.tokenizer: Any = open_clip.get_tokenizer(model_name)
        self.embed_dim = int(self.clip_model.visual.output_dim)

        for param in self.clip_model.parameters():
            param.requires_grad = False

        targets = lora_targets if lora_targets is not None else default_targets
        n_replaced = inject_lora(
            self.clip_model,
            targets,
            lora_r,
            lora_alpha,
            lora_dropout,
        )

        if n_replaced == 0 and lora_targets is not None:
            print(
                "[CLIPLoRA] Warning: no LoRA target matched. "
                f"Retrying with defaults: {default_targets}"
            )
            n_replaced = inject_lora(
                self.clip_model,
                default_targets,
                lora_r,
                lora_alpha,
                lora_dropout,
            )

        print(f"[CLIPLoRA] Injected LoRA into {n_replaced} layers (r={lora_r}, α={lora_alpha})")

        self.logit_scale = nn.Parameter(torch.tensor(2.6593))
        self.fusion = build_fusion(
            fusion_strategy,
            self.embed_dim,
            self.embed_dim,
            fusion_hidden,
        )

        self.to(device)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.clip_model.encode_image(images.to(self.device)), dim=-1)

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts).to(self.device)
        return F.normalize(self.clip_model.encode_text(tokens), dim=-1)

    def get_candidates(self, captions: List[str], device: str = "") -> torch.Tensor:
        return self.encode_texts(captions)

    def compute_loss(
        self,
        query_embeds: torch.Tensor,
        caption_embeds: torch.Tensor,
    ) -> torch.Tensor:
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits = logit_scale * (query_embeds @ caption_embeds.T)
        labels = torch.arange(logits.size(0), device=logits.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0

    def forward(
        self,
        images: torch.Tensor,
        meme_captions: List[str],
        titles: Optional[List[str]] = None,
        input_type: int = 1,
    ) -> torch.Tensor:
        if input_type == 1:
            query = self.encode_images(images)
        else:
            assert titles is not None
            query = F.normalize(self.fusion(self.encode_images(images), self.encode_texts(titles)), dim=-1)

        caption_emb = self.encode_texts(meme_captions)
        return self.compute_loss(query, caption_emb)

    def trainable_params(self):
        return (param for param in self.parameters() if param.requires_grad)

    def trainable_param_count(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def total_param_count(self) -> int:
        return sum(param.numel() for param in self.parameters())

    def get_preprocess(self):
        return self.preprocess
