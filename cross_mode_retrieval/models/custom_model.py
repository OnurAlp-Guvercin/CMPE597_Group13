from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from transformers import AutoModel, AutoTokenizer

from models.fusion import build_fusion


_BACKBONE_FACTORY = {
    "resnet18": (tv_models.resnet18, tv_models.ResNet18_Weights.DEFAULT),
    "resnet34": (tv_models.resnet34, tv_models.ResNet34_Weights.DEFAULT),
    "resnet50": (tv_models.resnet50, tv_models.ResNet50_Weights.DEFAULT),
    "resnet101": (tv_models.resnet101, tv_models.ResNet101_Weights.DEFAULT),
}


def _freeze(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


class ImageEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        backbone_name: str = "resnet50",
    ):
        super().__init__()

        if backbone_name not in _BACKBONE_FACTORY:
            valid = ", ".join(_BACKBONE_FACTORY.keys())
            raise ValueError(f"Unsupported image_backbone={backbone_name!r}. Choose one of: {valid}.")

        builder, weights = _BACKBONE_FACTORY[backbone_name]
        backbone = builder(weights=weights)

        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()  # type: ignore[assignment]
        self.backbone = backbone

        if freeze_backbone:
            _freeze(self.backbone)

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, embed_dim * 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return self.proj(features)


class TextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        embed_dim: int = 256,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size

        if freeze_backbone:
            _freeze(self.bert)

        self.proj = nn.Sequential(
            nn.Linear(hidden, embed_dim * 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, texts: List[str], device: torch.device) -> torch.Tensor:
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)
        outputs = self.bert(**tokens)
        cls = outputs.last_hidden_state[:, 0]
        return self.proj(cls)


class CustomDualEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        image_backbone: str = "resnet50",
        text_model: str = "distilbert-base-uncased",
        dropout: float = 0.1,
        temperature: float = 0.07,
        fusion_strategy: str = "concat_project",
        fusion_hidden: int = 512,
        freeze_image: bool = False,
        freeze_text: bool = False,
    ):
        super().__init__()
        self.temperature = temperature
        self.embed_dim = embed_dim

        self.image_encoder = ImageEncoder(
            embed_dim,
            dropout,
            freeze_backbone=freeze_image,
            backbone_name=image_backbone,
        )
        self.text_encoder = TextEncoder(
            text_model,
            embed_dim,
            dropout,
            freeze_backbone=freeze_text,
        )

        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))
        self.fusion = build_fusion(fusion_strategy, embed_dim, embed_dim, fusion_hidden)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.image_encoder(images), dim=-1)

    def encode_text(self, texts: List[str], device: torch.device) -> torch.Tensor:
        return F.normalize(self.text_encoder(texts, device), dim=-1)

    def get_query_type1(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_image(images)

    def get_query_type2(self, images: torch.Tensor, titles: List[str]) -> torch.Tensor:
        img_emb = self.encode_image(images)
        title_emb = self.encode_text(titles, images.device)
        fused = self.fusion(img_emb, title_emb)
        return F.normalize(fused, dim=-1)

    def get_candidates(
        self,
        captions: List[str],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        return self.encode_text(captions, device)

    def compute_loss(
        self,
        query_embeds: torch.Tensor,
        caption_embeds: torch.Tensor,
    ) -> torch.Tensor:
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits = logit_scale * (query_embeds @ caption_embeds.T)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2.0

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
            assert titles is not None, "Titles required for Type 2"
            query = self.get_query_type2(images, titles)

        caption_emb = self.encode_text(meme_captions, images.device)
        return self.compute_loss(query, caption_emb)
