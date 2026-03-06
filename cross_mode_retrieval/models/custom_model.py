import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from transformers import AutoModel, AutoTokenizer
from typing import List, Optional

from models.fusion import build_fusion


# Image Encoder
class ImageEncoder(nn.Module):
    """Torchvision CNN backbone → projection head."""

    def __init__(
        self,
        embed_dim: int = 256,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        backbone_name: str = "resnet50",
    ):
        super().__init__()

        if backbone_name == "resnet18":
            backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
        elif backbone_name == "resnet34":
            backbone = tv_models.resnet34(weights=tv_models.ResNet34_Weights.DEFAULT)
        elif backbone_name == "resnet50":
            backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
        elif backbone_name == "resnet101":
            backbone = tv_models.resnet101(weights=tv_models.ResNet101_Weights.DEFAULT)
        else:
            raise ValueError(
                f"Unsupported image_backbone={backbone_name!r}. "
                "Choose one of: resnet18, resnet34, resnet50, resnet101."
            )

        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()  # type: ignore[assignment]  # remove classification head
        self.backbone = backbone

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, embed_dim * 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,3,H,W] → [B,D]
        features = self.backbone(x)          # [B, 2048]
        return self.proj(features)           # [B, embed_dim]


# Text Encoder
class TextEncoder(nn.Module):
    """DistilBERT [CLS] → projection head."""

    def __init__(self, model_name: str = "distilbert-base-uncased",
                 embed_dim: int = 256, dropout: float = 0.1,
                 freeze_backbone: bool = False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size  # 768
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if freeze_backbone:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(hidden, embed_dim * 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, texts: List[str], device: torch.device) -> torch.Tensor:
        tokens = self.tokenizer(
            texts, padding=True, truncation=True, max_length=128,
            return_tensors="pt",
        ).to(device)
        out = self.bert(**tokens)
        cls = out.last_hidden_state[:, 0]    # [CLS] token
        return self.proj(cls)                # [B, embed_dim]


# Full dual-encoder model
class CustomDualEncoder(nn.Module):
    """
    Dual-encoder with optional fusion for Type 2 queries.

    loss = symmetric InfoNCE  (image↔caption + caption↔image) / 2
    """

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
            text_model, embed_dim, dropout, freeze_backbone=freeze_text,
        )

        # Learnable log-temperature (as in CLIP)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))

        # Fusion for Type 2
        self.fusion = build_fusion(fusion_strategy, embed_dim, embed_dim, fusion_hidden)

    # encoding helpers
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.image_encoder(images), dim=-1)

    def encode_text(self, texts: List[str], device: torch.device) -> torch.Tensor:
        return F.normalize(self.text_encoder(texts, device), dim=-1)

    # query builders
    def get_query_type1(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_image(images)

    def get_query_type2(
        self, images: torch.Tensor, titles: List[str],
    ) -> torch.Tensor:
        img_emb = self.encode_image(images)
        title_emb = self.encode_text(titles, images.device)
        fused = self.fusion(img_emb, title_emb)
        return F.normalize(fused, dim=-1)

    def get_candidates(
        self, captions: List[str], device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        return self.encode_text(captions, device)

    # contrastive loss
    def compute_loss(
        self,
        query_embeds: torch.Tensor,
        caption_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Symmetric InfoNCE (NT-Xent) loss.
        query_embeds  : [B, D]  (already L2-normalised)
        caption_embeds: [B, D]  (already L2-normalised)
        """
        # This is for numerical stability, as in CLIP paper
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        # Diagonal entries are positives, off-diagonal are negatives
        logits = logit_scale * (query_embeds @ caption_embeds.T)  # [B, B]
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2.0

    # forward (training)
    def forward(
        self,
        images: torch.Tensor,
        meme_captions: List[str],
        titles: Optional[List[str]] = None,
        input_type: int = 1,
    ) -> torch.Tensor:
        """
        Returns scalar loss.

        Args:
            images:        [B, 3, H, W]
            meme_captions: list[str] of length B
            titles:        list[str] of length B  (Type 2 only)
            input_type:    1 (image only) or 2 (image + title)
        """
        if input_type == 1:
            query = self.get_query_type1(images)
        else:
            assert titles is not None, "Titles required for Type 2"
            query = self.get_query_type2(images, titles)

        caption_emb = self.encode_text(meme_captions, images.device)
        return self.compute_loss(query, caption_emb)
