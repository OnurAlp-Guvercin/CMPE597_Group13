import torch
import torch.nn as nn
import open_clip
from typing import Any, List

from models.fusion import build_fusion


class CLIPZeroShot(nn.Module):
    """
    Thin wrapper around an OpenCLIP model.

    query modes
    -----------
    type1 : encode the meme image only
    type2 : encode the meme image + title (fused)
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        fusion_strategy: str = "concat_project",
        fusion_hidden: int = 512,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device,
        )
        self.clip_model: Any = clip_model
        self.tokenizer: Any = open_clip.get_tokenizer(model_name)

        # Determine CLIP's embedding dimension dynamically
        self.embed_dim = int(self.clip_model.visual.output_dim)

        # Fusion module for Type 2 (image + title)
        self.fusion = build_fusion(
            fusion_strategy, self.embed_dim, self.embed_dim, fusion_hidden,
        )
        self.fusion.to(device)

        # Freeze everything except fusion
        for p in self.clip_model.parameters():
            p.requires_grad = False

    # Encoding helpers
    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of pre-processed image tensors → [B, D]."""
        return self.clip_model.encode_image(images.to(self.device))

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Tokenise & encode a list of strings → [B, D]."""
        tokens = self.tokenizer(texts).to(self.device)
        return self.clip_model.encode_text(tokens)

    # Query builders
    def get_query_type1(self, images: torch.Tensor) -> torch.Tensor:
        """Type 1: query = image embedding."""
        return self.encode_images(images)

    def get_query_type2(
        self, images: torch.Tensor, titles: List[str],
    ) -> torch.Tensor:
        """Type 2: query = fused(image, title)."""
        img_emb = self.encode_images(images)
        title_emb = self.encode_texts(titles)
        return self.fusion(img_emb, title_emb)

    # Convenience: get candidates
    def get_candidates(self, captions: List[str], device: str = "") -> torch.Tensor:
        """Encode meme captions as retrieval candidates."""
        return self.encode_texts(captions)

    def get_preprocess(self):
        """Return the image pre-processing transform expected by this CLIP model."""
        return self.preprocess
