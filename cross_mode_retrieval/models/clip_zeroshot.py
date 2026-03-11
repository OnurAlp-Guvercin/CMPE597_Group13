from typing import Any, List

import open_clip
import torch
import torch.nn as nn

from models.fusion import build_fusion


class CLIPZeroShot(nn.Module):
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
            model_name,
            pretrained=pretrained,
            device=device,
        )
        self.clip_model: Any = clip_model
        self.tokenizer: Any = open_clip.get_tokenizer(model_name)
        self.embed_dim = int(self.clip_model.visual.output_dim)

        self.fusion = build_fusion(
            fusion_strategy,
            self.embed_dim,
            self.embed_dim,
            fusion_hidden,
        ).to(device)

        for param in self.clip_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        return self.clip_model.encode_image(images.to(self.device))

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts).to(self.device)
        return self.clip_model.encode_text(tokens)

    def get_candidates(self, captions: List[str], device: str = "") -> torch.Tensor:
        return self.encode_texts(captions)

    def get_preprocess(self):
        return self.preprocess
