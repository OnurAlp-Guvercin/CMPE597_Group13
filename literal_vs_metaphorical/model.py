import torch
import torch.nn as nn


class LiteralMetaphorClassifier(nn.Module):
    def __init__(
        self,
        clip_model,
        fusion="concat",
        hidden=512,
        dropout=0.1
    ):
        super().__init__()

        self.clip = clip_model
        self.fusion = fusion

        dim = clip_model.config.projection_dim

        if fusion == "concat":
            in_dim = dim * 2

        elif fusion == "multiply":
            in_dim = dim

        elif fusion == "bilinear":
            self.bilinear = nn.Bilinear(dim, dim, dim)
            in_dim = dim

        elif fusion == "attention":
            self.attn = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=4,
                batch_first=True
            )
            in_dim = dim

        else:
            raise ValueError("Unknown fusion type")

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, inputs):
        outputs = self.clip(**inputs)

        img = outputs.image_embeds
        txt = outputs.text_embeds

        # ---- fusion strategies ----
        if self.fusion == "concat":
            fused = torch.cat([img, txt], dim=1)

        elif self.fusion == "multiply":
            fused = img * txt

        elif self.fusion == "bilinear":
            fused = self.bilinear(img, txt)

        elif self.fusion == "attention":
            # treat embeddings as sequence length 1
            img = img.unsqueeze(1)
            txt = txt.unsqueeze(1)

            attn_out, _ = self.attn(img, txt, txt)
            fused = attn_out.squeeze(1)

        return self.classifier(fused).squeeze(1)