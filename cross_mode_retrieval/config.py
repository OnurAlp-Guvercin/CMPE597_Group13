import os
from dataclasses import dataclass, field
from typing import List

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
HISTORY_DIR = os.path.join(OUTPUT_DIR, "history")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)


@dataclass
class DataConfig:
    """Dataset paths and split settings."""
    train_json: str = os.path.join(DATA_DIR, "memes-train.json")
    test_json: str = os.path.join(DATA_DIR, "memes-test.json")
    image_dir: str = IMAGE_DIR
    val_ratio: float = 0.1  # train split'ten validation oranı
    seed: int = 42
    image_size: int = 224  # tüm görseller bu boyuta çekilir


@dataclass
class TrainConfig:
    """Shared training hyperparameters."""
    batch_size: int = 64
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 15
    patience: int = 4  # early stopping sabrı
    temperature: float = 0.07  # InfoNCE sıcaklığı
    embed_dim: int = 256  # ortak embedding boyutu
    seed: int = 42
    device: str = "cuda"  # CUDA yoksa main.py içinde cpu'ya düşer
    fp16: bool = False  # mixed precision aç/kapat
    warmup_epochs: int = 2  # cosine schedule öncesi warmup


@dataclass
class CLIPZeroShotConfig:
    """Zero-shot CLIP settings."""
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "openai"


@dataclass
class CustomModelConfig:
    """Custom dual-encoder settings."""
    image_backbone: str = "resnet50"
    text_model: str = "bert-base-uncased"
    embed_dim: int = 256
    freeze_image_backbone: bool = False
    freeze_text_backbone: bool = False
    dropout: float = 0.1


@dataclass
class LoRAFinetuneConfig:
    """LoRA fine-tuning settings."""
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["attn.out_proj", "mlp.c_fc", "mlp.c_proj"]
    )
    learning_rate: float = 1e-4
    epochs: int = 8
    embed_dim: int = 512  # CLIP ViT-B/32 text/image çıkış boyutu


@dataclass
class FusionConfig:
    """Fusion settings for Type 2 queries."""
    strategy: str = "weighted_sum"  # concat_project | cross_attention | add | weighted_sum | gated
    hidden_dim: int = 512  # concat_project için ara katman boyutu
