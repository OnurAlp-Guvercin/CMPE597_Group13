import os
from dataclasses import dataclass, field
from typing import List

# Paths
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
    """Settings that control dataset loading & splits."""
    train_json: str = os.path.join(DATA_DIR, "memes-train.json")
    test_json: str = os.path.join(DATA_DIR, "memes-test.json")
    image_dir: str = IMAGE_DIR
    val_ratio: float = 0.1          # fraction of train used as validation
    seed: int = 42
    image_size: int = 224


@dataclass
class TrainConfig:
    """Shared training hyper-parameters."""
    batch_size: int = 192
    num_workers: int = 12
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 15
    patience: int = 4               # early-stopping patience
    temperature: float = 0.07       # contrastive loss temperature
    embed_dim: int = 256             # projection dimension
    seed: int = 42
    device: str = "cuda"             # overridden if CUDA unavailable
    fp16: bool = True                # mixed-precision training


@dataclass
class CLIPZeroShotConfig:
    """Config for zero-shot CLIP evaluation."""
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "openai"


@dataclass
class CustomModelConfig:
    """Config for the custom dual-encoder trained from scratch."""
    image_backbone: str = "resnet50"   # torchvision backbone
    text_model: str = "distilbert-base-uncased"
    embed_dim: int = 256
    freeze_image_backbone: bool = False
    freeze_text_backbone: bool = False
    dropout: float = 0.1


@dataclass
class LoRAFinetuneConfig:
    """Config for LoRA-based CLIP fine-tuning."""
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
    embed_dim: int = 512


@dataclass
class FusionConfig:
    """Settings for image+title fusion (Type 2 input)."""
    strategy: str = "concat_project"   # "concat_project" | "cross_attention" | "add"
    hidden_dim: int = 512
