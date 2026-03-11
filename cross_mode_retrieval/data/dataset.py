import json
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms


def default_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class MemecapDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        image_dir: str,
        transform: Optional[Callable] = None,
        image_size: int = 224,
        use_img_caption: bool = False,
    ):
        with open(json_path, "r", encoding="utf-8") as f:
            self.samples: List[Dict] = json.load(f)

        self.image_dir = image_dir
        self.transform = transform or default_transform(image_size)
        self.image_size = image_size
        self.use_img_caption = use_img_caption

    @staticmethod
    def _to_text(value: object, first_item: bool = False) -> str:
        if first_item and isinstance(value, list):
            value = value[0] if value else ""
        if isinstance(value, list):
            value = value[0] if value else ""
        if value is None:
            return ""
        if isinstance(value, float) and np.isnan(value):
            return ""
        return value if isinstance(value, str) else str(value)

    def _load_image(self, fname: str) -> Image.Image:
        path = os.path.join(self.image_dir, fname)
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        entry = self.samples[idx]
        get = entry.get
        image = self.transform(self._load_image(get("img_fname", "")))

        meme_caption = self._to_text(get("meme_captions", [""]), first_item=True)
        img_caption = self._to_text(get("img_captions", [""]), first_item=True)
        title = self._to_text(get("title", ""))

        if self.use_img_caption:
            title = f"{title} [SEP] {img_caption}" if title else img_caption

        return {
            "image": image,
            "title": title,
            "meme_caption": meme_caption,
            "img_caption": img_caption,
            "index": idx,
        }


def train_val_split(
    dataset: MemecapDataset,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Subset, Subset]:
    
    rng = np.random.RandomState(seed)
    n = len(dataset)
    indices = rng.permutation(n).tolist()
    val_size = int(n * val_ratio)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def collate_fn(batch: List[Dict]) -> Dict:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "title": [item["title"] for item in batch],
        "meme_caption": [item["meme_caption"] for item in batch],
        "img_caption": [item["img_caption"] for item in batch],
        "index": torch.tensor([item["index"] for item in batch], dtype=torch.long),
    }
