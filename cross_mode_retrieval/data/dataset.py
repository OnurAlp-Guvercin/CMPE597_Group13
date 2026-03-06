import json
import os
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms
import numpy as np


# Default image transform (ImageNet stats)
def default_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class MemecapDataset(Dataset):
    """
    Full MemeCap dataset for one split (train or test).
    Images that are missing on disk are replaced by a blank tensor.
    """

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
    def _to_text(value: object) -> str:
        """Normalize potentially dirty JSON values into a safe string."""
        if value is None:
            return ""
        if isinstance(value, float) and np.isnan(value):
            return ""
        if isinstance(value, str):
            return value
        return str(value)

    @classmethod
    def _first_text(cls, value: object) -> str:
        """Extract first element from list-like text fields and normalize."""
        if isinstance(value, list):
            if not value:
                return ""
            return cls._to_text(value[0])
        return cls._to_text(value)

    # helpers
    def _load_image(self, fname: str) -> Image.Image:
        path = os.path.join(self.image_dir, fname)
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Missing / corrupt image → blank placeholder
            img = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
        return img

    # Dataset API
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        entry = self.samples[idx]
        image = self._load_image(entry.get("img_fname", ""))
        image = self.transform(image)

        # Take the first caption / img_caption if there are multiple
        meme_caption = self._first_text(entry.get("meme_captions", [""]))
        img_caption = self._first_text(entry.get("img_captions", [""]))
        title = self._to_text(entry.get("title", ""))

        out = {
            "image": image,
            "title": title,
            "meme_caption": meme_caption,
            "img_caption": img_caption,
            "index": idx,
        }

        # Optionally append the literal image caption to the title for richer context
        if self.use_img_caption:
            out["title"] = f"{title} [SEP] {img_caption}" if title else img_caption

        return out


# Train / val split helper
def train_val_split(
    dataset: MemecapDataset,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Subset, Subset]:
    """Randomly split dataset into train / val subsets."""
    rng = np.random.RandomState(seed)
    n = len(dataset)
    indices = rng.permutation(n).tolist()
    val_size = int(n * val_ratio)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


# Collate for raw-string batches 
def collate_fn(batch: List[Dict]) -> Dict:
    """Stack images into a tensor; keep texts as lists of strings."""
    images = torch.stack([b["image"] for b in batch])
    titles = [b["title"] for b in batch]
    meme_captions = [b["meme_caption"] for b in batch]
    img_captions = [b["img_caption"] for b in batch]
    indices = torch.tensor([b["index"] for b in batch], dtype=torch.long)
    return {
        "image": images,
        "title": titles,
        "meme_caption": meme_captions,
        "img_caption": img_captions,
        "index": indices,
    }
