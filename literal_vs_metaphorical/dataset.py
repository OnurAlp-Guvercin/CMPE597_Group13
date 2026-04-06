import json
import os
from PIL import Image
from torch.utils.data import Dataset
import torch


class MemeLiteralDataset(Dataset):
    def __init__(self, json_path, image_dir, processor, log_skipped=True):
        with open(json_path, "r") as f:
            data = json.load(f)

        self.samples = []
        self.processor = processor
        self.skipped_images = []

        for item in data:
            img_path = os.path.join(image_dir, item["img_fname"])

            # skip missing images
            if not os.path.exists(img_path):
                if log_skipped:
                    print(f"WARNING: missing file {img_path}")
                self.skipped_images.append(img_path)
                continue

            # literal captions
            for cap in item.get("img_captions", []):
                self.samples.append((img_path, cap, 0))

            # metaphorical captions
            for cap in item.get("meme_captions", []):
                self.samples.append((img_path, cap, 1))

        print(f"Loaded {len(self.samples)} samples from {json_path}")
        if self.skipped_images and log_skipped:
            print(f"Skipped {len(self.skipped_images)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        inputs = self.processor(
            text=caption,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # squeeze batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        return inputs, torch.tensor(label, dtype=torch.float)