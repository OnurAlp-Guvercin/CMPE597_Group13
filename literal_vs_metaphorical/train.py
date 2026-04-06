from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn

from transformers import CLIPProcessor, CLIPModel

from dataset import MemeLiteralDataset
from model import LiteralMetaphorClassifier
from evaluate import evaluate
from collate import collate_fn
from experiment_logger import save_experiment
import config


def main():
    device = config.DEVICE

    # load CLIP
    processor = CLIPProcessor.from_pretrained(config.MODEL_NAME)
    clip_model = CLIPModel.from_pretrained(config.MODEL_NAME).to(device)

    train_dataset = MemeLiteralDataset(
        config.TRAIN_JSON,
        config.IMAGE_DIR,
        processor
    )

    test_dataset = MemeLiteralDataset(
        config.TEST_JSON,
        config.IMAGE_DIR,
        processor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    model = LiteralMetaphorClassifier(
        clip_model,
        fusion=config.FUSION,
        hidden=config.HIDDEN,
        dropout=config.DROPOUT
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0

        for inputs, labels in train_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.float().to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

        results = evaluate(
            model,
            test_loader,
            device,
            save_path=f"predictions_{config.FUSION}.json"
        )

        save_experiment(
            {
                "fusion": config.FUSION,
                "hidden": config.HIDDEN,
                "dropout": config.DROPOUT,
                "lr": config.LR
            },
            results
        )


if __name__ == "__main__":
    main()