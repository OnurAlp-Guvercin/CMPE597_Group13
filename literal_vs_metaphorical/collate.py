import torch

def collate_fn(batch):
    inputs = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])

    pixel_values = torch.stack([x["pixel_values"] for x in inputs])

    input_ids = torch.nn.utils.rnn.pad_sequence(
        [x["input_ids"] for x in inputs],
        batch_first=True,
        padding_value=0
    )

    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [x["attention_mask"] for x in inputs],
        batch_first=True,
        padding_value=0
    )

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }, labels