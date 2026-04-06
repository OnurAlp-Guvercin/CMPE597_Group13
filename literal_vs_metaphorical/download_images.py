import json
import os
import requests
from tqdm import tqdm

TRAIN_JSON = "data/memes-trainval.json"
TEST_JSON = "data/memes-test.json"

SAVE_DIR = "images"
os.makedirs(SAVE_DIR, exist_ok=True)


def load_all_items():
    items = []

    for path in [TRAIN_JSON, TEST_JSON]:
        with open(path) as f:
            items.extend(json.load(f))

    return items


def download():
    items = load_all_items()

    for item in tqdm(items):
        url = item["url"]
        fname = item["img_fname"]
        path = os.path.join(SAVE_DIR, fname)

        if os.path.exists(path):
            continue

        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()

            with open(path, "wb") as f:
                f.write(r.content)

        except Exception as e:
            print("failed:", url)


if __name__ == "__main__":
    download()