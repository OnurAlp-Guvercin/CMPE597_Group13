import json
import requests
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# MemeCap GitHub raw URLs
REPO_RAW = "https://raw.githubusercontent.com/eujhwang/meme-cap/main/data"

TRAIN_URL = f"{REPO_RAW}/memes-trainval.json"
TEST_URL = f"{REPO_RAW}/memes-test.json"

DATA_DIR = Path(__file__).resolve().parent
IMAGE_DIR = DATA_DIR / "images"


def download_file(url: str, dest: Path, timeout: int = 30) -> bool:
    """Download a single file. Returns True on success."""
    if dest.exists():
        return True
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  [WARN] Failed to download {url}: {e}")
        return False


def download_jsons():
    """Download train / test JSON files."""
    for url, name in [(TRAIN_URL, "memes-trainval.json"), (TEST_URL, "memes-test.json")]:
        dest = DATA_DIR / name
        if dest.exists():
            print(f"  {name} already exists, skipping.")
            continue
        print(f"  Downloading {name} …")
        download_file(url, dest)


def download_images(json_path: Path, max_workers: int = 8):
    """Download meme images referenced in a JSON split."""
    with open(json_path) as f:
        samples = json.load(f)

    tasks = []
    for sample in samples:
        img_fname = sample.get("img_fname", "")
        url = sample.get("url", "")
        if not url or not img_fname:
            continue
        dest = IMAGE_DIR / img_fname
        if dest.exists():
            continue
        tasks.append((url, dest))

    if not tasks:
        print(f"  All images for {json_path.name} already present.")
        return

    print(f"  Downloading {len(tasks)} images for {json_path.name} …")
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    success, fail = 0, 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(download_file, url, dest): (url, dest)
                   for url, dest in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Images"):
            if future.result():
                success += 1
            else:
                fail += 1
    print(f"  Done: {success} downloaded, {fail} failed.")


def main():
    print("=== MemeCap Data Downloader ===")

    download_jsons()
    for name in ["memes-trainval.json", "memes-test.json"]:
        jp = DATA_DIR / name
        if jp.exists():
            download_images(jp)
    print("=== Finished ===")


if __name__ == "__main__":
    main()
