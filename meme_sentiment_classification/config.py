import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_JSON = os.path.join(DATA_DIR, "memes-trainval.json")
TEST_JSON = os.path.join(DATA_DIR, "memes-test.json")

# ── Task 2.3(a): Sentiment annotation ────────────────────────────────────────
# Emotion model: 7 classes — anger, disgust, fear, joy, neutral, sadness, surprise
SENTIMENT_MODEL = "j-hartmann/emotion-english-distilroberta-base"
SENTIMENT_BATCH_SIZE = 64
ANNOTATED_TRAIN_JSON = os.path.join(OUTPUT_DIR, "annotated-trainval.json")
ANNOTATED_TEST_JSON = os.path.join(OUTPUT_DIR, "annotated-test.json")

# Number of random samples to print for manual label-noise inspection
MANUAL_REVIEW_N = 50

SEED = 42

# ── Tasks 2.3(b,c): Classification (future) ──────────────────────────────────
CLIP_MODEL = "openai/clip-vit-base-patch32"
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 20
HIDDEN = 512
DROPOUT = 0.3

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
