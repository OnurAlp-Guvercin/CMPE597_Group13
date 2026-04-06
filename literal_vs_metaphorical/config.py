import torch

MODEL_NAME = "openai/clip-vit-base-patch32"

TRAIN_JSON = "data/memes-trainval.json"
TEST_JSON = "data/memes-test.json"

IMAGE_DIR = "images/"

BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 10
FUSION = "bilinear" 
HIDDEN = 512
DROPOUT = 0.3

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"