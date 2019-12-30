import os

IMAGE_WIDTH_SIZE = 128
IMAGE_HEIGHT_SIZE = 128
IMAGE_DEPTH = 3

BATCH_SIZE = 40
EPOCH_SIZE = 50

MODEL_PRE_VGG19 = "MODEL_PRE_VGG19"
MODEL_NEW_VGG19 = "MODEL_NEW_VGG19"
MODEL_PRE_RESNET50 = "MOEL_PRE_RESNET50"
MODEL_NEW_RESNET50 = "MOEL_NEW_RESNET50"

DATA_ALL_PATH = os.path.join("data", "all")
DATA_TRAIN_PATH = os.path.join("data", "train")
DATA_VAL_PATH = os.path.join("data", "val")
DATA_TEST_PATH = os.path.join("data", "test")
DATA_LABEL_CSV = os.path.join("data", "labeling_data.csv")

MODEL_SAVE_PATH = "model\\"
