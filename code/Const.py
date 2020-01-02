import os

IMAGE_WIDTH_SIZE = 128
IMAGE_HEIGHT_SIZE = 128
IMAGE_DEPTH = 3

TRAIN_BIAS = 0.6
VAL_BIAS = 0.2
TEST_BIAS = 0.2

DATA_SIZE = 50000
BATCH_SIZE = 20
EPOCH_SIZE = 50


MODEL_PRE_VGG19 = "MODEL_PRE_VGG19"
MODEL_NEW_VGG19 = "MODEL_NEW_VGG19"
MODEL_PRE_RESNET50 = "MOEL_PRE_RESNET50"
MODEL_NEW_RESNET50 = "MOEL_NEW_RESNET50"

ABS_PATH = os.path.abspath("medi_project").split("\\")
MAIN_PATH = "\\".join(ABS_PATH[0:len(ABS_PATH)-2])
DATA_ALL_PATH = os.path.join(MAIN_PATH, "data", "all")
DATA_TRAIN_PATH = os.path.join(MAIN_PATH, "data", "train")
DATA_VAL_PATH = os.path.join(MAIN_PATH, "data", "val")
DATA_TEST_PATH = os.path.join(MAIN_PATH, "data", "test")
DATA_LABEL_CSV = os.path.join(MAIN_PATH, "data", "labeling_data.csv")

# MODEL_SAVE_PATH = os.path.join(MAIN_PATH, "model")
MODEL_SAVE_PATH = f"D:\\medi_test\\model"
