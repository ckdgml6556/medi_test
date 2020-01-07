import os

# 노말과 업노말 6개의 클래스 전부 사용
TYPE_ALL = 6
# 노말과 업노말 2개의 클래스로 분류 시 사용
TYPE_NOMAL = 2
# 업노말 5개의 클래스로 분류 시 사용
TYPE_ABNOMAL = 5

CURRENT_TYPE = TYPE_ALL
CLASS_COUNT = CURRENT_TYPE

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

if CURRENT_TYPE == TYPE_ALL:
    DATA_PATH = "data"
elif CURRENT_TYPE == TYPE_NOMAL:
    DATA_PATH = "data_nomal"
else:
    DATA_PATH = "data_abnomal"

ABS_PATH = os.path.abspath("medi_project").split("\\")
MAIN_PATH = "\\".join(ABS_PATH[0:len(ABS_PATH) - 2])
DATA_ALL_PATH = os.path.join(MAIN_PATH, DATA_PATH, "all")
DATA_TRAIN_PATH = os.path.join(MAIN_PATH, DATA_PATH, "train")
DATA_VAL_PATH = os.path.join(MAIN_PATH, DATA_PATH, "val")
DATA_TEST_PATH = os.path.join(MAIN_PATH, DATA_PATH, "test")
DATA_LABEL_CSV = os.path.join(MAIN_PATH, DATA_PATH, "labeling_data.csv")

DATA_LABEL_CSV = os.path.join(MAIN_PATH, "data", "labeling_data.csv")

# MODEL_SAVE_PATH = os.path.join(MAIN_PATH, "model")
MODEL_MAIN_PATH = "D:\\medi_test\\model"
if CURRENT_TYPE == TYPE_ALL:
    MODEL_TYPE_PATH = "all"
elif CURRENT_TYPE == TYPE_NOMAL:
    MODEL_TYPE_PATH = "nomal"
else:
    MODEL_TYPE_PATH = "abnomal"
MODEL_SAVE_PATH = os.path.join(MODEL_MAIN_PATH, MODEL_TYPE_PATH)

