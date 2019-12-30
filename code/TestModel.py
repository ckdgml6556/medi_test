import os
import Const
import glob
import shutil
from keras import Model
from keras.models import model_from_json
from keras import losses
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator

# 옵티마이져
# optimizer = SGD(lr=0.0001)
optimizer = Adam(lr=0.0001)

def testModel(net_type):
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        Const.DATA_TEST_PATH,
        target_size=(Const.IMAGE_WIDTH_SIZE, Const.IMAGE_HEIGHT_SIZE),
        batch_size=Const.BATCH_SIZE,
        class_mode="categorical"
    )

    model_path = os.path.join(Const.MODEL_SAVE_PATH, net_type)

    json_file = open(f"{model_path}\\model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    model_list = glob.glob(f"{model_path}\\*.h5")
    best_score = 0.0
    best_model = ""
    for model in model_list:
        print(model)
        loaded_model.load_weights(model)
        loaded_model.compile(optimizer=optimizer, loss=losses.categorical_crossentropy, metrics=['accuracy'])
        print("-- Evaluate --")
        scores = loaded_model.evaluate_generator(test_generator)
        if scores[1] > best_score:
            print(f"change best score is {scores[1] * 100}/{model} saved")
            best_score = round(scores[1], 5) * 100
            best_model = model
        #클래스별 예측값 보기
        # output = loaded_model.predict_generator(test_generator)
        # print(output)
    saveBestModel(net_type, best_score, best_model)


def saveBestModel(net_type, accuracy, model_name):
    # model_file = f"{os.path.join(Const.MODEL_SAVE_PATH, net_type)}\\{model_name}"
    new_model_file = f"{Const.MODEL_SAVE_PATH}\\best_model\\{net_type}_{accuracy}_{model_name}"
    if not os.path.isfile(new_model_file):
        shutil.copy2(model_name,new_model_file)

testModel(Const.MODEL_NEW_RESNET50)
