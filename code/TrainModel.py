import os
import tensorflow as tf
from keras.layers import Input, Dropout
from keras import Model
from keras import backend as K
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras import losses
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import ResNet50,VGG19
import Const
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

file_list = os.listdir(Const.DATA_TRAIN_PATH)

label_dict = ["nomal", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]

train_nomal_dir = os.path.join(Const.DATA_TRAIN_PATH, "nomal")
train_epidural_dir = os.path.join(Const.DATA_TRAIN_PATH, "epidural")
train_intraparenchymal_dir = os.path.join(Const.DATA_TRAIN_PATH, "intraparenchymal")
train_intraventricular_dir = os.path.join(Const.DATA_TRAIN_PATH, "intraventricular")
train_subarachnoid_dir = os.path.join(Const.DATA_TRAIN_PATH, "subarachnoid")
train_subdural_dir = os.path.join(Const.DATA_TRAIN_PATH, "subdural")

# 옵티마이져
# optimizer = SGD(lr=0.0001)
optimizer = Adam(lr=0.0001)



def get_session():
    config = tf.compat.v1.ConfigProto(
        device_count={'GPU': 1}
    )
    tf.compat.v1.Session(config=config)


def recall(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    # round : 반올림한다
    y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn)

    # (True Positive + False Negative) = 실제 값이 1(Positive) 전체
    count_true_positive_false_negative = K.sum(y_target_yn)

    # Recall =  (True Positive) / (True Positive + False Negative)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())

    # return a single tensor value
    return recall


def precision(y_target, y_pred):
    # clip(t, clip_value_min, clip_value_max) : clip_value_min~clip_value_max 이외 가장자리를 깎아 낸다
    # round : 반올림한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))  # 예측값을 0(Negative) 또는 1(Positive)로 설정한다
    y_target_yn = K.round(K.clip(y_target, 0, 1))  # 실제값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn)

    # (True Positive + False Positive) = 예측 값이 1(Positive) 전체
    count_true_positive_false_positive = K.sum(y_pred_yn)

    # Precision = (True Positive) / (True Positive + False Positive)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())

    # return a single tensor value
    return precision


def f1score(y_target, y_pred):
    _recall = recall(y_target, y_pred)
    _precision = precision(y_target, y_pred)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    _f1score = (2 * _recall * _precision) / (_recall + _precision + K.epsilon())

    # return a single tensor value
    return _f1score

def trainningModel(net_type):
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        Const.DATA_TRAIN_PATH,
        target_size=(Const.IMAGE_WIDTH_SIZE, Const.IMAGE_HEIGHT_SIZE),
        batch_size=Const.BATCH_SIZE,
        class_mode="categorical"
    )

    vali_datagen = ImageDataGenerator(rescale=1. / 255)
    vali_generator = vali_datagen.flow_from_directory(
        Const.DATA_VAL_PATH,
        target_size=(Const.IMAGE_WIDTH_SIZE, Const.IMAGE_HEIGHT_SIZE),
        batch_size=Const.BATCH_SIZE,
        class_mode="categorical"
    )

    model_svae_dir = f"{Const.MODEL_SAVE_PATH}\\{net_type}"
    if not os.path.exists(model_svae_dir):
        os.makedirs(model_svae_dir)

    if net_type in Const.MODEL_PRE_VGG19:
        net_model = VGG19.getVGG19Model()
    elif net_type in Const.MODEL_NEW_VGG19:
        net_model = VGG19.getNewVGG19Model()
    elif net_type in Const.MODEL_NEW_RESNET50:
        net_model = ResNet50.getNewResNet50()

    net_model.compile(optimizer=optimizer,
                      loss=losses.categorical_crossentropy,
                      metrics = ["accuracy", precision_score, recall_score, f1_score])

    model_json_file = f"{model_svae_dir}\\model.json"
    if not os.path.isfile(model_json_file):
        model_json = net_model.to_json()
        with open(model_json_file , "w") as json_file:
            json_file.write(model_json)


    file_path = f"{model_svae_dir}\\{net_type}"+"-{epoch:02d}-{val_accuracy:.2f}.h5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_accuracy', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='max',)
    early = EarlyStopping(monitor='accuracy', min_delta=0, patience=200, verbose=1, mode='auto')

    history = net_model.fit_generator(
        steps_per_epoch=len(train_generator.filenames)/Const.BATCH_SIZE,
        generator=train_generator,
        validation_data=vali_generator,
        validation_steps=len(vali_generator.filenames)/Const.BATCH_SIZE,
        epochs=Const.EPOCH_SIZE,
        callbacks=[checkpoint, early]
    )
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["precision_score"])
    plt.plot(history.history["recall_score"])
    plt.plot(history.history["f1_score"])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Accuracy", "Validation Accuracy", "loss", "Validation Loss"])
    plt.show()

# getResNet50Model()
# getVGG16Model()
# get_session()
# trainningModel(Const.MODEL_NEW_VGG19)
# trainningModel(Const.MODEL_PRE_VGG19)
trainningModel(Const.MODEL_NEW_RESNET50)
