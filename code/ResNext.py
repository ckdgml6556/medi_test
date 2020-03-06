from keras_applications import resnext
import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D,
import Const

def getPreResNeXt():
    pre_net = resnext.ResNeXt101(
        backend=keras.backend,
        layers=keras.layers,
        models=keras.models,
        utils=keras.utils,
        input_shape=(Const.IMAGE_WIDTH_SIZE, Const.IMAGE_HEIGHT_SIZE, Const.IMAGE_DEPTH),
        include_top=True,
    )

    pre_net.summary()
    model = Sequential()
    model.add(pre_net)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(Const.CLASS_COUNT, activation='softmax'))

    # model.add(Flatten())
    # model.add(Dropout(0.5))
    # model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    # model.add(Dropout(0.5))
    # model.add(Dense(Const.CLASS_COUNT, activation='softmax'))

    return model