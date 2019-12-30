from keras.applications import VGG19
from keras.layers import Input, Dropout
from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten,MaxPooling2D
from keras import losses
import Const


def getVGG19Model():
    pre_model = VGG19(
        weights='imagenet',
        include_top=False,
        input_shape=(Const.IMAGE_WIDTH_SIZE, Const.IMAGE_HEIGHT_SIZE, Const.IMAGE_DEPTH)
    )
    # model.add(Flatten())
    # model.add(Dense(units=4096, activation="relu"))
    # model.add(Dense(units=4096, activation="relu"))
    # model.add(Dense(units=6, activation="softmax"))

    # Stacking a new simple convolutional network on top of it

    # lx_dict['block2_pool'].output
    x = pre_model.output
    x = Flatten()(x)
    x = Dense(units=4096, activation="relu")(x)
    x = Dense(units=4096, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(units=6, activation="softmax")(x)

    # Creating new model. Please note that this is NOT a Sequential() model.
    custom_model = Model(inputs=pre_model.input, output=x)
    custom_model.summary()
    # Make sure that the pre-trained bottom layers are not trainable
    for layer in custom_model.layers[:6]:
        layer.trainable = False

    return custom_model


def getNewVGG19Model():
    model = Sequential()
    model.add(Conv2D(input_shape=(Const.IMAGE_WIDTH_SIZE, Const.IMAGE_HEIGHT_SIZE, Const.IMAGE_DEPTH), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=6, activation="softmax"))

    return model

