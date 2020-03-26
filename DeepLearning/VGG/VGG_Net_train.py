import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import data_preprocessing as dc
from tensorflow import keras

x_train, tt, rt, at, x_valid, tv, rv, av = dc.data_chuli(2000, 200)
y_train = np.concatenate((tt, rt, at), axis=1)
y_valid = np.concatenate((tv, rv, av), axis=1)


# VGG16
class VGG_Net(keras.models.Model):
    def __init__(self, num_classes=1500):
        super(VGG_Net, self).__init__()
        self.layer1 = keras.models.Sequential([
            keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='same')
        ])
        self.layer2=keras.models.Sequential([
            keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='same')
        ])
        self.layer3=keras.models.Sequential([
            keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='same')
        ])
        self.layer4=keras.models.Sequential([
            keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
            keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
            keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
            keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='same')
        ])
        self.layer5=keras.models.Sequential([
            keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
            keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
            keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
            keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='same')
        ])
        self.layer6=keras.models.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(num_classes, activation='relu'),
            keras.layers.Dropout(rate=0.5),
            keras.layers.Dense(num_classes, activation='sigmoid'),
        ])

    def call(self, input):
        x=self.layer1(input)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.layer5(x)
        x=self.layer6(x)

        return x


def vggnet():
    return VGG_Net()


# 创建vggnet网络
model = vggnet()
model.build(input_shape=(None, 20, 20, 1))
model.summary()
# 设置loss函数和optimizer
model.compile(loss="mean_squared_error", optimizer='adam')

# 用于设置训练出的模型参数的保存路径和其它相关参数
logdir = './vggnet_callbacks'
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir, "multilevel_model.h5")
callbacks = [
    keras.callbacks.TensorBoard(logdir),
    keras.callbacks.ModelCheckpoint(output_model_file,
                                    save_best_only=True),
    # keras.callbacks.EarlyStopping(patience=5,min_delta=1e-3)
]

# 进行训练
history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
                    epochs=100, batch_size=64, callbacks=callbacks)


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 0.5)
    plt.show()


# 画出loss曲线图
plot_learning_curves(history)
