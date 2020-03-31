import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tensorflow.keras.layers import Activation
import tensorflow as tf
import data_preprocessing as dc
from tensorflow import keras

# 读取训练集和验证集中的数据，包括二维码矩阵参数(x_train，x_valid)，
# 透射谱(tt,tv)，反射谱(rt,rv)和吸收谱(at,av)，
# 每个频谱表示为一个(1,500)的行向量,并将透射谱，
# 反射谱和吸收谱连接起来成为一个（1,1500）的行向量(y_train，y_valid)。
# x_train和x_valid的shape为(None,20,20,1),y_train和y_valid的shape(None,1500)
x_train, tt, rt, at, x_valid, tv, rv, av = dc.data_chuli(2000, 200)
y_train = np.concatenate((tt, rt, at), axis=1)
y_valid = np.concatenate((tv, rv, av), axis=1)


def se_block(x, reduction=4):
    filters = x._shape_val[-1]
    out = keras.layers.GlobalAveragePooling2D()(x)
    out = keras.layers.Dense(int(filters / reduction), activation='relu')(out)
    out = keras.layers.Dense(filters, activation='hard_sigmoid')(out)
    print(out.shape)
    out = keras.layers.Reshape((1, 1, -1))(out)
    out = keras.layers.multiply([x, out])
    return out


def Hswish(x):
    return x * tf.nn.relu6(x + 3) / 6.0


def Bneck(x, kernel_size, input_size, expand_size, output_size, activation, stride, use_se):
    out = keras.layers.Conv2D(filters=expand_size, kernel_size=1, strides=1, use_bias=False)(x)
    out = keras.layers.BatchNormalization()(out, training=True)
    out = activation(out)
    out = keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding='same')(out)
    out = keras.layers.BatchNormalization()(out, training=True)
    if use_se:
        out = se_block(out)
    out = keras.layers.Conv2D(filters=output_size, kernel_size=1, strides=1, padding='same')(out)
    out = keras.layers.BatchNormalization()(out, training=True)
    if stride == 1 and input_size == output_size:
        short_cut = keras.layers.Conv2D(filters=output_size, kernel_size=1, strides=1, padding='same')(x)
        out = keras.layers.Add()([out, short_cut])
    return out


# mobilenet_v3模型
def MobileNetv3(num_classes=1500, input_shape=(20, 20, 1)):
    x = keras.Input(shape=input_shape)
    out = keras.layers.Conv2D(16, 3, strides=2, padding='same', use_bias=False)(x)
    out = keras.layers.BatchNormalization()(out, training=True)
    out = Activation(Hswish)(out)

    out = Bneck(out, 3, 16, 16, 16, keras.layers.ReLU(), 2, True)
    out = Bneck(out, 3, 16, 72, 24, keras.layers.ReLU(), 2, False)
    out = Bneck(out, 3, 24, 88, 24, keras.layers.ReLU(), 1, False)
    out = Bneck(out, 5, 24, 96, 40, Activation(Hswish), 2, True)
    out = Bneck(out, 5, 40, 240, 40, Activation(Hswish), 1, True)
    out = Bneck(out, 5, 40, 240, 40, Activation(Hswish), 1, True)
    out = Bneck(out, 5, 40, 120, 48, Activation(Hswish), 1, True)
    out = Bneck(out, 5, 48, 144, 48, Activation(Hswish), 1, True)
    out = Bneck(out, 5, 48, 288, 96, Activation(Hswish), 2, True)
    out = Bneck(out, 5, 96, 576, 96, Activation(Hswish), 1, True)
    out = Bneck(out, 5, 96, 576, 96, Activation(Hswish), 1, True)

    out = keras.layers.Conv2D(filters=576, kernel_size=1, strides=1)(out)
    out = keras.layers.BatchNormalization()(out, training=True)
    out = Activation(Hswish)(out)
    out = se_block(out)
    out = keras.layers.AveragePooling2D(pool_size=(1, 1))(out)
    out = keras.layers.Conv2D(filters=1280, kernel_size=1, strides=1)(out)
    out = keras.layers.Activation(Hswish)(out)
    out = keras.layers.Conv2D(filters=num_classes, kernel_size=1, strides=1, activation='sigmoid')(out)
    out = keras.layers.Flatten()(out)
    model = keras.Model(inputs=x, outputs=out)
    return model


# 创建mobilenetv3网络
model = MobileNetv3()
model.summary()
# 设置loss函数和optimizer
model.compile(loss="mean_squared_error", optimizer='adam')

# 用于设置训练出的模型参数的保存路径和其它相关参数
logdir = './mobilenet_callbacks'
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
