import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Activation
import tensorflow as tf
import data_preprocessing as dc
from tensorflow import keras

# 拿训练好的模型进行测试的网络结构
# 读取用于测试的数据，包括二维码矩阵参数(x_valid)，透射谱(tv)，反射谱(rv)和吸收谱(av)，
# 每个频谱表示为一个(1,500)的行向量,并将透射谱，
# 反射谱和吸收谱连接起来成为一个（1,1500）的行向量(y_test)。
# x_valid的shape为(None,20,20,1),y_valid的shape(None,1500)
_, _, _, _, x_valid, tv, rv, av = dc.data_chuli(2000, 200)
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
    return x * tf.nn.relu6(x + 3) / 6


def Bneck(x, kernel_size, input_size, expand_size, output_size, activation, stride, use_se):
    out = keras.layers.Conv2D(filters=expand_size, kernel_size=1, strides=1, use_bias=False)(x)
    out = keras.layers.BatchNormalization()(out, training=False)
    out = activation(out)
    out = keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding='same')(out)
    out = keras.layers.BatchNormalization()(out, training=False)
    if use_se:
        out = se_block(out)
    out = keras.layers.Conv2D(filters=output_size, kernel_size=1, strides=1, padding='same')(out)
    out = keras.layers.BatchNormalization()(out, training=False)
    if stride == 1 and input_size == output_size:
        short_cut = keras.layers.Conv2D(filters=output_size, kernel_size=1, strides=1, padding='same')(x)
        out = keras.layers.Add()([out, short_cut])
    return out


# mobilenet_v3模型
def MobileNetv3(num_classes=1500, input_shape=(20, 20, 1)):
    x = keras.Input(shape=input_shape)
    out = keras.layers.Conv2D(16, 3, strides=2, padding='same', use_bias=False)(x)
    out = keras.layers.BatchNormalization()(out, training=False)
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
    out = keras.layers.BatchNormalization()(out, training=False)
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
model.load_weights('./mobilenet_callbacks/multilevel_model.h5')
y_predicted = model(x_valid).numpy()

# 计算loss 欧氏距离的平方
metric = keras.metrics.MeanSquaredError()
metric(y_predicted, y_valid)
print("----------------------------------------------------------------------------")
print("loss为: ", metric.result().numpy())


# 画出预测的频谱和实际频谱之间的对比图，一次画5组频谱之间的对比图
# 频谱图是透射谱，反射谱和吸收谱的拼接图
def plot_learning_curves(y_predicted, y_valid):
    fig, ax = plt.subplots(2, 5, figsize=(50, 20))
    lm = np.array(range(0, 1500, 1))
    for k in range(5):
        ax[0, k].cla()
        ax[0, k].plot(lm, y_predicted[k])

    for l in range(5):
        ax[1, l].cla()
        ax[1, l].plot(lm, y_valid[l])

    plt.show()


plot_learning_curves(y_predicted[:5], y_valid[:5])
