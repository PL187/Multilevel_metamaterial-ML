import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import data_preprocessing as dc
from tensorflow import keras
from BasicBlockm_train import BasicBlock

# 读取训练集和验证集中的数据，包括二维码矩阵参数(x_train，x_valid)，
# 透射谱(tt,tv)，反射谱(rt,rv)和吸收谱(at,av)，
# 每个频谱表示为一个(1,500)的行向量,并将透射谱，
# 反射谱和吸收谱连接起来成为一个（1,1500）的行向量(y_train，y_valid)。
# x_train和x_valid的shape为(None,20,20,1),y_train和y_valid的shape(None,1500)
x_train, tt, rt, at, x_valid, tv, rv, av = dc.data_chuli(2000, 200)
y_train = np.concatenate((tt, rt, at), axis=1)
y_valid = np.concatenate((tv, rv, av), axis=1)


# resnet18模型
class ResNet(keras.models.Model):
    def __init__(self, layer_dims, num_classes=1500):
        super(ResNet, self).__init__()
        # 进入残差块前的预处理
        self.stem1 = keras.layers.Conv2D(64, (3, 3), strides=(1, 1))
        self.stem2 = keras.layers.BatchNormalization()
        self.stem3 = keras.models.Sequential([
            keras.layers.Activation('relu'),
            keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
        ])
        # 定义残差块
        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)
        # 最后经过池化层和全连接层输出预测的频谱
        self.avgpool = keras.layers.GlobalAveragePooling2D()
        self.fc = keras.layers.Dense(num_classes, activation='sigmoid')

    def call(self, input, training=True):
        # training用于设置Batchnormalization的相关参数
        x = self.stem1(input)
        x = self.stem2(x, training=training)
        x = self.stem3(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x

    # 用于构造残差块，filter_num表示该残差块输出数据的通道数，(None,x,x,filter_num)
    # blocks表明该残差块由多少个基本的残差单元构成，这里的残差单元是指BasicBlockm_test.py中
    # 定义的。
    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = keras.Sequential()
        res_blocks.add(BasicBlock(filter_num, stride))
        for pre in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))
        return res_blocks


def resnet():
    return ResNet([2, 2, 2, 2])


# 创建resnet18网络
model = resnet()
model.build(input_shape=(None, 20, 20, 1))
model.summary()
# 设置loss函数和optimizer
model.compile(loss="mean_squared_error", optimizer='adam')

# 用于设置训练出的模型参数的保存路径和其它相关参数
logdir = './Resent_callbacks'
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
