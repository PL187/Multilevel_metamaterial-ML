import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import data_preprocessing as dc
from tensorflow import keras
from BasicBlockm_test import BasicBlock

# 拿训练好的模型进行测试的网络结构
# 读取用于测试的数据，包括二维码矩阵参数(x_valid)，透射谱(tv)，反射谱(rv)和吸收谱(av)，
# 每个频谱表示为一个(1,500)的行向量,并将透射谱，
# 反射谱和吸收谱连接起来成为一个（1,1500）的行向量(y_test)。
# x_valid的shape为(None,20,20,1),y_valid的shape(None,1500)
_, _, _, _, x_valid, tv, rv, av = dc.data_chuli(2000, 200)
y_valid = np.concatenate((tv, rv, av), axis=1)


class ResNet(keras.models.Model):
    def __init__(self, layer_dims, num_classes=1500):
        super(ResNet, self).__init__()
        self.stem1 = keras.layers.Conv2D(64, (3, 3), strides=(1, 1))
        self.stem2 = keras.layers.BatchNormalization()
        self.stem3 = keras.models.Sequential([
            keras.layers.Activation('relu'),
            keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
        ])

        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        self.avgpool = keras.layers.GlobalAveragePooling2D()
        self.fc = keras.layers.Dense(num_classes, activation='sigmoid')

    def call(self, input, training=False):
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

    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = keras.Sequential()
        res_blocks.add(BasicBlock(filter_num, stride))
        for pre in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))
        return res_blocks


def resnet():
    return ResNet([2, 2, 2, 2])


model = resnet()
model.build(input_shape=(None, 20, 20, 1))
model.load_weights('./Resent_callbacks/multilevel_model.h5')
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
