import matplotlib.pyplot as plt
import numpy as np
import data_preprocessing as dc
from tensorflow import keras

_, _, _, _, x_valid, tv, rv, av = dc.data_chuli(2000, 200)
y_valid = np.concatenate((tv, rv, av), axis=1)


# VGG网络
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


# 创建VGG网络
model = vggnet()
model.build(input_shape=(None, 20, 20, 1))
model.load_weights('./vggnet_callbacks/multilevel_model.h5')
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
