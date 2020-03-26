import tensorflow as tf
from tensorflow import keras

# 用于构建训练时使用的resnet18网络
class BasicBlock(keras.layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = keras.layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = keras.layers.BatchNormalization()
        self.relu = keras.layers.Activation('relu')

        self.conv2 = keras.layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = keras.layers.BatchNormalization()

        if stride != 1:
            self.downsample = keras.models.Sequential()
            self.downsample.add(keras.layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x: x

    def call(self, input, training=True):
        out = self.conv1(input)
        out = self.bn1(out,training=training)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out,training=training)

        identity = self.downsample(input)
        output = keras.layers.add([out, identity])
        output = tf.nn.relu(output)
        return output
