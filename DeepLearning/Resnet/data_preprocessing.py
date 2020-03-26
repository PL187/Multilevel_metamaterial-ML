import numpy as np

# 用于从文件夹中读入训练数据，并进行预处理 。
# train_nums表示用于训练的样本数，valid_nums表示用于测试的样本数。
def data_chuli(train_nums, valid_nums):
    transmission_data_set2 = []
    transmission_data2 = []
    matrix_data2 = []
    matrix_data_set2 = []

    # 读取用于训练的二维码结构参数
    for i in range(train_nums):
        f = open("data/matrix/ma_" + str(i + 1) + ".txt", 'r')
        for line in f:
            list1 = line.strip('\n').split('\t')
            list1 = list(map(float, list1))
            matrix_data2.append(list1)
        matrix_data_set2.append(matrix_data2)
        matrix_data2 = []
    matrix_train = np.array(matrix_data_set2).reshape(-1, 20, 20, 1)

    # 读取用于测试的二维码结构参数
    matrix_data_set2 = []
    for i in range(valid_nums):
        f = open("data/matrix/ma_" + str(i + train_nums + 1) + ".txt", 'r')
        for line in f:
            list1 = line.strip('\n').split('\t')
            list1 = list(map(float, list1))
            matrix_data2.append(list1)
        matrix_data_set2.append(matrix_data2)
        matrix_data2 = []
    matrix_valid = np.array(matrix_data_set2).reshape(-1, 20, 20, 1)

    # 读取用于训练的透射谱
    for i in range(train_nums):
        f = open("data/transmission/tr_" + str(i + 1) + ".txt", 'r')
        line = f.readline()
        list2 = line.strip('\n').split('\t')
        list2 = list(map(float, list2))
        transmission_data2.append(list2)
        transmission_data_set2.append(transmission_data2)
        transmission_data2 = []
    transmission_train = -np.array(transmission_data_set2).reshape(-1, 500)

    # 读取用于测试的透射谱
    transmission_data_set2 = []
    for i in range(valid_nums):
        f = open("data/transmission/tr_" + str(i + train_nums + 1) + ".txt", 'r')
        line = f.readline()
        list2 = line.strip('\n').split('\t')
        list2 = list(map(float, list2))
        transmission_data2.append(list2)
        transmission_data_set2.append(transmission_data2)
        transmission_data2 = []
    transmission_valid = -np.array(transmission_data_set2).reshape(-1, 500)

    # 读取用于训练的反射谱
    transmission_data_set2 = []
    for i in range(train_nums):
        f = open("data/reflection/rf_" + str(i + 1) + ".txt", 'r')
        line = f.readline()
        list2 = line.strip('\n').split('\t')
        list2 = list(map(float, list2))
        transmission_data2.append(list2)
        transmission_data_set2.append(transmission_data2)
        transmission_data2 = []
    reflection_train = np.array(transmission_data_set2).reshape(-1, 500)

    # 读取用于测试的反射谱
    transmission_data_set2 = []
    for i in range(valid_nums):
        f = open("data/reflection/rf_" + str(i + train_nums + 1) + ".txt", 'r')
        line = f.readline()
        list2 = line.strip('\n').split('\t')
        list2 = list(map(float, list2))
        transmission_data2.append(list2)
        transmission_data_set2.append(transmission_data2)
        transmission_data2 = []
    reflection_valid = np.array(transmission_data_set2).reshape(-1, 500)

    # 读取用于训练的吸收谱
    transmission_data_set2 = []
    for i in range(train_nums):
        f = open("data/absorption/x_" + str(i + 1) + ".txt", 'r')
        line = f.readline()
        list2 = line.strip('\n').split('\t')
        list2 = list(map(float, list2))
        transmission_data2.append(list2)
        transmission_data_set2.append(transmission_data2)
        transmission_data2 = []
    absorption_train = np.array(transmission_data_set2).reshape(-1, 500)

    # 读取用于测试的吸收谱
    transmission_data_set2 = []
    for i in range(valid_nums):
        f = open("data/absorption/x_" + str(i + train_nums + 1) + ".txt", 'r')
        line = f.readline()
        list2 = line.strip('\n').split('\t')
        list2 = list(map(float, list2))
        transmission_data2.append(list2)
        transmission_data_set2.append(transmission_data2)
        transmission_data2 = []
    absorption_valid = np.array(transmission_data_set2).reshape(-1, 500)

    return matrix_train, transmission_train, reflection_train, absorption_train, matrix_valid, transmission_valid, reflection_valid, absorption_valid
