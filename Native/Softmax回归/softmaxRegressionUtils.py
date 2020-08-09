import numpy as np
import random as rd


def load_data(input_file):
    """
    导入函数
    :param input_file:样本位置
    :return:特征、标签、类别个数
    """
    f = open(input_file)
    feature_data = []
    label_data = []
    for line in f.readlines():
        feature_tmp = [1]
        lines = line.strip().split("\t")
        for i in range(len(lines) - 1):
            feature_tmp.append(float(lines[i]))
        label_data.append(int(lines[-1]))
        feature_data.append(feature_tmp)
    f.close()
    return np.mat(feature_data), np.mat(label_data).T, len(set(label_data))


def load_data_test(num, m):
    """
    导入测试数据
    :param num:生成测试样本个数
    :param m: 样本维数
    :return: 生成测试样本
    """
    testDataSet = np.mat(np.ones((num, m)))
    for i in range(num):
        # 生成随机数
        testDataSet[i, 1] = rd.random() * 6 - 3
        testDataSet[i, 2] = rd.random() * 15
    return testDataSet


def load_weights(weights_path):
    """
    导入训练好的Softmax模型
    :param weights_path:权重位置
    :return: 权重矩阵，行数，列数
    """
    f = open(weights_path)
    w = []
    for line in f.readlines():
        w_tmp = []
        lines = line.strip().split("\t")
        for x in lines:
            w_tmp.append(float(x))
        w.append(w_tmp)
    f.close()
    weights = np.mat(w)
    m, n = np.shape(weights)
    return weights, m, n


def save_model(file_name, weights):
    """
    保存最终模型
    :param file_name:文件名
    :param weights: softmax模型
    """
    f_w = open(file_name, "w")
    m, n = np.shape(weights)
    for i in range(m):
        w_tmp = []
        for j in range(n):
            w_tmp.append(str(weights[i, j]))
        f_w.write("\t".join(w_tmp) + "\n")
    f_w.close()


def save_result(file_name, result):
    """
    保存最终预测的结果
    :param file_name:文件名
    :param result: 预测结果
    """
    f_result = open(file_name, "w")
    m = np.shape(result)[0]
    for i in range(m):
        f_result.write(str(result[i, 0]) + "\n")
    f_result.close()


def predict(test_data, weights):
    """
    利用训练好的Softmax模型堆测试数据进行预测
    :param test_data: 测试数据特征
    :param weights: 模型权重
    :return: 所属类别
    """
    h = test_data * weights
    return h.argmax(axis=1)
