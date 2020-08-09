import numpy as np
from 逻辑回归.sigmoid import sig
from random import normalvariate


def stoc_grad_ascent(data_matrix, class_labels, k, max_iter, alpha):
    """
    随机梯度下降训练FM模型
    :param data_matrix:特征
    :param class_labels: 标签
    :param k: v的维数
    :param max_iter:最大迭代次数
    :param alpha: 学习率
    :return: 权重
    """
    m, n = np.shape(data_matrix)
    # 初始化参数
    w = np.zeros((n, 1))
    # 偏置量
    w0 = 0
    # 初始化v
    v = initialize_v(n, k)
    # 训练
    for it in range(max_iter):
        for x in range(m):
            iter_1 = data_matrix[x] * v
            # multiply对应元素相乘
            iter_2 = np.multiply(data_matrix[x], data_matrix[x]) * np.multiply(v, v)
            # 完成交叉项
            interaction = np.sum(np.multiply(iter_1, iter_1) - iter_2) / 2.
            # 计算预测输出
            p = w0 + data_matrix[x] * w + interaction
            loss = sig(class_labels[x] * p[0, 0]) - 1
            w0 = w0 - alpha * loss * class_labels[x]
            for i in range(n):
                if data_matrix[x, i] != 0:
                    w[i, 0] = w[i, 0] - alpha * loss * class_labels[x] * data_matrix[x, i]
                    for j in range(k):
                        v[i, j] = v[i, j] - alpha * loss * class_labels[x] * \
                                  (data_matrix[x, i] * iter_1[0, j] - v[i, j] *
                                   data_matrix[x, i] * data_matrix[x, i])
        # 计算损失函数
        if it % 1000 == 0:
            print("迭代次数：%s，误差比例：%.15f" %
                  (it, get_cost(get_prediction(np.mat(data_matrix), w0, w, v), class_labels)))
    return w0, w, v


def initialize_v(n, k):
    """
    初始化交叉项
    :param n: 特征个数
    :param k: FM模型的度
    :return: 交叉项的系数权重
    """
    v = np.mat(np.zeros((n, k)))
    for i in range(n):
        for j in range(k):
            v[i, j] = normalvariate(0, 0.2)
    return v


def get_cost(predict, class_labels):
    """
    计算预测准确性
    :param predict:预测值
    :param class_labels: 标签
    :return: 计算损失函数的值
    """
    m = len(predict)
    error = 0.0
    for i in range(m):
        error -= np.log(sig(predict[i] * class_labels[i]))
    return error


def get_prediction(data_matrix, w0, w, v):
    """
    得到预测值
    :param data_matrix:特征
    :param w0: 常数项权重
    :param w: 一次项权重
    :param v: 交叉项权重
    :return: 预测的结果
    """
    m = np.shape(data_matrix)[0]
    result = []
    for x in range(m):
        iter_1 = data_matrix[x] * v
        iter_2 = np.multiply(data_matrix[x], data_matrix[x]) * np.multiply(v, v)
        # 完成交叉项
        interaction = np.sum(np.multiply(iter_1, iter_1) - iter_2) / 2.
        # 预测输出
        p = w0 + data_matrix[x] * w + interaction
        pre = sig(p[0, 0])
        result.append(pre)
    return result


def getAccuracy(predict, classLabels):
    """
    计算预测准确性
    :param predict:预测值
    :param classLabels: 标签
    :return: 错误率
    """
    m = len(predict)
    allItem = 0
    error = 0
    for i in range(m):
        allItem += 1
        if float(predict[i]) < 0.5 and classLabels[i] == 1.0:
            error += 1
        elif float(predict[i]) >= 0.5 and classLabels[i] == -1.0:
            error += 1
        else:
            continue
    return float(error) / allItem
