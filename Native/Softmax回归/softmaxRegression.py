import numpy as np


def gradient_ascent(feature_data, label_data, k, maxCycle, aplha):
    """
    利用梯度下降法训练Softmax模型
    :param feature_data: 特征
    :param label_data: 标签
    :param k: 类别个数
    :param maxCycle: 最大迭代次数
    :param aplha: 学习率
    :return: 权重
    """
    m, n = np.shape(feature_data)
    # 权重初始化
    weights = np.mat(np.ones((n, k)))
    i = 0
    while i <= maxCycle:
        err = np.exp(feature_data * weights)
        if i % 100 == 0:
            print("当前迭代次数=%s,误差比例=%.15f" % (str(i), float(cost(err, label_data))))
        row_sum = -err.sum(axis=1)
        row_sum = row_sum.repeat(k, axis=1)
        err = err / row_sum
        for x in range(m):
            err[x, label_data[x, 0]] += 1
        weights = weights + (aplha / m) * feature_data.T * err
        i += 1
    return weights


def cost(err, label_data):
    """
    计算损失函数值
    :param err:exp的值
    :param label_data:标签的值
    :return: 损失函数的值
    """
    m = np.shape(err)[0]
    sum_cost = 0.0
    for i in range(m):
        if err[i, label_data[i, 0]] / np.sum(err[i, :]) > 0:
            sum_cost -= np.log(err[i, label_data[i, 0]] / np.sum(err[i, :]))
        else:
            sum_cost -= 0
    return sum_cost / m
