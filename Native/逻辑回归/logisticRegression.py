import numpy as np
from 逻辑回归.sigmoid import sig


def lr_train_bgd(feature, label, maxCycle, alpha):
    """
    梯度下降训练逻辑回归模型
    :param feature: 特征
    :param label:标签
    :param maxCycle: 最大迭代次数
    :param alpha: 学习率
    :return: 权重
    """
    # 查看特征维度
    n = np.shape(feature)[1]
    # 创建权重矩阵，初始化0
    w = np.mat(np.ones((n, 1)))
    # 标准正态分布矩阵
    # w = np.random.normal(size=(n, 1))
    i = 0
    # 限制最大迭代次数范围
    while i <= maxCycle:
        i += 1
        # 计算Sigmoid值，这里是用特征+权重的方式
        h = sig(feature * w)
        err = label - h
        if i % 100 == 0:
            print("当前迭代次数=%s,误差比例=%s" % (str(i), str(error_rate(h, label))))
        # 权重修正
        w = w + alpha * feature.T * err
    return w


def error_rate(h, label):
    """
    计算当前的损失函数值
    :param h: 预测值
    :param label: 实际值
    :return:
    """
    m = np.shape(h)[0]
    sum_err = 0.0
    for i in range(m):
        if h[i, 0] > 0 and (1 - h[i, 0]) > 0:
            sum_err -= (label[i, 0] * np.log(h[i, 0]) + (1 - label[i, 0]) * np.log(1 - h[i, 0]))
        else:
            sum_err -= 0
    return sum_err / m


def predict(data, w):
    """
    对测试数据进行测试
    :param data:测试数据的特征
    :param w:模型的参数
    :return:预测结果
    """
    h = sig(data * w.T)
    m = np.shape(h)[0]
    for i in range(m):
        if h[i, 0] < 0.5:
            h[i, 0] = 0.0
        else:
            h[i, 0] = 1.0
    return h
