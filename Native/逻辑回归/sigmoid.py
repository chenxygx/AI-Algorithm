import numpy as np


def sig(x):
    """
    Sigmoid函数
    :param x: 待分配参数
    :return: Sigmoid值
    """
    return 1.0 / (1 + np.exp(-x))
