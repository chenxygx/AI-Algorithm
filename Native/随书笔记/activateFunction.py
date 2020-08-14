import numpy as np


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def identity(x):
    return x
