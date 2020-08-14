import numpy as np
import pickle
from 随书笔记.activateFunction import sigmoid, softmax
from 识别MNIST.mnist import load_mnist
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False


def load():
    (x_train, l_train), (x_test, l_test) \
        = load_mnist(normalize=False, flatten=False, one_hot_label=False)

    print(x_train.shape)
    print(l_train.shape)
    print(x_test.shape)
    print(l_test.shape)

    img = x_test[6:10]
    label = l_test[6:10]

    for i in range(0, 4):
        plt.subplot(2, 2, (i + 1))
        plt.imshow(img[i].reshape(28, 28))
        plt.title(label[i])
    plt.show()


def read():
    x, t = get_data()
    network = init_network()
    accuracy_cnt = 0
    errors = []
    errors_value = []
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y)
        if p == t[i]:
            accuracy_cnt += 1
        else:
            errors.append(i)
            errors_value.append(p)
    print("精度：" + str(float(accuracy_cnt) / len(x)))
    show_error(x, t, errors, errors_value)


def show_error(x, t, errors, errors_value):
    for i in range(0, 36):
        plt.subplot(6, 6, (i + 1))
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.imshow(x[errors[i]].reshape(28, 28))
        plt.title("实际%s-预测%s" % (str(t[errors[i]]), str(errors_value[i])))

    plt.show()


def get_data():
    (X_train, t_train), (X_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return X_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y


if __name__ == "__main__":
    # load()
    read()
