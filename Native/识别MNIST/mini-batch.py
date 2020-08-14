import numpy as np
from 识别MNIST.mnist import load_mnist


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def mini():
    (X_train, t_train), (X_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    train_size = X_train.shape[0]
    batch_size = 10
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = X_train[batch_mask]
    t_batch = t_train[batch_mask]
    print(x_batch)


if __name__ == "__main__":
    mini()
