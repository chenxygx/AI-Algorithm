import numpy as np

from mnist import load_mnist


# def img_show(img):
#     pil_img = Image.fromarray(np.uint8(img))
#     pil_img.show()


def load():
    (x_train, l_train), (x_test, l_test) \
        = load_mnist(normalize=False, flatten=True, one_hot_label=False)

    print(x_train.shape)
    print(l_train.shape)
    print(x_test.shape)
    print(l_test.shape)

    img = x_train[0]
    label = l_train[0]
    print(label)

    img = img.reshape(28, 28)
    print img.shape

    # img_show(img)


if __name__ == "__main__":
    load()
