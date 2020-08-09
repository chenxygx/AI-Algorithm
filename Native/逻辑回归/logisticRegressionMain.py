import numpy as np

from 逻辑回归.logisticRegression import lr_train_bgd, predict
from 逻辑回归.logisticRegressionUtils import load_data_train, load_data_test, save_model, load_weight, save_result


def train():
    # 导入训练数据
    print("加载数据中")
    feature, label = load_data_train("../Data/LogisticRegression.txt")
    # 训练模型，计算权重和偏置
    print("训练模型中")
    maxCycle = 1000
    alpha = 0.01
    w = lr_train_bgd(feature, label, maxCycle, alpha)
    # 保存最终模型
    print("保存模型中")
    save_model("../Result/LogisticRegressionWeights", w)
    print("保存完成")


def test():
    # 导入训练模型
    print("导入训练模型")
    w = load_weight("../Result/LogisticRegressionWeights")
    n = np.shape(w)[1]
    # 导入测试数据
    print("导入测试数据")
    test_data = load_data_test("../Data/LogisticRegression_test_data", n)
    # 对测试数据进行预测
    print("数据预测")
    h = predict(test_data, w)
    # 保存最终结果
    print("结果保存")
    save_result("../Result/LogisticRegressionResult", h)


if __name__ == "__main__":
    train()
    print("训练结束-开始测试")
    test()
