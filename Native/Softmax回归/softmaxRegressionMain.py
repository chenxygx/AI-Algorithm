from Softmax回归.softmaxRegressionUtils import load_data, load_weights, \
    load_data_test, predict, save_model, save_result
from Softmax回归.softmaxRegression import gradient_ascent


def train():
    input_file = "../Data/SoftInput.txt"
    # 导入训练数据
    print("导入训练数据")
    feature, label, k = load_data(input_file)
    # 训练Softmax模型
    print("训练Softmax模型")
    maxCycle = 50000
    alpha = 0.2
    weights = gradient_ascent(feature, label, k, maxCycle, alpha)
    # 保存模型
    print("保存模型")
    save_model("../Result/softmax_weights", weights)


def test():
    input_file = "../Result/softmax_weights"
    # 导入模型
    print("导入模型")
    w, m, n = load_weights(input_file)
    # 导入测试数据
    print("导入测试数据")
    test_data = load_data_test(4000, m)
    # 预测
    print("进行预测")
    result = predict(test_data, w)
    # 保存结果
    print("保存结果")
    save_result("../Result/softmax_result", result)


if __name__ == "__main__":
    train()
    print("预测")
    test()
