from 因子分解机.FactorizationMachineUtils import loadDataSetTrain, loadModel, save_result, save_model, loadDataSetTest
from 因子分解机.FactorizationMachine import stoc_grad_ascent, getAccuracy, get_prediction
import numpy as np


def train():
    print("导入训练数据")
    dataTrain, labelTrain = loadDataSetTrain("../Data/FM_data.txt")
    print("训练模型")
    w0, w, v = stoc_grad_ascent(np.mat(dataTrain), labelTrain, 3, 10000, 0.01)
    # 得到准确性
    predict_result = get_prediction(np.mat(dataTrain), w0, w, v)
    print("训练误差%s" % (1 - getAccuracy(predict_result, labelTrain)))
    print("保存")
    save_model("../Result/FM_weights", w0, w, v)


def test():
    print("导入测试数据")
    dataTest = loadDataSetTest("../Data/FM_test_data.txt")
    print("导入FM模型")
    w0, w, v = loadModel("../Result/FM_weights")
    print("预测")
    result = get_prediction(dataTest, w0, w, v)
    print("保存")
    save_result("../Result/FM_result", result)


if __name__ == "__main__":
    train()
    print("开始测试")
    test()
