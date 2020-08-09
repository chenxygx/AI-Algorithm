import numpy as np


def save_model(file_name, w0, w, v):
    """
    保存训练模型
    :param file_name:文件名
    :param w0: 偏置量
    :param w: 一次项权重
    :param v: 交叉项权重
    """
    f = open(file_name, "w")
    # 保存
    f.write(str(w0) + "\n")
    # 保存一次项权重
    w_array = []
    m = np.shape(w)[0]
    for i in range(m):
        w_array.append(str(w[i, 0]))
    f.write("\t".join(w_array) + "\n")
    # 保存交叉项权重
    m1, n1 = np.shape(v)
    for i in range(m1):
        v_tmp = []
        for j in range(n1):
            v_tmp.append(str(v[i, j]))
        f.write("\t".join(v_tmp) + "\n")
    f.close()


def loadDataSetTrain(data):
    """导入训练数据
    input:  data(string)训练数据
    output: dataMat(list)特征
            labelMat(list)标签
    """
    dataMat = []
    labelMat = []
    fr = open(data)  # 打开文件
    for line in fr.readlines():
        lines = line.strip().split("\t")
        lineArr = []

        for i in range(len(lines) - 1):
            lineArr.append(float(lines[i]))
        dataMat.append(lineArr)

        labelMat.append(float(lines[-1]) * 2 - 1)  # 转换成{-1,1}
    fr.close()
    return dataMat, labelMat


def loadDataSetTest(data):
    """导入测试数据集
    input:  data(string)测试数据
    output: dataMat(list)特征
    """
    dataMat = []
    fr = open(data)  # 打开文件
    for line in fr.readlines():
        lines = line.strip().split("\t")
        lineArr = []

        for i in range(len(lines)):
            lineArr.append(float(lines[i]))
        dataMat.append(lineArr)

    fr.close()
    return dataMat


def loadModel(model_file):
    """
    导入FM模型
    :param model_file:模型
    :return: FM模型参数
    """
    f = open(model_file)
    line_index = 0
    w0 = 0.0
    w = []
    v = []
    for line in f.readlines():
        lines = line.strip().split("\t")
        if line_index == 0:
            w0 = float(lines[0].strip())
        elif line_index == 1:
            for x in lines:
                w.append(float(x.strip()))
        else:
            v_tmp = []
            for x in lines:
                v_tmp.append(float(x.strip()))
            v.append(v_tmp)
        line_index += 1
    f.close()
    return w0, np.mat(w).T, np.mat(v)


def save_result(file_name, result):
    """
    保存最终预测结果
    :param file_name:保存文件名
    :param result: 数据预测结果
    """
    f = open(file_name, "w")
    f.write("\n".join(str(x) for x in result))
    f.close()
