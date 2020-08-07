import numpy as np


def load_data_train(file_name):
    """
    加载数据文件
    :param file_name: 文件位置
    :return: feature_data特征、label_data标签
    """
    # 打开文件，文件前两列是训练数据，后一列是标签
    f = open(file_name)
    feature_data = []
    label_data = []
    for line in f.readlines():
        feature_tmp = []
        label_tmp = []
        # 空格分隔
        lines = line.strip().split("\t")
        # 偏置项
        feature_tmp.append(1)
        for i in range(len(lines) - 1):
            feature_tmp.append(float(lines[i]))
        label_tmp.append(float(lines[-1]))
        feature_data.append(feature_tmp)
        label_data.append(label_tmp)
    f.close()
    return np.mat(feature_data), np.mat(label_data)


def load_data_test(file_name, n):
    """
    导入数据
    :param file_name:测试集位置
    :param n:特征个数
    :return:测试集特征
    """
    f = open(file_name)
    feature_data = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split("\t")
        if len(lines) != (n - 1):
            continue
        feature_tmp.append(1)
        for x in lines:
            feature_tmp.append(float(x))
        feature_data.append(feature_tmp)
    f.close()
    return np.mat(feature_data)


def load_weight(w):
    """
    导入训练模型
    :param w:权重所在位置文件
    :return:权重矩阵
    """
    f = open(w)
    w = []
    for line in f.readlines():
        lines = line.strip().split("\t")
        w_tmp = []
        for x in lines:
            w_tmp.append(float(x))
        w.append(w_tmp)
    f.close()
    return np.mat(w)


def save_model(file_name, w):
    """
    保存最终模型
    :param file_name:模型文件名
    :param w: LR模型权重
    """
    m = np.shape(w)[0]
    f_w = open(file_name, "w")
    w_array = []
    for i in range(m):
        w_array.append(str(w[i, 0]))
    f_w.write("\t".join(w_array))
    f_w.close()


def save_result(file_name, result):
    """
    保存最终测试结果
    :param file_name:文件名
    :param result: 预测结果
    """
    m = np.shape(result)[0]
    tmp = []
    for i in range(m):
        tmp.append(str(result[i, 0]))
    f_result = open(file_name, "w")
    f_result.write("\t".join(tmp))
    f_result.close()
