# encoding: utf-8
# -------------------------------------------------------------------------------
# Name:         TrAdaboost
# Description:
# Author:       xueyunfei
# Date:         2019/11/10
# -------------------------------------------------------------------------------
import os
import sys

import numpy as np
from sklearn import svm, tree, linear_model
from sklearn.model_selection import train_test_split
import datetime
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics


def trainWeightedClassifier(data_training, labels_training, weights):
    # model = svm.LinearSVC(verbose=0, max_iter=5000)
    # model = linear_model.LogisticRegression()
    model = tree.DecisionTreeClassifier(criterion="gini", splitter="random", max_depth=5)
    model.fit(data_training, labels_training, sample_weight=weights)
    return model

# 返回 文件列表Filelist,包含文件名（完整路径）
def get_filelist(dir, Filelist):
    newDir = dir
    if os.path.isfile(dir):
        Filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            get_filelist(newDir, Filelist)
    return Filelist


# 读入原始数据
def handle_data_function(train_data_name_list):
    i = 1
    try:
        # 将数据格式从np.ndarray转换为pd.dataframe方便机器学习库
        train_data = pd.read_csv(train_data_name_list[0])

        while i < len(train_data_name_list):
            temp_data = pd.read_csv(train_data_name_list[i])
            train_data = train_data.append(temp_data)
            i += 1

        print("data loaded.")
        train_data.drop(train_data.columns[0], axis=1, inplace=True)
        data_label_0 = train_data.pop("41769")
        data_label = np.abs(data_label_0[:] - 1)
        data_label_1 = train_data.pop("41770")
        data_label_2 = train_data.pop("41771")
        data_label_3 = train_data.pop("41772")
        data_label_former = train_data.pop("41773")
        return [np.asarray(train_data), np.asarray(data_label)]
    except Exception as hf_e:
        print("data handle error：")
        print(train_data_name_list[i])
        print(hf_e)


def predict_final(model_list, beta_t_list, data, label, threshold):
    res = 1
    for i in range(len(model_list)):
        h_t = model_list[i].predict([data])[0]
        res = res / beta_t_list[i] ** h_t
    if res >= threshold:
        label_predict = 1
    else:
        label_predict = 0
    if label_predict == label:
        return [1, label_predict]
    else:
        return [0, label_predict]


def error_calculate(model, training_data_target, training_labels_target, weights):
    total = np.sum(weights)
    labels_predict = model.predict(training_data_target)
    error = np.sum(weights / total * np.abs(labels_predict - training_labels_target))
    return error

def KLsandu(P, Q):
    p1 = P.sum() / P.shape[0]
    q1 = Q.sum() / Q.shape[0]
    P01 = [1 - p1, p1]
    Q01 = [1 - q1, q1]
    kl = 0
    for i in range(2):
        kl += P01[i] * np.log(P01[i] / Q01[i])
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ': ' + '数据集KL散度: ', kl)


def TrAdaBoost(N=100):
    # 数据处理
    # 辅助域数据路径
    # T_data_path_name = str(sys.argv[1])
    # T_name_list = get_filelist(T_data_path_name, [])
    # # 去除2018年数据
    # T_name_list_split_2018 = T_name_list[0:48]
    # training_data_source, training_labels_source = handle_data_function(T_name_list_split_2018)  # 与新数据分布不相同的数据集
    # # 源域有少量标签训练数据路径
    # S_data_path_name = str(sys.argv[2])
    # S_name_list = get_filelist(S_data_path_name, [])
    # data_target, labels_target = handle_data_function(S_name_list)  # 与新数据分布相同的数据集

    T_data_path_name = str(sys.argv[1])
    T_name_list = get_filelist(T_data_path_name, [])
    datas, labels = handle_data_function(T_name_list)
    training_data_source, data_target, training_labels_source, labels_target = train_test_split(datas, labels, test_size=0.4, random_state=33)

    KLsandu(training_labels_source, labels_target)
    print("与新数据分布不同的数据集大小", training_data_source.shape)
    print("与新数据分布相同的数据集大小", data_target.shape)
    imputer_S = preprocessing.Imputer(missing_values=-1000, strategy='most_frequent', axis=0)
    imputer_S.fit(data_target, labels_target)

    training_data_source = imputer_S.transform(training_data_source)
    data_target = imputer_S.transform(data_target)
    print("数据分割")
    training_data_target, test_data_target, training_labels_target, test_labels_target = train_test_split(data_target,
                                                                                                          labels_target,
                                                                                                          test_size=0.25)
    print("与新数据分布相同的数据集用于训练的数据集大小", training_data_target.shape)
    print("与新数据分布不同的数据集用于测试的数据集大小", test_data_target.shape)
    # 合成训练数据
    training_data = np.r_[training_data_source, training_data_target]
    training_labels = np.r_[training_labels_source, training_labels_target]

    # 对比试验 baseline方法
    # svm_0 = svm.LinearSVC(verbose=0, max_iter=5000)
    svm_0 = trainWeightedClassifier(training_data, training_labels, None)
    print('——————————————————————————————————————————————')
    print('训练数据用目标域和源域的情况')
    print('The mean accuracy is ' + str(svm_0.score(test_data_target, test_labels_target)))
    print('The error rate is ' + str(1 - svm_0.score(test_data_target, test_labels_target)))
    # svm_y_score_0 = svm_0.decision_function(test_data_target)  # 得到预测的损失值
    svm_y_score_0 = svm_0.predict_proba(test_data_target)
    svm_fpr_0, svm_tpr_0, svm_threasholds_0 = metrics.roc_curve(test_labels_target,
                                                                svm_y_score_0[:, 1], pos_label=1)  # 计算ROC的值,svm_threasholds为阈值
    svm_auc_0 = metrics.auc(svm_fpr_0, svm_tpr_0)
    print('The auc is ', svm_auc_0)

    # svm_1 = svm.LinearSVC(verbose=0, max_iter=5000)
    svm_1 = trainWeightedClassifier(training_data_target, training_labels_target, None)
    print('——————————————————————————————————————————————')
    print('训练数据仅用目标域的情况')
    print('The mean accuracy is ' + str(svm_1.score(test_data_target, test_labels_target)))
    print('The error rate is ' + str(1 - svm_1.score(test_data_target, test_labels_target)))
    # svm_y_score_1 = svm_1.decision_function(test_data_target)  # 得到预测的损失值
    svm_y_score_1 = svm_1.predict_proba(test_data_target)
    svm_fpr_1, svm_tpr_1, svm_threasholds_1 = metrics.roc_curve(test_labels_target,
                                                                svm_y_score_1[:, 1], pos_label=1) # 计算ROC的值,svm_threasholds为阈值
    svm_auc_1 = metrics.auc(svm_fpr_1, svm_tpr_1)
    print('The auc is ', svm_auc_1)

    # svm_2 = svm.LinearSVC(verbose=0, max_iter=5000)
    svm_2 = trainWeightedClassifier(training_data_source, training_labels_source, None)
    print('——————————————————————————————————————————————')
    print('训练数据仅用源域的情况')
    print('The mean accuracy is ' + str(svm_2.score(test_data_target, test_labels_target)))
    print('The error rate is ' + str(1 - svm_2.score(test_data_target, test_labels_target)))
    # svm_y_score_2 = svm_2.decision_function(test_data_target)  # 得到预测的损失值
    svm_y_score_2 = svm_2.predict_proba(test_data_target)
    svm_fpr_2, svm_tpr_2, svm_threasholds_2 = metrics.roc_curve(test_labels_target,
                                                                svm_y_score_2[:, 1], pos_label=1)  # 计算ROC的值,svm_threasholds为阈值
    svm_auc_2 = metrics.auc(svm_fpr_2, svm_tpr_2)
    print('The auc is ', svm_auc_2)
    print('——————————————————————————————————————————————')

    # 训练主循环
    n_source = len(training_data_source)
    m_target = len(training_data_target)
    # 初始化权重
    weights = np.concatenate((np.ones(n_source) / n_source, np.ones(m_target) / m_target))
    beta_t_list = list()
    model_list = list()
    beta = 1.0 / (1.0 + np.sqrt(2 * np.log(n_source) / N))
    for t in range(N):
        p_t = weights / sum(weights)
        model = trainWeightedClassifier(training_data, training_labels, p_t)

        # 加权的错误率
        error_self = error_calculate(model, training_data_target, training_labels_target, weights[-m_target:])

        # 计算参数
        if error_self > 0.5:
            error_self = 0.5
        elif error_self == 0:
            N = t  # 防止过拟合
            break

        beta_t = error_self / (1 - error_self)
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ': ' + '第' + str(t) + '轮的加权错误率为: ', error_self)

        # 源域
        mm = 0
        for i in range(n_source):
            if model.predict([training_data_source[i]])[0] != training_labels_source[i]:
                weights[i] = weights[i] * beta
                mm += 1
        print("源域域预测值和真实值不相等的样本个数:", mm)
        # 目标域
        nn = 0
        for i in range(m_target):
            if model.predict([training_data_target[i]])[0] != training_labels_target[i]:
                weights[i + n_source] = weights[i + n_source] / beta_t
                nn += 1

        print("目标域预测值和真实值不相等的样本个数:", nn)
        # 记录当前的参数
        beta_t_list += [beta_t]
        model_list += [model]

    # 测试最后输出的模型
    count_accu = 0
    index_half = int(np.ceil(N / 2))
    threshold = 1
    for beta_t in beta_t_list[index_half:]:
        threshold = threshold / np.sqrt(beta_t)

    predict = np.zeros([len(test_data_target)])
    for i in range(len(test_data_target)):
        result = predict_final(model_list[index_half:], beta_t_list[index_half:], test_data_target[i],
                               test_labels_target[i], threshold)
        count_accu += result[0]
        predict[i] = result[1]
    error_final = 1.0 - count_accu / float(len(test_data_target))
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ': ' + '模型最后的准确率为: ' + str(1 - error_final))
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ': ' + '模型最后的错误率为: ' + str(error_final))
    print(metrics.classification_report(test_labels_target, predict, target_names=['NOAKI', 'AKI']))

    fpr, tpr, thresholds = metrics.roc_curve(y_true=test_labels_target, y_score=predict, pos_label=1)
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ': ' + '模型最后的auc为: ', metrics.auc(fpr, tpr))
    print("N", N)


if __name__ == '__main__':
    TrAdaBoost(int(sys.argv[2]))
