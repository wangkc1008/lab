"""
created by PyCharm
date: 2021/4/14
time: 17:16
user: wkc
"""
import torch

import pandas as pd
import numpy as np
import os
import shap
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
#plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
#plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#import matplotlib as mpl
plt.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
plt.rcParams['font.size'] = 12  # 字体大小
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


model = torch.load('./model/result_ALL_20210420_214233.pt')

def data_load(data):
    X, y = np.array(data.drop(labels=['label'], axis=1)), np.array(data['label'].apply(int))
    print('原数据X_size:%s, y_size:%s' % (X.shape, y.shape))
    print('原数据y分布', Counter(y))
    X_text, X_digital = X[:, 0:200], X[:, 768:]
    print('文本数据', X_text.shape)
    print('数值数据', X_digital.shape)
    X_sub = np.hstack((X_text, X_digital))
    print('分割后X_size:%s, y_size:%s' % (X_sub.shape, y.shape))

    SMO = SMOTE(random_state=666)
    X_res, y_res = SMO.fit_resample(X_sub, y)
    print('插值后y分布', Counter(y_res))

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=np.random.seed())
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    return X_train, X_test


def batch_predict(data, model=model):
    """
    model: pytorch训练的模型, **这里需要有默认的模型**
    data: 需要预测的数据
    """
    X_text = torch.from_numpy(data[:, 0:200]).float()
    X_digital = torch.from_numpy(data[:, 200:]).float()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X_text = X_text.to(device)
    X_digital = X_digital.to(device)
    logits = model(X_text, X_digital)
    probs = torch.nn.functional.softmax(logits, dim=1)
    return torch.argmax(probs, dim=1).detach().cpu().numpy()
#    return probs.detach().cpu().numpy()


def img_plot(X_test):
    explainer = shap.KernelExplainer(batch_predict, X_test)
    print(explainer.expected_value)

    # shap_values = explainer.shap_values(X_test[5])
    # print(shap_values)
    #
    # shap.force_plot(base_value=explainer.expected_value[1],
    #                 shap_values=shap_values[1],
    #                 feature_names=np.concatenate([data.columns.values[:50], data.columns.values[768:-1]]),
    #                 features=X_test[5],
    #                 matplotlib=True
    #                 )
    shap_values = explainer.shap_values(X_test)
    print(shap_values)
    shap.summary_plot(shap_values=shap_values,
                      features=X_test,
                      feature_names=np.concatenate([data.columns.values[:200], data.columns.values[768:-1]]),
                      plot_type='dot',
                      show=False,
                      plot_size=[23, 10]
                      )
    plt.savefig('./result_200_dot_all.png')


if __name__ == '__main__':
    shap.initjs()
    data = pd.read_csv('./total_data_xinjiao_20210125_ANSI - pinyin.csv', encoding='gbk')
    print('原数据size', data.shape)
    X_train, X_test = data_load(data)
    img_plot(X_test)
