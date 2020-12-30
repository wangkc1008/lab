"""
created by PyCharm
date: 2020/9/12
time: 22:30
user: hxf
"""

from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score,roc_auc_score
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


data = pd.read_csv('C:/Users/wangkc/Desktop/胡喜风预测论文/新建文件夹/mixed_data_100_5079_replace_20200912.csv')
print('data', data.shape)
data_unstructure, data_structure = data.iloc[:, 0:200], data.iloc[:, 200:]
print('data_unstructure', data_unstructure.shape)
print('data_structure', data_structure.shape)

corr_data = data_structure.corr()['flag'].sort_values(ascending=False)
data_structure_corr = data_structure[corr_data[(corr_data > 0.1) | (corr_data < -0.1)].index]
print('data_structure_corr', data_structure_corr.shape)

new_data = pd.concat([data_unstructure, data_structure_corr], axis=1)
print('new_data', new_data.shape)
print(new_data.head())

X, y = np.array(new_data.drop(labels=['flag'], axis=1)), np.array(new_data['flag'].apply(int))
print(X.shape, y.shape)
y_hot = to_categorical(y, 2)
print(X.shape, y_hot.shape)
X_unstructure, X_structure = X[:, 0:200], X[:, 200:]
print(X_unstructure[:1, :])
print(X_structure[:1, :])

batch_size = 100
epoch = 60
accuracy_rate_1 = 0.5
accuracy_rate_2 = 0.5

def build_model():

    input1 = layers.Input(shape=(X_unstructure.shape[1],))
    input2 = layers.Input(shape=(X_structure.shape[1],))

    input1_bn = layers.BatchNormalization()(input1)
    x1_1 = layers.Dense(600, activation='tanh')(input1_bn)
    input2_bn = layers.BatchNormalization()(input2)
    x2_1 = layers.Dense(200, activation='tanh')(input2_bn)

    x_connect = tf.concat([x1_1, x2_1], 1)
    x_connect_1 = layers.Dense(600, activation='tanh')(x_connect)
    x_connect_2 = layers.Dense(200, activation='tanh')(x_connect_1)
    x_connect_2_d = layers.Dropout(0.3)(x_connect_2)
    #x_connect_3 = layers.Dense(1)(x_connect_2)
    # w = tf.reduce_mean(x_connect_3, axis=1)
    # w = tf.reshape(x_connect_3, (batch_size, 1))
    #x1_1 = x_connect_3 * x1_1 + x1_1
    #x2_1 = x_connect_3 * x2_1 + x2_1

    x1_2 = layers.Dense(600, activation='tanh', kernel_regularizer=regularizers.l2())(x_connect_2_d)
    x2_2 = layers.Dense(200, activation='tanh', kernel_regularizer=regularizers.l2())(x_connect_2_d)
    x1_2_d = layers.Dropout(0.3)(x1_2)
    x2_2_d = layers.Dropout(0.3)(x2_2)

    x1_3 = layers.Dense(600, activation='tanh', kernel_regularizer=regularizers.l2())(x1_2)
    x2_3 = layers.Dense(200, activation='tanh', kernel_regularizer=regularizers.l2())(x2_2)
    x1_3_d = layers.Dropout(0.3)(x1_3)
    x2_3_d = layers.Dropout(0.3)(x2_3)

    y1_ = layers.Dense(2, activation='softmax')(x1_3)
    y2_ = layers.Dense(2, activation='softmax')(x2_3)

    model = models.Model(inputs=[input1, input2], outputs=[y1_, y2_])
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


class Metrics(Callback):

    def __init__(self, val_x_text, val_x_digital, val_y):
        super().__init__()
        self._val_x_text = val_x_text
        self._val_x_digital = val_x_digital
        self._val_y = val_y
        self.val_text_res = []
        self.val_digital_res = []

    def on_train_begin(self, logs=None):
        self.val_text_res = []
        self.val_digital_res = []

    def on_epoch_end(self, epoch, logs=None):
        # {'loss': 15.950776100158691, 'dense_6_loss': 0.4059295058250427, 'dense_7_loss': 0.4466106593608856, 'dense_6_accuracy': 0.8135796785354614, 'dense_7_accuracy': 0.792706310749054, 'val_loss': 16.693078994750977, 'val_dense_6_loss': 0.5566453337669373, 'val_dense_7_loss': 1.1686965227127075, 'val_dense_6_accuracy': 0.7262930870056152, 'val_dense_7_accuracy': 0.693965494632721}
        val_predict = (np.asarray(self.model.predict([self._val_x_text, self._val_x_digital]))).round()
        val_predict_text = np.argmax(val_predict[0], axis=1)  # 得到验证集中文本数据的预测结果
        val_predict_digital = np.argmax(val_predict[1], axis=1)  # 得到验证集中数值数据的预测结果
        val_targ = self._val_y

        # 计算文本数据的准确率、精准率、召回率、f1值、auc
        _val_text_acc = list(logs.values())[-2]
        _val_text_f1 = f1_score(val_targ, val_predict_text)
        _val_text_recall = recall_score(val_targ, val_predict_text)
        _val_text_precision = precision_score(val_targ, val_predict_text)
        _val_text_rocauc = roc_auc_score(val_targ, val_predict_text)
        self.val_text_res.append([_val_text_acc, _val_text_precision, _val_text_recall, _val_text_f1,_val_text_rocauc])
        print('\r\n')
        print(" TEXT — val_acc: %f — val_precision: %f — val_recall: %f — val_f1: %f -val_rauc: %f" % (_val_text_acc, _val_text_precision, _val_text_recall, _val_text_f1,_val_text_rocauc))

        # 计算数值数据的准确率、精准率、召回率、f1值、auc
        _val_digital_acc = list(logs.values())[-1]
        _val_digital_f1 = f1_score(val_targ, val_predict_digital)
        _val_digital_recall = recall_score(val_targ, val_predict_digital)
        _val_digital_precision = precision_score(val_targ, val_predict_digital)
        _val_digital_rocauc = roc_auc_score(val_targ, val_predict_digital)
        self.val_digital_res.append([_val_digital_acc, _val_digital_precision, _val_digital_recall, _val_digital_f1,_val_digital_rocauc])
        print(" DIGITAL — val_acc: %f — val_precision: %f — val_recall %f — val_f1: %f -val_rauc: %f" % (_val_digital_acc, _val_digital_precision, _val_digital_recall, _val_digital_f1,_val_digital_rocauc))




        return

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=np.random.seed(666))
standard_scaler = StandardScaler()
cvscores = []

for train, test in kfold.split(X, y):
    model = build_model()

    # standard_scaler.fit(X[train])
    # X_train = standard_scaler.transform(X[train])
    # X_test = standard_scaler.transform(X[test])
    X_unstructure_train = X_unstructure[train]  # 文本训练集
    X_structure_train = X_structure[train]      # 数值训练集
    X_unstructure_test = X_unstructure[test]    # 文本测试集
    X_structure_test = X_structure[test]        # 数值测试集

    f1 = Metrics(X_unstructure_test, X_structure_test, y[test])
    model.fit([X_unstructure_train, X_structure_train],
              [y_hot[train], y_hot[train]],
              epochs=epoch,
              batch_size=batch_size,
              callbacks=[f1],
              validation_data=([X_unstructure_test, X_structure_test], [y_hot[test], y_hot[test]])
              )
    # 验证集中文本数据epoch结果
    val_text_res = np.array(f1.val_text_res)
    print('---------------TEXT-----------------')
    for i in range(val_text_res.shape[1]):
        print(val_text_res[np.argmax(val_text_res[:, i])])

    # 验证集中数值数据epoch结果
    val_digital_res = np.array(f1.val_digital_res)
    print('---------------DIGITAL-----------------')
    for i in range(val_digital_res.shape[1]):
        print(val_digital_res[np.argmax(val_digital_res[:, i])])
    #scores => loss: 6.8549 - dense_9_loss: 0.5180 - dense_10_loss: 0.4828 - dense_9_accuracy: 0.7996 - dense_10_accuracy: 0.8017
    scores = model.evaluate([X_unstructure_test, X_structure_test], [y_hot[test], y_hot[test]])
    accuracy = accuracy_rate_1 * scores[3]*100 + accuracy_rate_2 * scores[4]*100
    print("accuracy: %.2f%%; %s: %.2f%%; %s: %.2f%%" % (accuracy, model.metrics_names[3], scores[3]*100, model.metrics_names[4], scores[4]*100))
    cvscores.append([scores[3] * 100, scores[4] * 100])
cvscores = np.array(cvscores)
accuracy_total = accuracy_rate_1 * np.mean(cvscores[:, 0]) + accuracy_rate_2 * np.mean(cvscores[:, 1])
print("accuracy: %.2f%%, accuracy_text: %.2f%%-%.2f%%, %.2f%% (+/- %.2f%%); accuracy_digital: %.2f%%-%.2f%%, %.2f%% (+/- %.2f%%)"
      % (accuracy_total, np.min(cvscores[:, 0]), np.max(cvscores[:, 0]), np.mean(cvscores[:, 0]), np.std(cvscores[:, 0]),
         np.min(cvscores[:, 1]), np.max(cvscores[:, 1]), np.mean(cvscores[:, 1]), np.std(cvscores[:, 1])))