"""
created by PyCharm
date: 2020/9/12
time: 22:30
user: wkc
"""

from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


data = pd.read_csv('C:/Users/wangkc/anacoda/laboratory/hxf/multimodal/mixed_data_100_5079_replace_20200912.csv')
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
epoch = 100

def build_model():
    model = models.Sequential()
    model.add(layers.BatchNormalization())
    model.add(layers.Input(shape=(X.shape[1],)))
    # model.add(layers.Dense(369, activation='tanh', input_shape=(X.shape[1],)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(200, activation='tanh', kernel_regularizer=regularizers.l2()))
    model.add(layers.Dropout(0.3))
    # model.add(layers.BatchNormalization())
    model.add(layers.Dense(200, activation='tanh', kernel_regularizer=regularizers.l2()))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


class Metrics(Callback):

    def __init__(self, val_x, val_y):
        super().__init__()
        self._val_x = val_x
        self._val_y = val_y
        self.val_res = []

    def on_train_begin(self, logs=None):
        self.val_res = []

    def on_epoch_end(self, epoch, logs=None):
        # {'loss': 3.837848424911499, 'accuracy': 0.8030230402946472, 'val_loss': 3.848667621612549, 'val_accuracy': 0.7693965435028076}
        val_predict = (np.asarray(self.model.predict_classes(self._val_x))).round()
        val_targ = self._val_y
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_res.append([logs['val_accuracy'], _val_precision, _val_recall, _val_f1])
        print(" — val_acc: %f — val_precision: %f — val_recall %f — val_f1: %f" % (logs['val_accuracy'], _val_precision, _val_recall, _val_f1))
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
    X_structure_train = X_structure[train]  # 数值训练集
    X_unstructure_test = X_unstructure[test]  # 文本测试集
    X_structure_test = X_structure[test]  # 数值测试集

    f1 = Metrics(X_structure_test, y[test])
    model.fit(X_structure_train,
              y_hot[train],
              epochs=epoch,
              batch_size=batch_size,
              callbacks=[f1],
              validation_data=(X_structure_test, y_hot[test])
              )

    val_res = np.array(f1.val_res)
    for i in range(val_res.shape[1]):
        print(val_res[np.argmax(val_res[:, i])])
