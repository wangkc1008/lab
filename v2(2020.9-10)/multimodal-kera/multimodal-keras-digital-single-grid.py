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
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


data = pd.read_csv('E:/hxf_prediction/mutil-task/multimodal(1)/multimodal/mixed_data_100_5079_replace_20200912.csv')
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


def build_model(hidden1, hidden2, dropout_rate, activation):
    model = models.Sequential()
    model.add(layers.Input(shape=(X_structure.shape[1],)))
    # model.add(layers.Dense(369, activation='tanh', input_shape=(X.shape[1],)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(hidden1, activation=activation, kernel_regularizer=regularizers.l2()))
    model.add(layers.Dropout(dropout_rate))
    # model.add(layers.BatchNormalization())
    model.add(layers.Dense(hidden2, activation=activation, kernel_regularizer=regularizers.l2()))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=build_model)
batch_size = [20,40, 60, 80, 100]
epochs = [10, 50, 100]
hidden1 = np.arange(200, 300, 50)
hidden2 = np.arange(100, 200, 50)
dropout_rate = np.arange(0.1, 0.6, 0.1)
activation = ['tanh', 'relu', 'sigmoid']
param_grid = dict(batch_size=batch_size, nb_epoch=epochs, hidden1=hidden1, hidden2=hidden2, dropout_rate=dropout_rate, activation=activation)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X_structure, y_hot)

# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# for params, mean_score, scores in grid_result.grid_scores_:
#     print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))