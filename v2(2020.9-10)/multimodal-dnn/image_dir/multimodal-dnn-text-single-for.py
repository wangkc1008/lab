import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DiseaseData:
    def __init__(self, train_model=True, need_shuffle=True):
        if train_model:
            self._data_structure = X_train_structure_standard  # 结构化数值归一化数据
            self._data_unstructure = X_train_unstructure_standard  # 非结构化文本数据
            self._labels = y_train
        else:
            self._data_structure = X_test_structure_standard  # 结构化数值归一化数据
            self._data_unstructure = X_test_unstructure_standard  # 非结构化文本数据
            self._labels = y_test
        #         print(self._data_structure.shape)
        #         print(self._data_unstructure.shape)
        #         print(self._labels.shape)

        self._num_examples = self._data_structure.shape[0]
        self._need_shuffle = need_shuffle
        self._indicator = 0
        if self._need_shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        # [0,1,2,3,4,5] -> [5,3,2,4,0,1]
        p = np.random.permutation(self._num_examples)
        self._data_structure = self._data_structure[p]
        self._data_unstructure = self._data_unstructure[p]
        self._labels = self._labels[p]

    def next_batch(self, batch_size):
        """return batch_size examples as a batch."""
        end_indicator = self._indicator + batch_size
        if end_indicator > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0
                end_indicator = batch_size
            else:
                raise Exception("have no more examples")
        if end_indicator > self._num_examples:
            raise Exception("batch size is larger than all examples")
        batch_data_structure = self._data_structure[self._indicator: end_indicator]
        batch_data_unstructure = self._data_unstructure[self._indicator: end_indicator]
        batch_labels = self._labels[self._indicator: end_indicator]
        self._indicator = end_indicator
        return batch_data_unstructure, batch_data_structure, batch_labels


# data = pd.read_csv('C:/Users/wangkc/Desktop/胡喜风预测论文/mixed_data_0717_drop_replace_fill_100_5079.csv')
data = pd.read_csv('C:/Users/wangkc/anacoda/laboratory/hxf/multimodal/mixed_data_100_5079_replace_20200912.csv')
print(data.shape)

# data = data.drop(labels=['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
data_unstructure, data_structure = data.iloc[:, 0:200], data.iloc[:, 200:]
# 对结构化数值数据进行相关性分析
corr_data = data_structure.corr()['flag'].sort_values(ascending=False)
data_structure_corr = data_structure[corr_data[(corr_data > 0.1) | (corr_data < -0.1)].index]
new_data = pd.concat([data_unstructure, data_structure_corr], axis=1)
print(new_data.shape)

# 训练、测试数据分割
X, y = np.array(new_data.drop(labels=['flag'], axis=1)), np.array(new_data['flag'].apply(int))
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666, test_size=0.33)
X_train_unstructure, X_train_structure = X_train[:, 0:200], X_train[:, 200:]
X_test_unstructure, X_test_structure = X_test[:, 0:200], X_test[:, 200:]

# 结构化数值数据归一化
standard_scaler = StandardScaler()
standard_scaler.fit(X_train_structure)
X_train_structure_standard = standard_scaler.transform(X_train_structure)
X_test_structure_standard = standard_scaler.transform(X_test_structure)

standard_scaler.fit(X_train_unstructure)
X_train_unstructure_standard = standard_scaler.transform(X_train_unstructure)
X_test_unstructure_standard = standard_scaler.transform(X_test_unstructure)

# learning_rate = 1e-6
# dropout_rate = 0.3
batch_size = 100
total_epoch = 500

activation_list = {'relu': tf.nn.relu, 'tanh': tf.nn.tanh, 'sigmod': tf.nn.sigmoid}
learning_rate_list = [1e-5, 1e-6]
hidden1_list = np.arange(200, 700, 100)
hidden2_list = np.arange(200, 700, 100)

best_param_score = 0.0
current_time = time.time()

for act_name, activation in activation_list.items():
    for learning_rate in learning_rate_list:
        for hidden1_neural_num in hidden1_list:
            for hidden2_neural_num in hidden2_list:
                name_scope = 'train_op_' + act_name + '_' + str(learning_rate) + '_' + str(hidden1_neural_num) + '_' + str(hidden2_neural_num)
                print('train_op: %s' % (name_scope))
                print('耗时: %d' % (round(time.time() - current_time, 2)))
                current_time = time.time()

                train_data = DiseaseData(True, True)
                # 结构化数值数据
                x = tf.placeholder(tf.float32, [None, X_train_unstructure.shape[1]])
                # x = tf.layers.dropout(x, dropout_rate)
                y = tf.placeholder(tf.int64, [None])

                hidden1_2 = tf.layers.dense(x, hidden1_neural_num, activation=activation)
                # hidden1_2 = tf.layers.dropout(hidden1_2, dropout_rate)
                hidden2_2 = tf.layers.dense(hidden1_2, hidden2_neural_num, activation=activation)
                # hidden2_2 = tf.layers.dropout(hidden2_2, dropout_rate)

                y_2 = tf.layers.dense(hidden2_2, 2)

                # 计算损失
                loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_2)
                # 计算准确率
                accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_2, 1), y), tf.float64))

                with tf.name_scope(name_scope):
                    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

                init = tf.global_variables_initializer()
                with tf.Session() as sess:
                    sess.run(init)

                    iter = 0
                    epoch_acc_val = []
                    for epoch in range(total_epoch):
                        for i in range(int(X_train.shape[0] / batch_size)):
                            batch_data_unstructure, batch_data_structure, batch_labels = train_data.next_batch(batch_size)
                            loss_val, acc_val, _ = sess.run(
                                [loss, accuracy, train_op],
                                feed_dict={
                                    x: batch_data_unstructure,  # 非结构化文本数据
                                    y: batch_labels})
                            # print('[Train] epoch: %d, step: %d, loss: %4.5f, acc: %4.5f' % (epoch+1, i+1, loss_val, acc_val))

                            iter += 1
                            if iter % int(X_train.shape[0] / batch_size) == 0:
                                test_data = DiseaseData(False, False)
                                all_test_acc_val = []
                                for j in range(int(X_test.shape[0] / batch_size)):
                                    test_batch_data_unstructure, test_batch_data_structure, test_batch_labels \
                                        = test_data.next_batch(batch_size)
                                    test_acc_val = sess.run(
                                        [accuracy],
                                        feed_dict={
                                            x: test_batch_data_unstructure,  # 非结构化文本数据
                                            y: test_batch_labels
                                        })
                                    all_test_acc_val.append(test_acc_val)
                                all_test_acc_val_mean = np.mean(all_test_acc_val)
                                if all_test_acc_val_mean > best_param_score:
                                    best_param_score = all_test_acc_val_mean
                                    print('[Test ] act_name: %s, learning_rate: %s, hidden1_neural_num: %d, hidden2_neural_num: %d, epoch: %d, acc: %4.5f'
                                          % (act_name, learning_rate, hidden1_neural_num, hidden2_neural_num, epoch, all_test_acc_val_mean))
