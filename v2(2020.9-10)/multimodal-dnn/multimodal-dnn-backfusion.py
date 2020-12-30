import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os
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
print('data', data.shape)

# data = data.drop(labels=['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
data_unstructure, data_structure = data.iloc[:, 0:200], data.iloc[:, 200:]
# 对结构化数值数据进行相关性分析
corr_data = data_structure.corr()['flag'].sort_values(ascending=False)
data_structure_corr = data_structure[corr_data[(corr_data > 0.1) | (corr_data < -0.1)].index]
new_data = pd.concat([data_unstructure, data_structure_corr], axis=1)
print('new_data', new_data.shape)

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

X_train = np.concatenate((X_train_unstructure_standard, X_train_structure_standard), axis=1)
X_test = np.concatenate((X_test_unstructure_standard, X_test_structure_standard), axis=1)

learning_rate = 1e-5
batch_size = 100
total_epoch = 1000
dropout_rate = 0.3

current_time = time.time()

train_data = DiseaseData(True, True)
# 结构化数值数据
x1 = tf.placeholder(tf.float32, [None, X_train_structure.shape[1]])
# x1_d = tf.layers.dropout(x1, dropout_rate)
# 非结构化文本数据
x2 = tf.placeholder(tf.float32, [None, X_train_unstructure.shape[1]])
# x2_d = tf.layers.dropout(x2, dropout_rate)

y = tf.placeholder(tf.int64, [None])

hidden1_1 = tf.layers.dense(x1, 300, activation=tf.nn.tanh)
hidden2_1 = tf.layers.dense(x2, 600, activation=tf.nn.relu)

hidden1_1_d = tf.layers.dropout(hidden1_1, dropout_rate)
hidden2_1_d = tf.layers.dropout(hidden2_1, dropout_rate)

hidden_concat = tf.concat([hidden1_1_d, hidden2_1_d], 1)
hidden_concat_1 = tf.layers.dense(hidden_concat, 100, activation=tf.nn.relu)
hidden_concat_2 = tf.layers.dense(hidden_concat_1, 100, activation=tf.nn.relu)
hidden_concat_3 = tf.layers.dense(hidden_concat_2, 50, activation=tf.nn.relu)
hidden_res = tf.layers.dense(hidden_concat_3, 10)
w = tf.reduce_mean(hidden_res, axis=1)
w = tf.reshape(w, (batch_size, 1))
hidden1_1 = w * hidden1_1 + hidden1_1
hidden2_1 = w * hidden2_1 + hidden2_1

# 对第一个隐藏层加dropout
# hidden1_1_d = tf.layers.dropout(hidden1_1, dropout_rate)
# hidden2_1_d = tf.layers.dropout(hidden2_1, dropout_rate)

hidden1_2 = tf.layers.dense(hidden1_1_d, 300, activation=tf.nn.tanh)
hidden2_2 = tf.layers.dense(hidden2_1_d, 600, activation=tf.nn.relu)
# 对第二个隐藏层加dropout
# hidden1_2_d = tf.layers.dropout(hidden1_2, dropout_rate)
# hidden2_2_d = tf.layers.dropout(hidden2_2, dropout_rate)

y_1 = tf.layers.dense(hidden1_2, 2)
y_2 = tf.layers.dense(hidden2_2, 2)

# 计算损失
loss1 = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_1)
loss2 = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_2)
# 计算准确率
accuracy1 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_1, 1), y), tf.float64))
accuracy2 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_2, 1), y), tf.float64))

loss = 0.5 * loss1 + 0.5 * loss2
accuracy = 0.3 * accuracy1 + 0.7 * accuracy2

with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    iter = 0
    epoch_acc_val = []
    for epoch in range(total_epoch):
        for i in range(int(X_train.shape[0] / batch_size)):
            batch_data_unstructure, batch_data_structure, batch_labels = train_data.next_batch(batch_size)
            loss_val, loss_val_1, loss_val_2, acc_val, acc_val_1, acc_val_2,  _ = sess.run(
                [loss, loss1, loss2, accuracy, accuracy1, accuracy2, train_op],
                feed_dict={
                    x1: batch_data_structure,    # 结构化数值数据
                    x2: batch_data_unstructure,  # 非结构化文本数据
                    y: batch_labels})
            # print('[Train] epoch: %d, step: %d, loss: %4.5f, acc: %4.5f' % (epoch+1, i+1, loss_val, acc_val))
            # print('[Train] epoch: %d, step: %d, loss: %4.5f, loss_1: %4.5f, loss_2: %4.5f, acc: %4.5f, acc_1: %4.5f, acc_2: %4.5f'
            #       % (epoch+1, i+1, loss_val, loss_val_1, loss_val_2, acc_val, acc_val_1, acc_val_2))
            iter += 1
            if iter % int(X_train.shape[0] / batch_size) == 0:
                test_data = DiseaseData(False, False)
                all_test_acc_val = []
                all_test_acc_val_1 = []
                all_test_acc_val_2 = []
                for j in range(int(X_test.shape[0] / batch_size)):
                    test_batch_data_unstructure, test_batch_data_structure, test_batch_labels = test_data.next_batch(batch_size)
                    test_acc_val, test_acc_val_1, test_acc_val_2 = sess.run(
                        [accuracy, accuracy1, accuracy2],
                        feed_dict={
                            x1: test_batch_data_structure,    # 结构化数值数据
                            x2: test_batch_data_unstructure,  # 非结构化数值数据
                            y: test_batch_labels
                        })
                    all_test_acc_val.append(test_acc_val)
                    all_test_acc_val_1.append(test_acc_val_1)
                    all_test_acc_val_2.append(test_acc_val_2)
                print('[Test ] epoch: %d, acc: %4.5f, acc1: %4.5f, acc2: %4.5f'
                      % (epoch+1, np.mean(all_test_acc_val), np.mean(all_test_acc_val_1), np.mean(all_test_acc_val_2)))
                epoch_acc_val.append([epoch+1, np.mean(all_test_acc_val), np.mean(all_test_acc_val_1), np.mean(all_test_acc_val_2)])
    plt.plot(np.array(epoch_acc_val)[:, 0], np.array(epoch_acc_val)[:, 1], color='red', linestyle='-')
    plt.plot(np.array(epoch_acc_val)[:, 0], np.array(epoch_acc_val)[:, 2], color='blue', linestyle='-')
    plt.plot(np.array(epoch_acc_val)[:, 0], np.array(epoch_acc_val)[:, 3], color='green', linestyle='-')
    plt.savefig("./image_dir/" + os.path.splitext(os.path.basename(__file__))[0] + "-" + str(total_epoch) + ".png")
    plt.show()


