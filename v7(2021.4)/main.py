"""
created by PyCharm
date: 2021/4/11
time: 18:12
user: hxf
"""

import pandas as pd
import numpy as np
import math
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import logging

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as Data

import time
from medical_model import Network
from process_utils import save_result, RunBuilder
from collections import OrderedDict
import config
from model_log import log_setting


def run(start, end, flag):
    data = pd.read_csv(config.data_path, encoding='gbk')
    logging.debug('原数据size: %s' % str(data.shape))
    X, y = np.array(data.drop(labels=['label'], axis=1)), np.array(data['label'].apply(int))
    logging.debug('原数据X_size: %s, y_size: %s' % (X.shape, y.shape))
    logging.debug('原数据y分布: %s' % Counter(y))
    if isinstance(start, int):
        if end == -1:
            X_text, X_digital = X[:, 0:768], X[:, start:]
        else:
            X_text, X_digital = X[:, 0:768], X[:, start:end]
    else:
        X_text, X_digital = X[:, 0:768], X[:, start[0]:end[0]]
        for i in range(len(start)):
            if i > 0:
                if end[i] == -1:
                    X_digital_tmp = X[:, start[i]:]
                else:
                    X_digital_tmp = X[:, start[i]:end[i]]
                X_digital = np.concatenate([X_digital, X_digital_tmp], axis=1)
    logging.debug('文本数据: %s' % str(X_text.shape))
    logging.debug('数值数据: %s' % str(X_digital.shape))
    X_sub = np.hstack((X_text, X_digital))
    logging.debug('分割后X_size:%s, y_size:%s' % (X_sub.shape, y.shape))

    SMO = SMOTE(random_state=666)
    X_res, y_res = SMO.fit_resample(X_sub, y)
    logging.debug('插值后y分布: %s' % Counter(y_res))

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=np.random.seed())
    logging.debug('%s_%s_%s_%s' % (X_train.shape, X_test.shape, y_train.shape, y_test.shape))

    train_params = config.train_params
    test_params = config.test_params

    run_count = 0
    run_data = []
    model = None
    hightest_accuracy = 0
    test_run = next(iter(RunBuilder.get_run(test_params)))

    for run in RunBuilder.get_run(train_params):

        device = torch.device(run.device)
        network = Network(X_text.shape[1], X_digital.shape[1]).to(device)
        train_loader = Data.DataLoader(
            Data.TensorDataset(torch.tensor(X_train).to(torch.float32), torch.tensor(y_train)),
            batch_size=run.batch_size,
            num_workers=run.num_workers,
            shuffle=run.shuffle
        )
        test_loader = Data.DataLoader(
            Data.TensorDataset(torch.tensor(X_test).to(torch.float32), torch.tensor(y_test)),
            batch_size=test_run.batch_size,
            num_workers=test_run.num_workers,
            shuffle=test_run.shuffle
        )
        optimizer = optim.Adam(network.parameters(), lr=run.lr)

        run_start_time = time.time()
        run_count += 1
        epoch_count = 0
        test_epoch_count = 0
        tb = SummaryWriter(comment=f'-{run}-{flag}')

        network.train()

        for epoch in range(config.epoch):

            epoch_start_time = time.time()
            epoch_count += 1
            epoch_loss = 0
            epoch_correct_num = 0
            epoch_precision_score = 0
            epoch_recall_score = 0
            epoch_f1_score = 0
            epoch_auc_score = 0
            for batch in train_loader:
                X_batch_train = batch[0].to(device)
                labels_train = batch[1].to(device)
                X_text_train, X_digital_train = X_batch_train[:, :768], X_batch_train[:, 768:]
                preds = network(X_text_train, X_digital_train)  # 前向传播 根据权重参数进行预测
                loss = F.cross_entropy(preds, labels_train)  # 计算损失 构建计算图

                optimizer.zero_grad()  # pytorch会积累梯度 在优化每个batch的权重的梯度之前将之前权重的梯度置为0
                loss.backward()  # 在最后一个张量上调用反向传播方法 在计算图中计算权重梯度
                optimizer.step()  # 使用预先设置的learning_rate的梯度来更新权重参数

                epoch_loss += loss.item() * train_loader.batch_size
                epoch_correct_num += preds.argmax(dim=1).eq(labels_train).sum().item()
                epoch_precision_score += precision_score(labels_train.to('cpu'), preds.argmax(dim=1).to('cpu'))
                epoch_recall_score += recall_score(labels_train.to('cpu'), preds.argmax(dim=1).to('cpu'))
                epoch_f1_score += f1_score(labels_train.to('cpu'), preds.argmax(dim=1).to('cpu'))
                epoch_auc_score += roc_auc_score(labels_train.to('cpu'), preds.argmax(dim=1).to('cpu'))

            epoch_duration = time.time() - epoch_start_time
            run_duration = time.time() - run_start_time

            loss = epoch_loss / len(train_loader.dataset)
            accuracy = epoch_correct_num / len(train_loader.dataset)
            precision = epoch_precision_score / math.ceil(len(train_loader.dataset) / run.batch_size)
            recall = epoch_recall_score / math.ceil(len(train_loader.dataset) / run.batch_size)
            f1 = epoch_f1_score / math.ceil(len(train_loader.dataset) / run.batch_size)
            auc = epoch_auc_score / math.ceil(len(train_loader.dataset) / run.batch_size)

            tb.add_scalar('Train Loss', loss, epoch_count)
            tb.add_scalar('Train Accuracy', accuracy, epoch_count)
            tb.add_scalar('Train Precision', precision, epoch_count)
            tb.add_scalar('Train Recall', recall, epoch_count)
            tb.add_scalar('Train F1', f1, epoch_count)
            tb.add_scalar('Train AUC', auc, epoch_count)

            for name, param in network.named_parameters():  # 将network中的每一层参数都存入tensorboard
                tb.add_histogram(name, param, epoch_count)
                tb.add_histogram(f'{name}.grad', param.grad, epoch_count)

            # 保存训练参数
            results = OrderedDict()
            results['flag'] = flag
            results['current'] = 'Train'
            results['run'] = run_count
            results['epoch'] = epoch_count
            results['loss'] = np.round(loss, 6)
            results['accuracy'] = np.round(accuracy, 6)
            results['precision'] = np.round(precision, 6)
            results['recall'] = np.round(recall, 6)
            results['f1'] = np.round(f1, 6)
            results['auc'] = np.round(auc, 6)
            results['epoch_duration'] = np.round(epoch_duration, 6)
            results['run_duration'] = np.round(run_duration, 6)
            for k, v in run._asdict().items():
                results[k] = v
            if (run_count == 1) and (epoch_count == 1):
                logging.info('  '.join(results.keys()))
            logging.info('  '.join([str(item) for item in results.values()]))
            run_data.append(results)

            #  对测试集进行预测
            if epoch_count % config.test_num == 0:
                test_epoch_start_time = time.time()
                test_epoch_count += 1
                test_epoch_loss = 0
                test_epoch_correct_num = 0
                test_epoch_precision_score = 0
                test_epoch_recall_score = 0
                test_epoch_f1_score = 0
                test_epoch_auc_score = 0

                network.eval()

                for batch in test_loader:
                    X_batch_test = batch[0].to(device)
                    labels_test = batch[1].to(device)
                    X_text_test, X_digital_test = X_batch_test[:, :768], X_batch_test[:, 768:]
                    preds = network(X_text_test, X_digital_test)  # 前向传播 根据权重参数进行预测
                    test_loss = F.cross_entropy(preds, labels_test)  # 计算损失 构建计算图

                    test_epoch_loss += test_loss.item() * test_loader.batch_size
                    test_epoch_correct_num += preds.argmax(dim=1).eq(labels_test).sum().item()
                    test_epoch_precision_score += precision_score(labels_test.to('cpu'), preds.argmax(dim=1).to('cpu'))
                    test_epoch_recall_score += recall_score(labels_test.to('cpu'), preds.argmax(dim=1).to('cpu'))
                    test_epoch_f1_score += f1_score(labels_test.to('cpu'), preds.argmax(dim=1).to('cpu'))
                    test_epoch_auc_score += roc_auc_score(labels_test.to('cpu'), preds.argmax(dim=1).to('cpu'))

                test_epoch_duration = time.time() - test_epoch_start_time
                test_run_duration = time.time() - run_start_time

                test_loss = test_epoch_loss / len(test_loader.dataset)
                test_accuracy = test_epoch_correct_num / len(test_loader.dataset)
                test_precision = test_epoch_precision_score / math.ceil(len(test_loader.dataset) / test_run.batch_size)
                test_recall = test_epoch_recall_score / math.ceil(len(test_loader.dataset) / test_run.batch_size)
                test_f1 = test_epoch_f1_score / math.ceil(len(test_loader.dataset) / test_run.batch_size)
                test_auc = test_epoch_auc_score / math.ceil(len(test_loader.dataset) / test_run.batch_size)

                tb.add_scalar('Test Loss', test_loss, test_epoch_count)
                tb.add_scalar('Test Accuracy', test_accuracy, test_epoch_count)
                tb.add_scalar('Test Precision', test_precision, test_epoch_count)
                tb.add_scalar('Test Recall', test_recall, test_epoch_count)
                tb.add_scalar('Test F1', test_f1, test_epoch_count)
                tb.add_scalar('Test AUC', test_auc, test_epoch_count)

                results = OrderedDict()
                results['flag'] = flag
                results['current'] = 'Test'
                results['run'] = run_count
                results['epoch'] = test_epoch_count
                results['loss'] = np.round(test_loss, 6)
                results['accuracy'] = np.round(test_accuracy, 6)
                results['precision'] = np.round(test_precision, 6)
                results['recall'] = np.round(test_recall, 6)
                results['f1'] = np.round(test_f1, 6)
                results['auc'] = np.round(test_auc, 6)
                results['epoch_duration'] = np.round(test_epoch_duration, 6)
                results['run_duration'] = np.round(test_run_duration, 6)
                for k, v in test_run._asdict().items():
                    if k == 'lr': v = run.lr
                    results[k] = v

                logging.info('  '.join([str(item) for item in results.values()]))
                run_data.append(results)

        if test_accuracy > hightest_accuracy:
            hightest_accuracy = test_accuracy
            model = network
        tb.close()
    save_result(model, run_data, flag)
    logging.info(model)
    return run_data


def evaluate(run_data):
    run_data_df = pd.DataFrame(run_data)
    run_data_df_evaluate = run_data_df.iloc[
                           [
                               run_data_df[run_data_df['current'] == 'Test']['accuracy'].sort_values(
                                   ascending=False).index[0],
                               run_data_df[run_data_df['current'] == 'Test']['precision'].sort_values(
                                   ascending=False).index[0],
                               run_data_df[run_data_df['current'] == 'Test']['recall'].sort_values(
                                   ascending=False).index[0],
                               run_data_df[run_data_df['current'] == 'Test']['f1'].sort_values(ascending=False).index[
                                   0],
                               run_data_df[run_data_df['current'] == 'Test']['auc'].sort_values(ascending=False).index[
                                   0]
                           ],
                           :
                           ]
    results = pd.read_csv(config.results_file)
    results = pd.concat([results, run_data_df_evaluate])
    results.to_csv(config.results_file, index=False)


if __name__ == '__main__':
    params = config.params
    log_setting('medical_predict')
    for item in params:
        evaluate(run(item[0], item[1], item[2]))




