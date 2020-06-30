"""
created by PyCharm
date: 2020/6/12
time: 12:43
user: wkc
"""
import pandas as pd
import numpy as np
import re

filename1 = './data_20200630_1.csv'
filename2 = './data_20200630_2.csv'
filename3 = './data_20200630_3.csv'
filename4 = './data_20200630_4.csv'
filename5 = './data_20200630_5.csv'
filename3_label = './data_20200630_3_label.csv'
filename4_label = './data_20200630_4_label.csv'
filename_total_origin = './total_20200630_origin.csv'
filename_total = './total_20200630.csv'


class FileProcess:

    def label_processing(self):
        """
        处理label
        :return:
        """
        chart_data = pd.read_csv(filename3)
        print(chart_data.shape)
        label_set = chart_data['itemid'].drop_duplicates()
        print(label_set)
        print(len(label_set))
        label_set.to_csv(filename3_label, index=False)  # 写入label

        lab_data = pd.read_csv(filename4)
        print(lab_data.shape)
        lab_label_set = lab_data['itemid'].drop_duplicates()
        print(lab_label_set)
        print(len(lab_label_set))
        lab_label_set.to_csv(filename4_label, index=False)

    def data_process(self):
        """
        处理数据
        :return:
        """
        basic_data = pd.read_csv(filename1)  # 读取数据
        print('basic_data size：', basic_data.shape)
        other_data = pd.read_csv(filename2)
        print('other_data size：', other_data.shape)
        weight_data = pd.read_csv(filename5)
        print('weight_data size：', weight_data.shape)
        res = pd.merge(basic_data, other_data, on='subject_id', how='inner')  # 基础数据合并
        res = pd.merge(res, weight_data, on='subject_id', how='left')  # 基础数据合并
        print('res size：', res.shape)

        chart_label = pd.read_csv(filename3_label)  # 读取label数据
        print('chart_label size：', chart_label.shape)
        lab_label = pd.read_csv(filename4_label)
        print('lab_label size：', lab_label.shape)
        res_label = pd.concat([chart_label, lab_label], axis=0)  # 拼接label
        print('res_label size：', res_label.shape)

        zeros = np.zeros((res.shape[0], res_label.shape[0]))  # 生成 4290*3307 的0矩阵
        print('zeros size：', zeros.shape)
        zeros = pd.DataFrame(zeros, columns=res_label['itemid'])
        nan_mat = zeros.replace(0, '')

        new_res = pd.concat([res, nan_mat], axis=1)  # 基础数据与0矩阵合并
        new_res = new_res.set_index(new_res['subject_id'])
        print('new_res size：', new_res.shape)
        new_res.to_csv(filename_total_origin)

        chart_data = pd.read_csv(filename3)  # 读取chart_event和lab_event数据
        print('chart_data size：', chart_data.shape)
        lab_data = pd.read_csv(filename4)
        print('lab_data size：', lab_data.shape)

        # lab_data['res_value'] = lab_data['value'].astype('str') + ' ' + lab_data['valueuom']  # valuenum与valueuom拼接
        for i in range(lab_data.shape[0]):
            if i % 100000 == 0:
                print('lab_data 当前行数：%d' % i)
            new_res.at[lab_data.iat[i, 0], lab_data.iat[i, 2]] = lab_data.iat[i, 3]

        new_res.to_csv(filename_total)

        for i in range(chart_data.shape[0]):
            if i % 100000 == 0:
                print('chart_data 当前行数：%d' % i)
            new_res.at[chart_data.iat[i, 0], chart_data.iat[i, 2]] = chart_data.iat[i, 3]

        new_res.to_csv(filename_total)


if __name__ == '__main__':
    process = FileProcess()
    # process.label_processing()  # 先执行这一步
    process.data_process()