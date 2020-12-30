"""
created by PyCharm
date: 2020/6/12
time: 12:43
user: wkc
"""
import pandas as pd
import numpy as np
import re

hadm_id_file = './hadm_id_20200709'
admission_information_file = './admission_information_20200709'
demographics_file = './demographics_20200709'
treatment_information_file = './treatment_information_20200709'
lab_value_file = './lab_value_20200709'
chart_value_file = './chart_value_20200709'

lab_label_file = './lab_label_20200709'
chart_label_file = './chart_label_20200709'

filename_total_origin = './total_20200709_origin.csv'
filename_total = './total_20200709.csv'


class FileProcess:

    def label_processing(self):
        """
        处理label
        :return:
        """
        chart_data = pd.read_csv(lab_value_file)
        print(chart_data.shape)
        label_set = chart_data['itemid'].drop_duplicates()
        print(label_set)
        print(len(label_set))
        label_set.to_csv(lab_label_file, index=False)  # 写入label

        lab_data = pd.read_csv(chart_value_file)
        print(lab_data.shape)
        lab_label_set = lab_data['itemid'].drop_duplicates()
        print(lab_label_set)
        print(len(lab_label_set))
        lab_label_set.to_csv(chart_label_file, index=False)

    def data_process(self):
        """
        处理数据
        :return:
        """
        hadm_id_data = pd.read_csv(hadm_id_file)                                        # 读取hadm_id_file数据  hadm_id,subject_id
        print('hadm_id_data size：', hadm_id_data.shape)
        admission_information_data = pd.read_csv(admission_information_file)            # 读取admission_information_file数据 hadm_id,admittime,dischtime,deathtime,discharge_location,religion,marital_status,ethnicity,diagnosis
        print('admission_information_data size：', admission_information_data.shape)
        demographics_data = pd.read_csv(demographics_file)                              # 读取demographics_file数据 subject_id,gender,dob,dod,expire_flag
        print('demographics_data size：', demographics_data.shape)
        treatment_information_data = pd.read_csv(treatment_information_file)            # 读取treatment_information_file数据 hadm_id,descriptin
        treatment_information_data = treatment_information_data[(~treatment_information_data['hadm_id'].duplicated())]
        print('treatment_information_data size：', treatment_information_data.shape)

        res = pd.merge(hadm_id_data, admission_information_data, on='hadm_id', how='inner')  # 基础数据合并
        print('res size：', res.shape)
        res = pd.merge(res, demographics_data, on='subject_id', how='left')
        print('res size：', res.shape)
        res = pd.merge(res, treatment_information_data, on='hadm_id', how='left')
        print('res size：', res.shape)

        chart_label = pd.read_csv(chart_label_file)  # 读取label数据
        print('chart_label size：', chart_label.shape)
        lab_label = pd.read_csv(lab_label_file)
        print('lab_label size：', lab_label.shape)
        res_label = pd.concat([chart_label, lab_label], axis=0)  # 拼接label
        print('res_label size：', res_label.shape)

        zeros = np.zeros((res.shape[0], res_label.shape[0]))  # 0矩阵
        print('zeros size：', zeros.shape)
        zeros = pd.DataFrame(zeros, columns=res_label['itemid'])
        nan_mat = zeros.replace(0, '')

        new_res = pd.concat([res, nan_mat], axis=1)  # 基础数据与0矩阵合并
        new_res = new_res.set_index(new_res['hadm_id'])
        print('new_res size：', new_res.shape)
        new_res.to_csv(filename_total_origin)

        chart_data = pd.read_csv(chart_value_file)  # 读取chart_event和lab_event数据
        print('chart_data size：', chart_data.shape)
        lab_data = pd.read_csv(lab_value_file)
        print('lab_data size：', lab_data.shape)

        # lab_data['res_value'] = lab_data['value'].astype('str') + ' ' + lab_data['valueuom']  # valuenum与valueuom拼接
        for i in range(lab_data.shape[0]):
            if i % 100000 == 0:
                print('lab_data 当前行数：%d' % i)
            new_res.at[lab_data.iat[i, 0], lab_data.iat[i, 1]] = lab_data.iat[i, 2]

        new_res.to_csv(filename_total)

        for i in range(chart_data.shape[0]):
            if i % 100000 == 0:
                print('chart_data 当前行数：%d' % i)
            new_res.at[chart_data.iat[i, 0], chart_data.iat[i, 1]] = chart_data.iat[i, 2]

        new_res.to_csv(filename_total)


if __name__ == '__main__':
    process = FileProcess()
    # process.label_processing()  # 先执行这一步
    process.data_process()
