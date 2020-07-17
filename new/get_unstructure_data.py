"""
created by PyCharm
date: 2020/7/14
time: 11:59
user: wkc
"""
import pandas as pd
import numpy as np
from hxf.testPostgreSQL import PostgreSQLOperate
import csv

file_name = './total_20200709.csv'
new_file_name = './total_20200714.csv'
text_file_name = './text_20200714.csv'
text_flag_file_name = './text_flag_20200714.csv'


def write_to_file(filename, label, res):
    with open(filename, mode='w+', encoding='utf-8', newline='') as f:
        write = csv.writer(f, dialect='excel')
        write.writerow(label)
        for row in res:
            write.writerow(row)
    f.close()


def get_data():
    """
    获取非结构化数据
    :return:
    """
    data = pd.read_csv(file_name)
    print(len(data['hadm_id'].values))
    hadm_id_str = ','.join([str(item) for item in data['hadm_id'].values])
    post = PostgreSQLOperate('127.0.0.1', 5432)
    post.connect('postgres', 'postgres', 'mimic')
    print('ready to read')
    res = post.select("select hadm_id, category, text from noteevents where hadm_id in (" + hadm_id_str + ") and category = 'Discharge summary' and description = 'Report'")
    print(len(res))
    write_to_file(text_file_name, ['hadm_id', 'category', 'text'], res)
    print('write done')


def get_common_data():
    """
    获取hadm_id差集
    :return:
    """
    text_data = pd.read_csv(text_file_name)
    text_data = text_data[(~text_data['hadm_id'].duplicated())]
    print('text_data size：', text_data.shape)
    text_hadm_id = text_data['hadm_id'].values
    total_data = pd.read_csv(file_name)
    total_hadm_id = total_data['hadm_id'].values
    print(np.setdiff1d(total_hadm_id, text_hadm_id))


def handle_data():
    """
    处理数据
    :return:
    """
    text_data = pd.read_csv(text_file_name)
    text_data = text_data[(~text_data['hadm_id'].duplicated())][['hadm_id', 'text']]
    print('text_data size：', text_data.shape)
    total_data = pd.read_csv(new_file_name)
    basic_data = total_data[['hadm_id', 'flag']]
    # print(basic_data)
    res = pd.merge(text_data, basic_data, on='hadm_id', how='left')
    print(res.shape)
    res.to_csv(text_flag_file_name)


if __name__ == '__main__':
    # get_data()
    # get_common_data()  # 这个没用
    handle_data()  # 测试