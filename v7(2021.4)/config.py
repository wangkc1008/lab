"""
created by PyCharm
date: 2021/4/11
time: 20:14
user: hxf
"""
import numpy as np
from collections import OrderedDict

data_path = './total_data_xinjiao_20210125.csv'
run_data_path = './run_data'
model_path = './model'
results_file = './results.csv'
log_path = './log'
epoch = 8
test_num = 4
train_params = OrderedDict(
    lr=[.01, .001],
    batch_size=[50, 100],
    shuffle=[False],
    device=['cuda'],
    num_workers=[1]  # 有多少子进程被用来加载数据 默认为0即在主进程中加载数据 可以利用多核CPU的特点指定num_workers个数 提前将数据加载到内存中
)

test_params = OrderedDict(
    lr=[np.nan],
    batch_size=[50],
    shuffle=[False],
    device=['cuda'],
    num_workers=[1]  # 有多少子进程被用来加载数据 默认为0即在主进程中加载数据 可以利用多核CPU的特点指定num_workers个数 提前将数据加载到内存中
)

params = [
    [768, -1, 'ALL'],
    [768, 772, 'demography'],
    [772, 779, 'GRACE'],
    [779, 791, 'past_medical_history'],
    [791, 805, 'clinical_laboratory_results'],
    [805, -1, 'medication'],
    [
        [772, 791, 805],
        [779, 805, -1],
        'variable'
    ],
    [
        [768, 779],
        [772, 791],
        'stable'
    ],
]