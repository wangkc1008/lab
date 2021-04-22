"""
created by PyCharm
date: 2021/4/11
time: 18:15
user: hxf
"""
import pandas as pd
import os
import json
import datetime
from collections import namedtuple
from itertools import product
import torch
import config


class RunBuilder:
    @staticmethod
    def get_run(params):  # 静态方法，不需要实例化

        Run = namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs


def save_result(model, run_data, flag):
    """
    运行结果保存
        默认文件路径 ./run_data
        默认模型路径 ./model
    :param model: 模型
    :param run_data: 运行数据
    :param flag: 当前模型标签
    """
    result_dir = config.run_data_path
    model_dir = config.model_path
    name = 'result'

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    time_index = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # 保存运行文件
    run_data_path = os.path.join(result_dir, name)
    pd.DataFrame(run_data).to_csv(f'{run_data_path}_{time_index}.csv', index=False)
    with open(f'{run_data_path}_{time_index}.json', 'w', encoding='utf-8') as f:
        json.dump(run_data, f, ensure_ascii=False, indent=4)

    # 保存运行模型
    model_path = os.path.join(model_dir, name)
    torch.save(model, f'{model_path}_{flag}_{time_index}.pt')
