"""
created by PyCharm
date: 2021/4/20
time: 15:55
user: hxf
"""

import logging
import time
import os
import config


def log_setting(log_name):
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path = config.log_path
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log_name = log_path + '/' + log_name + '_' + rq + '.log'
    logging.basicConfig(
        level=logging.DEBUG,  # 控制台打印的日志级别
        filename=log_name,
        filemode='a',
        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'  # 日志格式
    )