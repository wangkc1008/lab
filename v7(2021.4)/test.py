"""
created by PyCharm
date: 2021/4/20
time: 17:04
user: wkc
"""

import logging
from model_log import log_setting


def just_a_test():
    for i in range(10000):
        logging.info(i)


if __name__ == '__main__':
    log_setting('just_a_test')
    just_a_test()
