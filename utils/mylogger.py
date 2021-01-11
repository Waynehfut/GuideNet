# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：       log_utils
   Description :
   Author :          wayne
   Date:             18-10-20
   Create by :       PyCharm
   Check status:     https://waynehfut.github.io
-------------------------------------------------
"""
__author__ = 'Wayne'

import logging
import time
import os


def setup_logger(model_name, log_type, file_dir='log'):
    log_time = time.strftime("%Y_%m_%d_%H_%M_%S")
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    log_path = file_dir
    log_name = log_path + "/" + log_type + "_" + model_name + "_" + log_time + '.log'
    print("Start logging file at {}".format(log_name))
    fh = logging.FileHandler(log_name, 'a', encoding='utf-8')
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '[%(asctime)s] %(filename)s in %(funcName)s at line:%(lineno)d [%(levelname)s]%(message)s')

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    fh.close()
    ch.close()
