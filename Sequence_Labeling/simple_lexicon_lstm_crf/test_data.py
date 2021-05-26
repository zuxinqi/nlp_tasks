import numpy as np
import tensorflow as tf
import re,os,datetime,argparse
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from model import make_graph
from data_process import Date_Process
from evaluation_utils import Evaluation_Utils


class Lstm_Crf_Entity_Extraction(object):
    def __init__(self, config_path):
        """
        初始化模型，初始化数据处理，并解析config参数
        :param config_path:
        """
        self.dp = Date_Process()
        self.dp.parse_config(config_path)

    def data_process_init(self):
        """
        模型首次运行，执行此函数，初始化数据，并保存
        :return:
        """
        feature_selection_name = self.dp.config.get('DATA_PROCESS', 'feature_selection_name')
        self.dp.init(feature_selection_name)
        self.dp.save_model(self.dp.save_path)
if __name__ == '__main__':
    cfg_path = './default.cfg'
    LCEE = Lstm_Crf_Entity_Extraction(cfg_path)

    # 初始化数据，开启训练
    LCEE.data_process_init()
    # LCEE.train()