# -*- coding:utf-8 -*-
import os,pickle,datetime
import numpy as np
from model import create_model
from data_process import Date_Process

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Bert_Whitening_Similarity_Calculation(object):
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
        vocab_file = self.dp.config.get('DATA_PROCESS', 'vocab_file')
        self.dp.init(vocab_file)
        self.dp.save_model(self.dp.save_path)

    def data_process_reload(self):
        """
        模型重复训练，运行此函数，恢复初处理好的数据
        :return:
        """
        self.dp.load_model(self.dp.save_path)

    # 训练模型
    def train(self):
        param = self.dp.param
        # 制作训练集
        all_token_ids = self.dp.parse_data(self.dp.raw_data, self.dp.tokenizer, param.max_seq_length)
        # 构造模型
        encoder = create_model(param.bert_config_file,param.init_checkpoint)

        all_vecs = encoder.predict([all_token_ids,
                                    np.zeros_like(all_token_ids)],
                                   verbose=True)
        kernel, bias = self.dp.compute_kernel_bias(all_vecs)

        res_all_vecs = self.dp.transform_and_normalize(all_vecs, kernel, bias)

        with open(param.model_save_path, "wb") as f:
            pickle.dump([self.dp.raw_data, kernel, bias, res_all_vecs], f)


    def predict(self,words, return_quantity=10, threshold=0):
        """
        传入文本，输出与其相似度在前TopK的文本
        :param words: list 传入的文本的列表
        :param return_quantity: int 返回文本数量
        :param threshold: int 最低相似度阈值，低于此值的结果不返回
        :return:
        """
        # 恢复模型参数，并针对测试修改部分参数
        param = self.dp.param
        with open(param.model_save_path, "rb") as f:
            word_list, kernel, bias, res_all_vecs = pickle.load(f)
        encoder = create_model(param.bert_config_file, param.init_checkpoint)

        test_token_ids = self.dp.parse_data(words, self.dp.tokenizer, param.max_seq_length)
        test_vecs = encoder.predict([test_token_ids,
                                    np.zeros_like(test_token_ids)],
                                   verbose=True)
        res_test_vecs = self.dp.transform_and_normalize(test_vecs, kernel, bias)
        similarities = self.dp.get_similarities(res_all_vecs,res_test_vecs[0])
        sorted_res = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
        result_similarity = []
        for i in sorted_res[:int(return_quantity)]:
            if i[1] > threshold:
                result_similarity.append([word_list[i[0]], i[1]])
        return result_similarity


if __name__ == '__main__':

    words = ['小米8 全面屏游戏智能手机 6GB+128GB 黑色 全网通4G 双卡双待  拍照手机']
    cfg_path = './default.cfg'

    BWSC = Bert_Whitening_Similarity_Calculation(cfg_path)

    # 初始化数据，开启训练
    # BWSC.data_process_init()
    # BWSC.train()
    # result_similarity = BWSC.predict(words)
    # for sentence,similarity in result_similarity:
    #     print(sentence+"  "+str(similarity))

    # # 加载已处理好的数据，开启训练
    BWSC.data_process_reload()
    result_similarity = BWSC.predict(words)
    for sentence,similarity in result_similarity:
        print(sentence+"  "+str(similarity))


