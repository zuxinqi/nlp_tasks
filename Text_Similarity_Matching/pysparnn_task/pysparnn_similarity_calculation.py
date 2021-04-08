# -*- coding:utf-8 -*-
import os
from data_process import Date_Process

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Pysparnn_Similarity_Calculation(object):
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
        self.dp.init()
        self.dp.save_model(self.dp.save_path)

    def data_process_reload(self):
        """
        模型重复训练，运行此函数，恢复初处理好的数据
        :return:
        """
        self.dp.load_model(self.dp.save_path)
        self.dp.tokenizer_init()

    # 针对一句话进行分词
    def split_one_words(self, word, tokenizer, stopwords):
        cut_word = [i for i in tokenizer.cut(word) if i not in stopwords and i != ' ']
        cut_word = " ".join(cut_word)
        return cut_word

    def predict(self,words, return_quantity=10, k_clusters=10):
        """
        传入文本，输出与其相似度在前TopK的文本
        :param words: list 传入的文本的列表
        :param return_quantity: int 返回文本数量
        :param k_clusters: int 簇的个数，多一些相对精准
        :return:
        """
        # 恢复模型参数，并针对测试修改部分参数
        cut_words = self.split_one_words(words, self.dp.tokenizer, self.dp.stopwords)
        search_features_vec = self.dp.tv.transform([cut_words])
        # k 返回个数, k_clusters 簇的个数，多一些相对精准
        result_similarity = self.dp.cp.search(search_features_vec, k=return_quantity, k_clusters=k_clusters,
                                           return_distance=True)
        return result_similarity


if __name__ == '__main__':

    words = '小米8 全面屏游戏智能手机 6GB+128GB 黑色 全网通4G 双卡双待  拍照手机'
    cfg_path = './default.cfg'


    PSC = Pysparnn_Similarity_Calculation(cfg_path)

    # 初始化数据，开启训练
    PSC.data_process_init()
    result_similarity = PSC.predict(words)
    for similarity, sentence in result_similarity[0]:
        print(sentence+"  "+str(similarity))

    # 加载已处理好的数据，开启训练
    # PSC.data_process_reload()
    # result_similarity = PSC.predict(words)
    # for similarity, sentence in result_similarity[0]:
    #     print(sentence+"  "+str(similarity))


