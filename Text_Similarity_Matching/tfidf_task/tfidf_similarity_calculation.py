# -*- coding:utf-8 -*-
import os
from data_process import Date_Process

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Tfidf_Similarity_Calculation(object):
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
        return cut_word

    def predict(self,words, return_quantity=10, threshold=0):
        """
        传入文本，输出与其相似度在前TopK的文本
        :param words: list 传入的文本的列表
        :param return_quantity: int 返回文本数量
        :param threshold: int 最低相似度阈值，低于此值的结果不返回
        :return:
        """
        # 恢复模型参数，并针对测试修改部分参数
        cut_words = self.split_one_words(words, self.dp.tokenizer, self.dp.stopwords)
        bow_vector = self.dp.dictionary.doc2bow(cut_words)
        bow_tfidf = self.dp.tfidf[bow_vector]
        similarities = self.dp.sparse_matrix.get_similarities(bow_tfidf)
        sorted_res = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
        result_similarity = []
        for i in sorted_res[:int(return_quantity)]:
            if i[1] > threshold:
                result_similarity.append([self.dp.word_list[i[0]], i[1]])
        return result_similarity


if __name__ == '__main__':

    words = '小米8 全面屏游戏智能手机 6GB+128GB 黑色 全网通4G 双卡双待  拍照手机'
    cfg_path = './default.cfg'


    TSC = Tfidf_Similarity_Calculation(cfg_path)

    # 初始化数据，开启训练
    # TSC.data_process_init()
    # result_similarity = TSC.predict(words)
    # for sentence,similarity in result_similarity:
    #     print(sentence+"  "+str(similarity))

    # 加载已处理好的数据，开启训练
    TSC.data_process_reload()
    result_similarity = TSC.predict(words)
    for sentence,similarity in result_similarity:
        print(sentence+"  "+str(similarity))


