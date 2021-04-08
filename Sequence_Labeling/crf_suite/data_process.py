import numpy as np
import sys,pickle,random,configparser
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.append('./Base_Line')
sys.path.append('../../Base_Line')
from base_data_process import Base_Process

class parameters():
    def __init__(self):
        self.learning_rate = 0.0001
        # self.is_training = True
        # self.update_embedding = True
        # self.num_train_epochs = 10
        # self.batch_size = 64
        # self.shuffle = True
        # self.dropout_rate = 0.9
        # self.display_per_step = 100
        # self.evaluation_per_step = 500
        # self.require_improvement = 1

class Date_Process(Base_Process):

    # 初始化模型,构造分词器、训练语料分词、生成embedding矩阵
    def init(self, feature_selection_name="word2vec"):
        """
        初始化参数，主要为4步：
        1、初始化分词器
        2、读取数据
        3、数据处理
        4、产生预训练的embedding矩阵
        :param feature_selection_name: 使用的预训练矩阵名称
        :return:
        """
        self.tokenizer, self.stopwords = self.make_tokenizer()
        self.read_data()
        self.data_process()

    def parse_config(self,config_path):
        """
        解析config文件，根据不同的参数，解析情况可能不一致
        :return:
        """
        # 基础分词器参数
        self.config = configparser.ConfigParser()
        self.config.read(config_path,encoding="utf-8-sig")
        self.uerdict_path = self.config.get('DEFAULT', 'uerdict_path')
        self.stopwords_path = self.config.get('DEFAULT', 'stopwords_path')
        self.tokenizer_name = self.config.get('DEFAULT', 'tokenizer_name')

        # 数据处理所需参数
        self.file_path = self.config.get('DATA_PROCESS', 'file_path')
        self.save_path = self.config.get('DATA_PROCESS', 'save_path')
        self.word2vec_embed_file = self.config.get('DATA_PROCESS', 'word2vec_embed_file')
        self.fasttext_embed_file = self.config.get('DATA_PROCESS', 'fasttext_embed_file')

        # 模型所需参数
        self.param = parameters()

    # 读取数据
    def read_data(self):
        with open(self.file_path, 'rb') as f:
            self.raw_data = pickle.load(f)

    # 数据处理（头条）
    def data_process(self):
        """
        获得word2id，tag2id，将原始数据按比例切割。
        :return:
        """
        all_tag = []
        all_tag += ["[SEP]", "[CLS]", "X"]
        for sentences,tags in self.raw_data:
            for tag in tags:
                if tag not in all_tag:
                    all_tag.append(tag)
        self.tag2id = {all_tag[i]: i+1 for i in range(len(all_tag))}
        self.id2tag = {v: k for k, v in self.tag2id.items()}


        if "MSRA" in self.file_path:
            train = -4631
            for sentences, tags in self.raw_data[:train]:
                one_line = []
                for s, t in zip(sentences, tags):
                    one_line.append([s, t])
                self.train_data.append(one_line)
            for sentences, tags in self.raw_data[train:]:
                one_line = []
                for s, t in zip(sentences, tags):
                    one_line.append([s, t])
                self.test_data.append(one_line)
        else:
            random.seed(1)
            random.shuffle(self.raw_data)
            data_len = len(self.raw_data)
            train = int(data_len * 0.8)
            for sentences, tags in self.raw_data[:train]:
                one_line = []
                for s, t in zip(sentences, tags):
                    one_line.append([s, t])
                self.train_data.append(one_line)
            for sentences, tags in self.raw_data[train:]:
                one_line = []
                for s, t in zip(sentences, tags):
                    one_line.append([s, t])
                self.test_data.append(one_line)

    # 解析数据，将数据变为数字表示
    def parse_data(self, data, word2id, tag2id):
        """
        将原始数据转变为数字id的形式，并进行填充
        :param data:
        :param word2id:
        :param tag2id:
        :param max_length:
        :return:
        """
        pass

    # 将数据pad，生成batch数据返回，这里没有取余数。
    def get_batch(self, data, batch_size, word2id, tag2id, max_seq_length,shuffle=False):
        """

        :param data:
        :param batch_size:
        :param vocab:
        :param tag2label:
        :param max_seq_length:
        :param shuffle:
        :return:
        """
        # 乱序没有加
        pass

    # 保存模型
    def save_model(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump((self.train_data, self.test_data,
                         self.tag2id, self.id2tag, self.word2id, self.id2word, self.embedding_mat), f)

    # 恢复模型
    def load_model(self, load_path):
        with open(load_path, 'rb') as f:
            self.train_data, self.test_data, \
            self.tag2id, self.id2tag, self.word2id, self.id2word, self.embedding_mat = pickle.load(f)



if __name__ == '__main__':
    dp = Date_Process()
    dp.parse_config("./default.cfg")
    dp.param.aaaa = 11
    print(dir(dp.param))
    for i in dir(dp.param):
        if "__" not in i:
            print(str(i)+"  "+str(getattr(dp.param, i)))
