import sys,pickle,configparser
import numpy as np
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding
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
    def init(self,vocab_file,do_lower_case=True):
        """
        初始化参数，主要为4步：
        1、初始化分词器
        2、读取数据
        3、数据处理
        4、产生预训练的embedding矩阵
        :param feature_selection_name: 使用的预训练矩阵名称
        :return:
        """
        self.tokenizer = Tokenizer(vocab_file, do_lower_case=do_lower_case)
        self.raw_data = self.read_data(self.file_path)


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

        # 模型所需参数
        self.param = parameters()
        self.param.max_seq_length = self.config.getint('MODEL', 'max_seq_length')
        self.param.bert_config_file = self.config.get('MODEL', 'bert_config_file')
        self.param.init_checkpoint = self.config.get('MODEL', 'init_checkpoint')
        self.param.model_save_path = self.config.get('MODEL', 'model_save_path')


    # 读取数据
    def read_data(self, file_path):
        """
        读取训练数据
        :param file_path: str 训练文件的路径
        :return: list 每个文档作为一个子元素放在列表中
        """
        word_list = []
        with open(file_path, 'r', encoding="utf-8-sig") as f:
            for line in f.readlines():
                word_list.append(line.strip())
        return word_list


    def parse_data(self,data, tokenizer, max_seq_length):
        """
        数据解析，变为token_ids
        :param data: 输入数据
        :param tokenizer: bert的tokenizer
        :param max_seq_length: 最大的文本长度
        :return: all_token_ids 所有句子变为token_ids的列表
        """
        all_token_ids = []
        for d in data:
            token_ids = tokenizer.encode(d, maxlen=max_seq_length)[0]
            all_token_ids.append(token_ids)
        all_token_ids = sequence_padding(all_token_ids)
        return all_token_ids


    def compute_kernel_bias(self, vecs):
        """计算kernel和bias
        最后的变换：y = (x + bias).dot(kernel)
        """
        #     vecs = np.concatenate(vecs, axis=0)
        mu = vecs.mean(axis=0, keepdims=True)
        cov = np.cov(vecs.T)
        u, s, vh = np.linalg.svd(cov)
        W = np.dot(u, np.diag(1 / np.sqrt(s)))
        # return None, None
        # return W, -mu
        return W[:, :256], -mu

    def transform_and_normalize(self, vecs, kernel=None, bias=None):
        """应用变换，然后标准化
        """
        if not (kernel is None or bias is None):
            vecs = (vecs + bias).dot(kernel)
        return vecs / (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5

    def cos_similarity(self, a, b):
        """
        进行余弦相似度计算
        :param a: 向量a
        :param b: 向量b
        :return:
        """
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        cos = np.dot(a, b) / (a_norm * b_norm)
        return cos

    def get_similarities(self,all_data,test_data):
        """
        单句与所有句子逐一做余弦相似度，返回相似度结果列表
        :param all_data: 所有句子的特征表示列表
        :param test_data: 单句的特征表示列表
        :return: similarities 单句与所有句子逐一计算余弦相似度，所有相似度放在列表中
        """
        similarities = []
        for i in all_data:
            similarity = self.cos_similarity(i,test_data)
            similarities.append(similarity)
        return similarities

    def save_model(self,save_path):
        with open(save_path, 'wb') as f:
            pickle.dump((self.raw_data, self.tokenizer), f)

    def load_model(self,load_path):
        with open(load_path, 'rb') as f:
            self.raw_data, self.tokenizer = pickle.load(f)
