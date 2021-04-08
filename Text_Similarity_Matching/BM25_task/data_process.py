import sys,pickle,configparser
from rank_bm25 import BM25Okapi
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
    def init(self):
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
        self.raw_data = self.read_data(self.file_path)
        self.bm25, self.word_list = self.data_process(self.raw_data, self.tokenizer, self.stopwords)


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

    # 输入数据，将其中句子进行分词
    def split_words(self, data, tokenizer, stopwords):
        """
        对传入数据进行分词
        :param data: list 传入数据
        :param tokenizer: Object 分词器
        :param stopwords: list 停用词表
        :return: word_list, cut_word_list 原始语句列表，分词语句列表
        """
        word_list = []
        cut_word_list = []
        for line in data:
            seg_list = tokenizer.cut(line.strip())
            seg_list = [i for i in seg_list if i not in stopwords and i != ' ']
            word_list.append(line.strip())
            cut_word_list.append(seg_list)
        return word_list, cut_word_list

    # 传入数据，初始化BM25算法对象
    def data_process(self,data,tokenizer,stopwords):
        """
        传入数据，初始化BM25算法对象
        :param data: list 输入数据
        :param tokenizer: Object 分词器
        :param stopwords: list 停用词表
        :return:
        """
        word_list, cut_word_list = self.split_words(data, tokenizer, stopwords)
        bm25 = BM25Okapi(cut_word_list)
        return bm25, word_list

    def save_model(self,save_path):
        with open(save_path, 'wb') as f:
            pickle.dump((self.bm25, self.word_list), f)

    def load_model(self,load_path):
        with open(load_path, 'rb') as f:
            self.bm25, self.word_list = pickle.load(f)
