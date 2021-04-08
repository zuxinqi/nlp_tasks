import numpy as np
import sys,pickle,random,configparser
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
    def init(self, vocab_file, do_lower_case=True):
        """
        初始化参数，主要为4步：
        1、初始化分词器
        2、读取数据
        3、数据处理
        :param feature_selection_name: 使用的预训练矩阵名称
        :return:
        """
        self.bert_tokenizer = self.bert_tokenizer_init(vocab_file, do_lower_case)
        self.read_data()
        self.data_process()

    def parse_config(self,config_path):
        """
        解析config文件，根据不同的参数，解析情况可能不一致
        :return:
        """
        # 基础分词器参数
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.uerdict_path = self.config.get('DEFAULT', 'uerdict_path')
        self.stopwords_path = self.config.get('DEFAULT', 'stopwords_path')
        self.tokenizer_name = self.config.get('DEFAULT', 'tokenizer_name')

        # 数据处理所需参数
        self.file_path = self.config.get('DATA_PROCESS', 'file_path')
        self.save_path = self.config.get('DATA_PROCESS', 'save_path')
        self.vocab_file = self.config.get('DATA_PROCESS', 'vocab_file')
        self.do_lower_case = self.config.get('DATA_PROCESS', 'do_lower_case')

        # 模型所需参数
        self.param = parameters()
        self.param.max_seq_length = self.config.getint('MODEL', 'max_seq_length')
        self.param.bert_config_file = self.config.get('MODEL', 'bert_config_file')
        self.param.init_checkpoint = self.config.get('MODEL', 'init_checkpoint')
        self.param.dropout_rate = self.config.getfloat('MODEL', 'dropout_rate')
        self.param.learning_rate = self.config.getfloat('MODEL', 'learning_rate')
        self.param.is_training = self.config.getboolean('MODEL', 'is_training')
        self.param.num_train_epochs = self.config.getint('MODEL', 'num_train_epochs')
        self.param.batch_size = self.config.getint('MODEL', 'batch_size')
        self.param.warmup_proportion = self.config.getfloat('MODEL', 'warmup_proportion')
        self.param.shuffle = self.config.getboolean('MODEL', 'shuffle')
        self.param.display_per_step = self.config.getint('MODEL', 'display_per_step')
        self.param.evaluation_per_step = self.config.getint('MODEL', 'evaluation_per_step')
        self.param.require_improvement = self.config.getint('MODEL', 'require_improvement')

    # 读取数据
    def read_data(self):
        with open(self.file_path, 'rb') as f:
            self.raw_train_data,self.raw_dev_data,self.raw_test_data  = pickle.load(f)

    # 数据处理（头条）
    def data_process(self):
        """
        获得word2vec，tag2vec，将原始数据按比例切割。
        :return:
        """
        self.tag2id = {"0": 0, "1": 1}
        self.id2tag = {v: k for k, v in self.tag2id.items()}

        self.train_data = self.raw_train_data
        self.dev_data = self.raw_dev_data
        self.test_data = self.raw_test_data

    # 解析数据，将数据变为数字表示
    def parse_data(self, data, bert_tokenizer, max_length):
        """
        将原始数据转变为数字id的形式，并进行填充
        根据原始句子长度制作mask_id和segment_id
        :param data:
        :param bert_tokenizer:
        :param max_length:
        :return:
        """
        all_seq = []
        for sentens1,sentens2, tag in data:
            sentens1_tokens = bert_tokenizer.tokenize(sentens1)
            sentens2_tokens = bert_tokenizer.tokenize(sentens2)


            sentens1_tokens.insert(0, "[CLS]")
            sentens1_tokens.append("[SEP]")
            sentens2_tokens.append("[SEP]")

            all_sentence = sentens1_tokens + sentens2_tokens
            segment_ids = [0] * len(sentens1_tokens) + [1] * (len(sentens2_tokens))
            task_ids = [1] * len(all_sentence)
            all_sentence = bert_tokenizer.convert_tokens_to_ids(all_sentence)

            if len(all_sentence) < max_length:
                pad = [0] * (max_length - len(all_sentence))
                all_sentence = all_sentence + pad
            else:
                all_sentence = all_sentence[:max_length]

            if len(segment_ids) < max_length:
                pad = [0] * (max_length - len(segment_ids))
                segment_ids = segment_ids + pad
            else:
                segment_ids = segment_ids[:max_length]

            if len(task_ids) < max_length:
                pad = [0] * (max_length - len(task_ids))
                task_ids = task_ids + pad
            else:
                task_ids = task_ids[:max_length]

            all_seq.append((all_sentence, segment_ids, task_ids, int(tag)))
        return all_seq

    # 将数据按批次输出
    def get_batch(self, data, batch_size,shuffle=False):
        """

        :param data:
        :param batch_size:
        :param shuffle:
        :return:
        """
        # 乱序没有加
        if shuffle:
            random.shuffle(data)
        for i in range(len(data) // batch_size):
            data_size = data[i * batch_size: (i + 1) * batch_size]
            seqs, segment_ids_,task_ids_,labels_ = [], [], [],[]
            for (all_sentence, segment_ids,task_ids,tag) in data_size:
                seqs.append(all_sentence)
                segment_ids_.append(segment_ids)
                task_ids_.append(task_ids)
                labels_.append(tag)
            yield np.array(seqs), np.array(segment_ids_),np.array(task_ids_),np.array(labels_)
        remainder = len(data) % batch_size
        if remainder !=0:
            data_size = data[-remainder:]
            seqs, segment_ids_, task_ids_, labels_ = [], [], [], []
            for (all_sentence, segment_ids, task_ids, tag) in data_size:
                seqs.append(all_sentence)
                segment_ids_.append(segment_ids)
                task_ids_.append(task_ids)
                labels_.append(tag)
            yield np.array(seqs), np.array(segment_ids_), np.array(task_ids_), np.array(
                labels_)

    # 保存模型
    def save_model(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump((self.train_data, self.test_data,
                         self.tag2id, self.id2tag, self.word2id, self.id2word, self.bert_tokenizer), f)

    # 恢复模型
    def load_model(self, load_path):
        with open(load_path, 'rb') as f:
            self.train_data, self.test_data, \
            self.tag2id, self.id2tag, self.word2id, self.id2word, self.bert_tokenizer = pickle.load(f)

