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
        4、产生预训练的embedding矩阵
        :param feature_selection_name: 使用的预训练矩阵名称
        :return:
        """
        self.bert_tokenizer = self.bert_tokenizer_init(vocab_file,do_lower_case)
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
        self.do_lower_case = self.config.getboolean('DATA_PROCESS', 'do_lower_case')

        # 模型所需参数
        self.param = parameters()
        self.param.max_seq_length = self.config.getint('MODEL', 'max_seq_length')
        self.param.bert_config_file = self.config.get('MODEL', 'bert_config_file')
        self.param.init_checkpoint = self.config.get('MODEL', 'init_checkpoint')
        self.param.dropout_rate = self.config.getfloat('MODEL', 'dropout_rate')
        self.param.use_l2_regularization = self.config.getboolean('MODEL', 'use_l2_regularization')
        self.param.learning_rate = self.config.getfloat('MODEL', 'learning_rate')
        self.param.do_lower_case = self.config.getboolean('MODEL', 'do_lower_case')
        self.param.is_training = self.config.getboolean('MODEL', 'is_training')
        self.param.use_hidden_layer = self.config.getboolean('MODEL', 'use_hidden_layer')
        self.param.hidden_num = self.config.getint('MODEL', 'hidden_num')
        self.param.num_train_epochs = self.config.getint('MODEL', 'num_train_epochs')
        self.param.batch_size = self.config.getint('MODEL', 'batch_size')
        self.param.warmup_proportion = self.config.getfloat('MODEL', 'warmup_proportion')
        self.param.shuffle = self.config.getboolean('MODEL', 'shuffle')
        self.param.display_per_step = self.config.getint('MODEL', 'display_per_step')
        self.param.evaluation_per_step = self.config.getint('MODEL', 'evaluation_per_step')
        self.param.require_improvement = self.config.getint('MODEL', 'require_improvement')

    # 数据处理（头条）
    def data_process(self):
        """
        获得word2vec，tag2vec，将原始数据按比例切割。
        :return:
        """
        all_tag = []
        for i in self.raw_data:
            if i[-1] not in all_tag:
                all_tag.append(i[-1])
        self.tag2id = {all_tag[i]: i for i in range(len(all_tag))}
        self.id2tag = {v: k for k, v in self.tag2id.items()}

        for snentence_1, snentence_2, tag in self.raw_data:
            self.all_data.append([snentence_2, tag])


        random.seed(1)
        random.shuffle(self.all_data)
        data_len = len(self.all_data)
        train = int(data_len * 0.8)
        self.train_data = self.all_data[:train]
        self.test_data = self.all_data[train:]

    # 解析数据，将数据变为数字表示
    def parse_data(self, data, bert_tokenizer, tag2id, max_length):
        """
        将原始数据转变为数字id的形式，并进行填充
        根据原始句子长度制作mask_id和segment_id
        :param data:
        :param word2id:
        :param tag2id:
        :param max_length:
        :return:
        """
        all_seq = []
        for words, tag in data:
            tokens = bert_tokenizer.tokenize(words)

            if len(tokens) > max_length - 2:
                tokens = tokens[0:(max_length - 2)]

            tokens.insert(0, "[CLS]")
            tokens.append("[SEP]")

            input_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
            task_ids = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            tag = tag2id[tag]

            while len(input_ids) < max_length:
                input_ids.append(0)
                task_ids.append(0)
                segment_ids.append(0)


            all_seq.append((input_ids, task_ids, segment_ids, tag))
        return all_seq

    # 将数据按批次输出
    def get_batch(self,data, batch_size, shuffle):
        """

        :param data:
        :param batch_size:
        :param vocab:
        :param tag2label:
        :param shuffle:
        :return:
        """
        if shuffle:
            random.shuffle(data)
        for i in range(len(data) // batch_size):
            data_size = data[i * batch_size: (i + 1) * batch_size]
            input_list, mask_list, segment_list, label_list = [], [], [], []
            for (input_ids, input_mask, segment_ids, label_ids) in data_size:
                input_list.append(input_ids)
                mask_list.append(input_mask)
                segment_list.append(segment_ids)
                label_list.append(label_ids)
            yield np.array(input_list), np.array(mask_list), np.array(segment_list), np.array(
                label_list)
        remainder = len(data) % batch_size
        if remainder != 0:
            data_size = data[-remainder:]
            input_list, mask_list, segment_list, label_list = [], [], [], []
            for (input_ids, input_mask, segment_ids, label_ids) in data_size:
                input_list.append(input_ids)
                mask_list.append(input_mask)
                segment_list.append(segment_ids)
                label_list.append(label_ids)
            yield np.array(input_list), np.array(mask_list), np.array(segment_list), np.array(
                label_list)

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



if __name__ == '__main__':
    pass
