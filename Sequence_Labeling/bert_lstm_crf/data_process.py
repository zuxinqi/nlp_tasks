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
    def init(self, vocab_file):
        """
        初始化参数，主要为4步：
        1、初始化分词器
        2、读取数据
        3、数据处理
        4、产生预训练的embedding矩阵
        :param feature_selection_name: 使用的预训练矩阵名称
        :return:
        """
        self.bert_tokenizer = self.bert_tokenizer_init(vocab_file)
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

        # 模型所需参数
        self.param = parameters()
        self.param.max_seq_length = self.config.getint('MODEL', 'max_seq_length')
        self.param.is_training = self.config.getboolean('MODEL', 'is_training')
        self.param.bert_config_file = self.config.get('MODEL', 'bert_config_file')
        self.param.init_checkpoint = self.config.get('MODEL', 'init_checkpoint')
        self.param.dropout_rate = self.config.getfloat('MODEL', 'dropout_rate')
        self.param.lstm_size = self.config.getint('MODEL', 'lstm_size')
        self.param.cell = self.config.get('MODEL', 'cell')
        self.param.cell_nums = self.config.getint('MODEL', 'cell_nums')
        self.param.learning_rate = self.config.getfloat('MODEL', 'learning_rate')
        self.param.num_train_epochs = self.config.getint('MODEL', 'num_train_epochs')
        self.param.warmup_proportion = self.config.getfloat('MODEL', 'warmup_proportion')
        self.param.batch_size = self.config.getint('MODEL', 'batch_size')
        self.param.shuffle = self.config.getboolean('MODEL', 'shuffle')
        self.param.display_per_step = self.config.getint('MODEL', 'display_per_step')
        self.param.evaluation_per_step = self.config.getint('MODEL', 'evaluation_per_step')
        self.param.require_improvement = self.config.getint('MODEL', 'require_improvement')


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
            train = -4636
            self.train_data = self.raw_data[:train]
            self.test_data = self.raw_data[train:]
        else:
            random.seed(1)
            random.shuffle(self.raw_data)
            data_len = len(self.raw_data)
            train = int(data_len * 0.8)
            self.train_data = self.raw_data[:train]
            self.test_data = self.raw_data[train:]

    # 解析数据，将数据变为数字表示
    def parse_data(self, data, bert_tokenizer, tag2id, max_length):
        """
        将原始数据转变为数字id的形式，并进行填充
        :param data:
        :param word2id:
        :param tag2id:
        :param max_length:
        :return:
        """
        all_seq = []
        for sentences, tags in data:
            tokens = []
            labels = []
            for i, word in enumerate(sentences):
                # 分词，如果是中文，就是分字,但是对于一些不在BERT的vocab.txt中得字符会被进行WordPice处理（例如中文的引号），可以将所有的分字操作替换为list(input)
                token = bert_tokenizer.tokenize(word)
                tokens.extend(token)
                label_1 = tags[i]
                for m in range(len(token)):
                    if m == 0:
                        labels.append(label_1)
                    else:  # 一般不会出现else
                        labels.append("X")
            # tokens = tokenizer.tokenize(example.text)
            # 序列截断
            if len(tokens) >= max_length - 2:
                tokens = tokens[0:(max_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
                labels = labels[0:(max_length - 2)]
            ntokens = []
            segment_ids = []
            label_ids = []
            ntokens.append("[CLS]")  # 句子开始设置CLS 标志
            segment_ids.append(0)
            # append("O") or append("[CLS]") not sure!
            label_ids.append(tag2id["[CLS]"])  # O OR CLS 没有任何影响，不过我觉得O 会减少标签个数,不过拒收和句尾使用不同的标志来标注，使用LCS 也没毛病
            for i, token in enumerate(tokens):
                ntokens.append(token)
                segment_ids.append(0)
                label_ids.append(tag2id[labels[i]])
            ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
            segment_ids.append(0)
            label_ids.append(tag2id["[SEP]"])
            input_ids = bert_tokenizer.convert_tokens_to_ids(ntokens)  # 将序列中的字(ntokens)转化为ID形式
            input_mask = [1] * len(input_ids)
            # label_mask = [1] * len(input_ids)
            # padding, 使用
            while len(input_ids) < max_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                # we don't concerned about it!
                label_ids.append(0)
                ntokens.append("**NULL**")
                # label_mask.append(0)
            # print(len(input_ids))
            assert len(input_ids) == max_length
            assert len(input_mask) == max_length
            assert len(segment_ids) == max_length
            assert len(label_ids) == max_length
            all_seq.append([input_ids,input_mask,segment_ids,label_ids])
        return all_seq


    # 将数据pad，生成batch数据返回，这里没有取余数。
    def get_batch(self, data, batch_size, shuffle=False):
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
        if shuffle:
            random.shuffle(data)
        for i in range(len(data) // batch_size):
            data_size = data[i * batch_size: (i + 1) * batch_size]
            seqs, masks,segments, labels = [], [], [], []
            for (input_ids,input_mask,segment_ids,label_ids) in data_size:
                seqs.append(input_ids)
                masks.append(input_mask)
                segments.append(segment_ids)
                labels.append(label_ids)
            yield np.array(seqs), np.array(masks), np.array(segments),np.array(labels)
        remainder = len(data) % batch_size
        if remainder != 0:
            data_size = data[-remainder:]
            seqs, masks, segments, labels = [], [], [], []
            for (input_ids, input_mask, segment_ids, label_ids) in data_size:
                seqs.append(input_ids)
                masks.append(input_mask)
                segments.append(segment_ids)
                labels.append(label_ids)
            yield np.array(seqs), np.array(masks), np.array(segments), np.array(labels)

    # 保存模型
    def save_model(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump((self.train_data, self.test_data,
                         self.tag2id, self.id2tag, self.bert_tokenizer), f)

    # 恢复模型
    def load_model(self, load_path):
        with open(load_path, 'rb') as f:
            self.train_data, self.test_data, \
            self.tag2id, self.id2tag, self.bert_tokenizer = pickle.load(f)



if __name__ == '__main__':
    dp = Date_Process()
    dp.parse_config("./default.cfg")
    dp.param.aaaa = 11
    print(dir(dp.param))
    for i in dir(dp.param):
        if "__" not in i:
            print(str(i)+"  "+str(getattr(dp.param, i)))
