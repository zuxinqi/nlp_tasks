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
        if feature_selection_name == "word2vec":
            self.embedding_mat,self.embed_size = self.make_TX_embedding(self.word2id,self.word2vec_embed_file)
        elif feature_selection_name == "fasttext":
            self.embedding_mat,self.embed_size = self.make_fasttext_embedding(self.word2id,self.fasttext_embed_file)
        elif feature_selection_name == "self_word2vec":
            # 单独调用self_word2vec接口。
            pass
        else:
            # 默认word2vec
            self.embedding_mat,self.embed_size = self.make_TX_embedding(self.word2id,self.word2vec_embed_file)

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
        self.word2vec_embed_file = self.config.get('DATA_PROCESS', 'word2vec_embed_file')
        self.fasttext_embed_file = self.config.get('DATA_PROCESS', 'fasttext_embed_file')

        # 模型所需参数
        self.param = parameters()
        self.param.max_seq_length = self.config.getint('MODEL', 'max_seq_length')
        self.param.is_training = self.config.getboolean('MODEL', 'is_training')
        self.param.update_embedding = self.config.getboolean('MODEL', 'update_embedding')
        self.param.lstm_hidden_size = self.config.getint('MODEL', 'lstm_hidden_size')
        self.param.fc_hidden_size = self.config.getint('MODEL', 'fc_hidden_size')
        self.param.cell_nums = self.config.getint('MODEL', 'cell_nums')
        self.param.use_cnn = self.config.getboolean('MODEL', 'use_cnn')
        self.param.use_lstm = self.config.getboolean('MODEL', 'use_lstm')
        self.param.kernel_nums = self.config.getint('MODEL', 'kernel_nums')
        self.param.kernel_size = self.config.getint('MODEL', 'kernel_size')
        self.param.learning_rate = self.config.getfloat('MODEL', 'learning_rate')
        self.param.clip = self.config.getint('MODEL', 'clip')
        self.param.dropout_rate = self.config.getfloat('MODEL', 'dropout_rate')
        self.param.num_train_epochs = self.config.getint('MODEL', 'num_train_epochs')
        self.param.batch_size = self.config.getint('MODEL', 'batch_size')
        self.param.shuffle = self.config.getboolean('MODEL', 'shuffle')
        self.param.use_l2_regularization = self.config.getboolean('MODEL', 'use_l2_regularization')
        self.param.use_decay_learning_rate = self.config.getboolean('MODEL', 'use_decay_learning_rate')
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

        # 产生tag2id
        all_tag = ["O"]
        for sentences, tags in self.raw_data:
            for tag in tags:
                if len(tag)>1:
                    tag = "_".join(tag.split("_")[1:])
                if tag not in all_tag:
                    all_tag.append(tag)
        self.tag2id = {all_tag[i]: i for i in range(len(all_tag))}
        self.id2tag = {v: k for k, v in self.tag2id.items()}

        # 产生tag2id
        all_tag2 = ["O"]
        for sentences, tags in self.raw_data:
            for tag in tags:
                if tag not in all_tag2:
                    all_tag2.append(tag)
        self.BIO_tag2id = {all_tag2[i]: i for i in range(len(all_tag2))}
        self.BIO_id2tag = {v: k for k, v in self.BIO_tag2id.items()}


        # 产生word2id
        word_list = []
        word_list.append('<PAD>')
        word_list.append('<UNK>')
        for sentences, tags in self.raw_data:
            for word in sentences:
                if word not in word_list:
                    word_list.append(word)
        for index, word in enumerate(word_list):
            self.word2id[word] = index
            self.id2word[index] = word

        if "MSRA" in self.file_path:
            train = -4631
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
    def parse_data(self, data, word2id):
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
            sents = [word2id[sent] if sent in word2id else word2id['<UNK>'] for sent in sentences]
            all_seq.append((sents, tags, len(sents)))
        return all_seq

    def find_labels_index(self,labels,max_seq_length):
        res_dict = {}
        for index, one_label in enumerate(labels):
            i = 0
            one_label = one_label[:max_seq_length]
            while i < len(one_label):
                if one_label[i] == 'O':
                    j = i + 1
                    while j < len(one_label) and one_label[j] == 'O':
                        j += 1
                else:
                    if one_label[i][0] != 'B':
                        #                 print(tags[i][0] + ' error start')
                        j = i + 1
                    else:
                        # if tags[i][2:] not in res_dict:
                        #     res_dict[tags[i][2:]] = []
                        j = i + 1
                        while j < len(one_label) and one_label[j][0] == 'I' and one_label[j][2:] == one_label[i][2:]:
                            j += 1
                        res_dict[(index, i, j-1)] = one_label[i][2:]
                i = j
        return res_dict

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
        if shuffle:
            random.shuffle(data)
        pad = word2id['<PAD>']
        for i in range(len(data) // batch_size):
            data_size = data[i * batch_size: (i + 1) * batch_size]
            seqs, labels, sentence_legth = [], [], []
            for (sent_, tag_, seq_legth) in data_size:
                seqs.append(sent_)
                labels.append(tag_)
                sentence_legth.append(seq_legth)
            max_l = max_seq_length
            res_seq = []
            for sent in seqs:
                if len(sent) > max_l:
                    res_seq.append(sent[:max_l])
                    continue
                sent_new = np.concatenate((sent, np.tile(pad, max_l - len(sent))), axis=0)  # 以pad的形式补充成等长的帧数
                res_seq.append(sent_new)

            res_labels = np.zeros((batch_size,max_seq_length,max_seq_length))
            res_dict = self.find_labels_index(labels,max_seq_length)
            for k,v in res_dict.items():
                res_labels[k[0]][k[1]][k[2]] = tag2id[v]

            new_sentence_legth = []
            for sentence_len in sentence_legth:
                if sentence_len > max_l:
                    sentence_len = max_l
                new_sentence_legth.append(sentence_len)

            yield np.array(res_seq), np.array(res_labels), new_sentence_legth
        remainder = len(data) % batch_size
        if remainder != 0:
            data_size = data[-remainder:]
            seqs, labels, sentence_legth = [], [], []
            for (sent_, tag_, seq_legth) in data_size:
                seqs.append(sent_)
                labels.append(tag_)
                sentence_legth.append(seq_legth)
            max_l = max_seq_length
            res_seq = []
            for sent in seqs:
                if len(sent) > max_l:
                    res_seq.append(sent[:max_l])
                    continue
                sent_new = np.concatenate((sent, np.tile(pad, max_l - len(sent))), axis=0)  # 以pad的形式补充成等长的帧数
                res_seq.append(sent_new)

            res_labels = np.zeros((remainder, max_seq_length, max_seq_length))
            res_dict = self.find_labels_index(labels,max_seq_length)
            for k, v in res_dict.items():
                res_labels[k[0]][k[1]][k[2]] = tag2id[v]

            new_sentence_legth = []
            for sentence_len in sentence_legth:
                if sentence_len > max_l:
                    sentence_len = max_l
                new_sentence_legth.append(sentence_len)

            yield np.array(res_seq), np.array(res_labels), new_sentence_legth

    # 保存模型
    def save_model(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump((self.train_data, self.test_data,
                         self.tag2id, self.id2tag,self.BIO_tag2id,self.BIO_id2tag, self.word2id, self.id2word, self.embedding_mat), f)

    # 恢复模型
    def load_model(self, load_path):
        with open(load_path, 'rb') as f:
            self.train_data, self.test_data, \
            self.tag2id, self.id2tag, self.BIO_tag2id,self.BIO_id2tag,self.word2id, self.id2word, self.embedding_mat = pickle.load(f)



if __name__ == '__main__':
    dp = Date_Process()
    dp.parse_config("./default.cfg")
    dp.param.aaaa = 11
    print(dir(dp.param))
    for i in dir(dp.param):
        if "__" not in i:
            print(str(i)+"  "+str(getattr(dp.param, i)))
