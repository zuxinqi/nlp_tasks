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
        self.param.num_filters = self.config.getint('MODEL', 'num_filters')
        self.param.kernel_size = self.config.getint('MODEL', 'kernel_size')
        self.param.embedding_dim = self.config.getint('MODEL', 'embedding_dim')
        self.param.is_training = self.config.getboolean('MODEL', 'is_training')
        self.param.update_embedding = self.config.getboolean('MODEL', 'update_embedding')
        self.param.use_l2_regularization = self.config.getboolean('MODEL', 'use_l2_regularization')
        self.param.learning_rate = self.config.getfloat('MODEL', 'learning_rate')
        self.param.use_decay_learning_rate = self.config.getboolean('MODEL', 'use_decay_learning_rate')
        self.param.num_train_epochs = self.config.getint('MODEL', 'num_train_epochs')
        self.param.batch_size = self.config.getint('MODEL', 'batch_size')
        self.param.shuffle = self.config.getboolean('MODEL', 'shuffle')
        self.param.dropout_rate = self.config.getfloat('MODEL', 'dropout_rate')
        self.param.display_per_step = self.config.getint('MODEL', 'display_per_step')
        self.param.evaluation_per_step = self.config.getint('MODEL', 'evaluation_per_step')
        self.param.require_improvement = self.config.getint('MODEL', 'require_improvement')


    # 数据处理（头条）
    def data_process(self):
        """
        获得word2id，tag2id，将原始数据按比例切割。
        :return:
        """
        all_tag = []
        for i in self.raw_data:
            if i[-1] not in all_tag:
                all_tag.append(i[-1])
        self.tag2id = {all_tag[i]: i for i in range(len(all_tag))}
        self.id2tag = {v: k for k, v in self.tag2id.items()}

        for snentence_1, snentence_2, tag in self.raw_data:
            seg_list = self.tokenizer.cut(snentence_2)
            seg_list = [i for i in seg_list if i not in self.stopwords]
            self.all_data.append([seg_list, tag])

        if self.tokenizer_name != "single":
            # 如果不是按字作为特征，则引入tfidf的初始化帮助清洗数据
            all_texts = [" ".join(i[0]) for i in self.all_data]
            tfidf_vec = TfidfVectorizer(max_features=15000, max_df=0.8, min_df=4)
            tfidf_matrix = tfidf_vec.fit_transform(all_texts)
            word_list = []
            word_list.append('<PAD>')
            word_list.append('<UNK>')
            for word in tfidf_vec.vocabulary_.keys():
                if word not in word_list:
                    word_list.append(word)
            for index, word in enumerate(word_list):
                self.word2id[word] = index
                self.id2word[index] = word
        else:
            word_list = []
            word_list.append('<PAD>')
            word_list.append('<UNK>')
            for seg_list, tag in self.all_data:
                for word in seg_list:
                    if word not in word_list:
                        word_list.append(word)
            for index, word in enumerate(word_list):
                self.word2id[word] = index
                self.id2word[index] = word


        random.seed(1)
        random.shuffle(self.all_data)
        data_len = len(self.all_data)
        train = int(data_len * 0.8)
        self.train_data = self.all_data[:train]
        self.test_data = self.all_data[train:]

    # 解析数据，将数据变为数字表示
    def parse_data(self, data, word2id, tag2id, max_length):
        """
        将原始数据转变为数字id的形式，并进行填充
        :param data:
        :param word2id:
        :param tag2id:
        :param max_length:
        :return:
        """
        all_seq = []
        for words, tag in data:
            sents = [word2id[word] if word in word2id else word2id["<UNK>"] for word in words]
            tag = tag2id[tag]
            original_length = len(sents)
            if original_length < max_length:
                pad = [word2id['<PAD>']] * (max_length - original_length)
                sents = sents + pad
            if original_length > max_length:
                sents = sents[:max_length]
                original_length = max_length
            all_seq.append((sents, tag, original_length))
        return all_seq

    # 将数据按批次输出
    def get_batch(self, data, batch_size, max_seq_length, shuffle=False):
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
            seqs, labels, sentence_legth = [], [], []
            for (sent_, tag_, seq_legth) in data_size:
                seqs.append(sent_)
                labels.append(tag_)
                if seq_legth > max_seq_length:
                    seq_legth = max_seq_length
                sentence_legth.append(seq_legth)
            yield np.array(seqs), np.array(labels), sentence_legth
        remainder = len(data) % batch_size
        if remainder != 0:
            data_size = data[-remainder:]
            seqs, labels, sentence_legth = [], [], []
            for (sent_, tag_, seq_legth) in data_size:
                seqs.append(sent_)
                labels.append(tag_)
                if seq_legth > max_seq_length:
                    seq_legth = max_seq_length
                sentence_legth.append(seq_legth)
            yield np.array(seqs), np.array(labels), sentence_legth

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
