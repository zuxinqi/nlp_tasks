import numpy as np
import sys,copy,pickle,random,configparser
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
    def init(self, embedding_file, vocab_file, embed_size,feature_selection_name="word2vec"):
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
            self.embedding_mat = self.load_char_embed(embedding_file, vocab_file, embed_size)
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
        self.embedding_file = self.config.get('DATA_PROCESS', 'embedding_file')
        self.vocab_file = self.config.get('DATA_PROCESS', 'vocab_file')
        self.embed_size = self.config.getint('DATA_PROCESS', 'embed_size')

        # 模型所需参数
        self.param = parameters()
        self.param.max_seq_length = self.config.getint('MODEL', 'max_seq_length')
        self.param.is_training = self.config.getboolean('MODEL', 'is_training')
        self.param.update_embedding = self.config.getboolean('MODEL', 'update_embedding')
        self.param.embedding_hidden_size = self.config.getint('MODEL', 'embedding_hidden_size')
        self.param.hidden_units = self.config.getint('MODEL', 'hidden_units')
        self.param.filters = self.config.getint('MODEL', 'filters')
        self.param.kernel_size = self.config.getint('MODEL', 'kernel_size')
        self.param.distance_selection = self.config.get('MODEL', 'distance_selection')
        self.param.learning_rate = self.config.getfloat('MODEL', 'learning_rate')
        self.param.dropout_rate = self.config.getfloat('MODEL', 'dropout_rate')
        self.param.num_train_epochs = self.config.getint('MODEL', 'num_train_epochs')
        self.param.batch_size = self.config.getint('MODEL', 'batch_size')
        self.param.shuffle = self.config.getboolean('MODEL', 'shuffle')
        self.param.display_per_step = self.config.getint('MODEL', 'display_per_step')
        self.param.evaluation_per_step = self.config.getint('MODEL', 'evaluation_per_step')
        self.param.require_improvement = self.config.getint('MODEL', 'require_improvement')

    # 加载自定义的词向量
    def load_char_embed(self,embedding_file, vocab_file, embed_size):
        """
        加载自定义的词向量
        :param embedding_file: 自定义词向量保存的路径
        :param vocab_file: 词表路径
        :param embed_size:  词向量的长度
        :return:
        """
        def get_coefs(word, *arr):
            return word, np.asarray(arr, dtype='float32')

        embeddings_index = dict(
            get_coefs(*o.split(" ")) for o in open(embedding_file, encoding="utf8", errors='ignore') if len(o) > 100)

        all_embs = np.stack(embeddings_index.values())
        emb_mean, emb_std = np.mean(all_embs), np.std(all_embs)
        vocab = [line.strip() for line in open(vocab_file, encoding='utf-8').readlines()]
        word2idx = {word: index for index, word in enumerate(vocab, start=1)}
        word2idx['<PAD>'] = 0
        word2idx['<UNK>'] = len(word2idx)

        embedding_matrix = np.random.normal(emb_mean, emb_std, (len(word2idx), embed_size))
        print(embedding_matrix.shape)
        for word, i in word2idx.items():
            if i >= len(word2idx): continue
            try:
                embedding_vector = embeddings_index.get(word)
                if not np.mean(embedding_vector): print(embedding_vector)
                embedding_matrix[i] = embedding_vector
            except:
                pass
        embedding_matrix[0] = np.zeros(embed_size)

        self.word2id = word2idx
        self.id2word = {v: k for k, v in self.word2id.items()}

        print(np.mean(embedding_matrix), np.std(embedding_matrix))

        return embedding_matrix

    # 读取数据
    def read_data(self):
        with open(self.file_path, 'rb') as f:
            self.raw_train_data,self.raw_dev_data,self.raw_test_data  = pickle.load(f)


    # 数据处理（头条）
    def data_process(self):
        """
        获得tag2id，将原始数据进行解析。
        word2id将在加载自定义词向量的时候获得
        :return:
        """

        # 产生tag2id
        self.tag2id = {"0": 0, "1": 1}
        self.id2tag = {v: k for k, v in self.tag2id.items()}

        # 句子太短不去停用词
        self.stopwords = []

        for sentence1, sentence2, tag in self.raw_train_data:
            sentence1 = self.tokenizer.cut(sentence1)
            sentence1 = [i for i in sentence1 if i not in self.stopwords]
            sentence2 = self.tokenizer.cut(sentence2)
            sentence2 = [i for i in sentence2 if i not in self.stopwords]

            self.train_data.append([sentence1, sentence2, tag])

        for sentence1, sentence2, tag in self.raw_dev_data:
            sentence1 = self.tokenizer.cut(sentence1)
            sentence1 = [i for i in sentence1 if i not in self.stopwords]
            sentence2 = self.tokenizer.cut(sentence2)
            sentence2 = [i for i in sentence2 if i not in self.stopwords]
            self.dev_data.append([sentence1, sentence2, tag])

        for sentence1, sentence2, tag in self.raw_test_data:
            sentence1 = self.tokenizer.cut(sentence1)
            sentence1 = [i for i in sentence1 if i not in self.stopwords]
            sentence2 = self.tokenizer.cut(sentence2)
            sentence2 = [i for i in sentence2 if i not in self.stopwords]
            self.test_data.append([sentence1, sentence2, tag])

    # 解析数据，将数据变为数字表示
    def parse_data(self, data, word2id, max_seq_length):
        """
        将原始数据转变为数字id的形式，并进行填充
        :param data: 输入数据
        :param word2id: 字典，类似{"word1":0,"word2":1,"word3":2......}
        :param max_seq_length: 设置好的最大文本长度
        :return:
        """
        all_seq = []
        for sentens1,sentens2, tag in data:
            sents1 = [word2id[word] if word in word2id else word2id["<UNK>"] for word in sentens1]
            sents2 = [word2id[word] if word in word2id else word2id["<UNK>"] for word in sentens2]
            original_length1 = len(sents1)
            original_length2 = len(sents2)
            if original_length1 < max_seq_length:
                pad = [word2id['<PAD>']] * (max_seq_length - original_length1)
                sents1 = sents1 + pad

            if original_length2 < max_seq_length:
                pad = [word2id['<PAD>']] * (max_seq_length - original_length2)
                sents2 = sents2 + pad

            if original_length1 > max_seq_length:
                sents1 = sents1[:max_seq_length]

            if original_length2 > max_seq_length:
                sents2 = sents2[:max_seq_length]

            all_seq.append((sents1, sents2, int(tag)))
        return all_seq


    # 将数据pad，生成batch数据返回，这里没有取余数。
    def get_batch(self, data, batch_size, shuffle=False):
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
            seqs1, seqs2, labels = [], [], []
            for (sent_1, sent_2, tag_) in data_size:
                seqs1.append(sent_1)
                seqs2.append(sent_2)
                labels.append(tag_)
            yield np.array(seqs1), np.array(seqs2), np.array(labels)
        remainder = len(data) % batch_size
        if remainder != 0:
            data_size = data[-remainder:]
            seqs1, seqs2, labels = [], [], []
            for (sent_1, sent_2, tag_) in data_size:
                seqs1.append(sent_1)
                seqs2.append(sent_2)
                labels.append(tag_)
            yield np.array(seqs1), np.array(seqs2), np.array(labels)

    # 保存模型
    def save_model(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump((self.train_data, self.test_data, self.word2id, self.id2word,
                         self.tag2id, self.id2tag,  self.embedding_mat), f)

    # 恢复模型
    def load_model(self, load_path):
        with open(load_path, 'rb') as f:
            self.train_data, self.test_data, self.word2id, self.id2word, \
            self.tag2id, self.id2tag, self.embedding_mat = pickle.load(f)
