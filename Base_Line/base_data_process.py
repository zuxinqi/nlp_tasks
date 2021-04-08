import copy,zhconv,jieba,pkuseg
import random,pickle,configparser
import numpy as np
from single_tokenizer import Single_Tokenizer
from gensim.models import FastText

class parameters():
    # 定义模型参数的类
    def __init__(self):
        self.learning_rate = 0.0001

class Base_Process():
    def __init__(self):
        """
        定义初始化值
        """
        self.file_path = ""
        self.config = None
        self.tokenizer_name = ""
        self.stopwords_path = ""
        self.uerdict_path = ""
        self.tokenizer = None
        self.bert_tokenizer = None
        self.stopwords = []
        self.raw_data = []
        self.all_data = []
        self.train_data = []
        self.dev_data = []
        self.test_data = []
        self.tag2id = {}
        self.id2tag = {}
        self.word2id = {}
        self.id2word = {}
        self.max_seq_length = 0
        self.embed_size = 0
        self.word2vec_embed_file =""
        self.fasttext_embed_file =""
        self.embedding_mat = None

    def parse_config(self,config_path):
        """
        解析config文件，根据不同的参数，解析情况可能不一致，此为示例，基本都要重写此函数
        :return:
        """
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.uerdict_path = self.config.get('DEFAULT', 'uerdict_path')
        self.stopwords_path = self.config.get('DEFAULT', 'stopwords_path')
        self.tokenizer_name = self.config.get('DEFAULT', 'tokenizer_name')

        self.param = parameters()
        self.param.max_seq_length = self.config.getint('MODEL', 'max_seq_length')

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

    # 选择分词器
    def make_tokenizer(self):
        """
        设置初始的分词器，目前支持jieba和pkuseg两种
        支持添加自定义词典，在此函数中也会读取stopwords并返回
        :return:
        """
        if self.tokenizer_name not in ["jieba", "pkuseg", "single"]:
            self.tokenizer_name = "jieba"
        if self.tokenizer_name == "jieba":
            jieba.set_dictionary("./dict.txt")
            if self.uerdict_path != "":
                # 加载自定义词典
                jieba.load_userdict(self.uerdict_path)
            if self.stopwords_path != "":
                # 将停用词读出放在stopwords这个列表中
                stopwords = [line.strip() for line in open(self.stopwords_path, 'r', encoding='utf-8').readlines()]
            else:
                stopwords = []
            return jieba, stopwords
        elif self.tokenizer_name == "pkuseg":
            if self.uerdict_path != "":
                lexicon = [line.strip().split(" ")[0] for line in
                           open("./userdict.txt", 'r', encoding='utf-8').readlines() if
                           len(line.strip().split(" ")[0]) != 0]
                # 加载自定义词典
                seg = pkuseg.pkuseg(user_dict=lexicon)
            else:
                seg = pkuseg.pkuseg()
            if self.stopwords_path != "":
                # 将停用词读出放在stopwords这个列表中
                stopwords = [line.strip() for line in open(self.stopwords_path, 'r', encoding='utf-8').readlines()]
            else:
                stopwords = []
            return seg, stopwords
        elif self.tokenizer_name == "single":
            st = Single_Tokenizer()
            if self.stopwords_path != "":
                # 将停用词读出放在stopwords这个列表中
                stopwords = [line.strip() for line in open(self.stopwords_path, 'r', encoding='utf-8').readlines()]
            else:
                stopwords = []
            return st, stopwords



    # 初始化分词器
    def tokenizer_init(self):
        """
        预测的时候调用此接口，用来初始化分词器
        :return:
        """
        self.tokenizer, self.stopwords = self.make_tokenizer()

    def make_word_convert(self, text):
        """
        将传进来的文字，同一转为简体中文
        :param text:
        :return:
        """
        res_text = zhconv.convert(text, 'zh-cn')
        return res_text

    # 初始化bert词典
    def bert_vocab_init(self, vocab_file):
        """
        将Bert的词表读进来，产生word2id
        :param vocab_file:
        :return:
        """
        vocab = {}
        index = 0
        with open(vocab_file, "r", encoding="utf-8") as reader:
            while True:
                token = reader.readline()
                if not token:
                    break
                token = token.strip()
                vocab[token] = index
                index += 1
        return vocab

    # 初始化bert分词器
    def bert_tokenizer_init(self,vocab_file,do_lower_case=True):
        import tokenization
        tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file,
            do_lower_case=do_lower_case)
        return tokenizer

    # 将文本转为id
    def convert_tokens_to_ids(self, vocab, tokens, do_lower_case,unk_token="[UNK]"):
        """Converts a sequence of tokens into ids using the vocab."""
        # 这里空格 使用[unused2]占位符替换
        ids = []
        for token in tokens:
            if do_lower_case:
                if token != "[CLS]" and token != "[SEP]":
                    token = token.lower()
            if token in vocab:
                ids.append(vocab[token])
            elif token.split() == []:
                ids.append(vocab["[unused2]"])
            else:
                ids.append(vocab[unk_token])
        return ids

    # 将文本转为id
    def convert_ids_to_tokens(self, vocab, id_vocab, ids):
        """Converts a sequence of tokens into ids using the vocab."""
        # 这里的 [unused2] 用空格替换
        tokens = []
        for id in ids:
            if id == vocab["[unused2]"]:
                tokens.append(" ")
                continue
            if id in id_vocab:
                tokens.append(id_vocab[id])
        return tokens


    # 读取数据
    def read_data(self):
        """
        读取训练数据，随着数据的不同，此接口实现方式也不同
        :return:
        """
        with open(self.file_path, 'rb') as f:
            self.raw_data = pickle.load(f)

    # 数据处理
    def data_process(self):
        pass

    # 解析数据，将数据变为数字表示
    def parse_data(self, data, word2id, tag2id, max_length):
        pass

    # 将数据按批次输出
    def get_batch(self, data, batch_size, max_seq_length, num_train_epochs,shuffle=False):
        """
        数据生成器，将数据切块，一般不同数据，生成器的写法不同

        :param data:
        :param batch_size:
        :param vocab:
        :param tag2label:
        :param shuffle:
        :return:
        """
        # 乱序没有加
        if shuffle:
            random.shuffle(data)
        num_train_steps = int(
            len(data) / batch_size * num_train_epochs)
        for i in range(len(data) // batch_size):
            data_size = data[i * batch_size: (i + 1) * batch_size]
            seqs1, seqs2, labels = [], [], []
            for (sent_1, sent_2, tag_) in data_size:
                seqs1.append(sent_1)
                seqs2.append(sent_2)
                labels.append(tag_)
            yield np.array(seqs1), np.array(seqs2), np.array(labels),num_train_steps
        remainder = len(data) % batch_size
        if remainder != 0:
            data_size = data[-remainder:]
            seqs1, seqs2, labels = [], [], []
            for (sent_1, sent_2, tag_) in data_size:
                seqs1.append(sent_1)
                seqs2.append(sent_2)
                labels.append(tag_)
            yield np.array(seqs1), np.array(seqs2), np.array(labels), num_train_steps

    # 获取腾讯word2vec向量矩阵
    def make_TX_embedding(self, word2id, WORD2VEC_FILE):
        """
        获取word2vec的值，这里使用TX的word2vec词向量
        :param word2id: 数据处理时生成的word2id
        :param WORD2VEC_FILE: word2vec文件的路径
        :return:
        """
        res_dick = {}
        embed_size = 200
        with open(WORD2VEC_FILE, "r", encoding="utf-8") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                word = line.split(" ")[0]
                vec = [float(i) for i in line.strip().split(" ")[1:]]
                if word in word2id:
                    try:
                        res_dick[word] = vec
                    except:
                        print(line)

        n = 0
        embedding_mat = np.zeros((len(word2id), 200))
        for word, idex in word2id.items():
            try:
                embedding_mat[idex] = res_dick[word]
            except:
                # 随机初始化向量
                n += 1
                random_num = np.random.uniform(-0.25, 0.25, (200))
                embedding_mat[idex] = random_num
        print(n)
        embedding_mat[word2id["<PAD>"]] = np.zeros(200)
        embedding_mat[word2id["<UNK>"]] = np.random.uniform(-0.25, 0.25, (200))
        return embedding_mat,embed_size

    # 获取fasttext向量矩阵
    def make_fasttext_embedding(self, word2id, FASTTEXT_FILE):
        """
        获取fasttext的值，这里使用fasttext官方预训练的fasttext词向量
        :param word2id: 数据处理时生成的word2id
        :param FASTTEXT_FILE: fasttext文件的路径
        :return:
        """
        embed_size = 300
        FASTTEXT = FastText.load_fasttext_format(FASTTEXT_FILE)
        n = 0
        embedding_mat = np.zeros((len(word2id), 300))
        for word, idex in word2id.items():
            try:
                embedding_mat[idex] = FASTTEXT[word]
            except:
                # 随机初始化向量
                # 这个得分在什么场景下，这个embdding是否可以参与训练
                n += 1
                random_num = np.random.uniform(-0.25, 0.25, (300))
                embedding_mat[idex] = random_num
        print(n)
        embedding_mat[word2id["<PAD>"]] = np.zeros(300)
        embedding_mat[word2id["<UNK>"]] = np.random.uniform(-0.25, 0.25, (300))
        return embedding_mat,embed_size

    def make_self_embedding(self, EMBEDDING_FILE, VOCAB_FILE, embed_size):
        """
        使用自训练的word2vec，需要调用自训练接口，在原始数据上训练一下
        :param EMBEDDING_FILE: 自训练的bin文件路径
        :param VOCAB_FILE: 自训练产生的词表
        :param embed_size: 自训练时设置的词向量长度
        :return:
        """
        def get_coefs(word, *arr):
            return word, np.asarray(arr, dtype='float32')

        embeddings_index = dict(
            get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o) > 100)

        all_embs = np.stack(embeddings_index.values())
        emb_mean, emb_std = np.mean(all_embs), np.std(all_embs)
        # '/root/zxq/模型总结/文本匹配/deep_text_matching-master/input/word2vec/char_vocab.txt'
        vocab = [line.strip() for line in open(VOCAB_FILE, encoding='utf-8').readlines()]
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

        return embedding_matrix, embed_size

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

    # 保存模型参数
    def save_param(self, param, param_save_path):
        with open(param_save_path, 'wb') as f:
            param_new = copy.deepcopy(param)
            param_new.embedding_mat = []
            pickle.dump(param_new, f)

    # 恢复模型参数
    def load_param(self, param_save_path):
        with open(param_save_path, 'rb') as f:
            param = pickle.load(f)
        return param
