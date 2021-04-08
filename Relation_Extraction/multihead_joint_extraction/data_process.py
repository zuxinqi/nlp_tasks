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
    def init(self, max_seq_length, feature_selection_name="word2vec"):
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
        self.data_process(max_seq_length)
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
        self.param.embedding_dropout = self.config.getfloat('MODEL', 'embedding_dropout')
        self.param.num_lstm_layers = self.config.getint('MODEL', 'num_lstm_layers')
        self.param.use_lstm_dropout = self.config.getboolean('MODEL', 'use_lstm_dropout')
        self.param.lstm_dropout = self.config.getfloat('MODEL', 'lstm_dropout')
        self.param.hidden_size_lstm = self.config.getint('MODEL', 'hidden_size_lstm')
        self.param.hidden_size_n1 = self.config.getint('MODEL', 'hidden_size_n1')
        self.param.rel_activation = self.config.get('MODEL', 'rel_activation')
        self.param.rel_dropout = self.config.getfloat('MODEL', 'rel_dropout')
        self.param.learning_rate = self.config.getfloat('MODEL', 'learning_rate')
        self.param.clip = self.config.getint('MODEL', 'clip')
        self.param.label_embeddings_size = self.config.getint('MODEL', 'label_embeddings_size')
        self.param.num_train_epochs = self.config.getint('MODEL', 'num_train_epochs')
        self.param.batch_size = self.config.getint('MODEL', 'batch_size')
        self.param.shuffle = self.config.getboolean('MODEL', 'shuffle')
        self.param.display_per_step = self.config.getint('MODEL', 'display_per_step')
        self.param.evaluation_per_step = self.config.getint('MODEL', 'evaluation_per_step')
        self.param.require_improvement = self.config.getint('MODEL', 'require_improvement')

    # 读取数据
    def read_data(self):
        with open(self.file_path, 'rb') as f:
            self.raw_train_data,self.raw_test_data,self.relation2id,self.id2relation = pickle.load(f)

    def parse_raw_data(self, raw_data, max_seq_length):
        """
        解析原始数据，返回特定格式数据
        :param raw_data: 原始数据
        :param max_seq_length: 文本最大长度
        :return:res_data 列表[[文本，实体标签，关系标签，文本长度][文本，实体标签，关系标签，文本长度]...]
        """
        res_data = []
        for one_data in raw_data:
            text = one_data["text"]
            if len(text) > max_seq_length:
                continue
            triple_list = one_data["triple_list"]
            rel_tags = np.zeros((max_seq_length, max_seq_length, len(self.rel_tag2id)))
            # rel_tags = np.zeros((len(text),len(text),len(self.rel_tag2id)))
            ner_tags = ["O"] * len(text)
            for s, r, o in triple_list:
                t1, t2, r_ = r.split("/")
                ss_pos = self.search_entity(text, s)
                oo_pos = self.search_entity(text, o)
                for entity, pos in ss_pos:
                    ner_tags[pos[0]] = "B_" + t1
                    ner_tags[pos[0] + 1:pos[1]] = ["I_" + t1] * (pos[1] - pos[0] - 1)
                for entity, pos in oo_pos:
                    ner_tags[pos[0]] = "B_" + t2
                    ner_tags[pos[0] + 1:pos[1]] = ["I_" + t2] * (pos[1] - pos[0] - 1)
                for s_entity, s_pos in ss_pos:
                    for o_entity, o_pos in oo_pos:
                        rel_tags[s_pos[1] - 1][o_pos[1] - 1][self.rel_tag2id[r]] = 1
            res_data.append([text, ner_tags, rel_tags, len(text)])
        return res_data

    def search_entity(self, text_str, entity):
        """
        寻找实体在句子的位置
        :param text_str: 句子
        :param entity: 实体
        :return:entity_list, 列表[[实体,[开始位置，终止位置]],[实体,[开始位置，终止位置]]....]
        """
        entity_list = []
        target_len = len(entity)
        for i in range(len(text_str)):
            if text_str[i: i + target_len] == entity:
                entity_list.append([entity, [i, i + target_len]])
        return entity_list

    # 数据处理（头条）
    def data_process(self,max_seq_length):
        """
        获得word2id，tag2id，将原始数据进行处理，返回可训练数据
        :return:
        """

        # 产生tag2id
        all_rel_tag = []
        all_ner_tag = []
        # self.raw_data = self.raw_train_data + self.raw_test_data
        for one_data in self.raw_train_data:
            triple_list = one_data["triple_list"]
            for s, r, o in triple_list:
                if r not in all_rel_tag:
                    all_rel_tag.append(r)
                t1, t2, r_ = r.split("/")
                if t1 not in all_ner_tag:
                    all_ner_tag.append(t1)
                if t2 not in all_ner_tag:
                    all_ner_tag.append(t2)
        self.rel_tag2id = {all_rel_tag[i]: i for i in range(len(all_rel_tag))}
        self.rel_id2tag = {v: k for k, v in self.rel_tag2id.items()}
        new_all_ner_tag = []
        new_all_ner_tag.append("O")
        for i in all_ner_tag:
            new_all_ner_tag.append("B_" + i)
            new_all_ner_tag.append("I_" + i)
        self.ner_tag2id = {new_all_ner_tag[i]: i for i in range(len(new_all_ner_tag))}
        self.ner_id2tag = {v: k for k, v in self.ner_tag2id.items()}


        self.train_data = self.parse_raw_data(self.raw_train_data, max_seq_length)
        self.test_data = self.parse_raw_data(self.raw_test_data, max_seq_length)

        self.all_data = self.train_data + self.test_data

        word_list = []
        word_list.append('<PAD>')
        word_list.append('<UNK>')
        for text, ner_tags, rel_tags, seq_len in self.all_data:
            for word in text:
                if word not in word_list:
                    word_list.append(word)
        for index, word in enumerate(word_list):
            self.word2id[word] = index
            self.id2word[index] = word

    # 解析数据，将数据变为数字表示
    def parse_data(self, all_data,max_seq_length):
        """
        将原始数据转变为数字id的形式，并进行填充
        :param data: 输入数据
        :param word2id: 字典，类似{"word1":0,"word2":1,"word3":2......}
        :param tag2id: 字典，类似{"tag1":0,"tag2":1,"tag3":2......}
        :param max_length: 最大文本长度
        :return: all_list, 列表 [[句子的ids,实体标签的ids,关系标签矩阵,文本长度]...]
        """
        all_list = []
        for text, ner_tags, rel_tags, seq_len in all_data:
            sents_ = [self.word2id[sent] if sent in self.word2id else self.word2id['<UNK>'] for sent in text]
            ner_tags_ = [self.ner_tag2id[tag] for tag in ner_tags]
            if len(sents_) < max_seq_length:
                sents_ = np.concatenate((sents_, np.tile(self.word2id['<PAD>'], max_seq_length - len(sents_))),
                                        axis=0)  # 以pad的形式补充成等长的帧数
                ner_tags_ = np.concatenate((ner_tags_, np.tile(self.ner_tag2id["O"], max_seq_length - len(ner_tags_))),
                                           axis=0)
            all_list.append([sents_, ner_tags_, rel_tags, seq_len])
        return all_list

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
            seqs, ner_labels,rel_labels, sentence_legth = [], [], [], []
            for (sents_,ner_tags_,rel_tags,seq_len) in data_size:
                seqs.append(sents_)
                ner_labels.append(ner_tags_)
                rel_labels.append(rel_tags)
                sentence_legth.append(seq_len)
            yield np.array(seqs), np.array(ner_labels), np.array(rel_labels),sentence_legth
        remainder = len(data) % batch_size
        if remainder != 0:
            data_size = data[-remainder:]
            seqs, ner_labels,rel_labels, sentence_legth = [], [], [], []
            for (sents_, ner_tags_, rel_tags, seq_len) in data_size:
                seqs.append(sents_)
                ner_labels.append(ner_tags_)
                rel_labels.append(rel_tags)
                sentence_legth.append(seq_len)
            yield np.array(seqs), np.array(ner_labels), np.array(rel_labels), sentence_legth


    # 保存模型
    def save_model(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump((self.word2id, self.id2word,
                         self.rel_tag2id, self.rel_id2tag, self.ner_tag2id, self.ner_id2tag, self.embedding_mat), f)

    # 恢复模型
    def load_model(self, load_path):
        with open(load_path, 'rb') as f:
            self.word2id, self.id2word, \
            self.rel_tag2id, self.rel_id2tag, self.ner_tag2id, self.ner_id2tag, self.embedding_mat = pickle.load(f)



if __name__ == '__main__':
    dp = Date_Process()
    dp.parse_config("./default.cfg")
    dp.param.aaaa = 11
    print(dir(dp.param))
    for i in dir(dp.param):
        if "__" not in i:
            print(str(i)+"  "+str(getattr(dp.param, i)))
