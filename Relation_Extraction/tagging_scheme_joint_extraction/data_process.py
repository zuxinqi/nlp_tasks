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
    def init(self, max_seq_length, vocab_file):
        """
        初始化参数，主要为4步：
        1、初始化分词器
        2、读取数据
        3、数据处理
        4、产生预训练的embedding矩阵
        :param feature_selection_name: 使用的预训练矩阵名称
        :return:
        """
        self.bert_tokenizer = self.bert_vocab_init(vocab_file)
        self.read_data()
        self.data_process(max_seq_length)

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

        # 模型所需参数
        self.param = parameters()
        self.param.max_seq_length = self.config.getint('MODEL', 'max_seq_length')
        self.param.bert_config_file = self.config.get('MODEL', 'bert_config_file')
        self.param.init_checkpoint = self.config.get('MODEL', 'init_checkpoint')
        self.param.use_lstm = self.config.getboolean('MODEL', 'use_lstm')
        self.param.use_lstm_dropout = self.config.getboolean('MODEL', 'use_lstm_dropout')
        self.param.hidden_size = self.config.getint('MODEL', 'hidden_size')
        self.param.cell_nums = self.config.getint('MODEL', 'cell_nums')
        self.param.lstm_dropout = self.config.getfloat('MODEL', 'lstm_dropout')
        self.param.embedding_dropout = self.config.getfloat('MODEL', 'embedding_dropout')
        self.param.learning_rate = self.config.getfloat('MODEL', 'learning_rate')
        self.param.warmup_proportion = self.config.getfloat('MODEL', 'warmup_proportion')
        self.param.is_training = self.config.getboolean('MODEL', 'is_training')
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
        :return:res_data 列表[[文本，实体标签，文本长度][文本，实体标签，文本长度]...]
        """
        res_data = []
        for one_data in raw_data:
            text = one_data["text"]
            # 因为后面会再添加两个元素
            if len(text) > max_seq_length - 2:
                continue
            triple_list = one_data["triple_list"]

            # ner_tags = np.zeros((len(text), len(self.ner_tag2id)))
            # 产生标签矩阵，默认“O”所在下标为1
            one_tags = np.zeros(len(self.ner_tag2id))
            one_tags[0] = 1
            ner_tags = np.tile(one_tags, (len(text), 1))

            for s, r, o in triple_list:
                ss_pos = self.search_entity(text, s)
                oo_pos = self.search_entity(text, o)
                for entity, pos in ss_pos:
                    # 将默认为1的下标抹除
                    ner_tags[pos[0]][0] = 0
                    ner_tags[pos[0] + 1:pos[1], 0] = 0
                    # 将真实标签位置置为1
                    ner_tags[pos[0]][self.ner_tag2id["B_S_" + r]] = 1
                    ner_tags[pos[0] + 1:pos[1], self.ner_tag2id["I"]] = 1
                for entity, pos in oo_pos:
                    # 将默认为1的下标抹除
                    ner_tags[pos[0]][0] = 0
                    ner_tags[pos[0] + 1:pos[1], 0] = 0
                    # 将真实标签位置置为1
                    ner_tags[pos[0]][self.ner_tag2id["B_O_" + r]] = 1
                    ner_tags[pos[0] + 1:pos[1], self.ner_tag2id["I"]] = 1

            res_data.append([text, ner_tags, len(text)])
        return res_data

    def search_entity(self, text_str, entity):
        """
        寻找实体在句子中的位置
        :param text_str: 语句
        :param entity:  实体
        :return: entity_list， 列表[实体，[实体起始位置，实体终止位置]]
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
        获得word2id，tag2id，将原始数据进行解析。
        :return:
        """

        # 产生tag2id
        self.raw_data = self.raw_train_data + self.raw_test_data
        all_ner_tag = []
        for one_data in self.raw_data:
            triple_list = one_data["triple_list"]
            for s, r, o in triple_list:
                if r not in all_ner_tag:
                    all_ner_tag.append(r)
        new_all_ner_tag = []
        new_all_ner_tag.append("O")
        new_all_ner_tag.append("I")
        new_all_ner_tag += ["[SEP]", "[CLS]", "X"]
        for i in all_ner_tag:
            new_all_ner_tag.append("B_S_" + i)
            new_all_ner_tag.append("B_O_" + i)
        self.ner_tag2id = {new_all_ner_tag[i]: i for i in range(len(new_all_ner_tag))}
        self.ner_id2tag = {v: k for k, v in self.ner_tag2id.items()}


        self.train_data = self.parse_raw_data(self.raw_train_data, max_seq_length)
        self.test_data = self.parse_raw_data(self.raw_test_data, max_seq_length)

        self.word2id = self.bert_tokenizer
        self.id2word = {v: k for k, v in self.word2id.items()}

    def parse_data(self,all_data,max_seq_length):
        """
        输入数据，将数据解析为id形式，长度变为一致
        :param all_data: 数据
        :param max_seq_length: 最大文本长度
        :return:all_list， 列表 [[sents_,ner_tags_,mask_ids_,segment_ids_,seq_len]...]
        :return:all_list， 列表 [[语句的ids,实体标签,输入的mask,输入的segment_ids_,文本长度]...]
        """
        all_list = []
        for text,ner_tags,seq_len in all_data:
            # sents_ = [self.word2id[sent] if sent in self.word2id else self.word2id['<UNK>'] for sent in text]
            text.insert(0, "[CLS]")
            text.append("[SEP]")
            sents_ = self.convert_tokens_to_ids(self.bert_tokenizer,text,True)
            seq_len = len(sents_)
            cls_pad = np.zeros(len(self.ner_tag2id))
            cls_pad[self.ner_tag2id["[CLS]"]] = 1
            ner_tags = np.insert(ner_tags, 0, cls_pad, axis=0)
            # ner_tags.insert(0,cls_pad)
            sep_pad = np.zeros(len(self.ner_tag2id))
            sep_pad[self.ner_tag2id["[SEP]"]] = 1
            sep_pad = np.reshape(sep_pad,(1,len(self.ner_tag2id)))
            # ner_tags.append(sep_pad)
            ner_tags = np.append(ner_tags, sep_pad, axis=0)
            # ner_tags_ = [self.ner_tag2id[tag] for tag in ner_tags]
            ner_tags_ = ner_tags
            mask_ids_ = [1]*len(sents_)
            pad_tag = np.zeros(len(self.ner_tag2id))
            pad_tag[self.ner_tag2id["O"]] = 1
            if len(sents_) < max_seq_length:
                sents_ = np.concatenate((sents_, np.tile(0, max_seq_length - len(sents_))), axis=0)  # 以pad的形式补充成等长的帧数
                ner_tags_ = np.concatenate((ner_tags_, np.tile(pad_tag, (max_seq_length - len(ner_tags_),1))), axis=0)
                mask_ids_ = np.concatenate((mask_ids_, np.tile(0, max_seq_length - len(mask_ids_))), axis=0)
            segment_ids_ = [0]*len(sents_)
            all_list.append([sents_,ner_tags_,mask_ids_,segment_ids_,seq_len])
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
            seqs, ner_labels,masks,segments, sentence_legth = [], [],[], [], []
            for (sents_,ner_tags_,mask_ids_,segment_ids_,seq_len) in data_size:
                seqs.append(sents_)
                ner_labels.append(ner_tags_)
                masks.append(mask_ids_)
                segments.append(segment_ids_)
                sentence_legth.append(seq_len)
            yield np.array(seqs), np.array(ner_labels),np.array(masks), np.array(segments),sentence_legth
        remainder = len(data) % batch_size
        if remainder != 0:
            data_size = data[-remainder:]
            seqs, ner_labels, masks, segments, sentence_legth = [], [], [], [], []
            for (sents_, ner_tags_, mask_ids_, segment_ids_, seq_len) in data_size:
                seqs.append(sents_)
                ner_labels.append(ner_tags_)
                masks.append(mask_ids_)
                segments.append(segment_ids_)
                sentence_legth.append(seq_len)
            yield np.array(seqs), np.array(ner_labels), np.array(masks), np.array(
                segments), sentence_legth


    # 保存模型
    def save_model(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump((self.train_data,self.test_data,self.word2id, self.id2word,self.bert_tokenizer,
                         self.ner_tag2id, self.ner_id2tag, self.embedding_mat), f)

    # 恢复模型
    def load_model(self, load_path):
        with open(load_path, 'rb') as f:
            self.train_data,self.test_data,self.word2id, self.id2word, self.bert_tokenizer,\
            self.ner_tag2id, self.ner_id2tag, self.embedding_mat = pickle.load(f)



if __name__ == '__main__':
    dp = Date_Process()
    dp.parse_config("./default.cfg")
    dp.param.aaaa = 11
    print(dir(dp.param))
    for i in dir(dp.param):
        if "__" not in i:
            print(str(i)+"  "+str(getattr(dp.param, i)))
