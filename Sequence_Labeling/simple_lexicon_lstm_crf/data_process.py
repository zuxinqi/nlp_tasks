import numpy as np
import sys,pickle,random,configparser
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.data import Data
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
    def init(self, char_emb, bichar_emb, gaz_file, train_file, dev_file, test_file):
        """
        初始化参数，主要为4步：
        1、初始化分词器
        2、读取数据
        3、数据处理
        4、产生预训练的embedding矩阵
        :param feature_selection_name: 使用的预训练矩阵名称
        :return:
        """
        self.data = Data()
        self.data_initialization(self.data, gaz_file, train_file, dev_file, test_file)
        self.data.generate_instance_with_gaz(train_file, 'train')
        self.data.generate_instance_with_gaz(dev_file, 'dev')
        self.data.generate_instance_with_gaz(test_file, 'test')
        self.data.build_word_pretrain_emb(char_emb)
        self.data.build_biword_pretrain_emb(bichar_emb)
        self.data.build_gaz_pretrain_emb(gaz_file)

    def data_initialization(self, data, gaz_file, train_file, dev_file, test_file):
        # 字、词、biword、label的对象全都创建
        data.build_alphabet(train_file)
        data.build_alphabet(dev_file)
        data.build_alphabet(test_file)
        # vab读出来
        data.build_gaz_file(gaz_file)
        data.build_gaz_alphabet(train_file, count=True)
        data.build_gaz_alphabet(dev_file, count=True)
        data.build_gaz_alphabet(test_file, count=True)
        data.fix_alphabet()
        return data

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
        self.param.is_training = self.config.getboolean('MODEL', 'is_training')
        self.param.update_embedding = self.config.getboolean('MODEL', 'update_embedding')
        self.param.use_biword = self.config.getboolean('MODEL', 'use_biword')
        self.param.use_char = self.config.getboolean('MODEL', 'use_char')
        self.param.use_count = self.config.getboolean('MODEL', 'use_count')
        self.param.hidden_size = self.config.getint('MODEL', 'hidden_size')
        self.param.cell_nums = self.config.getint('MODEL', 'cell_nums')
        self.param.use_cnn = self.config.getboolean('MODEL', 'use_cnn')
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
        all_tag = []
        for sentences, tags in self.raw_data:
            for tag in tags:
                if tag not in all_tag:
                    all_tag.append(tag)
        self.tag2id = {all_tag[i]: i for i in range(len(all_tag))}
        self.id2tag = {v: k for k, v in self.tag2id.items()}

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
    def parse_data(self, data, word2id, tag2id):
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
            labels = [tag2id[tag] for tag in tags]
            if len(sents) != len(labels):
                print('........')
            all_seq.append((sents, labels, len(sents)))
        return all_seq


    # 将数据pad，生成batch数据返回，这里没有取余数。
    def get_batch(self, data, batch_size,shuffle=False):
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
            sentence_lengths = [len(i[0]) for i in data_size]
            max_seq_len = max(sentence_lengths)
            gaz_num = [len(i[5][0][0]) for i in data_size]
            max_gaz_num = max(gaz_num)
            gaz_len = [len(i[7][0][0][0]) for i in data_size]
            max_gaz_len = max(gaz_len)

            word_Ids = np.zeros((batch_size, max_seq_len))
            biword_Ids = np.zeros((batch_size, max_seq_len))
            label_Ids = np.zeros((batch_size, max_seq_len))

            gazs = np.zeros((batch_size, max_seq_len, 4, max_gaz_num))
            gazs_count = np.zeros((batch_size, max_seq_len, 4, max_gaz_num))
            layergazmasks = np.zeros((batch_size, max_seq_len, 4, max_gaz_num))

            gaz_char_Ids = np.zeros((batch_size, max_seq_len, 4, max_gaz_num, max_gaz_len))
            gazchar_masks = np.zeros((batch_size, max_seq_len, 4, max_gaz_num, max_gaz_len))

            for index,(word_Id, biword_Id, char_Id, gaz_Id, label_Id, gaz, gaz_count, gaz_char_Id, layergazmask,gazchar_mask, bert_text_id) in enumerate(data_size):
                seqlen = len(word_Id)
                gaznum = len(gaz[0][0])
                gazlen = len(gaz_char_Id[0][0][0])

                word_Ids[index,:seqlen] = word_Id
                biword_Ids[index,:seqlen] = biword_Id
                label_Ids[index,:seqlen] = label_Id

                gazs[index,:seqlen, :, :gaznum] = gaz
                gazs_count[index,:seqlen, :, :gaznum] = gaz_count
                layergazmasks[index,:seqlen, :, :gaznum] = layergazmask

                gaz_char_Ids[index, :seqlen, :, :gaznum, :gazlen] = gaz_char_Id
                gazchar_masks[index, :seqlen, :, :gaznum, :gazlen] = gazchar_mask

            yield word_Ids,biword_Ids,label_Ids,gazs,gazs_count,layergazmasks,gaz_char_Ids,gazchar_masks,sentence_lengths
        remainder = len(data) % batch_size
        if remainder != 0:
            data_size = data[-remainder:]
            sentence_lengths = [len(i[0]) for i in data_size]
            max_seq_len = max(sentence_lengths)
            gaz_num = [len(i[5][0][0]) for i in data_size]
            max_gaz_num = max(gaz_num)
            gaz_len = [len(i[7][0][0][0]) for i in data_size]
            max_gaz_len = max(gaz_len)

            word_Ids = np.zeros((remainder, max_seq_len))
            biword_Ids = np.zeros((remainder, max_seq_len))
            label_Ids = np.zeros((remainder, max_seq_len))

            gazs = np.zeros((remainder, max_seq_len, 4, max_gaz_num))
            gazs_count = np.zeros((remainder, max_seq_len, 4, max_gaz_num))
            layergazmasks = np.zeros((remainder, max_seq_len, 4, max_gaz_num))

            gaz_char_Ids = np.zeros((remainder, max_seq_len, 4, max_gaz_num, max_gaz_len))
            gazchar_masks = np.zeros((remainder, max_seq_len, 4, max_gaz_num, max_gaz_len))

            for index, (
            word_Id, biword_Id, char_Id, gaz_Id, label_Id, gaz, gaz_count, gaz_char_Id, layergazmask, gazchar_mask,
            bert_text_id) in enumerate(data_size):
                seqlen = len(word_Id)
                gaznum = len(gaz[0][0])
                gazlen = len(gaz_char_Id[0][0][0])

                word_Ids[index, :seqlen] = word_Id
                biword_Ids[index, :seqlen] = biword_Id
                label_Ids[index, :seqlen] = label_Id

                gazs[index, :seqlen, :, :gaznum] = gaz
                gazs_count[index, :seqlen, :, :gaznum] = gaz_count
                layergazmasks[index, :seqlen, :, :gaznum] = layergazmask

                gaz_char_Ids[index, :seqlen, :, :gaznum, :gazlen] = gaz_char_Id
                gazchar_masks[index, :seqlen, :, :gaznum, :gazlen] = gazchar_mask

            yield word_Ids, biword_Ids, label_Ids, gazs, gazs_count, layergazmasks, gaz_char_Ids, gazchar_masks, sentence_lengths

    # 保存模型
    def save_model(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self.data, f)

    # 恢复模型
    def load_model(self, load_path):
        with open(load_path, 'rb') as f:
            self.data = pickle.load(f)



if __name__ == '__main__':
    dp = Date_Process()
    dp.parse_config("./default.cfg")
    dp.param.aaaa = 11
    print(dir(dp.param))
    for i in dir(dp.param):
        if "__" not in i:
            print(str(i)+"  "+str(getattr(dp.param, i)))
