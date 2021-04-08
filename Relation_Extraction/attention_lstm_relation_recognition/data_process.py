import numpy as np
import sys,copy,pickle,random,configparser
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
        self.param.dropout_rate = self.config.getfloat('MODEL', 'dropout_rate')
        self.param.pos_num = self.config.getint('MODEL', 'pos_num')
        self.param.pos_size = self.config.getint('MODEL', 'pos_size')
        self.param.gru_size = self.config.getint('MODEL', 'gru_size')
        self.param.num_layers = self.config.getint('MODEL', 'num_layers')
        self.param.learning_rate = self.config.getfloat('MODEL', 'learning_rate')
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

    def find_head_idx(self,source, target):
        """
        在句子中实体寻找下标
        :param source: 语句
        :param target: 实体
        :return: i, 实体在语句中的起始下标位置
        """
        target_len = len(target)
        for i in range(len(source)):
            if source[i: i + target_len] == target:
                return i
        return -1

    def find_head_idx_reverse(self,source, target):
        """
        从后向前在句子中实体寻找下标
        :param source: 语句
        :param target: 实体
        :return: i, 实体在语句中的起始下标位置
        """
        target_len = len(target)
        for i in range(len(source),0,-1):
            if source[i-target_len: i] == target:
                return i-target_len
        return -1

    def parse_raw_data(self, raw_data, max_seq_length):
        """
        解析原始数据，返回特定格式数据
        :param raw_data: 原始数据
        :param max_seq_length: 文本最大长度
        :return:res_data 列表[[subject实体，object实体,关系类别,文本],[subject实体，object实体,关系类别,文本]...]
        """
        res_data = []
        for data in raw_data:
            text = data["text"]
            s_list = []
            o_list = []
            so_list = []
            new_triple_list = data["triple_list"]
            for s, r, o in data["triple_list"]:
                if s not in s_list:
                    s_list.append(s)
                if o not in o_list:
                    o_list.append(o)
                if [s, o] not in so_list:
                    so_list.append([s, o])
            for s in s_list:
                for o in o_list:
                    if s != o:
                        if [s, o] not in so_list:
                            new_triple_list.append([s, "无关", o])
            for s, r, o in new_triple_list:
                s_index = self.find_head_idx(text, s)
                o_index = self.find_head_idx_reverse(text, o)
                one_text = copy.deepcopy(text)
                if s_index < o_index:
                    one_text.insert(s_index, "$")
                    one_text.insert(s_index + len(s) + 1, "$")
                    one_text.insert(o_index + 2, "#")
                    one_text.insert(o_index + len(o) + 3, "#")
                    s = ["$"] + s + ["$"]
                    o = ["#"] + o + ["#"]
                else:
                    one_text.insert(o_index, "#")
                    one_text.insert(o_index + len(o) + 1, "#")
                    one_text.insert(s_index + 2, "$")
                    one_text.insert(s_index + len(s) + 3, "$")
                    o = ["#"] + o + ["#"]
                    s = ["$"] + s + ["$"]
                one_text = "".join(one_text)
                ss = "".join(s)
                oo = "".join(o)
                one_text = "".join(one_text.split())
                ss = "".join(ss.split())
                oo = "".join(oo.split())
                if len(one_text) > max_seq_length:
                    continue
                res_data.append([ss, oo, r, one_text])
        return res_data

    def parse_test_data(self, data, max_seq_length):
        """
        切分句子，并给每个句子添加一个虚假tag
        :param raw_data: 原始数据
        :param max_seq_length: 文本最大长度
        :return:res_data 列表[[subject实体，object实体,关系类别,文本],[subject实体，object实体,关系类别,文本]...]
        """
        res_list = []
        entity_list = []
        raw_entity_list = []
        for one_data in data:
            # 通过原始的数据把text拿出来
            raw_one_data = one_data[0]
            text = raw_one_data["text"]
            # 建立s集和还有o集和
            s_set = set()
            o_set = set()
            predict_list = one_data[2]
            # s和o要区分出来
            s_allow_list = ["人物"]
            o_allow_list = ["Date", "地点", "国家", "学校"]
            # 取出s和o实体
            for pred in predict_list:
                if pred[0] in s_allow_list:
                    s_set.add("".join(pred[1]))
                if pred[0] in o_allow_list:
                    o_set.add("".join(pred[1]))
            # 保存下s、o的提取情况
            raw_entity_list.append(["".join(text), s_set, o_set])

            # 添加a1、a2，构造训练语句
            for s1 in s_set:
                for o1 in o_set:
                    s = list(s1)
                    o = list(o1)
                    s_index = self.find_head_idx(text, s)
                    o_index = self.find_head_idx_reverse(text, o)
                    one_text = copy.deepcopy(text)
                    if len(one_text) > max_seq_length:
                        one_text = one_text[:max_seq_length]
                    if s_index < o_index:
                        one_text.insert(s_index, "$")
                        one_text.insert(s_index + len(s) + 1, "$")
                        one_text.insert(o_index + 2, "#")
                        one_text.insert(o_index + len(o) + 3, "#")
                        s = ["$"] + s + ["$"]
                        o = ["#"] + o + ["#"]
                    else:
                        one_text.insert(o_index, "#")
                        one_text.insert(o_index + len(o) + 1, "#")
                        one_text.insert(s_index + 2, "$")
                        one_text.insert(s_index + len(s) + 3, "$")
                        o = ["#"] + o + ["#"]
                        s = ["$"] + s + ["$"]
                    one_text = "".join(one_text)
                    ss = "".join(s)
                    oo = "".join(o)
                    one_text = "".join(one_text.split())
                    ss = "".join(ss.split())
                    oo = "".join(oo.split())
                    res_list.append([ss, oo, "无关", one_text])
                    entity_list.append(("".join(text), s1, o1))
        return res_list,entity_list,raw_entity_list

    # 数据处理（头条）
    def data_process(self,max_seq_length):
        """
        获得wordid，tag2id，将原始数据进行解析。
        :return:
        """

        # 产生tag2id
        all_tag = []
        all_tag.append("无关")
        for one_data in self.raw_train_data:
            triple_list = one_data["triple_list"]
            for s, r, o in triple_list:
                if r not in all_tag:
                    all_tag.append(r)
        self.tag2id = {all_tag[i]: i for i in range(len(all_tag))}
        self.id2tag = {v: k for k, v in self.tag2id.items()}
        self.train_data = self.parse_raw_data(self.raw_train_data, max_seq_length)
        self.test_data = self.parse_raw_data(self.raw_test_data, max_seq_length)

        self.all_data = self.train_data + self.test_data

        word_list = []
        word_list.append('<PAD>')
        word_list.append('<UNK>')
        for ss, oo, r, text in self.all_data:
            for word in text:
                if word not in word_list:
                    word_list.append(word)
        for index, word in enumerate(word_list):
            self.word2id[word] = index
            self.id2word[index] = word

    # embedding the position
    def pos_embed(self,x):
        """
        把输入位置转变在-60-60之间
        :param x: 输入位置
        :return:
        """
        if x < -60:
            return 0
        if -60 <= x <= 60:
            return x + 61
        if x > 60:
            return 122

    # 解析数据，将数据变为数字表示
    def parse_data(self, data, word2id, tag2id, max_seq_length):
        """
        将原始数据转变为数字id的形式，并进行填充
        :param data: 输入的数据
        :param word2id: 字典，类似{"word1":0,"word2":1,"word3":2......}
        :param tag2id: 字典，类似{"tag1":0,"tag2":1,"tag3":2......}
        :param max_length: 最大文本长度
        :return:all_list, 列表 [[words, rel_e1s, rel_e2s, label]...]
        :return:all_list, 列表 [[语句的ids, subject实体位置ids,类似[0,0,1,...], object实体位置ids,类似[0,0,1,...], 实体类别标签]...]
        """
        all_list = []
        for content in data:
            en1 = content[0]
            en2 = content[1]

            relation = tag2id[content[2]]
            label = [0 for i in range(len(tag2id))]
            label[relation] = 1
            sentence = content[3]
            # For Chinese
            en1pos = sentence.find(en1)
            if en1pos == -1:
                print(en1)
                print(sentence)
                print("未找到11111111111")
                en1pos = 0
            en2pos = sentence.find(en2)
            if en2pos == -1:
                print(en2)
                print(sentence)
                print("未找到22222222222")
                en2pos = 0

            words = []
            rel_e1s = []
            rel_e2s = []
            # Embeding the position
            for i in range(max_seq_length):
                word = word2id['<PAD>']
                rel_e1 = self.pos_embed(i - en1pos)
                rel_e2 = self.pos_embed(i - en2pos)
                words.append(word)
                rel_e1s.append(rel_e1)
                rel_e2s.append(rel_e2)

            for i in range(min(max_seq_length, len(sentence))):
                if sentence[i] not in word2id:
                    word = word2id['<UNK>']
                else:
                    word = word2id[sentence[i]]

                words[i] = word

            all_list.append([words, rel_e1s, rel_e2s, label])
        return all_list

    # 将数据pad，生成batch数据返回，这里没有取余数。
    def get_batch(self, data, batch_size, shuffle=False):
        """

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
        for i in range(len(data) // batch_size):
            data_size = data[i * batch_size: (i + 1) * batch_size]
            total_num = 0
            seqs, e1_list, e2_list, total_shape, labels = [], [], [], [], []
            for (sent_, e1_, e2_, tag_) in data_size:
                seqs.append(sent_)
                e1_list.append(e1_)
                e2_list.append(e2_)
                labels.append(tag_)
                total_shape.append(total_num)
                total_num += 1
            total_shape.append(total_num)
            yield np.array(seqs), np.array(e1_list), np.array(e2_list), np.array(total_shape), np.array(labels)


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



if __name__ == '__main__':
    dp = Date_Process()
    dp.parse_config("./default.cfg")
    dp.param.aaaa = 11
    print(dir(dp.param))
    for i in dir(dp.param):
        if "__" not in i:
            print(str(i)+"  "+str(getattr(dp.param, i)))
