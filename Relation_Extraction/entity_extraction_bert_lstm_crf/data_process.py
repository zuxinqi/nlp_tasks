import numpy as np
import sys,pickle,random,configparser,copy
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
        self.bert_tokenizer = self.bert_vocab_init(vocab_file)
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
            self.raw_train_data, self.raw_test_data, self.relation2id, self.id2relation = pickle.load(f)

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

    def parse_raw_data(self, raw_data):
        """
        解析原始数据，返回特定格式数据
        :param raw_data: 原始数据
        :return:res_data 列表[[文本，实体标签][文本，实体标签]...]
        """
        res_data = []
        n = 0
        for one_data in raw_data:
            raw_text = one_data["text"]
            text = copy.deepcopy(raw_text)
            triple_list = one_data["triple_list"]
            ner_tags = ["O"] * len(text)

            entity_set = set()
            for s, r, o in triple_list:
                s_tag, o_tag, r_ = r.split("/")
                entity_set.add(("".join(s),s_tag))
                entity_set.add(("".join(o),o_tag))
            for entity,tag in entity_set:
                entity = list(entity)
                pos_list = self.search_entity(text, entity)
                for pos in pos_list:
                    if ner_tags[pos[1][0]:pos[1][1]] != ["O"]*(pos[1][1]-pos[1][0]):
                        n+=1
                        continue
                    ner_tags[pos[1][0]] = "B_"+tag
                    ner_tags[pos[1][0]+1:pos[1][1]] = ["I_"+tag]*(pos[1][1]-pos[1][0]-1)
            res_data.append([text, ner_tags])
        print(n)
        return res_data

    # 数据处理（头条）
    def data_process(self):
        """
        获得word2id，tag2id，将原始数据按比例切割。
        :return:
        """
        all_tag = []
        for one_data in self.raw_train_data:
            triple_list = one_data["triple_list"]
            for s, r, o in triple_list:
                t1, t2, r_ = r.split("/")
                if t1 not in all_tag:
                    all_tag.append(t1)
                if t2 not in all_tag:
                    all_tag.append(t2)
        new_all_ner_tag = []
        new_all_ner_tag += ["O","[SEP]", "[CLS]", "X"]
        for i in all_tag:
            new_all_ner_tag.append("B_" + i)
            new_all_ner_tag.append("I_" + i)
        self.tag2id = {new_all_ner_tag[i]: i for i in range(len(new_all_ner_tag))}
        self.id2tag = {v: k for k, v in self.tag2id.items()}

        self.train_data = self.parse_raw_data(self.raw_train_data)
        self.test_data = self.parse_raw_data(self.raw_test_data)


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
            # 这里的sentences和tags都是列表
            # 序列截断
            if len(sentences) >= max_length - 2:
                sentences = sentences[0:(max_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
                tags = tags[0:(max_length - 2)]

            sentences.insert(0, "[CLS]")
            sentences.append("[SEP]")
            tags.insert(0, "[CLS]")
            tags.append("[SEP]")
            input_ids = self.convert_tokens_to_ids(bert_tokenizer, sentences, True)
            label_ids = [tag2id[i] for i in tags]
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < max_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                label_ids.append(0)

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
