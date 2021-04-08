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
        self.bert_tokenizer = self.bert_tokenizer_init(vocab_file)
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
        self.param.is_training = self.config.getboolean('MODEL', 'is_training')
        self.param.fc_output_dim = self.config.getint('MODEL', 'fc_output_dim')
        self.param.dropout_rate = self.config.getfloat('MODEL', 'dropout_rate')
        self.param.learning_rate = self.config.getfloat('MODEL', 'learning_rate')
        self.param.warmup_proportion = self.config.getfloat('MODEL', 'warmup_proportion')
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
        :return:res_data 列表[[文本，关系类别][文本，关系类别]...]
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
                    one_text.insert(s_index, " a1 ")
                    one_text.insert(s_index + len(s) + 1, " a2 ")
                    one_text.insert(o_index + 2, " b1 ")
                    one_text.insert(o_index + len(o) + 3, " b2 ")
                else:
                    one_text.insert(o_index, " b1 ")
                    one_text.insert(o_index + len(o) + 1, " b2 ")
                    one_text.insert(s_index + 2, " a1 ")
                    one_text.insert(s_index + len(s) + 3, " a2 ")
                one_text = "".join(one_text)
                if len(one_text) > max_seq_length - 18:
                    continue
                res_data.append([one_text, r])
        return res_data

    # 解析传入的待测试数据
    def parse_test_data(self, data, max_seq_length):
        """
        切分句子，并给每个句子添加一个虚假tag
        :param words:
        :return:
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
            raw_entity_list.append(["".join(text),s_set,o_set])

            # 添加a1、a2，构造训练语句
            for s1 in s_set:
                for o1 in o_set:
                    s = list(s1)
                    o = list(o1)
                    s_index = self.find_head_idx(text, s)
                    o_index = self.find_head_idx_reverse(text, o)
                    one_text = copy.deepcopy(text)
                    if len(one_text) > max_seq_length - 18:
                        # 如果有长度超的，这里不能删掉，因此进行截断
                        one_text = one_text[:(max_seq_length - 18)]
                    if s_index < o_index:
                        one_text.insert(s_index, " a1 ")
                        one_text.insert(s_index + len(s) + 1, " a2 ")
                        one_text.insert(o_index + 2, " b1 ")
                        one_text.insert(o_index + len(o) + 3, " b2 ")
                    else:
                        one_text.insert(o_index, " b1 ")
                        one_text.insert(o_index + len(o) + 1, " b2 ")
                        one_text.insert(s_index + 2, " a1 ")
                        one_text.insert(s_index + len(s) + 3, " a2 ")
                    one_text = "".join(one_text)
                    res_list.append([one_text, "无关"])
                    entity_list.append(("".join(text),"".join(s),"".join(o)))
        return res_list,entity_list,raw_entity_list

    # 数据处理（头条）
    def data_process(self,max_seq_length):
        """
        获得word2vec，tag2vec，将原始数据解析。
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

    def parse_data(self, all_data, max_seq_length):
        """
        输入数据，将数据解析为id形式，长度变为一致
        :param all_data: 数据
        :param max_seq_length: 最大文本长度
        :return: all_list, 列表 [[input_ids, input_mask, segment_ids, label_id,e1_mask,e2_mask]...]
        :return: all_list, 列表 [[输入语句id,输入的mask,输入的segment_id,关系类别标签,第一个实体的mask,第二个实体的mask]...]
        """
        all_list = []
        for one_text, r in all_data:
            raw_tokens = self.bert_tokenizer.tokenize(one_text)
            e11_p = raw_tokens.index("a1")  # the start position of entity1
            e12_p = raw_tokens.index("a2")  # the end position of entity1
            e21_p = raw_tokens.index("b1")  # the start position of entity2
            e22_p = raw_tokens.index("b2")  # the end position of entity2

            # Replace the token
            raw_tokens[e11_p] = "$"
            raw_tokens[e12_p] = "$"
            raw_tokens[e21_p] = "#"
            raw_tokens[e22_p] = "#"

            # Add 1 because of the [CLS] token
            e11_p += 1
            e12_p += 1
            e21_p += 1
            e22_p += 1

            tokens = []
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in raw_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            # e1 mask, e2 mask
            e1_mask = [0] * len(input_ids)
            e2_mask = [0] * len(input_ids)
            for i in range(e11_p, e12_p + 1):
                e1_mask[i] = 1
            for i in range(e21_p, e22_p + 1):
                e2_mask[i] = 1

            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                e1_mask.append(0)
                e2_mask.append(0)

            label_id = self.tag2id[r]

            all_list.append([input_ids, input_mask, segment_ids, label_id,e1_mask,e2_mask])
        return all_list

    # 将数据pad，生成batch数据返回，这里没有取余数。
    def get_batch(self, data, batch_size, shuffle):
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
            input_list, mask_list, segment_list, label_list, e1_mask_list, e2_mask_list = [], [], [], [], [], []
            for (input_ids, input_mask, segment_ids, label_ids, e1_mask, e2_mask) in data_size:
                input_list.append(input_ids)
                mask_list.append(input_mask)
                segment_list.append(segment_ids)
                label_list.append(label_ids)
                e1_mask_list.append(e1_mask)
                e2_mask_list.append(e2_mask)
            yield np.array(input_list), np.array(mask_list), np.array(segment_list), np.array(
                label_list), np.array(e1_mask_list), np.array(e2_mask_list)
        remainder = len(data) % batch_size
        if remainder != 0:
            data_size = data[-remainder:]
            if remainder == 0:
                data_size = []
            input_list, mask_list, segment_list, label_list, e1_mask_list, e2_mask_list = [], [], [], [], [], []
            for (input_ids, input_mask, segment_ids, label_ids, e1_mask, e2_mask) in data_size:
                input_list.append(input_ids)
                mask_list.append(input_mask)
                segment_list.append(segment_ids)
                label_list.append(label_ids)
                e1_mask_list.append(e1_mask)
                e2_mask_list.append(e2_mask)
            yield np.array(input_list), np.array(mask_list), np.array(segment_list), np.array(
                label_list), np.array(e1_mask_list), np.array(e2_mask_list)

    # 保存模型
    def save_model(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump((self.train_data, self.test_data,self.bert_tokenizer, self.tag2id, self.id2tag), f)

    # 恢复模型
    def load_model(self, load_path):
        with open(load_path, 'rb') as f:
            self.train_data, self.test_data, self.bert_tokenizer,self.tag2id, self.id2tag = pickle.load(f)



if __name__ == '__main__':
    dp = Date_Process()
    dp.parse_config("./default.cfg")
    dp.param.aaaa = 11
    print(dir(dp.param))
    for i in dir(dp.param):
        if "__" not in i:
            print(str(i)+"  "+str(getattr(dp.param, i)))
