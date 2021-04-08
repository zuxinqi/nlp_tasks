import numpy as np
from random import choice
from utils import get_tokenizer
import sys,pickle,random,configparser
sys.path.append('./Base_Line')
sys.path.append('../../Base_Line')
from base_data_process import Base_Process

BERT_MAX_LEN = 512
RANDOM_SEED = 2019

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
        self.bert_tokenizer = get_tokenizer(vocab_file)
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
        self.param.bert_config_file = self.config.get('MODEL', 'bert_config_file')
        self.param.init_checkpoint = self.config.get('MODEL', 'init_checkpoint')
        self.param.learning_rate = self.config.getfloat('MODEL', 'learning_rate')
        self.param.num_train_epochs = self.config.getint('MODEL', 'num_train_epochs')
        self.param.batch_size = self.config.getint('MODEL', 'batch_size')


    # 读取数据
    def read_data(self):
        with open(self.file_path, 'rb') as f:
            self.train_data,self.test_data,self.relation2id,self.id2relation = pickle.load(f)

    def to_tuple(self, sent):
        triple_list = []
        for triple in sent['triple_list']:
            triple_list.append((tuple(triple[0]), triple[1], tuple(triple[2])))
        sent['triple_list'] = triple_list

    # 数据处理（头条）
    def data_process(self):
        """
        将原始数据转变为元组
        :return:
        """
        # 随机打乱一下train_data
        random_order = list(range(len(self.train_data)))
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(random_order)
        self.train_data = [self.train_data[i] for i in random_order]

        for sent in self.train_data:
            self.to_tuple(sent)
        for sent in self.test_data:
            self.to_tuple(sent)

        print("train_data len:", len(self.train_data))
        print("test_data len:", len(self.test_data))


    # 保存模型
    def save_model(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump((self.train_data, self.test_data,
                         self.relation2id,self.id2relation, self.bert_tokenizer), f)

    # 恢复模型
    def load_model(self, load_path):
        with open(load_path, 'rb') as f:
            self.train_data, self.test_data, \
            self.relation2id,self.id2relation, self.bert_tokenizer = pickle.load(f)




class data_generator:
    def __init__(self, data, tokenizer, rel2id, num_rels, maxlen, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.rel2id = rel2id
        self.num_rels = num_rels
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps

    def find_head_idx(self, source, target):
        target_len = len(target)
        for i in range(len(source)):
            if source[i: i + target_len] == target:
                return i
        return -1

    def seq_padding(self, batch, padding=0):
        length_batch = [len(seq) for seq in batch]
        max_length = max(length_batch)
        return np.array([
            np.concatenate([seq, [padding] * (max_length - len(seq))]) if len(seq) < max_length else seq for seq in
            batch
        ])

    def __iter__(self):
        # 生成器，生成训练数据
        while True:
            idxs = list(range(len(self.data)))
            np.random.seed(RANDOM_SEED)
            np.random.shuffle(idxs)
            tokens_batch, segments_batch, sub_heads_batch, sub_tails_batch, sub_head_batch, sub_tail_batch, obj_heads_batch, obj_tails_batch = [], [], [], [], [], [], [], []
            for idx in idxs:
                line = self.data[idx]
                text = line['text'][:self.maxlen]
                # text = ' '.join(line['text'].split()[:self.maxlen])
                tokens = self.tokenizer.tokenize(text)
                if len(tokens) > BERT_MAX_LEN:
                    tokens = tokens[:BERT_MAX_LEN]
                text_len = len(tokens)

                s2ro_map = {}
                for triple in line['triple_list']:
                    triple = (self.tokenizer.tokenize(triple[0])[1:-1], triple[1], self.tokenizer.tokenize(triple[2])[1:-1])
                    sub_head_idx = self.find_head_idx(tokens, triple[0])
                    obj_head_idx = self.find_head_idx(tokens, triple[2])
                    if sub_head_idx != -1 and obj_head_idx != -1:
                        sub = (sub_head_idx, sub_head_idx + len(triple[0]) - 1)
                        if sub not in s2ro_map:
                            s2ro_map[sub] = []
                        s2ro_map[sub].append((obj_head_idx,
                                           obj_head_idx + len(triple[2]) - 1,
                                           self.rel2id[triple[1]]))

                if s2ro_map:
                    token_ids, segment_ids = self.tokenizer.encode(first=text)
                    if len(token_ids) > text_len:
                        token_ids = token_ids[:text_len]
                        segment_ids = segment_ids[:text_len]
                    tokens_batch.append(token_ids)
                    segments_batch.append(segment_ids)
                    sub_heads, sub_tails = np.zeros(text_len), np.zeros(text_len)
                    for s in s2ro_map:
                        sub_heads[s[0]] = 1
                        sub_tails[s[1]] = 1
                    sub_head, sub_tail = choice(list(s2ro_map.keys()))
                    obj_heads, obj_tails = np.zeros((text_len, self.num_rels)), np.zeros((text_len, self.num_rels))
                    for ro in s2ro_map.get((sub_head, sub_tail), []):
                        obj_heads[ro[0]][ro[2]] = 1
                        obj_tails[ro[1]][ro[2]] = 1
                    sub_heads_batch.append(sub_heads)
                    sub_tails_batch.append(sub_tails)
                    sub_head_batch.append([sub_head])
                    sub_tail_batch.append([sub_tail])
                    obj_heads_batch.append(obj_heads)
                    obj_tails_batch.append(obj_tails)
                    if len(tokens_batch) == self.batch_size or idx == idxs[-1]:
                        tokens_batch = self.seq_padding(tokens_batch)
                        segments_batch = self.seq_padding(segments_batch)
                        sub_heads_batch = self.seq_padding(sub_heads_batch)
                        sub_tails_batch = self.seq_padding(sub_tails_batch)
                        obj_heads_batch = self.seq_padding(obj_heads_batch, np.zeros(self.num_rels))
                        obj_tails_batch = self.seq_padding(obj_tails_batch, np.zeros(self.num_rels))
                        sub_head_batch, sub_tail_batch = np.array(sub_head_batch), np.array(sub_tail_batch)
                        yield [tokens_batch, segments_batch, sub_heads_batch, sub_tails_batch, sub_head_batch, sub_tail_batch, obj_heads_batch, obj_tails_batch], None
                        tokens_batch, segments_batch, sub_heads_batch, sub_tails_batch, sub_head_batch, sub_tail_batch, obj_heads_batch, obj_tails_batch, = [], [], [], [], [], [], [], []



if __name__ == '__main__':
    dp = Date_Process()
    dp.parse_config("./default.cfg")
    dp.param.aaaa = 11
    print(dir(dp.param))
    for i in dir(dp.param):
        if "__" not in i:
            print(str(i)+"  "+str(getattr(dp.param, i)))
