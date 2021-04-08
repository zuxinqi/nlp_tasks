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
        self.param.learning_rate = self.config.getfloat('MODEL', 'learning_rate')
        self.param.num_train_epochs = self.config.getint('MODEL', 'num_train_epochs')
        self.param.batch_size = self.config.getint('MODEL', 'batch_size')
        self.param.shuffle = self.config.getboolean('MODEL', 'shuffle')
        self.param.warmup_proportion = self.config.getfloat('MODEL', 'warmup_proportion')
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
        all_tag = []
        all_tag += ["[SEP]", "[CLS]", "X"]
        for sentences,tags in self.raw_data:
            for tag in tags:
                if tag not in all_tag:
                    all_tag.append(tag)
        self.tag2id = {all_tag[i]: i+1 for i in range(len(all_tag))}
        self.id2tag = {v: k for k, v in self.tag2id.items()}


        if "MSRA" in self.file_path:
            train = -4636
            self.train_data = self.raw_data[:train]
            self.test_data = self.raw_data[train:]
        else:
            random.seed(1)
            random.shuffle(self.raw_data)
            data_len = len(self.raw_data)
            train = int(data_len * 0.8)
            self.train_data = self.raw_data[:train]
            self.test_data = self.raw_data[train:]

    def convert_tokens_to_ids(self, vocab, tokens, do_lower_case,unk_token="[UNK]"):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            if do_lower_case:
                if token != "[CLS]" and token != "[SEP]":
                    token = token.lower()
            if token in vocab:
                ids.append(vocab[token])
            else:
                ids.append(vocab[unk_token])
        return ids

    def gen_query_map_dict(self,ner_query_map):
        slot_query_tag_dict = {}
        for slot_tag in ner_query_map.get("tags"):
            slot_query = ner_query_map.get("natural_query").get(slot_tag)
            slot_query_tokenize = [w for w in slot_query]
            slot_query_tokenize.insert(0, "[CLS]")
            slot_query_tokenize.append("[SEP]")
            slot_query_tag_dict.update({slot_tag: slot_query_tokenize})
        return slot_query_tag_dict

    def find_tag_start_end_index(self, tag, label_list):
        start_index_tag = [0] * len(label_list)
        end_index_tag = [0] * len(label_list)
        # MSRA用的
        # start_tag = "B-" + tag
        # end_tag = "I-" + tag
        start_tag = "B_" + tag.upper()
        end_tag = "I_" + tag.upper()
        for i in range(len(start_index_tag)):
            if label_list[i].upper() == start_tag:
                # begin
                start_index_tag[i] = 1
                if i == len(start_index_tag) - 1:
                    # last tag
                    end_index_tag[i] = 1
                else:
                    if label_list[i + 1].upper() != end_tag:
                        end_index_tag[i] = 1
            elif label_list[i].upper() == end_tag:
                if i == len(start_index_tag) - 1:
                    # last tag
                    end_index_tag[i] = 1
                else:
                    if label_list[i + 1].upper() != end_tag:
                        end_index_tag[i] = 1
        return start_index_tag, end_index_tag


    def parse_data(self, data,ner_query_map,bert_tokenizer,max_length):
        query_map_dict = self.gen_query_map_dict(ner_query_map)
        all_list = []
        for sentences, tags in data:
            for slot_tag in query_map_dict:
                slot_query = query_map_dict.get(slot_tag)
                slot_query = [w for w in slot_query]
                x_merge = slot_query + sentences
                query_len = len(slot_query)
                text_len = len(x_merge)
                token_type_ids = [0] * len(slot_query) + [1] * (len(sentences))
                x_merge = self.convert_tokens_to_ids(bert_tokenizer,x_merge,True)
                if len(x_merge) > max_length:
                    continue
                start_index_tag, end_index_tag = self.find_tag_start_end_index(slot_tag, tags)
                start_index_tag = [0] * len(slot_query) + start_index_tag
                # print(len(start_index_tag))
                end_index_tag = [0] * len(slot_query) + end_index_tag

                if len(x_merge) < max_length:
                    pad = [0] * (max_length - len(x_merge))
                    x_merge = x_merge + pad
                else:
                    x_merge = x_merge[:max_length]

                if len(start_index_tag) < max_length:
                    pad = [0] * (max_length - len(start_index_tag))
                    start_index_tag = start_index_tag + pad
                else:
                    start_index_tag = start_index_tag[:max_length]

                if len(end_index_tag) < max_length:
                    pad = [0] * (max_length - len(end_index_tag))
                    end_index_tag = end_index_tag + pad
                else:
                    end_index_tag = end_index_tag[:max_length]

                if len(token_type_ids) < max_length:
                    pad = [0] * (max_length - len(token_type_ids))
                    token_type_ids = token_type_ids + pad
                else:
                    token_type_ids = token_type_ids[:max_length]

                if text_len > max_length:
                    text_len = max_length

                all_list.append([x_merge,start_index_tag,end_index_tag,token_type_ids,query_len,text_len])
        return all_list

    # 将数据按批次输出
    def get_batch(self, data, batch_size,shuffle=False):
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
            seqs, s_labels,e_labels,token_types, query_lens,text_lens = [], [], [],[], [], []
            for (x_merge,start_index_tag,end_index_tag,token_type_ids,query_len,text_len) in data_size:
                seqs.append(x_merge)
                s_labels.append(start_index_tag)
                e_labels.append(end_index_tag)
                token_types.append(token_type_ids)
                query_lens.append(query_len)
                text_lens.append(text_len)
            yield np.array(seqs), np.array(s_labels), np.array(e_labels),np.array(token_types),query_lens,text_lens
        remainder = len(data) % batch_size
        if remainder !=0:
            data_size = data[-remainder:]
            seqs, s_labels, e_labels, token_types, query_lens, text_lens = [], [], [], [], [], []
            for (x_merge, start_index_tag, end_index_tag, token_type_ids, query_len, text_len) in data_size:
                seqs.append(x_merge)
                s_labels.append(start_index_tag)
                e_labels.append(end_index_tag)
                token_types.append(token_type_ids)
                query_lens.append(query_len)
                text_lens.append(text_len)
            yield np.array(seqs), np.array(s_labels), np.array(e_labels), np.array(
                token_types), query_lens, text_lens


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
