import sys,pickle
from gensim.models import Word2Vec
sys.path.append('./Base_Line')
sys.path.append('../Base_Line')
sys.path.append('../../Base_Line')
from base_data_process import Base_Process

class make_word2vec(Base_Process):
    def __init__(self,file_path,tokenizer_name,uerdict_pat="",stopwords_path=""):
        self.file_path = file_path
        self.tokenizer_name = tokenizer_name
        self.uerdict_path = uerdict_pat
        self.stopwords_path = stopwords_path
        self.all_data = []

    # 初始化模型,构造分词器、训练语料分词、生成embedding矩阵
    def init(self, task_name):
        self.tokenizer, self.stopwords = self.make_tokenizer()
        self.read_data(task_name)


    # 读取数据
    def read_data(self,task_name):
        if task_name == "文本匹配":
            with open(self.file_path, 'rb') as f:
                self.raw_train_data,self.raw_dev_data,self.raw_test_data = pickle.load(f)
                self.raw_data = self.raw_train_data + self.raw_dev_data + self.raw_test_data
                for sentence1, sentence2, tag in self.raw_train_data:
                    self.all_data.append(sentence1)
                    self.all_data.append(sentence2)

    def train_word2vec(self,size,window,min_count,workers,model_path,bin_model_path,word_vocab_path):
        train_word_data = []
        for sentence in self.all_data:
            seg_list = list(self.tokenizer.cut(sentence.replace(" ", "")))
            train_word_data.append(seg_list)

        model = Word2Vec(train_word_data, size=size, window=window, min_count=min_count, workers=workers)

        model.save(model_path)
        model.wv.save_word2vec_format(bin_model_path, binary=False)
        word_set = set()
        for sample in train_word_data:
            for word in sample:
                word_set.add(word)
        with open(word_vocab_path, 'w', encoding='utf8') as f:
            f.write("\n".join(sorted(list(word_set), reverse=True)))


if __name__ == '__main__':
    file_path = '/root/zxq/all_models/Text_Matching/数据准备/wz_data.pkl'
    tokenizer_name = "single"
    task_name = "文本匹配"


    size = 100
    window = 5
    min_count = 3
    workers = 8
    model_path = "./char2vec.model"
    bin_model_path = "./char2vec.bin"
    word_vocab_path = "./char_vocab.txt"

    MW = make_word2vec(file_path,tokenizer_name)
    MW.init(task_name)
    MW.train_word2vec(size, window, min_count, workers, model_path, bin_model_path, word_vocab_path)
