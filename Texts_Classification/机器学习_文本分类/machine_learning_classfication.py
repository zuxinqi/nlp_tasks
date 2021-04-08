import numpy as np
import os,datetime,pickle,argparse
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from data_process import Date_Process

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Machine_Learning_Classfication(object):
    def __init__(self, config_path):
        """
        初始化模型，初始化数据处理，并解析config参数
        :param config_path:
        """
        self.dp = Date_Process()
        self.dp.parse_config(config_path)

    def data_process_init(self):
        """
        模型首次运行，执行此函数，初始化数据，并保存
        :return:
        """
        self.dp.init()
        self.dp.save_model(self.dp.save_path)

    def data_process_reload(self):
        """
        模型重复训练，运行此函数，恢复初处理好的数据
        :return:
        """
        self.dp.load_model(self.dp.save_path)
        self.dp.tokenizer_init()

    def write_param(self,param,acc,f):
        """
        比较出最佳训练acc，将loss和acc写入到文档记录
        将模型当前参数及参数值写入到文档记录
        :param param:
        :param acc_dict:
        :param f:
        :return:
        """
        f.write(
            "the acc is {:.4}".format(acc))
        f.write("\n")
        f.write("the paramers is: " + "\n")
        for i in dir(param):
            if "__" not in i and "embedding_mat" not in i:
                f.write("_"+str(i) + "    " + str(getattr(param, i)) + "\n")

    # 训练模型
    def train(self):
        """
        训练模型
        :param :
        :return:
        """
        # 将模型参数到处，添加一些当前模型所需参数，并保存
        param = self.dp.param
        time_now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
        param.train_log = "./train_logs/train_log_" + time_now + ".txt"
        param.save_path = "./checkpoints/ml_model_" + time_now + ".pkl"

        # 保存模型及参数
        param_save_path = "./param_dir/param_" + time_now + ".pkl"
        self.dp.save_param(param, param_save_path)

        # 训练Tfidf模型
        tfidf_vec = TfidfVectorizer(max_df=param.max_df, min_df=param.min_df, ngram_range=tuple(param.ngram_range))
        all_texts = [" ".join(i[0]) for i in self.dp.all_data]
        tfidf_matrix = tfidf_vec.fit_transform(all_texts)

        # 选择机器学习模型
        if param.classifier_name == "SGD":
            classifier = SGDClassifier()
        elif param.classifier_name == "SVC":
            classifier = SVC()
        elif param.classifier_name == "GBDT":
            classifier = GradientBoostingClassifier()
        else:
            # 默认SGD
            classifier = SGDClassifier()

        # 将训练集数据特征化并训练
        train_data = [" ".join(i[0]) for i in self.dp.train_data]
        train_tfidf_data = tfidf_vec.transform(train_data)
        train_tfidf_tag = [self.dp.tag2id[i[1]] for i in self.dp.train_data]
        classifier.fit(train_tfidf_data, train_tfidf_tag)

        # 保存模型
        with open(param.save_path, "wb") as f:
            pickle.dump((tfidf_vec,classifier),f)

        # 将测试集数据特征化并测试
        test_data = [" ".join(i[0]) for i in self.dp.test_data]
        test_tfidf_data = tfidf_vec.transform(test_data)
        test_tfidf_tag = [self.dp.tag2id[i[1]] for i in self.dp.test_data]
        predicted = classifier.predict(test_tfidf_data)

        all_list = np.concatenate((test_tfidf_tag, predicted), axis=0)
        all_list = np.unique(all_list)
        target_names = [self.dp.id2tag[i] for i in all_list]

        acc = accuracy_score(test_tfidf_tag, predicted)
        f1 = classification_report(test_tfidf_tag, predicted, target_names=target_names)

        with open(param.train_log, "w", encoding="utf-8") as f:
            self.write_param(param, acc, f)
        print(
            'accuracy: {:.4}  '.format(acc))
        print(f1)


    # 对整个测试集进行测试
    def make_all_predict(self,param_save_path):
        """
        对整个训练集进行测试，可改变参数文件
        :param param_save_path: 保存的参数文件路径
        :return:
        """
        # 恢复模型参数，并针对测试修改部分参数
        param = self.dp.load_param(param_save_path)
        # 加载模型
        with open(param.save_path, "rb") as f:
            tfidf_vec, classifier = pickle.load(f)
        # 模型测试
        test_data = [" ".join(i[0]) for i in self.dp.test_data]
        test_tfidf_data = tfidf_vec.transform(test_data)
        test_tfidf_tag = [self.dp.tag2id[i[1]] for i in self.dp.test_data]
        predicted = classifier.predict(test_tfidf_data)

        all_list = np.concatenate((test_tfidf_tag, predicted), axis=0)
        all_list = np.unique(all_list)
        target_names = [self.dp.id2tag[i] for i in all_list]

        acc = accuracy_score(test_tfidf_tag, predicted)
        f1 = classification_report(test_tfidf_tag, predicted, target_names=target_names)

        print(
            'accuracy: {:.4}  '.format(acc))
        print(f1)


    # 解析传入的待测试数据
    def parse_words(self, words):
        """
        切分句子，并给每个句子添加一个虚假tag
        :param words:
        :return:
        """
        words_list = []
        for word in words:
            seg_list = self.dp.tokenizer.cut(word)
            seg_list = [i for i in seg_list if i not in self.dp.stopwords]
            words_list.append([seg_list,self.dp.id2tag[0]])
        return words_list


    # 模型预测
    def predict(self,words,param_save_path):
        """
        :param words: 传入待预测的数据
        :return:
        """
        # 恢复模型参数，并针对测试修改部分参数
        param = self.dp.load_param(param_save_path)
        # 加载模型
        with open(param.save_path, "rb") as f:
            tfidf_vec, classifier = pickle.load(f)
        # 模型测试
        words_list = self.parse_words(words)
        test_data = [" ".join(i[0]) for i in words_list]
        test_tfidf_data = tfidf_vec.transform(test_data)
        predicted = classifier.predict(test_tfidf_data)

        predicted = [self.dp.id2tag[i] for i in predicted]
        return predicted



if __name__ == '__main__':
    words = ['《DNF》玩家感叹；当时迫不及待头脑发热了一下，现在后悔莫及！。关键字: dnf',
             '王者荣耀的哪个英雄或者哪个技能最让你怀念呢？',
             '刘德华的吃相、刘晓庆的吃相、徐峥的吃相，最后一个最丑让人恶心']
    # words = '《DNF》玩家感叹；当时迫不及待头脑发热了一下，现在后悔莫及！。关键字: dnf'
    cfg_path = './default.cfg'
    param_save_path = './param_dir/param_2021_02_22_15_24.pkl'

    # 创建保存文件
    for i in ["./param_dir", "./checkpoints", "./train_logs"]:
        if not os.path.exists(i):
            os.makedirs(i)

    MLC = Machine_Learning_Classfication(cfg_path)

    # 初始化数据，开启训练
    # MLC.data_process_init()
    # MLC.train()

    # 加载已处理好的数据，开启训练
    MLC.data_process_reload()
    MLC.train()

    # 根据训练好的模型参数，预测整个测试集
    # MLC.data_process_reload()
    # MLC.make_all_predict(param_save_path)

    # 根据训练好的模型参数，预测样本
    # MLC.data_process_reload()
    # predicted = MLC.predict(words,param_save_path)
    # print(predicted)

    # 晚上多组跑的时候可以添加此函数修改模型参数
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--square", type=int,
    #                     help="display a square of a given number")
    # parser.add_argument("--v", action="store_true",
    #                     help="increase output verbosity")
    # parser.add_argument("--square2", type=int,
    #                     help="display a square of a given number")
    # args = parser.parse_args()
    # answer = args.square ** 2
    # if args.v:
    #     print("the square of {} equals {}".format(args.square, answer))
    # else:
    #     print(answer)
    # print(args.v)
    # print(args.square2)