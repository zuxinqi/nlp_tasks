import numpy as np
import sklearn_crfsuite
import os, datetime, pickle, argparse
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from data_process import Date_Process
from evaluation_utils import Evaluation_Utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Crf_Suite_Entity_Extraction(object):
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

    def word2features_old(self,sent, i):
        word = sent[i][0]

        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
        }
        if i > 0:
            word1 = sent[i - 1][0]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
            })
        else:
            features['BOS'] = True

        if i < len(sent) - 1:
            word1 = sent[i + 1][0]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
            })
        else:
            features['EOS'] = True

        return features

    def word2features(self,sent, i):
        word = sent[i][0]

        features = [
            'bias',
            'word= ' + word,
            'word.islower{}'.format(word.islower()),
            'word.isupper{}'.format(word.isupper()),
            'word.isdigit{}'.format(word.isdigit()),
        ]
        if i > 0:
            word1 = sent[i - 1][0]
            words = word1 + word
            features.extend([
                '-1:word= ' + word1,
                '-1:words= ' + words,
                '-1:word.isupper{}'.format(word1.isupper()),
                '-1:word.isdigit{}'.format(word1.isdigit()),
            ])
        else:
            features.append('BOS')
        if i > 1:
            word1 = sent[i - 2][0]
            word2 = sent[i - 1][0]
            words = word1 + word2 + word
            features.extend([
                '-2:word= ' + word1,
                '-2:words= ' + words,
                '-2:word.isupper{}'.format(word1.isupper()),
                '-2:word.isdigit{}'.format(word1.isdigit()),
            ])

        # if i > 2:
        #     word1 = sent[i - 3][0]
        #     word2 = sent[i - 2][0]
        #     word3 = sent[i - 1][0]
        #     words = word1 + word2 + word3 + word
        #     features.extend([
        #         '-3:word= ' + word1,
        #         '-3:words= ' + words,
        #         '-3:word.isupper{}'.format(word1.isupper()),
        #         '-3:word.isdigit{}'.format(word1.isdigit()),
        #     ])
        if i < len(sent) - 1:
            word1 = sent[i + 1][0]
            words = word + word1
            features.extend([
                '+1:word= ' + word1,
                '+1:words= ' + words,
                '+1:word.isupper{}'.format(word1.isupper()),
                '+1:word.isdigit{}'.format(word1.isdigit()),
            ])
        else:
            features.append('EOS')
        if i < len(sent) - 2:
            word1 = sent[i + 2][0]
            word2 = sent[i + 1][0]
            words = word + word1 + word2
            features.extend([
                '+2:word= ' + word1,
                '+2:words= ' + words,
                '+2:word.isupper{}'.format(word1.isupper()),
                '+2:word.isdigit{}'.format(word1.isdigit()),
            ])

        # if i < len(sent) - 3:
        #     word1 = sent[i + 3][0]
        #     word2 = sent[i + 2][0]
        #     word3 = sent[i + 1][0]
        #     words = word + word1 + word2 + word3
        #     features.extend([
        #         '+3:word= ' + word1,
        #         '+3:words= ' + words,
        #         '+3:word.isupper{}'.format(word1.isupper()),
        #         '+3:word.isdigit{}'.format(word1.isdigit()),
        #     ])
        return features

    def sent2features(self,sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    def sent2labels(self,sent):
        return [label for token, label in sent]

    def sent2tokens(self,sent):
        return [token for token, label in sent]

    # 训练模型
    def train(self):
        """
        训练模型
        :param param_save_path:
        :return:
        """
        # 将模型参数导出，添加一些当前模型所需参数，并保存
        param = self.dp.param
        time_now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
        param.train_log = "./train_logs/train_log_" + time_now + ".txt"
        param.save_path = "./checkpoints/ml_model_" + time_now + ".pkl"
        param_save_path = "./param_dir/param_" + time_now + ".pkl"
        self.dp.save_param(param, param_save_path)

        # 定义评价函数
        eu = Evaluation_Utils()

        # 将数据特征化
        X_train = [self.sent2features(s) for s in self.dp.train_data]
        y_train = [self.sent2labels(s) for s in self.dp.train_data]

        X_test = [self.sent2features(s) for s in self.dp.test_data]
        y_test = [self.sent2labels(s) for s in self.dp.test_data]

        # 调用sklearn_crfsuite接口
        entity_extraction_model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        # 训练模型
        entity_extraction_model.fit(X_train, y_train)
        y_pred = entity_extraction_model.predict(X_test)

        # 保存模型
        with open(param.save_path, "wb") as f:
            pickle.dump(entity_extraction_model, f)

        y_test_ = []
        for i in y_test:
            i = [self.dp.tag2id[data] for data in i]
            y_test_ += i

        y_pred_ = []
        for i in y_pred:
            i = [self.dp.tag2id[data] for data in i]
            y_pred_ += i

        all_list = np.concatenate((y_test_, y_pred_), axis=0)
        all_list = np.unique(all_list)
        target_names = [self.dp.id2tag[i] for i in all_list]
        acc = accuracy_score(y_test_, y_pred_)
        print('  accuracy: {:.4}  '.format(acc))
        print(classification_report(y_test_, y_pred_, target_names=target_names))

        repo_res_dict = eu.calculate_report(y_test_, y_pred_, self.dp.id2tag)
        for key in repo_res_dict.keys():
            print(
                '{}   Rec:{:6.2%}  Pre:{:6.2%}   f1:{:6.2%}  count:  {}'.format(key,
                                                                                repo_res_dict[key][
                                                                                    0],
                                                                                repo_res_dict[key][
                                                                                    1],
                                                                                repo_res_dict[key][
                                                                                    2],
                                                                                repo_res_dict[key][
                                                                                    3]))
        with open(param.train_log, "w", encoding="utf-8") as f:
            self.write_param(param, acc, f)

    # 对整个测试集进行测试
    def make_all_predict(self,param_save_path,optional_reload_path=""):
        """
        对整个训练集进行测试，可改变参数文件
        :param param_save_path: 保存的参数文件路径
        :param optional_reload_path: 自选的加载的参数文件路径，如果为空字符串，则自动选择最低loss模型
        :return:
        """
        # 恢复模型参数，并针对测试修改部分参数
        param = self.dp.load_param(param_save_path)

        # 定义评价函数
        eu = Evaluation_Utils()

        # 恢复模型
        with open(param.save_path, "rb") as f:
            entity_extraction_model = pickle.load(f)

        # 加载模型
        X_test = [self.sent2features(s) for s in self.dp.test_data]
        y_test = [self.sent2labels(s) for s in self.dp.test_data]
        y_pred = entity_extraction_model.predict(X_test)

        y_test_ = []
        for i in y_test:
            i = [self.dp.tag2id[data] for data in i]
            y_test_ += i

        y_pred_ = []
        for i in y_pred:
            i = [self.dp.tag2id[data] for data in i]
            y_pred_ += i

        all_list = np.concatenate((y_test_, y_pred_), axis=0)
        all_list = np.unique(all_list)
        target_names = [self.dp.id2tag[i] for i in all_list]
        acc = accuracy_score(y_test_, y_pred_)
        print('  accuracy: {:.4}  '.format(acc))
        print(classification_report(y_test_, y_pred_, target_names=target_names))

        repo_res_dict = eu.calculate_report(y_test_, y_pred_, self.dp.id2tag)
        for key in repo_res_dict.keys():
            print(
                '{}   Rec:{:6.2%}  Pre:{:6.2%}   f1:{:6.2%}  count:  {}'.format(key,
                                                                                repo_res_dict[key][
                                                                                    0],
                                                                                repo_res_dict[key][
                                                                                    1],
                                                                                repo_res_dict[key][
                                                                                    2],
                                                                                repo_res_dict[key][
                                                                                    3]))

    # 解析预测数据
    def predict_parse_data(self, words):
        # for sentences, tags in input_data:
        res_data = []
        for word in words:
            # "游戏"事随便给的，预防报错
            sentences = list(word)
            tags = ["O"] * len(sentences)
            one_line = []
            for s, t in zip(sentences, tags):
                one_line.append([s, t])
            res_data.append(one_line)
        return res_data


    # 模型预测
    def predict(self,words,param_save_path,optional_reload_path=""):
        """
        :param words: 传入待预测的数据
        :param param_save_path:自选的加载的参数文件路径，如果为空字符串，则自动选择最低loss模型
        :return:
        """
        # 恢复模型参数，并针对测试修改部分参数
        param = self.dp.load_param(param_save_path)
        # 定义评价函数
        eu = Evaluation_Utils()

        # 恢复模型
        with open(param.save_path, "rb") as f:
            entity_extraction_model = pickle.load(f)

        # 加载数据
        test_data = self.predict_parse_data(words)
        X_test = [self.sent2features(s) for s in test_data]
        y_pred = entity_extraction_model.predict(X_test)
        all_entity_dict = []
        all_entity_list = []
        for one_token_, pred in zip(words, y_pred):
            res_dict = eu.transfrom2seq(pred)
            entity_dict, entity_list = eu.extract_entity(one_token_, res_dict)
            all_entity_dict.append(entity_dict)
            all_entity_list.append(entity_list)
        return all_entity_dict, all_entity_list





if __name__ == '__main__':
    words = ['株洲网讯（株洲晚报记者胡乐）昨天上午，图书馆株洲市民大讲堂走进株洲职业技术学院，邀请了湖南省作家协会会员、湖南省杂文学会理事张人杰作题为《互联网浪潮与网络文化》的讲座，三百多名师生与社会人士到场听讲。',
             '浙江省公安厅十分重视，副厅长叶寒冰要求各地多警联动，精确制导，缜密缉拿，确保解决在当地。',
             '这时台湾的中央研究院院长的事还没有发生，他决意回台湾，除了前面所说的研究和著作的关系以外，争取言论自由显然也是一个很重要的原因。']
    # words = '《DNF》玩家感叹；当时迫不及待头脑发热了一下，现在后悔莫及！。关键字: dnf'
    cfg_path = './default.cfg'
    param_save_path = 'param_dir/param_2021_02_20_17_03.pkl'
    # optional_reload_path = ""
    optional_reload_path = "model_12_-_4.8949.ckpt"

    # 创建保存文件
    for i in ["./param_dir", "./checkpoints", "./train_logs"]:
        if not os.path.exists(i):
            os.makedirs(i)

    CSEE = Crf_Suite_Entity_Extraction(cfg_path)

    # 初始化数据，开启训练
    # CSEE.data_process_init()
    # CSEE.train()

    # 加载已处理好的数据，开启训练
    # CSEE.data_process_reload()
    # CSEE.train()

    # 根据训练好的模型参数，预测整个测试集
    CSEE.data_process_reload()
    CSEE.make_all_predict(param_save_path,optional_reload_path)

    # 根据训练好的模型参数，预测样本
    # CSEE.data_process_reload()
    # all_entity_dict, all_entity_list = CSEE.predict(words,param_save_path,optional_reload_path)
    # for i,j in zip(all_entity_dict,all_entity_list):
    #     print(i)
    #     print(j)

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