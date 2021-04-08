import re,os,datetime,argparse
from model import E2EModel, Evaluate
from utils import extract_items, get_tokenizer, metric, predict_one
from data_process import Date_Process, data_generator

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class CasRel_Relation_Extraction(object):
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
        vocab_file = self.dp.config.get('DATA_PROCESS', 'vocab_file')
        self.dp.init(vocab_file)
        self.dp.save_model(self.dp.save_path)

    def data_process_reload(self):
        """
        模型重复训练，运行此函数，恢复初处理好的数据
        :return:
        """
        self.dp.load_model(self.dp.save_path)

    def write_param(self,param,f):
        """
        比较出最佳训练acc，将loss和acc写入到文档记录
        将模型当前参数及参数值写入到文档记录
        :param param:
        :param acc_dict:
        :param f:
        :return:
        """
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
        # 将模型参数导出，添加一些当前模型所需参数，并保存
        param = self.dp.param
        param.num_rels = len(self.dp.relation2id)
        time_now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
        param.save_weights_path = "./checkpoints/h5_" + time_now + "_best_model.weights"
        param.train_log = "./train_logs/train_log_" + time_now + ".txt"
        param_save_path = "./param_dir/param_" + time_now + ".pkl"
        param.test_result_path = 'results/test_result_' + time_now + '.json'
        self.dp.save_param(param, param_save_path)

        # 加载模型
        subject_model, object_model, hbt_model = E2EModel(param)
        # 加载数据生成器
        data_manager = data_generator(self.dp.train_data, self.dp.bert_tokenizer, self.dp.relation2id, param.num_rels, param.max_seq_length, param.batch_size)
        # 评价函数
        evaluator = Evaluate(subject_model, object_model, self.dp.bert_tokenizer, self.dp.id2relation, self.dp.test_data, param.save_weights_path, param.train_log)
        # 一个epoch运行的步数
        STEPS = len(self.dp.train_data) // param.batch_size
        # 模型训练
        hbt_model.fit_generator(data_manager.__iter__(),
                                steps_per_epoch=STEPS,
                                epochs=param.num_train_epochs,
                                callbacks=[evaluator]
                                )

        with open(param.train_log, "a+", encoding="utf-8") as f:
            self.write_param(param, f)


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
        subject_model, object_model, hbt_model = E2EModel(param)
        # param.save_weights_path = "./checkpoints/h5_2021_03_01_16_50_best_model.weights"
        # 模型参数恢复
        hbt_model.load_weights(param.save_weights_path)

        # 获取测试数据
        isExactMatch = True
        precision, recall, f1_score = metric(subject_model, object_model, self.dp.test_data, self.dp.id2relation, self.dp.bert_tokenizer, isExactMatch,param.test_result_path)
        print(f'{precision}\t{recall}\t{f1_score}')

    # 模型预测
    def predict(self,words,param_save_path):
        """
        :param words: 传入待预测的数据
        :param param_save_path:自选的加载的参数文件路径，如果为空字符串，则自动选择最低loss模型
        :return:
        """
        # 恢复模型参数，并针对测试修改部分参数
        param = self.dp.load_param(param_save_path)
        param.save_weights_path = "./checkpoints/h5_2021_03_01_16_50_best_model.weights"
        # 加载模型
        subject_model, object_model, hbt_model = E2EModel(param)

        # 模型参数恢复
        hbt_model.load_weights(param.save_weights_path)

        # 模型预测
        res_list = predict_one(subject_model, object_model, words, self.dp.id2relation, self.dp.bert_tokenizer)

        return res_list





if __name__ == '__main__':
    # 测试语句
    words = ["布丹出生于1824年的法国画家",
             "黄主文，1941年8月20日出生，台湾省桃园县人，台湾大学法律系毕业、司法官训练所第8期结业、司法官高等考试及格",
             "杨维桢（1296—1370），元末明初著名诗人、文学家、书画家和戏曲家",
             "张柏芝，1980年5月24日出生于中国香港，毕业于澳大利亚Rmit Holmes College，中国香港女演员、歌手"
             ]

    cfg_path = './default.cfg'
    param_save_path = './param_dir/param_2021_03_01_10_29.pkl'


    # ner_query_map = {
    #     "tags": ["ORG", "PER", "LOC"],
    #     "natural_query": {
    #         "ORG": "找出公司，商业机构，社会组织等组织机构",
    #         "LOC": "找出国家，城市，山川等抽象或具体的地点",
    #         "PER": "找出真实和虚构的人名"
    #     }
    # }


    # 创建保存文件
    for i in ["./param_dir", "./checkpoints", "./train_logs", "./results"]:
        if not os.path.exists(i):
            os.makedirs(i)

    CRRE = CasRel_Relation_Extraction(cfg_path)

    # 初始化数据，开启训练
    # CRRE.data_process_init()
    # CRRE.train()

    # 加载已处理好的数据，开启训练
    # CRRE.data_process_reload()
    # CRRE.train()

    # 根据训练好的模型参数，预测整个测试集
    CRRE.data_process_reload()
    CRRE.make_all_predict(param_save_path)

    # 根据训练好的模型参数，预测样本
    # CRRE.data_process_reload()
    # res_list = CRRE.predict(words,param_save_path)
    # for line,triple_list in res_list:
    #     print(line)
    #     for j in triple_list:
    #         print("".join(j[0]), j[1], "".join(j[2]))
    #     print()

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