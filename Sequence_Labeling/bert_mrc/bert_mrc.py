import numpy as np
import tensorflow as tf
import re,os,datetime,argparse
import modeling
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from model import make_graph
from data_process import Date_Process
from evaluation_utils import Evaluation_Utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Bert_Mrc_Entity_Extraction(object):
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

    def write_param(self,param,acc_dict,f):
        """
        比较出最佳训练acc，将loss和acc写入到文档记录
        将模型当前参数及参数值写入到文档记录
        :param param:
        :param acc_dict:
        :param f:
        :return:
        """
        best_acc = 0
        for i in acc_dict:
            if i > best_acc:
                best_acc = i
        epoch = acc_dict[best_acc][0]
        loss = acc_dict[best_acc][1]
        f.write(
            "the best acc is {:.4}, the loss is {:.4}, from number {} epoch.".format(best_acc,loss,epoch+1))
        f.write("\n")
        f.write("the paramers is: " + "\n")
        for i in dir(param):
            if "__" not in i and "embedding_mat" not in i:
                f.write("_"+str(i) + "    " + str(getattr(param, i)) + "\n")

    # 训练模型
    def train(self, ner_query_map):
        """
        训练模型
        :param :
        :return:
        """
        # 将模型参数导出，添加一些当前模型所需参数，并保存
        param = self.dp.param
        param.ner_query_map = ner_query_map
        param.num_classes = 2 #span模式只有0、1二分类
        param.num_train_steps = int(
            len(self.dp.train_data) / param.batch_size * param.num_train_epochs)
        param.num_warmup_steps = int(param.num_train_steps * param.warmup_proportion)
        param.bert_config = modeling.BertConfig.from_json_file(param.bert_config_file)
        time_now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
        param.save_checkpoints_path = "./checkpoints/checkpoint_" + time_now + "/"
        param.train_log = "./train_logs/train_log_" + time_now + ".txt"
        param_save_path = "./param_dir/param_" + time_now + ".pkl"
        self.dp.save_param(param, param_save_path)

        # 定义评价函数
        eu = Evaluation_Utils()
        # 加载模型
        graph, input_ids, start_labels, end_labels, token_type_ids_list, query_len_list, \
        text_length_list, predict_start_ids, predict_end_ids, \
        global_add, total_loss, train_op = make_graph(param)
        # 制作训练集和测试集
        train_seq_data = self.dp.parse_data(self.dp.train_data, param.ner_query_map, self.dp.bert_tokenizer,
                                            param.max_seq_length)
        test_seq_data = self.dp.parse_data(self.dp.test_data, param.ner_query_map, self.dp.bert_tokenizer,
                                            param.max_seq_length)
        # 训练模型
        with tf.Session(graph=graph) as sess:
            if param.is_training:
                saver = tf.train.Saver(tf.global_variables(),max_to_keep=10)
                dev_best_loss = float('inf')
                last_improve = 0
                acc_dict = {}
                init = tf.global_variables_initializer()
                sess.run(init)
                # saver.restore(sess, "./checkpoints/checkpoint_2021_02_07_13_01/model_50_-_0.5583.ckpt")

                for epoch in range(param.num_train_epochs):
                    n = 0
                    all_acc = 0
                    count_loss = 0
                    for seqs_, s_labels_, e_labels_, token_types_, query_lens_, text_lens_ in self.dp.get_batch(train_seq_data, param.batch_size, param.shuffle):
                        _, l, global_nums, predict_start_ids_, predict_end_ids_ = sess.run(
                            [train_op, total_loss, global_add, predict_start_ids, predict_end_ids], {
                                input_ids: seqs_,
                                start_labels: s_labels_,
                                end_labels: e_labels_,
                                token_type_ids_list: token_types_,
                                query_len_list: query_lens_,
                                text_length_list: text_lens_
                            })

                        if global_nums % param.display_per_step == 0:
                            # 获取真实序列、标签长度。
                            label_list, pred_list, all_l_seq, all_p_seq = eu.make_mask(seqs_, text_lens_,
                                                                                                     query_lens_,
                                                                                                     s_labels_,
                                                                                                     e_labels_,
                                                                                                     predict_start_ids_,
                                                                                                     predict_end_ids_)
                            all_list = np.concatenate((label_list, pred_list), axis=0)
                            all_list = np.unique(all_list)
                            id2tag_ = {0: "0", 1: "1"}
                            target_names = [id2tag_[i] for i in all_list]
                            acc = accuracy_score(label_list, pred_list)
                            print(
                                'epoch {}, global_step {}, loss: {:.4}, accuracy: {:.4}  '.format(epoch + 1,
                                                                                                  global_nums + 1,
                                                                                                  l, acc))
                            print(classification_report(label_list, pred_list, target_names=target_names))
                            repo_res_dict = eu.calculate_report(all_l_seq, all_p_seq, self.dp.id2tag)
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
                            print()
                            print()
                        if global_nums % param.evaluation_per_step == 0:
                            print('-----------------valudation---------------')
                            seqs_, s_labels_, e_labels_, token_types_, query_lens_, text_lens_ = next(
                                self.dp.get_batch(test_seq_data, param.batch_size, param.shuffle))
                            l, global_nums, predict_start_ids_, predict_end_ids_ = sess.run(
                                [total_loss, global_add, predict_start_ids, predict_end_ids],
                                feed_dict={
                                    input_ids: seqs_,
                                    start_labels: s_labels_,
                                    end_labels: e_labels_,
                                    token_type_ids_list: token_types_,
                                    query_len_list: query_lens_,
                                    text_length_list: text_lens_
                                })

                            # 获取真实序列、标签长度。
                            label_list, pred_list, all_l_seq, all_p_seq = eu.make_mask(seqs_, text_lens_,
                                                                                                     query_lens_,
                                                                                                     s_labels_,
                                                                                                     e_labels_,
                                                                                                     predict_start_ids_,
                                                                                                     predict_end_ids_)
                            all_list = np.concatenate((label_list, pred_list), axis=0)
                            all_list = np.unique(all_list)
                            id2tag_ = {0: "0", 1: "1"}
                            target_names = [id2tag_[i] for i in all_list]
                            acc = accuracy_score(label_list, pred_list)
                            print(
                                'epoch {}, global_step {}, loss: {:.4}, accuracy: {:.4}'.format(epoch + 1,
                                                                                                global_nums + 1,
                                                                                                l, acc))
                            print(classification_report(label_list, pred_list, target_names=target_names))
                            repo_res_dict = eu.calculate_report(all_l_seq, all_p_seq, self.dp.id2tag)
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
                            print('-----------------valudation---------------')
                            print()
                            print()
                    for seqs_, s_labels_, e_labels_, token_types_, query_lens_, text_lens_ in self.dp.get_batch(test_seq_data, param.batch_size, param.shuffle):
                        l, global_nums, predict_start_ids_, predict_end_ids_ = sess.run(
                            [total_loss, global_add, predict_start_ids, predict_end_ids],
                            feed_dict={
                                input_ids: seqs_,
                                start_labels: s_labels_,
                                end_labels: e_labels_,
                                token_type_ids_list: token_types_,
                                query_len_list: query_lens_,
                                text_length_list: text_lens_
                            })

                        # 获取真实序列、标签长度。
                        label_list, pred_list, all_l_seq, all_p_seq = eu.make_mask(seqs_, text_lens_,
                                                                                   query_lens_,
                                                                                   s_labels_,
                                                                                   e_labels_,
                                                                                   predict_start_ids_,
                                                                                   predict_end_ids_)
                        acc_ = accuracy_score(label_list, pred_list)
                        count_loss += l
                        all_acc += acc_
                        n += 1
                    acc_dict[all_acc/n] = [epoch, count_loss/n]

                    if count_loss < dev_best_loss:
                        dev_best_loss = count_loss
                        last_improve = epoch
                        saver.save(sess, param.save_checkpoints_path + 'model' + "_" + str(epoch + 1) + "___" + str(
                            round(count_loss/n, 4)) + '.ckpt')

                        with open(param.train_log, "a+", encoding="utf-8") as f:
                            f.write("this number {} loss is {:.4} the acc is {:.4} ****".format(epoch + 1, count_loss / n,
                                                                                     all_acc / n))
                            f.write("\n")

                        print()
                        print("****************************")
                        print(
                            "this number {} loss reached a new location, the count_loss is {:.4} the acc is {:.4} ".format(epoch + 1,count_loss/n,all_acc/n))
                        print("****************************")
                        print()
                    else:
                        saver.save(sess, param.save_checkpoints_path + 'model' + "_" + str(epoch + 1) + "_-_" + str(
                            round(count_loss / n, 4)) + '.ckpt')
                        with open(param.train_log, "a+", encoding="utf-8") as f:
                            f.write("this number {} loss is {:.4} the acc is {:.4} ".format(epoch + 1, count_loss / n,
                                                                                            all_acc / n))
                            f.write("\n")
                        print()
                        print("----------------------------")
                        print("this number {} loss not reached a new location, the count_loss is {:.4} the acc is {:.4} ".format(epoch + 1,count_loss/n,all_acc/n))
                        print("----------------------------")
                        print()

                    if epoch - last_improve > param.require_improvement:
                        # 验证集loss超过1000batch没下降，结束训练
                        print("No optimization for a long time, auto-stopping...")
                        break

                print('-----------------test---------------')
                all_label_list = []
                all_pred_list = []
                all_all_l_seq = []
                all_all_p_seq = []
                m = 0
                all_loss = 0
                for seqs_, s_labels_, e_labels_, token_types_, query_lens_, text_lens_ in self.dp.get_batch(
                        test_seq_data, param.batch_size, param.shuffle):
                    l, global_nums, predict_start_ids_, predict_end_ids_ = sess.run(
                        [total_loss, global_add, predict_start_ids, predict_end_ids],
                        feed_dict={
                            input_ids: seqs_,
                            start_labels: s_labels_,
                            end_labels: e_labels_,
                            token_type_ids_list: token_types_,
                            query_len_list: query_lens_,
                            text_length_list: text_lens_
                        })
                    # 获取真实序列、标签长度。
                    label_list, pred_list, all_l_seq, all_p_seq = eu.make_mask(seqs_, text_lens_,
                                                                               query_lens_,
                                                                               s_labels_,
                                                                               e_labels_,
                                                                               predict_start_ids_,
                                                                               predict_end_ids_)
                    all_label_list += label_list
                    all_pred_list += pred_list
                    all_all_l_seq += all_l_seq
                    all_all_p_seq += all_p_seq
                    m += 1
                    all_loss += l
                average_loss = all_loss / m
                all_list = np.concatenate((all_label_list, all_pred_list), axis=0)
                all_list = np.unique(all_list)
                id2tag_ = {0: "0", 1: "1"}
                target_names = [id2tag_[i] for i in all_list]
                acc = accuracy_score(all_label_list, all_pred_list)
                print(
                    'loss: {:.4}, accuracy: {:.4}  '.format(average_loss, acc))
                print(classification_report(all_label_list, all_pred_list, target_names=target_names))
                repo_res_dict = eu.calculate_report(all_all_l_seq, all_all_p_seq, self.dp.id2tag)
                for key in repo_res_dict.keys():
                    print(
                        '{}   Rec:{:6.2%}  Pre:{:6.2%}   f1:{:6.2%}  count:  {}'.format(key, repo_res_dict[key][0],
                                                                                        repo_res_dict[key][1],
                                                                                        repo_res_dict[key][2],
                                                                                        repo_res_dict[key][3]))
                print()
                with open(param.train_log, "a+", encoding="utf-8") as f:
                    self.write_param(param,acc_dict,f)

                print('-----------------test---------------')



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
        param.dropout_rate = 1.0
        param.is_training = False
        param.shuffle = False

        # 定义评价函数
        eu = Evaluation_Utils()
        # 加载模型
        graph, input_ids, start_labels, end_labels, token_type_ids_list, query_len_list, \
        text_length_list, predict_start_ids, predict_end_ids, \
        global_add, total_loss, train_op = make_graph(param)
        # 获取测试数据
        test_seq_data = self.dp.parse_data(self.dp.test_data, param.ner_query_map, self.dp.bert_tokenizer,
                                           param.max_seq_length)

        # 获取最小loss的ckpt文件路径
        dirs = os.listdir(param.save_checkpoints_path)
        print(dirs)
        dict_ = {}
        for file_name in dirs:
            res = re.match("model_(\d+)_([_\-])_(.+).ckpt.meta", file_name)
            if res != None:
                dict_[res.group(3)] = [res.group(1), res.group(2)]
        list_ = [float(i) for i in dict_.keys()]
        list_.sort()
        epoch, connector = dict_[str(list_[0])]
        count_loss = list_[0]
        checkpoint_path = param.save_checkpoints_path + "model_" + str(epoch) + "_" + str(connector) + "_" + str(
            count_loss) + ".ckpt"
        print(checkpoint_path)
        # 测试模型
        with tf.Session(graph=graph) as sess:
            saver = tf.train.Saver()
            if optional_reload_path == "":
                saver.restore(sess, checkpoint_path)
            else:
                saver.restore(sess, param.save_checkpoints_path + optional_reload_path)
            print('-----------------test---------------')
            all_label_list = []
            all_pred_list = []
            all_all_l_seq = []
            all_all_p_seq = []
            m = 0
            all_loss = 0
            for seqs_, s_labels_, e_labels_, token_types_, query_lens_, text_lens_ in self.dp.get_batch(
                    test_seq_data, param.batch_size, param.shuffle):
                l, global_nums, predict_start_ids_, predict_end_ids_ = sess.run(
                    [total_loss, global_add, predict_start_ids, predict_end_ids],
                    feed_dict={
                        input_ids: seqs_,
                        start_labels: s_labels_,
                        end_labels: e_labels_,
                        token_type_ids_list: token_types_,
                        query_len_list: query_lens_,
                        text_length_list: text_lens_
                    })
                # 获取真实序列、标签长度。
                label_list, pred_list, all_l_seq, all_p_seq = eu.make_mask(seqs_, text_lens_,
                                                                           query_lens_,
                                                                           s_labels_,
                                                                           e_labels_,
                                                                           predict_start_ids_,
                                                                           predict_end_ids_)
                all_label_list += label_list
                all_pred_list += pred_list
                all_all_l_seq += all_l_seq
                all_all_p_seq += all_p_seq
                m += 1
                all_loss += l
            average_loss = all_loss / m
            all_list = np.concatenate((all_label_list, all_pred_list), axis=0)
            all_list = np.unique(all_list)
            id2tag_ = {0: "0", 1: "1"}
            target_names = [id2tag_[i] for i in all_list]
            acc = accuracy_score(all_label_list, all_pred_list)
            print(
                'loss: {:.4}, accuracy: {:.4}  '.format(average_loss, acc))
            print(classification_report(all_label_list, all_pred_list, target_names=target_names))
            repo_res_dict = eu.calculate_report(all_all_l_seq, all_all_p_seq, self.dp.id2tag)
            for key in repo_res_dict.keys():
                print(
                    '{}   Rec:{:6.2%}  Pre:{:6.2%}   f1:{:6.2%}  count:  {}'.format(key, repo_res_dict[key][0],
                                                                                    repo_res_dict[key][1],
                                                                                    repo_res_dict[key][2],
                                                                                    repo_res_dict[key][3]))
            print('-----------------test---------------')

    # 解析传入的待测试数据
    def parse_words(self, words):
        """
        切分句子，并给每个句子添加一个虚假tag
        :param words:
        :return:
        """
        res_data = []
        for word in words:
            # 标签暂时先给"O"
            one_word = list(word)
            tag = ["O"] * len(one_word)
            res_data.append([one_word, tag])
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
        param.dropout_rate = 1.0
        param.is_training = False
        param.shuffle = False
        # 定义评价函数
        eu = Evaluation_Utils()
        # 加载模型
        graph, input_ids, start_labels, end_labels, token_type_ids_list, query_len_list, \
        text_length_list, predict_start_ids, predict_end_ids, \
        global_add, total_loss, train_op = make_graph(param)
        # 获取测试数据
        words_list = self.parse_words(words)
        test_data = self.dp.parse_data(words_list, param.ner_query_map, self.dp.bert_tokenizer,
                                           param.max_seq_length)
        # 获取最小loss的ckpt文件路径
        dirs = os.listdir(param.save_checkpoints_path)
        print(dirs)
        dict_ = {}
        for file_name in dirs:
            res = re.match("model_(\d+)_([_\-])_(.+).ckpt.meta", file_name)
            if res != None:
                dict_[res.group(3)] = [res.group(1), res.group(2)]
        list_ = [float(i) for i in dict_.keys()]
        list_.sort()
        epoch, connector = dict_[str(list_[0])]
        count_loss = list_[0]
        checkpoint_path = param.save_checkpoints_path + "model_" + str(epoch) + "_" + str(connector) + "_" + str(
            count_loss) + ".ckpt"
        print(checkpoint_path)

        # 取出id2word
        id2token = {v: k for k, v in self.dp.bert_tokenizer.items()}

        all_entity_dict = []
        all_entity_list = []

        # 模型预测
        with tf.Session(graph=graph) as sess:
            saver = tf.train.Saver()
            if optional_reload_path == "":
                saver.restore(sess, checkpoint_path)
            else:
                saver.restore(sess, param.save_checkpoints_path + optional_reload_path)
            for seqs_, s_labels_, e_labels_, token_types_, query_lens_, text_lens_ in self.dp.get_batch(
                    test_data, param.batch_size, param.shuffle):
                l, global_nums, predict_start_ids_, predict_end_ids_ = sess.run(
                    [total_loss, global_add, predict_start_ids, predict_end_ids],
                    feed_dict={
                        input_ids: seqs_,
                        start_labels: s_labels_,
                        end_labels: e_labels_,
                        token_type_ids_list: token_types_,
                        query_len_list: query_lens_,
                        text_length_list: text_lens_
                    })
                label_list, pred_list, all_l_seq, all_p_seq = eu.make_mask_pred(seqs_, text_lens_,
                                                                           query_lens_,
                                                                           s_labels_,
                                                                           e_labels_,
                                                                           predict_start_ids_,
                                                                           predict_end_ids_)

                for one_token_, pred,query_lens_one,text_lens_one in zip(seqs_, all_p_seq,query_lens_, text_lens_):
                    # tags = [self.dp.id2tag[data] for data in pred]
                    tags = pred
                    # one_token_ = self.dp.bert_tokenizer.convert_ids_to_tokens(one_token_)

                    one_token_ = [id2token
                                  [token] for token in one_token_]
                    one_token_ = one_token_[query_lens_one:text_lens_one]
                    # one_token_ = "".join([self.dp.id2word[data] for data in one_token_])
                    res_dict = eu.transfrom2seq(tags)
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
    param_save_path = './param_dir/param_2021_02_26_08_51.pkl'
    # optional_reload_path = ""
    optional_reload_path = "model_5_-_0.2381.ckpt"

    ner_query_map = {
        "tags": ["product_name", "time", "person_name", "org_name", "location", "company_name"],
        "natural_query": {
            "product_name": "找出网站，软件，书籍，汽车，游戏，家电，手机软件等产品名称",
            "time": "找出年份，月份，小时，时间周期等时间名称",
            "person_name": "找出真实和虚构的人名",
            "org_name": "找出学校，医院，社会组织，国家机关等组织机构",
            "location": "找出国家，城市，山川等抽象或具体的地点",
            "company_name": "找出银行，工厂，公司，商业机构等企业名称"
        }
    }
    # ner_query_map = {
    #     "tags": ["ORG", "PER", "LOC"],
    #     "natural_query": {
    #         "ORG": "找出公司，商业机构，社会组织等组织机构",
    #         "LOC": "找出国家，城市，山川等抽象或具体的地点",
    #         "PER": "找出真实和虚构的人名"
    #     }
    # }


    # 创建保存文件
    for i in ["./param_dir", "./checkpoints", "./train_logs"]:
        if not os.path.exists(i):
            os.makedirs(i)

    BMEE = Bert_Mrc_Entity_Extraction(cfg_path)

    # 初始化数据，开启训练
    BMEE.data_process_init()
    BMEE.train(ner_query_map)

    # 加载已处理好的数据，开启训练
    # BMEE.data_process_reload()
    # BMEE.train(ner_query_map)

    # 根据训练好的模型参数，预测整个测试集
    # BMEE.data_process_reload()
    # BMEE.make_all_predict(param_save_path,optional_reload_path)

    # 根据训练好的模型参数，预测样本
    # BMEE.data_process_reload()
    # all_entity_dict, all_entity_list = BMEE.predict(words,param_save_path,optional_reload_path)
    # for i, j in zip(all_entity_dict, all_entity_list):
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