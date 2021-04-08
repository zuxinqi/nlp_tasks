import numpy as np
import tensorflow as tf
import re,os,datetime,argparse
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from model import make_graph
from data_process import Date_Process
from evaluation_utils import Evaluation_Utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Multihead_Joint_Relation_Extraction(object):
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
        max_seq_length = self.dp.config.getint('DATA_PROCESS', 'max_seq_length')
        feature_selection_name = self.dp.config.get('DATA_PROCESS', 'feature_selection_name')
        self.dp.init(max_seq_length, feature_selection_name)
        self.dp.save_model(self.dp.save_path)

    def data_process_reload(self):
        """
        模型重复训练，运行此函数，恢复初处理好的数据
        :return:
        """
        self.dp.load_model(self.dp.save_path)
        self.dp.tokenizer_init()

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
    def train(self):
        """
        训练模型
        :param :
        :return:
        """
        # 将模型参数导出，添加一些当前模型所需参数，并保存
        param = self.dp.param
        param.ner_num_tags = len(self.dp.ner_tag2id)
        param.rel_num_tags = len(self.dp.rel_tag2id)
        param.embedding_mat = self.dp.embedding_mat
        time_now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
        param.save_checkpoints_path = "./checkpoints/checkpoint_"+time_now+"/"
        param.train_log = "./train_logs/train_log_"+time_now+".txt"
        param_save_path = "./param_dir/param_"+time_now+".pkl"
        self.dp.save_param(param, param_save_path)

        # 定义评价函数
        eu = Evaluation_Utils()
        # 加载模型
        graph,input_ids,sequence_lengths,ner_label_ids,rel_label_ids,is_train,ner_pred,rel_predict,total_loss,train_op,global_add = make_graph(param)
        # 制作训练集和测试集
        train_seq_data = self.dp.parse_data(self.dp.train_data, param.max_seq_length)
        test_seq_data = self.dp.parse_data(self.dp.test_data, param.max_seq_length)
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
                    for seqs, ner_labels, rel_labels, sentence_legth in self.dp.get_batch(train_seq_data, param.batch_size,param.shuffle):
                        _, l, ner_p, rel_p, global_nums = sess.run(
                            [train_op,total_loss,ner_pred, rel_predict,global_add], {
                                input_ids: seqs,
                                ner_label_ids: ner_labels,
                                rel_label_ids: rel_labels,
                                sequence_lengths: sentence_legth,
                                is_train: 1
                            })

                        if global_nums % param.display_per_step == 0:
                            # 获取真实序列、标签长度。
                            recall, precision, f1_score = eu.make_rel_evaluation(rel_labels, rel_p)
                            print(
                                'epoch {}, global_step {}, loss: {:.4}  '.format(epoch + 1, global_nums + 1, l))
                            print()
                            print(
                                '1_recall: {:.4}, 1_precisionl: {:.4}, 1_f1: {:.4}  '.format(recall, precision,
                                                                                             f1_score))

                            label_list, pred_list = eu.make_mask(ner_labels, ner_p, sentence_legth)
                            all_list = np.concatenate((label_list, pred_list), axis=0)
                            all_list = np.unique(all_list)
                            target_names = [self.dp.ner_id2tag[i] for i in all_list]
                            acc = accuracy_score(label_list, pred_list)
                            print(
                                'epoch {}, global_step {}, loss: {:.4}, accuracy: {:.4}  '.format(epoch + 1,
                                                                                                  global_nums + 1,
                                                                                                  l, acc))
                            print(classification_report(label_list, pred_list, target_names=target_names))
                            repo_res_dict = eu.calculate_report(label_list, pred_list, self.dp.ner_id2tag)
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
                            seqs, ner_labels, rel_labels, sentence_legth = next(self.dp.get_batch(test_seq_data, param.batch_size, param.shuffle))
                            l, ner_p, rel_p = sess.run([total_loss, ner_pred, rel_predict],
                                                                      feed_dict={
                                                                                input_ids: seqs,
                                                                                ner_label_ids: ner_labels,
                                                                                rel_label_ids: rel_labels,
                                                                                sequence_lengths: sentence_legth,
                                                                                is_train: 0
                                                                                })
                            recall, precision, f1_score = eu.make_rel_evaluation(rel_labels, rel_p)
                            print(
                                'epoch {}, global_step {}, loss: {:.4}  '.format(epoch + 1, global_nums + 1, l))
                            print()
                            print(
                                '1_recall: {:.4}, 1_precisionl: {:.4}, 1_f1: {:.4}  '.format(recall, precision,
                                                                                             f1_score))

                            label_list, pred_list = eu.make_mask(ner_labels, ner_p, sentence_legth)
                            all_list = np.concatenate((label_list, pred_list), axis=0)
                            all_list = np.unique(all_list)
                            target_names = [self.dp.ner_id2tag[i] for i in all_list]
                            acc = accuracy_score(label_list, pred_list)
                            print(
                                'epoch {}, global_step {}, loss: {:.4}, accuracy: {:.4}  '.format(epoch + 1,
                                                                                                  global_nums + 1,
                                                                                                  l, acc))
                            print(classification_report(label_list, pred_list, target_names=target_names))
                            repo_res_dict = eu.calculate_report(label_list, pred_list, self.dp.ner_id2tag)
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
                    for seqs, ner_labels, rel_labels, sentence_legth in self.dp.get_batch(test_seq_data, param.batch_size, param.shuffle):
                        l, ner_p, rel_p = sess.run([total_loss, ner_pred, rel_predict],
                                                   feed_dict={
                                                       input_ids: seqs,
                                                       ner_label_ids: ner_labels,
                                                       rel_label_ids: rel_labels,
                                                       sequence_lengths: sentence_legth,
                                                       is_train: 0
                                                   })

                        # 获取真实序列、标签长度。
                        label_list, pred_list = eu.make_mask(ner_labels, ner_p, sentence_legth)
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
                correct_num, predict_num, gold_num = 1e-10, 1e-10, 1e-10
                correct_num_r, predict_num_r, gold_num_r = 1e-10, 1e-10, 1e-10
                ner_all_label_list = []
                ner_all_pred_list = []
                all_sequence_length = []
                m = 0
                all_loss = 0
                for seqs, ner_labels, rel_labels, sentence_legth in self.dp.get_batch(test_seq_data, param.batch_size,
                                                                                      param.shuffle):
                    l, ner_p, rel_p = sess.run([total_loss, ner_pred, rel_predict],
                                               feed_dict={
                                                   input_ids: seqs,
                                                   ner_label_ids: ner_labels,
                                                   rel_label_ids: rel_labels,
                                                   sequence_lengths: sentence_legth,
                                                   is_train: 0
                                               })
                    for i in range(len(ner_labels)):
                        data_ture = [ner_labels[i], rel_labels[i], seqs[i], sentence_legth[i]]
                        triples_set_true = eu.make_triples(data_ture, self.dp.ner_id2tag, self.dp.rel_id2tag,
                                                           self.dp.id2word)

                        data_pred = [ner_p[i], rel_p[i], seqs[i], sentence_legth[i]]
                        triples_set_pred = eu.make_triples(data_pred, self.dp.ner_id2tag, self.dp.rel_id2tag,
                                                           self.dp.id2word)

                        correct_num += len(triples_set_true & triples_set_pred)
                        gold_num += len(triples_set_true)
                        predict_num += len(triples_set_pred)

                    for i in range(len(rel_labels)):
                        new_rel_ids = np.where(rel_labels[i] == 1)
                        new_rel_ids = np.array(new_rel_ids).T
                        new_rel_ids_set = set(tuple(i) for i in new_rel_ids)

                        pred_rel_ids = np.where(rel_p[i] == 1)
                        pred_rel_ids = np.array(pred_rel_ids).T
                        pred_rel_ids_set = set(tuple(i) for i in pred_rel_ids)

                        correct_num_r += len(new_rel_ids_set & pred_rel_ids_set)
                        gold_num_r += len(new_rel_ids_set)
                        predict_num_r += len(pred_rel_ids_set)

                    ner_p = ner_p.tolist()
                    ner_labels = ner_labels.tolist()
                    ner_all_label_list += ner_labels
                    ner_all_pred_list += ner_p

                    all_sequence_length += sentence_legth
                    m += 1
                    all_loss += l

                average_loss = all_loss / m
                recall = correct_num / gold_num
                precision = correct_num / predict_num
                f1_score = 2 * precision * recall / (precision + recall)

                recall_r = correct_num_r / gold_num_r
                precision_r = correct_num_r / predict_num_r
                f1_score_r = 2 * precision_r * recall_r / (precision_r + recall_r)

                print(
                    'loss: {:.4}  '.format(average_loss))
                print()
                print(
                    'r_recall: {:.4}, r_precisionl: {:.4}, r_f1: {:.4}  '.format(recall_r, precision_r, f1_score_r))
                print()
                print(
                    'complete_recall: {:.4}, complete_precisionl: {:.4}, complete_f1: {:.4}  '.format(recall, precision,
                                                                                                      f1_score))
                ner_all_label_list, ner_all_pred_list = eu.make_mask(ner_all_label_list,
                                                                                   ner_all_pred_list,
                                                                                   all_sequence_length)
                ner_all_list = np.concatenate((ner_all_label_list, ner_all_pred_list), axis=0)
                ner_all_list = np.unique(ner_all_list)
                ner_target_names = [self.dp.ner_id2tag[i] for i in ner_all_list]
                print()
                print(classification_report(ner_all_label_list, ner_all_pred_list, target_names=ner_target_names))

                repo_res_dict = eu.calculate_report(ner_all_label_list, ner_all_pred_list, self.dp.ner_id2tag)

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
        param.embedding_mat = self.dp.embedding_mat
        param.embedding_dropout = 1.0
        param.lstm_dropout = 1.0
        param.rel_dropout = 1.0
        param.shuffle = False

        # 定义评价函数
        eu = Evaluation_Utils()
        # 加载模型
        graph,input_ids,sequence_lengths,ner_label_ids,rel_label_ids,is_train,ner_pred,rel_predict,total_loss,train_op,global_add = make_graph(param)
        # 获取测试数据
        test_seq_data = self.dp.parse_data(self.dp.test_data, param.max_seq_length)

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
            correct_num, predict_num, gold_num = 1e-10, 1e-10, 1e-10
            correct_num_r, predict_num_r, gold_num_r = 1e-10, 1e-10, 1e-10
            ner_all_label_list = []
            ner_all_pred_list = []
            all_sequence_length = []
            m = 0
            all_loss = 0
            for seqs, ner_labels, rel_labels, sentence_legth in self.dp.get_batch(test_seq_data, param.batch_size,
                                                                                  param.shuffle):
                l, ner_p, rel_p = sess.run([total_loss, ner_pred, rel_predict],
                                           feed_dict={
                                               input_ids: seqs,
                                               ner_label_ids: ner_labels,
                                               rel_label_ids: rel_labels,
                                               sequence_lengths: sentence_legth,
                                               is_train: 0
                                           })
                for i in range(len(ner_labels)):
                    data_ture = [ner_labels[i], rel_labels[i], seqs[i], sentence_legth[i]]
                    triples_set_true = eu.make_triples(data_ture, self.dp.ner_id2tag,self.dp.rel_id2tag,
                                                       self.dp.id2word)

                    data_pred = [ner_p[i], rel_p[i],seqs[i], sentence_legth[i]]
                    triples_set_pred = eu.make_triples(data_pred, self.dp.ner_id2tag,self.dp.rel_id2tag,
                                                       self.dp.id2word)

                    correct_num += len(triples_set_true & triples_set_pred)
                    gold_num += len(triples_set_true)
                    predict_num += len(triples_set_pred)

                for i in range(len(rel_labels)):
                    new_rel_ids = np.where(rel_labels[i] == 1)
                    new_rel_ids = np.array(new_rel_ids).T
                    new_rel_ids_set = set(tuple(i) for i in new_rel_ids)

                    pred_rel_ids = np.where(rel_p[i] == 1)
                    pred_rel_ids = np.array(pred_rel_ids).T
                    pred_rel_ids_set = set(tuple(i) for i in pred_rel_ids)

                    correct_num_r += len(new_rel_ids_set & pred_rel_ids_set)
                    gold_num_r += len(new_rel_ids_set)
                    predict_num_r += len(pred_rel_ids_set)

                ner_p = ner_p.tolist()
                ner_labels = ner_labels.tolist()
                ner_all_label_list += ner_labels
                ner_all_pred_list += ner_p

                all_sequence_length += sentence_legth
                m += 1
                all_loss += l

            average_loss = all_loss / m
            recall = correct_num / gold_num
            precision = correct_num / predict_num
            f1_score = 2 * precision * recall / (precision + recall)

            recall_r = correct_num_r / gold_num_r
            precision_r = correct_num_r / predict_num_r
            f1_score_r = 2 * precision_r * recall_r / (precision_r + recall_r)

            print(
                'loss: {:.4}  '.format(average_loss))
            print()
            print(
                'r_recall: {:.4}, r_precisionl: {:.4}, r_f1: {:.4}  '.format(recall_r, precision_r, f1_score_r))
            print()
            print(
                'complete_recall: {:.4}, complete_precisionl: {:.4}, complete_f1: {:.4}  '.format(recall, precision,
                                                                                                  f1_score))
            ner_all_label_list, ner_all_pred_list = eu.make_mask(ner_all_label_list,
                                                                 ner_all_pred_list,
                                                                 all_sequence_length)
            ner_all_list = np.concatenate((ner_all_label_list, ner_all_pred_list), axis=0)
            ner_all_list = np.unique(ner_all_list)
            ner_target_names = [self.dp.ner_id2tag[i] for i in ner_all_list]
            print()
            print(classification_report(ner_all_label_list, ner_all_pred_list, target_names=ner_target_names))

            repo_res_dict = eu.calculate_report(ner_all_label_list, ner_all_pred_list, self.dp.ner_id2tag)

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
            print('-----------------test---------------')

    # 解析传入的待测试数据
    def parse_words(self, words, max_seq_length):
        """
        切分句子，并给每个句子添加一个虚假tag
        :param words:
        :return:
        """
        res_data = []
        for word in words:
            if len(word) > max_seq_length:
                word = word[:max_seq_length]
            # 标签暂时先给"O"
            one_word = [i.lower() for i in word]
            ner_tags = ["O"] * len(one_word)
            rel_tags = np.zeros((max_seq_length, max_seq_length, len(self.dp.rel_tag2id)))
            res_data.append([one_word, ner_tags, rel_tags, len(one_word)])
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
        param.embedding_mat = self.dp.embedding_mat
        param.embedding_dropout = 1.0
        param.lstm_dropout = 1.0
        param.rel_dropout = 1.0
        param.shuffle = False
        # 定义评价函数
        eu = Evaluation_Utils()
        # 加载模型
        graph, input_ids, sequence_lengths, ner_label_ids, rel_label_ids, is_train, ner_pred, rel_predict, total_loss, train_op, global_add = make_graph(
            param)
        # 获取测试数据
        words_list = self.parse_words(words,param.max_seq_length)
        test_seq_data = self.dp.parse_data(words_list, param.max_seq_length)
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
        res_list = []
        with tf.Session(graph=graph) as sess:
            saver = tf.train.Saver()
            if optional_reload_path == "":
                saver.restore(sess, checkpoint_path)
            else:
                saver.restore(sess, param.save_checkpoints_path + optional_reload_path)
            print('-----------------test---------------')
            for seqs, ner_labels, rel_labels, sentence_legth in self.dp.get_batch(test_seq_data, param.batch_size,
                                                                                  param.shuffle):
                l, ner_p, rel_p = sess.run([total_loss, ner_pred, rel_predict],
                                           feed_dict={
                                               input_ids: seqs,
                                               ner_label_ids: ner_labels,
                                               rel_label_ids: rel_labels,
                                               sequence_lengths: sentence_legth,
                                               is_train: 0
                                           })
                for i in range(len(ner_labels)):
                    line = tuple([self.dp.id2word[i] for i in seqs[i][:sentence_legth[i]]])
                    data_pred = [ner_p[i], rel_p[i], seqs[i], sentence_legth[i]]
                    triples_set_pred = eu.make_triples(data_pred, self.dp.ner_id2tag,self.dp.rel_id2tag,self.dp.id2word)
                    res_list.append([line,triples_set_pred])
        return res_list





if __name__ == '__main__':
    words = ["布丹出生于1824年的法国画家",
             "黄主文，1941年8月20日出生，台湾省桃园县人，台湾大学法律系毕业、司法官训练所第8期结业、司法官高等考试及格",
             "杨维桢（1296—1370），元末明初著名诗人、文学家、书画家和戏曲家",
             "张柏芝，1980年5月24日出生于中国香港，毕业于澳大利亚Rmit Holmes College，中国香港女演员、歌手"
             ]
    # words = '《DNF》玩家感叹；当时迫不及待头脑发热了一下，现在后悔莫及！。关键字: dnf'
    cfg_path = './default.cfg'
    param_save_path = './param_dir/param_2021_03_02_15_21.pkl'
    optional_reload_path = ""
    # optional_reload_path = "model_11_-_5.6428.ckpt"

    # 创建保存文件
    for i in ["./param_dir", "./checkpoints", "./train_logs"]:
        if not os.path.exists(i):
            os.makedirs(i)

    MJRE = Multihead_Joint_Relation_Extraction(cfg_path)

    # 初始化数据，开启训练
    # MJRE.data_process_init()
    # MJRE.train()

    # 加载已处理好的数据，开启训练
    # MJRE.dp.read_data()
    # max_seq_length = MJRE.dp.config.getint('DATA_PROCESS', 'max_seq_length')
    # MJRE.dp.data_process(max_seq_length)
    # MJRE.data_process_reload()
    # MJRE.train()

    # 根据训练好的模型参数，预测整个测试集
    MJRE.dp.read_data()
    max_seq_length = MJRE.dp.config.getint('DATA_PROCESS', 'max_seq_length')
    MJRE.dp.data_process(max_seq_length)
    MJRE.data_process_reload()
    MJRE.make_all_predict(param_save_path,optional_reload_path)

    # 根据训练好的模型参数，预测样本
    # MJRE.data_process_reload()
    # res_list = MJRE.predict(words,param_save_path,optional_reload_path)
    # for line, triple_list in res_list:
    #     print("".join(line))
    #     # print(triple_list)
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