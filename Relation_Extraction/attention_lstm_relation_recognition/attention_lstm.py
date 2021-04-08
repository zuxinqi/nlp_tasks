import numpy as np
import tensorflow as tf
import re,os,datetime,pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from model import make_graph
from data_process import Date_Process

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Attention_Lstm_Relation_Recognition(object):
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
        param.num_classes = len(self.dp.tag2id)
        param.embedding_mat = self.dp.embedding_mat
        time_now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
        param.save_checkpoints_path = "./checkpoints/checkpoint_"+time_now+"/"
        param.train_log = "./train_logs/train_log_"+time_now+".txt"
        param_save_path = "./param_dir/param_"+time_now+".pkl"
        self.dp.save_param(param, param_save_path)

        # 加载模型
        graph,input_word,input_pos1,input_pos2,input_y,total_shape,predictions,true_,accuracy,acc,final_loss,train_op,global_add = make_graph(param)
        # 制作训练集和测试集
        train_seq_data = self.dp.parse_data(self.dp.train_data, self.dp.word2id, self.dp.tag2id, param.max_seq_length)
        test_seq_data = self.dp.parse_data(self.dp.test_data, self.dp.word2id, self.dp.tag2id, param.max_seq_length)
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
                    for res_seq, e1_list, e2_list, total_shape_, res_labels in self.dp.get_batch(train_seq_data, param.batch_size,
                                                                                                 param.shuffle):

                        _, l, acc_,  t, p, global_nums = sess.run(
                            [train_op, final_loss, acc, true_, predictions, global_add],
                            feed_dict={
                                input_word: res_seq,
                                input_pos1: e1_list,
                                input_pos2: e2_list,
                                total_shape: total_shape_,
                                input_y: res_labels
                            })

                        if global_nums % param.display_per_step == 0:
                            # 获取真实序列、标签长度。
                            all_list = np.concatenate((t, p), axis=0)
                            all_list = np.unique(all_list)
                            target_names = [self.dp.id2tag[i] for i in all_list]
                            acc_ = accuracy_score(t, p)
                            print(
                                'epoch {}, global_step {}, loss: {:.4}, accuracy: {:.4}  '.format(epoch + 1,
                                                                                                  global_nums + 1,
                                                                                                  l, acc_))
                            print(classification_report(t, p, target_names=target_names))
                            print()
                            print()
                        if global_nums % param.evaluation_per_step == 0:
                            print('-----------------valudation---------------')
                            res_seq, e1_list, e2_list, total_shape_, res_labels = next(self.dp.get_batch(test_seq_data, param.batch_size, param.shuffle))
                            l, acc_, t, p = sess.run(
                                [final_loss, acc, true_, predictions],
                                feed_dict={
                                    input_word: res_seq,
                                    input_pos1: e1_list,
                                    input_pos2: e2_list,
                                    total_shape: total_shape_,
                                    input_y: res_labels
                                })
                            # 获取真实序列、标签长度。
                            all_list = np.concatenate((t, p), axis=0)
                            all_list = np.unique(all_list)
                            target_names = [self.dp.id2tag[i] for i in all_list]
                            acc_ = accuracy_score(t, p)
                            print(
                                'epoch {}, global_step {}, loss: {:.4}, accuracy: {:.4}  '.format(epoch + 1,
                                                                                                  global_nums + 1,
                                                                                                  l, acc_))
                            print(classification_report(t, p, target_names=target_names))
                            print('-----------------valudation---------------')
                    for res_seq, e1_list, e2_list, total_shape_, res_labels in self.dp.get_batch(test_seq_data, param.batch_size, param.shuffle):
                        l, acc_, t, p = sess.run(
                            [final_loss, acc, true_, predictions],
                            feed_dict={
                                input_word: res_seq,
                                input_pos1: e1_list,
                                input_pos2: e2_list,
                                total_shape: total_shape_,
                                input_y: res_labels
                            })

                        # 获取真实序列、标签长度。
                        acc_ = accuracy_score(t, p)
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
                m = 0
                all_loss = 0
                for res_seq, e1_list, e2_list, total_shape_, res_labels in self.dp.get_batch(test_seq_data,
                                                                                             param.batch_size,
                                                                                             param.shuffle):
                    l, acc_, t, p = sess.run(
                        [final_loss, acc, true_, predictions],
                        feed_dict={
                            input_word: res_seq,
                            input_pos1: e1_list,
                            input_pos2: e2_list,
                            total_shape: total_shape_,
                            input_y: res_labels
                        })
                    all_label_list += t.tolist()
                    all_pred_list += p
                    m += 1
                    all_loss += l
                average_loss = all_loss / m
                all_list = np.concatenate((all_label_list, all_pred_list), axis=0)
                all_list = np.unique(all_list)
                target_names = [self.dp.id2tag[i] for i in all_list]
                acc_ = accuracy_score(all_label_list, all_pred_list)
                print(
                    'loss: {:.4}, accuracy: {:.4}  '.format(average_loss, acc_))
                print(classification_report(all_label_list, all_pred_list, target_names=target_names))
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
        param.dropout_rate = 1.0
        param.is_training = False
        param.shuffle = False

        # 加载模型
        graph, input_word, input_pos1, input_pos2, input_y, total_shape, predictions, true_, accuracy, acc, final_loss, train_op, global_add = make_graph(
            param)
        # 制作训练集和测试集
        test_seq_data = self.dp.parse_data(self.dp.test_data, self.dp.word2id, self.dp.tag2id, param.max_seq_length)

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
            m = 0
            all_loss = 0
            for res_seq, e1_list, e2_list, total_shape_, res_labels in self.dp.get_batch(test_seq_data,
                                                                                         param.batch_size,
                                                                                         param.shuffle):
                l, acc_, t, p = sess.run(
                    [final_loss, acc, true_, predictions],
                    feed_dict={
                        input_word: res_seq,
                        input_pos1: e1_list,
                        input_pos2: e2_list,
                        total_shape: total_shape_,
                        input_y: res_labels
                    })
                all_label_list += t.tolist()
                all_pred_list += p
                m += 1
                all_loss += l
            average_loss = all_loss / m
            all_list = np.concatenate((all_label_list, all_pred_list), axis=0)
            all_list = np.unique(all_list)
            target_names = [self.dp.id2tag[i] for i in all_list]
            acc_ = accuracy_score(all_label_list, all_pred_list)
            print(
                'loss: {:.4}, accuracy: {:.4}  '.format(average_loss, acc_))
            print(classification_report(all_label_list, all_pred_list, target_names=target_names))
            print('-----------------test---------------')


    def make_True_dict(self,data):
        res_dict = {}
        for one_data in data:
            # 第一个元素是真实标签的字典
            raw_one_data = one_data[0]
            text = "".join(raw_one_data["text"])
            new_true_triple_list = set()
            for s, r, o in raw_one_data["triple_list"]:
                new_true_triple_list.add(("".join(s), r, "".join(o)))
            res_dict[text] = new_true_triple_list
        return res_dict

    def calculate_f1(self,true_dict, pred_dict):
        correct_num, predict_num, gold_num = 1e-10, 1e-10, 1e-10
        for i in true_dict.keys():
            true_triple_list = true_dict[i]
            predict_triple_list = pred_dict[i]
            correct_num += len(true_triple_list & predict_triple_list)
            gold_num += len(true_triple_list)
            predict_num += len(predict_triple_list)
        recall = correct_num / gold_num
        precision = correct_num / predict_num
        f1_score = 2 * precision * recall / (precision + recall)
        return recall,precision,f1_score


    # 模型预测
    def predict_complete(self,file_path,param_save_path,id2tag,optional_reload_path=""):
        """
        :param words: 传入待预测的数据
        :param param_save_path:自选的加载的参数文件路径，如果为空字符串，则自动选择最低loss模型
        :return:
        """
        # 恢复模型参数，并针对测试修改部分参数
        param = self.dp.load_param(param_save_path)
        param.embedding_mat = self.dp.embedding_mat
        param.dropout_rate = 1.0
        param.is_training = False
        param.shuffle = False

        # 加载模型
        graph, input_word, input_pos1, input_pos2, input_y, total_shape, predictions, true_, accuracy, acc, final_loss, train_op, global_add = make_graph(
            param)

        # 加载测试数据
        with open(file_path, 'rb') as f:
            test_data = pickle.load(f)

        allow_size = len(test_data)//param.batch_size
        allow_size = allow_size*param.batch_size
        test_data = test_data[:allow_size]
        # 获取测试数据
        # test_list为添加a1 b1的数据
        # test_entity_list 为保存了[text,s,o]的数据
        # raw_entity_list 为保存了[text,s_set,o_set]的数据，目的为了检查实体抽取情况
        test_list,test_entity_list,raw_entity_list = self.dp.parse_test_data(test_data, param.max_seq_length)
        test_seq_data = self.dp.parse_data(test_list, self.dp.word2id, self.dp.tag2id, param.max_seq_length)
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
            all_pred_list = []
            for res_seq, e1_list, e2_list, total_shape_, res_labels in self.dp.get_batch(test_seq_data,
                                                                                         param.batch_size,
                                                                                         param.shuffle):
                l, acc_, t, p = sess.run(
                    [final_loss, acc, true_, predictions],
                    feed_dict={
                        input_word: res_seq,
                        input_pos1: e1_list,
                        input_pos2: e2_list,
                        total_shape: total_shape_,
                        input_y: res_labels
                    })
                pred_list = p
                all_pred_list += pred_list
        # 拿出测试数据的dict
        true_dict = self.make_True_dict(test_data)
        # 建立预测数据的dict
        pred_dict = {}
        for i in true_dict:
            triple_list = set()
            pred_dict[i] = triple_list
        for test_one_data,pred in zip(test_entity_list,all_pred_list):
            text = test_one_data[0]
            pred_r = id2tag[pred]
            if text not in pred_dict:
                print("真实数据和预测出来的数据文本text发生改变，请查看原因")
                raise ValueError
            if pred_r != "无关":
                s = test_one_data[1]
                o = test_one_data[2]
                pred_dict[text].add((s, pred_r, o))
        recall,precision,f1_score = self.calculate_f1(true_dict, pred_dict)
        print(
            'complete_recall: {:.4}, complete_precisionl: {:.4}, complete_f1: {:.4}  '.format(recall, precision,
                                                                                              f1_score))





if __name__ == '__main__':
    words = ["布丹出生于1824年的法国画家",
             "黄主文，1941年8月20日出生，台湾省桃园县人，台湾大学法律系毕业、司法官训练所第8期结业、司法官高等考试及格",
             "杨维桢（1296—1370），元末明初著名诗人、文学家、书画家和戏曲家",
             "张柏芝，1980年5月24日出生于中国香港，毕业于澳大利亚Rmit Holmes College，中国香港女演员、歌手"
             ]
    # words = '《DNF》玩家感叹；当时迫不及待头脑发热了一下，现在后悔莫及！。关键字: dnf'
    cfg_path = './default.cfg'
    param_save_path = './param_dir/param_2021_03_10_16_21.pkl'
    # optional_reload_path = ""
    optional_reload_path = "model_6_-_8.3785.ckpt"

    # 创建保存文件
    for i in ["./param_dir", "./checkpoints", "./train_logs"]:
        if not os.path.exists(i):
            os.makedirs(i)

    ALRR = Attention_Lstm_Relation_Recognition(cfg_path)

    # 初始化数据，开启训练
    # ALRR.data_process_init()
    # ALRR.train()

    # 加载已处理好的数据，开启训练
    # ALRR.data_process_reload()
    # ALRR.train()

    # 根据训练好的模型参数，预测整个测试集
    # ALRR.data_process_reload()
    # ALRR.make_all_predict(param_save_path,optional_reload_path)

    # 根据训练好的模型参数，预测整个测试集
    ALRR.data_process_reload()
    file_path = "/root/zxq/all_models/Relation_Extraction/entity_extraction_bert_lstm_crf/entity_list.pkl"
    ALRR.predict_complete(file_path,param_save_path,ALRR.dp.id2tag,optional_reload_path)


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