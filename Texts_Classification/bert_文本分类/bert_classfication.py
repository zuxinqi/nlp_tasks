import modeling
import re,os,datetime
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from model_new import make_graph
from data_process_with_tokenization import Date_Process

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Bert_Classfication(object):
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
        # 这里还可以设置英文字母大小写 do_lower_case
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
    def train(self):
        """
        训练模型
        :param :
        :return:
        """
        # 将模型参数到处，添加一些当前模型所需参数，并保存
        param = self.dp.param
        param.tag2id = self.dp.tag2id
        param.num_train_steps = int(
            len(self.dp.train_data) / param.batch_size * param.num_train_epochs)
        param.num_warmup_steps = int(param.num_train_steps * param.warmup_proportion)
        param.bert_config = modeling.BertConfig.from_json_file(param.bert_config_file)
        time_now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
        param.save_checkpoints_path = "./checkpoints/checkpoint_"+time_now+"/"
        param.train_log = "./train_logs/train_log_"+time_now+".txt"
        param_save_path = "./param_dir/param_" + time_now + ".pkl"
        self.dp.save_param(param, param_save_path)

        # 加载模型
        graph, input_ids, input_mask, segment_ids, label_ids, \
        dropout_rate, train_op, total_loss, logits, pred, true_, acc, global_add = make_graph(param)
        # 制作训练集和测试集
        train_seq_data = self.dp.parse_data(self.dp.train_data, self.dp.bert_tokenizer, self.dp.tag2id, param.max_seq_length)
        test_seq_data = self.dp.parse_data(self.dp.test_data, self.dp.bert_tokenizer, self.dp.tag2id, param.max_seq_length)
        # 训练模型
        with tf.Session(graph=graph) as sess:
            if param.is_training:
                saver = tf.train.Saver(tf.global_variables())
                dev_best_loss = float('inf')
                last_improve = 0
                acc_dict = {}
                init = tf.global_variables_initializer()
                sess.run(init)

                for epoch in range(param.num_train_epochs):
                    n = 0
                    all_acc = 0
                    count_loss = 0
                    for input_list, mask_list, segment_list, label_list in self.dp.get_batch(train_seq_data, param.batch_size, param.shuffle):
                        _, l, acc_, p, t, global_nums = sess.run([train_op, total_loss, acc, pred, true_, global_add],
                                                                 feed_dict={
                                                                     input_ids: input_list,
                                                                     input_mask: mask_list,
                                                                     segment_ids: segment_list,
                                                                     label_ids: label_list,
                                                                     dropout_rate:param.dropout_rate
                                                                 })

                        if global_nums % param.display_per_step == 0:
                            all_list = np.concatenate((label_list, p), axis=0)
                            all_list = np.unique(all_list)
                            target_names = [self.dp.id2tag[i] for i in all_list]
                            print(
                                'epoch {}, global_step {}, loss: {:.4}, accuracy: {:.4}  '.format(epoch + 1,
                                                                                                  global_nums + 1,
                                                                                                  l, acc_))
                            print(classification_report(label_list, p, target_names=target_names))
                        if global_nums % param.evaluation_per_step == 0:
                            print('-----------------valudation---------------')
                            input_list, mask_list, segment_list, label_list = next(self.dp.get_batch(test_seq_data, param.batch_size, param.shuffle))
                            l, acc_, p, t = sess.run([total_loss, acc, pred, true_],
                                                     feed_dict={
                                                         input_ids: input_list,
                                                         input_mask: mask_list,
                                                         segment_ids: segment_list,
                                                         label_ids: label_list,
                                                         dropout_rate: 1.0
                                                     })
                            all_list = np.concatenate((label_list, p), axis=0)
                            all_list = np.unique(all_list)
                            target_names = [self.dp.id2tag[i] for i in all_list]
                            print(
                                'epoch {}, global_step {}, loss: {:.4}, accuracy: {:.4} '.format(epoch + 1,
                                                                                                 global_nums + 1,
                                                                                                 l, acc_))
                            print(classification_report(label_list, p, target_names=target_names))
                            print('-----------------valudation---------------')
                    for input_list, mask_list, segment_list, label_list in self.dp.get_batch(test_seq_data, param.batch_size, param.shuffle):
                        l, acc_, p, t = sess.run([total_loss, acc, pred, true_], feed_dict={
                            input_ids: input_list,
                            input_mask: mask_list,
                            segment_ids: segment_list,
                            label_ids: label_list,
                            dropout_rate: 1.0
                        })
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

                for input_list, mask_list, segment_list, label_list in self.dp.get_batch(test_seq_data,
                                                                                         param.batch_size,
                                                                                         param.shuffle):
                    l, acc_, p, t = sess.run([total_loss, acc, pred, true_], feed_dict={
                        input_ids: input_list,
                        input_mask: mask_list,
                        segment_ids: segment_list,
                        label_ids: label_list,
                        dropout_rate: 1.0
                    })
                    all_label_list += label_list.tolist()
                    all_pred_list += p.tolist()
                all_list = np.concatenate((all_label_list, all_pred_list), axis=0)
                all_list = np.unique(all_list)
                target_names = [self.dp.id2tag[i] for i in all_list]
                acc_all = accuracy_score(all_label_list, all_pred_list)
                print(
                    'loss: {:.4}, accuracy: {:.4}  '.format(l, acc_all))
                print(classification_report(all_label_list, all_pred_list, target_names=target_names))
                with open(param.train_log, "a+", encoding="utf-8") as f:
                    self.write_param(param,acc_dict,f)

                print('-----------------test---------------')


    # 对整个测试集进行测试
    def make_all_predict(self,param_save_path,optional_reload_path=""):
        """
        对整个测试集进行测试，可改变参数文件
        :param param_save_path: 保存的参数文件路径
        :param optional_reload_path: 自选的加载的参数文件路径，如果为空字符串，则自动选择最低loss模型
        :return:
        """
        # 恢复模型参数，并针对测试修改部分参数
        param = self.dp.load_param(param_save_path)
        param.is_training = False
        param.shuffle = False
        # 加载模型
        graph, input_ids, input_mask, segment_ids, label_ids, \
        dropout_rate, train_op, total_loss, logits, pred, true_, acc, global_add = make_graph(param)
        # 获取测试数据
        test_seq_data = self.dp.parse_data(self.dp.test_data, self.dp.bert_tokenizer, self.dp.tag2id,
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
            n = 0
            all_loss = 0
            for input_list, mask_list, segment_list, label_list in self.dp.get_batch(test_seq_data,
                                                                                     param.batch_size,
                                                                                     param.shuffle):
                l, acc_, p, t = sess.run([total_loss, acc, pred, true_], feed_dict={
                    input_ids: input_list,
                    input_mask: mask_list,
                    segment_ids: segment_list,
                    label_ids: label_list,
                    dropout_rate: 1.0
                })
                n += 1
                all_loss += l
                all_label_list += label_list.tolist()
                all_pred_list += p.tolist()
            average_loss = all_loss/n
            all_list = np.concatenate((all_label_list, all_pred_list), axis=0)
            all_list = np.unique(all_list)
            target_names = [self.dp.id2tag[i] for i in all_list]
            acc_all = accuracy_score(all_label_list, all_pred_list)
            print(
                'loss: {:.4}, accuracy: {:.4}  '.format(average_loss, acc_all))
            print(classification_report(all_label_list, all_pred_list, target_names=target_names))
            print('-----------------test---------------')

    # 解析传入的待测试数据
    def parse_words(self, words):
        """
        切分句子，并给每个句子添加一个虚假tag
        :param words:
        :return:
        """
        words_list = []
        for word in words:
            seg_list = list(word)
            words_list.append([seg_list,self.dp.id2tag[0]])
        return words_list


    # 模型预测
    def predict(self,words,param_save_path,optional_reload_path=""):
        """
        :param words: 传入待预测的数据
        :param param_save_path:自选的加载的参数文件路径，如果为空字符串，则自动选择最低loss模型
        :return:
        """
        # 恢复模型参数，并针对测试修改部分参数
        param = self.dp.load_param(param_save_path)
        param.is_training = False
        param.shuffle = False
        # 加载模型
        graph, input_ids, input_mask, segment_ids, label_ids, \
        dropout_rate, train_op, total_loss, logits, pred, true_, acc, global_add = make_graph(param)
        # 获取测试数据
        words_list = self.parse_words(words)
        test_seq_data = self.dp.parse_data(words_list, self.dp.bert_tokenizer, param.max_seq_length)
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
        # 模型预测
        with tf.Session(graph=graph) as sess:
            saver = tf.train.Saver()
            if optional_reload_path == "":
                saver.restore(sess, checkpoint_path)
            else:
                saver.restore(sess, param.save_checkpoints_path + optional_reload_path)
            all_pred = []
            for input_list, mask_list, segment_list, label_list in self.dp.get_batch(test_seq_data,
                                                                                     param.batch_size,
                                                                                     param.shuffle):
                l, acc_, p, t = sess.run([total_loss, acc, pred, true_], feed_dict={
                    input_ids: input_list,
                    input_mask: mask_list,
                    segment_ids: segment_list,
                    label_ids: label_list,
                    dropout_rate: 1.0
                })
                all_pred += p.tolist()
        predicted = [self.dp.id2tag[i] for i in all_pred]
        return predicted


if __name__ == '__main__':
    words = ['《DNF》玩家感叹；当时迫不及待头脑发热了一下，现在后悔莫及！。关键字: dnf',
             '王者荣耀的哪个英雄或者哪个技能最让你怀念呢？',
             '刘德华的吃相、刘晓庆的吃相、徐峥的吃相，最后一个最丑让人恶心']
    # words = '《DNF》玩家感叹；当时迫不及待头脑发热了一下，现在后悔莫及！。关键字: dnf'
    cfg_path = './default.cfg'
    param_save_path = './param_dir/param_2021_02_22_15_24.pkl'
    # optional_reload_path = ""
    optional_reload_path = "model_4_-_0.3399.ckpt"
    # 创建保存文件
    for i in ["./param_dir","./checkpoints","./train_logs"]:
        if not os.path.exists(i):
            os.makedirs(i)


    BC = Bert_Classfication(cfg_path)

    # 初始化数据，开启训练
    # BC.data_process_init()
    # BC.train()

    # 加载已处理好的数据，开启训练
    # BC.data_process_reload()
    # BC.train()

    # 根据训练好的模型参数，预测整个测试集
    BC.data_process_reload()
    BC.make_all_predict(param_save_path,optional_reload_path)

    # 根据训练好的模型参数，预测样本
    # BC.data_process_reload()
    # predicted = BC.predict(words,param_save_path,optional_reload_path)
    # print(predicted)