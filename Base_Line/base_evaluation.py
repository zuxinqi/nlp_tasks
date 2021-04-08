import numpy as np
from tensorflow.contrib.crf import viterbi_decode

class Base_Evaluation():
    def __init__(self):
        self.name = "base_evaluation"

    def make_mask(self,logits_, labels_, sentence_legth, is_CRF=False, transition_params_=None,action="train"):
        """
        获取真实序列、标签长度。
        :param logits_: 预测的标签
        :param labels_: 真实的标签
        :param sentence_legth: 真实文本长度
        :param is_CRF: 是否使用维特比解码
        :param transition_params_: 传入的转移概率矩阵
        :param action: 是训练还是测试
        :return: label_list, pred_list，列表，既截断完的训练数据和预测数据
        """
        pred_list = []
        label_list = []
        for log, lab, seq_len in zip(logits_, labels_, sentence_legth):
            if is_CRF:
                viterbi_seq, _ = viterbi_decode(log[:seq_len], transition_params_)
            else:
                viterbi_seq = log[:seq_len]
            if action == "train":
                pred_list.extend(viterbi_seq)
                label_list.extend(lab[:seq_len])
            elif action == "pred":
                pred_list.append(viterbi_seq)
                label_list.append(lab[:seq_len])
            else:
                # 默认是train
                pred_list.extend(viterbi_seq)
                label_list.extend(lab[:seq_len])
        return label_list, pred_list

    # 将数据解析出来
    def transfrom2seq(self, tags):
        """
        将传入的序列解析，变为标签搭配下标的形式返回
        :param tags: 列表，类似[B-XXX,I-XXX,O,O,O]
        :return: res_dict,字典包含标签名及位置,例如:{XXX:[(2,4),(11,14)]}
        """
        res_dict = {}
        i = 0
        while i < len(tags):
            if tags[i] == 'O':
                j = i + 1
                while j < len(tags) and tags[j] == 'O':
                    j += 1
            else:
                if tags[i][0] != 'B':
                    #                 print(tags[i][0] + ' error start')
                    j = i + 1
                else:
                    if tags[i][2:] not in res_dict:
                        res_dict[tags[i][2:]] = []
                    j = i + 1
                    while j < len(tags) and tags[j][0] == 'I' and tags[j][2:] == tags[i][2:]:
                        j += 1
                    res_dict[tags[i][2:]].append((i, j))
            i = j
        return res_dict

    # 计算精确率、召回率、f1值
    def calculate_report(self,true_data, pred_data, id2tag):
        """
        传入正确的序列及预测的序列，计算召回率、精确率及f1
        :param true_data: 列表，类似[B-XXX,I-XXX,O,O,O]
        :param pred_data: 列表，类似[B-XXX,I-XXX,O,O,O]
        :param id2tag: 字典，类似{0:"O",1:"B-XXX",2:"I-XXX"......}
        :return:res_report_dict，字典包含召回率、精确率及f1，类似:{XXX:[recall,precision,f1]}
        """
        res_report_dict = {}
        for value in id2tag.values():
            if len(value) > 2:
                if value[2:] not in res_report_dict:
                    res_report_dict[value[2:]] = [0, 0, 0, 0]
        true_data = [id2tag[data] for data in true_data]
        pred_data = [id2tag[data] for data in pred_data]
        true_res_dict = self.transfrom2seq(true_data)
        pred_res_dict = self.transfrom2seq(pred_data)
        for key in true_res_dict.keys():
            if key not in pred_res_dict:
                res_report_dict[key][0] = 0
                res_report_dict[key][3] = len(true_res_dict[key])
            else:
                recall_name = 0
                res_report_dict[key][3] = len(true_res_dict[key])
                for v in true_res_dict[key]:
                    if v in pred_res_dict[key]:
                        recall_name += 1
                if len(true_res_dict[key]) > 0:
                    res_report_dict[key][0] = round(recall_name / len(true_res_dict[key]), 2)
                if len(pred_res_dict[key]) > 0:
                    res_report_dict[key][1] = round(recall_name / len(pred_res_dict[key]), 2)
        for key in res_report_dict.keys():
            res_report_dict[key][2] = round(
                2 * (res_report_dict[key][0] * res_report_dict[key][1]) / (res_report_dict[key][0]
                                                                           + res_report_dict[key][1] + 0.000001), 2)
        return res_report_dict

    def extract_entity(self,words, res_dict):
        """
        将测试语句的实体值及下标解析出来。
        :param words: 待预测解析的语句
        :param res_dict: 已经预测出的字典，类似{标签名称:[(位置11,位置12),(位置21,位置22)]}
        :return:entity_dict, entity_list，预测结果 类似字典{标签名称:(实体1,实体2,....)}  列表[[[标签名称,实体1],位置11,位置12],.....]
        """
        entity_dict = {}
        entity_list = []
        for key, values in res_dict.items():
            entity_dict[key] = []
            for value in values:
                entity_dict[key].append(words[value[0]:value[1]])
                entity_list.append([key, words[value[0]:value[1]], value[0], value[1]])
        return entity_dict, entity_list

    # 获取真实序列、标签长度。
    def make_rel_evaluation(self,rel_labels,rel_p):
        """
        计算关系分类的召回率、精确率及f1
        :param rel_labels:真实标签值 类似(maxlen,maxlen,tag_names)这种shape的数据
        :param rel_p:预测值 类似(maxlen,maxlen,tag_names)这种shape的数据
        :return:recall, precision, f1_score， 召回率、精确率及f1的值
        """
        correct_num, gold_num, predict_num = 1e-10, 1e-10, 1e-10
        for i in range(len(rel_labels)):
            new_rel_ids = np.where(rel_labels[i] == 1)
            new_rel_ids = np.array(new_rel_ids).T
            new_rel_ids_set = set(tuple(i) for i in new_rel_ids)

            pred_rel_ids = np.where(rel_p[i] == 1)
            pred_rel_ids = np.array(pred_rel_ids).T
            pred_rel_ids_set = set(tuple(i) for i in pred_rel_ids)

            correct_num += len(new_rel_ids_set & pred_rel_ids_set)
            gold_num += len(new_rel_ids_set)
            predict_num += len(pred_rel_ids_set)
        recall = correct_num / gold_num
        precision = correct_num / predict_num
        f1_score = 2 * precision * recall / (precision + recall)

        return recall, precision, f1_score

    def make_triples(self, data, ner_id2tag,rel_id2tag,id2word):
        """
        传入数据，解析出字符串形式的三元组
        :param data: 待解析的数据 [ner_ids,rel_ids,seqs,sentence_legth]
        :param ner_id2tag: 字典，类似{0:"O",1:"B-XXX",2:"I-XXX"......}
        :param rel_id2tag: 字典，类似{0:"rel1",1:"rel2",2:"rel3"......}
        :param id2word: 字典，类似{0:"word1",1:"word2",2:"word3"......}
        :return: triples_set, 字符串形式的关系抽取三元组{(s,r,o),(s,r,o),(s,r,o).....}
        """
        ner_ids = data[0]
        rel_ids = data[1]
        seqs = data[2]
        sentence_legth = data[3]
        line = tuple([id2word[i] for i in seqs[:sentence_legth]])
        ner_ids = [ner_id2tag[id_] for id_ in ner_ids]
        true_res_dict = self.transfrom2seq(ner_ids)
        all_entity_dict = {}
        for key, value_list in true_res_dict.items():
            for v in value_list:
                one_key = v[1] - 1
                if one_key not in all_entity_dict:
                    all_entity_dict[one_key] = v
        new_rel_ids = np.where(rel_ids == 1)
        new_rel_ids = np.array(new_rel_ids).T
        triples_set = set()
        for s, o, r in new_rel_ids:
            if s in all_entity_dict and o in all_entity_dict:
                triples_set.add((line[all_entity_dict[s][0]:all_entity_dict[s][1]], rel_id2tag[r], line[all_entity_dict[o][0]:all_entity_dict[o][1]] ))
        return triples_set