import sys
import numpy as np
sys.path.append('./Base_Line')
sys.path.append('../../Base_Line')
from base_evaluation import Base_Evaluation

class Evaluation_Utils(Base_Evaluation):
    def __init__(self):
        self.name = "evaluation_utils"

    # 将数据解析出来
    def transfrom2seq(self,tags):
        """
        将传入的序列解析，变为标签搭配下标的形式返回
        :param tags: 列表 [[0,0,0,0,1,0...],[0,1,0,0,0,0...],.....]
        :return: res_dict,字典包含标签名及位置,例如:{XXX:[(2,4),(11,14)]}
        """
        res_dict = {}
        for i in range(len(tags)):
            for j in range(len(tags[i])):
                if tags[i][j] not in ["O", "I", "[SEP]", "[CLS]", "X"]:
                    if tags[i][j][2:] not in res_dict:
                        res_dict[tags[i][j][2:]] = []
                    m = i + 1
                    while m < len(tags) and "I" in tags[m]:
                        m += 1
                    res_dict[tags[i][j][2:]].append((i, m))
        return res_dict

    def make_triples(self, data, ner_id2tag):
        """
        传入数据，解析出字符串形式的三元组
        :param data: 待解析的数据 [ner_ids,seqs]
        :param ner_id2tag: 字典，类似 {0:"O",1:"B-XXX",2:"I-XXX"......}
        :return: triples_set, 字符串形式的关系抽取三元组{(s,r,o),(s,r,o),(s,r,o).....}
        """
        ner_ids = data[0]
        line = data[1]
        ner_id_lists = np.where(ner_ids == 1)
        ner_id_lists = np.array(ner_id_lists).T
        ner_lists = [[] for i in range(len(ner_ids))]
        for id_list in ner_id_lists:
            ner_lists[id_list[0]].append(ner_id2tag[id_list[1]])
        res_dict = self.transfrom2seq(ner_lists)
        triples_set = set()
        for k, v in res_dict.items():
            if "S_" in k:
                s, r = k.split("_")
                for k1, v1 in res_dict.items():
                    if k1 == "O_" + r:
                        for i in v:
                            for j in v1:
                                triples_set.add((line[i[0]:i[1]], r, line[j[0]:j[1]]))
                                # 就多加了一个break
                                break
        return triples_set

    # 计算精确率、召回率、f1值
    def calculate_report(self, true_data, pred_data, id2tag):
        """
        传入正确的序列及预测的序列，计算召回率、精确率及f1
        :param true_data: [[0,0,0,0,1,0...],[0,1,0,0,0,0...],.....]
        :param pred_data: [[0,0,0,0,1,0...],[0,1,0,0,0,0...],.....]
        :param id2tag: {0:"O",1:"B-XXX",2:"I-XXX"......}
        :return:res_report_dict，字典包含召回率、精确率及f1，类似:{XXX:[recall,precision,f1]}
        """
        res_report_dict = {}
        for value in id2tag.values():
            if len(value) > 5:
                if value[2:] not in res_report_dict:
                    res_report_dict[value[2:]] = [0, 0, 0, 0]

        # 由于是做的sigmoid二分类，所以标签与原有的不同，需要解析成类似[B-XXX,I-XXX,O,O,O]这样的
        true_ner_id_lists = np.where(true_data == 1)
        true_ner_id_lists = np.array(true_ner_id_lists).T
        true_ner_lists = [[] for i in range(len(true_data))]
        for true_id_list in true_ner_id_lists:
            true_ner_lists[true_id_list[0]].append(id2tag[true_id_list[1]])
        true_res_dict = self.transfrom2seq(true_ner_lists)

        pred_ner_id_lists = np.where(pred_data == 1)
        pred_ner_id_lists = np.array(pred_ner_id_lists).T
        pred_ner_lists = [[] for i in range(len(pred_data))]
        for pred_id_list in pred_ner_id_lists:
            pred_ner_lists[pred_id_list[0]].append(id2tag[pred_id_list[1]])
        pred_res_dict = self.transfrom2seq(pred_ner_lists)

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