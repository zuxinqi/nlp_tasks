import sys
sys.path.append('./Base_Line')
sys.path.append('../../Base_Line')
from base_evaluation import Base_Evaluation
from tensorflow.contrib.crf import viterbi_decode

class Evaluation_Utils(Base_Evaluation):
    def __init__(self):
        self.name = "evaluation_utils"

    def make_seq(self, s, e, tag_name):
        seq = ["O"] * len(s)
        cusor = 0
        while cusor < len(s):
            if s[cusor] == 1:
                cusor2 = cusor
                while cusor2 < len(e):
                    if e[cusor2] == 1 and cusor2 == cusor:
                        seq[cusor] = "B_" + tag_name
                        break
                    if e[cusor2] == 1:
                        seq[cusor] = "B_" + tag_name
                        seq[cusor + 1:cusor2 + 1] = ["I_" + tag_name] * (cusor2 - cusor)
                        break
                    cusor2 += 1
            cusor += 1
        return seq

    # 获取真实序列、标签长度。
    def make_mask(self, seqs_, text_lens_, query_lens_, s_labels_, e_labels_, predict_start_ids_, predict_end_ids_):
        pred_list_s = []
        label_list_s = []
        pred_list_e = []
        label_list_e = []
        for i in range(len(s_labels_)):
            label_list_s.extend(s_labels_[i][query_lens_[i]:text_lens_[i]])
            pred_list_s.extend(predict_start_ids_[i][query_lens_[i]:text_lens_[i]])
            label_list_e.extend(e_labels_[i][query_lens_[i]:text_lens_[i]])
            pred_list_e.extend(predict_end_ids_[i][query_lens_[i]:text_lens_[i]])

        pred_list = pred_list_s + pred_list_e
        label_list = label_list_s + label_list_e
        all_l_seq = []
        all_p_seq = []
        seqs_ = seqs_.tolist()
        for i in range(len(s_labels_)):
            if [101, 2823, 1139, 5381, 4991] == seqs_[i][:5]:
                s_l = s_labels_[i][query_lens_[i]:text_lens_[i]]
                e_l = e_labels_[i][query_lens_[i]:text_lens_[i]]
                s_p = predict_start_ids_[i][query_lens_[i]:text_lens_[i]]
                e_p = predict_end_ids_[i][query_lens_[i]:text_lens_[i]]
                l_seq = self.make_seq(s_l, e_l, "product_name")
                p_seq = self.make_seq(s_p, e_p, "product_name")
                all_l_seq.extend(l_seq)
                all_p_seq.extend(p_seq)
            elif [101, 2823, 1139, 2399, 819] == seqs_[i][:5]:
                s_l = s_labels_[i][query_lens_[i]:text_lens_[i]]
                e_l = e_labels_[i][query_lens_[i]:text_lens_[i]]
                s_p = predict_start_ids_[i][query_lens_[i]:text_lens_[i]]
                e_p = predict_end_ids_[i][query_lens_[i]:text_lens_[i]]
                l_seq = self.make_seq(s_l, e_l, "time")
                p_seq = self.make_seq(s_p, e_p, "time")
                all_l_seq.extend(l_seq)
                all_p_seq.extend(p_seq)
            elif [101, 2823, 1139, 4696, 2141] == seqs_[i][:5]:
                s_l = s_labels_[i][query_lens_[i]:text_lens_[i]]
                e_l = e_labels_[i][query_lens_[i]:text_lens_[i]]
                s_p = predict_start_ids_[i][query_lens_[i]:text_lens_[i]]
                e_p = predict_end_ids_[i][query_lens_[i]:text_lens_[i]]
                l_seq = self.make_seq(s_l, e_l, "person_name")
                p_seq = self.make_seq(s_p, e_p, "person_name")
                all_l_seq.extend(l_seq)
                all_p_seq.extend(p_seq)
            elif [101, 2823, 1139, 2110, 3413] == seqs_[i][:5]:
                s_l = s_labels_[i][query_lens_[i]:text_lens_[i]]
                e_l = e_labels_[i][query_lens_[i]:text_lens_[i]]
                s_p = predict_start_ids_[i][query_lens_[i]:text_lens_[i]]
                e_p = predict_end_ids_[i][query_lens_[i]:text_lens_[i]]
                l_seq = self.make_seq(s_l, e_l, "org_name")
                p_seq = self.make_seq(s_p, e_p, "org_name")
                all_l_seq.extend(l_seq)
                all_p_seq.extend(p_seq)
            elif [101, 2823, 1139, 1744, 2157] == seqs_[i][:5]:
                s_l = s_labels_[i][query_lens_[i]:text_lens_[i]]
                e_l = e_labels_[i][query_lens_[i]:text_lens_[i]]
                s_p = predict_start_ids_[i][query_lens_[i]:text_lens_[i]]
                e_p = predict_end_ids_[i][query_lens_[i]:text_lens_[i]]
                l_seq = self.make_seq(s_l, e_l, "location")
                p_seq = self.make_seq(s_p, e_p, "location")
                all_l_seq.extend(l_seq)
                all_p_seq.extend(p_seq)
            elif [101, 2823, 1139, 7213, 6121] == seqs_[i][:5]:
                s_l = s_labels_[i][query_lens_[i]:text_lens_[i]]
                e_l = e_labels_[i][query_lens_[i]:text_lens_[i]]
                s_p = predict_start_ids_[i][query_lens_[i]:text_lens_[i]]
                e_p = predict_end_ids_[i][query_lens_[i]:text_lens_[i]]
                l_seq = self.make_seq(s_l, e_l, "company_name")
                p_seq = self.make_seq(s_p, e_p, "company_name")
                all_l_seq.extend(l_seq)
                all_p_seq.extend(p_seq)
        return label_list, pred_list, all_l_seq, all_p_seq


    # 获取真实序列、标签长度。
    def make_mask_pred(self, seqs_, text_lens_, query_lens_, s_labels_, e_labels_, predict_start_ids_, predict_end_ids_):
        pred_list_s = []
        label_list_s = []
        pred_list_e = []
        label_list_e = []
        for i in range(len(s_labels_)):
            label_list_s.extend(s_labels_[i][query_lens_[i]:text_lens_[i]])
            pred_list_s.extend(predict_start_ids_[i][query_lens_[i]:text_lens_[i]])
            label_list_e.extend(e_labels_[i][query_lens_[i]:text_lens_[i]])
            pred_list_e.extend(predict_end_ids_[i][query_lens_[i]:text_lens_[i]])

        pred_list = pred_list_s + pred_list_e
        label_list = label_list_s + label_list_e
        all_l_seq = []
        all_p_seq = []
        seqs_ = seqs_.tolist()
        for i in range(len(s_labels_)):
            if [101, 2823, 1139, 5381, 4991] == seqs_[i][:5]:
                s_l = s_labels_[i][query_lens_[i]:text_lens_[i]]
                e_l = e_labels_[i][query_lens_[i]:text_lens_[i]]
                s_p = predict_start_ids_[i][query_lens_[i]:text_lens_[i]]
                e_p = predict_end_ids_[i][query_lens_[i]:text_lens_[i]]
                l_seq = self.make_seq(s_l, e_l, "product_name")
                p_seq = self.make_seq(s_p, e_p, "product_name")
                all_l_seq.append(l_seq)
                all_p_seq.append(p_seq)
            elif [101, 2823, 1139, 2399, 819] == seqs_[i][:5]:
                s_l = s_labels_[i][query_lens_[i]:text_lens_[i]]
                e_l = e_labels_[i][query_lens_[i]:text_lens_[i]]
                s_p = predict_start_ids_[i][query_lens_[i]:text_lens_[i]]
                e_p = predict_end_ids_[i][query_lens_[i]:text_lens_[i]]
                l_seq = self.make_seq(s_l, e_l, "time")
                p_seq = self.make_seq(s_p, e_p, "time")
                all_l_seq.append(l_seq)
                all_p_seq.append(p_seq)
            elif [101, 2823, 1139, 4696, 2141] == seqs_[i][:5]:
                s_l = s_labels_[i][query_lens_[i]:text_lens_[i]]
                e_l = e_labels_[i][query_lens_[i]:text_lens_[i]]
                s_p = predict_start_ids_[i][query_lens_[i]:text_lens_[i]]
                e_p = predict_end_ids_[i][query_lens_[i]:text_lens_[i]]
                l_seq = self.make_seq(s_l, e_l, "person_name")
                p_seq = self.make_seq(s_p, e_p, "person_name")
                all_l_seq.append(l_seq)
                all_p_seq.append(p_seq)
            elif [101, 2823, 1139, 2110, 3413] == seqs_[i][:5]:
                s_l = s_labels_[i][query_lens_[i]:text_lens_[i]]
                e_l = e_labels_[i][query_lens_[i]:text_lens_[i]]
                s_p = predict_start_ids_[i][query_lens_[i]:text_lens_[i]]
                e_p = predict_end_ids_[i][query_lens_[i]:text_lens_[i]]
                l_seq = self.make_seq(s_l, e_l, "org_name")
                p_seq = self.make_seq(s_p, e_p, "org_name")
                all_l_seq.append(l_seq)
                all_p_seq.append(p_seq)
            elif [101, 2823, 1139, 1744, 2157] == seqs_[i][:5]:
                s_l = s_labels_[i][query_lens_[i]:text_lens_[i]]
                e_l = e_labels_[i][query_lens_[i]:text_lens_[i]]
                s_p = predict_start_ids_[i][query_lens_[i]:text_lens_[i]]
                e_p = predict_end_ids_[i][query_lens_[i]:text_lens_[i]]
                l_seq = self.make_seq(s_l, e_l, "location")
                p_seq = self.make_seq(s_p, e_p, "location")
                all_l_seq.append(l_seq)
                all_p_seq.append(p_seq)
            elif [101, 2823, 1139, 7213, 6121] == seqs_[i][:5]:
                s_l = s_labels_[i][query_lens_[i]:text_lens_[i]]
                e_l = e_labels_[i][query_lens_[i]:text_lens_[i]]
                s_p = predict_start_ids_[i][query_lens_[i]:text_lens_[i]]
                e_p = predict_end_ids_[i][query_lens_[i]:text_lens_[i]]
                l_seq = self.make_seq(s_l, e_l, "company_name")
                p_seq = self.make_seq(s_p, e_p, "company_name")
                all_l_seq.append(l_seq)
                all_p_seq.append(p_seq)
        return label_list, pred_list, all_l_seq, all_p_seq


    # 将数据解析出来
    def transfrom2seq(self, tags):
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
    def calculate_report(self, true_data, pred_data, id2tag):
        res_report_dict = {}
        for value in id2tag.values():
            if len(value) > 2:
                if value[2:] not in res_report_dict:
                    res_report_dict[value[2:]] = [0, 0, 0, 0]
        # true_data = [id2tag[data] for data in true_data]
        # pred_data = [id2tag[data] for data in pred_data]
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
        entity_dict = {}
        entity_list = []
        for key, values in res_dict.items():
            entity_dict[key] = []
            for value in values:
                entity_dict[key].append("".join(words[value[0]:value[1]]))
                entity_list.append([key, "".join(words[value[0]:value[1]]), value[0], value[1]])
        return entity_dict, entity_list