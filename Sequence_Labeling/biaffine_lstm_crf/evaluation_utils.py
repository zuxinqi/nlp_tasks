import sys
import numpy as np
sys.path.append('./Base_Line')
sys.path.append('../../Base_Line')
from base_evaluation import Base_Evaluation

class Evaluation_Utils(Base_Evaluation):
    def __init__(self):
        self.name = "evaluation_utils"

    def make_tags(self, index_res, labels, res_labels, id2tag):
        for i in np.array(index_res).T:
            tag = id2tag[labels[i[0]][i[1]][i[2]]]
            if (i[2] - i[1]) == 0:
                res_labels[i[0]][i[1]:i[2] + 1] = ["B_" + tag]
            else:
                res_labels[i[0]][i[1]:i[1] + 1] = ["B_" + tag]
                res_labels[i[0]][i[1] + 1:i[2] + 1] = (i[2] - i[1]) * ["I_" + tag]
        return res_labels

    def make_mask(self, labels_, preds_, sentence_legth, id2tag, max_seq_length, action="train"):
        labels_list = []
        preds_list = []
        res_labels_list = [["O"] * max_seq_length for i in range(len(labels_))]
        res_preds_list = [["O"] * max_seq_length for i in range(len(labels_))]

        index_labels_list = np.where(labels_ != 0)
        index_preds_list = np.where(preds_ != 0)
        res_labels_list = self.make_tags(index_labels_list, labels_, res_labels_list, id2tag)
        res_preds_list = self.make_tags(index_preds_list, preds_, res_preds_list, id2tag)

        for lab, pred, seq_len in zip(res_labels_list, res_preds_list, sentence_legth):
            if action == "train":
                labels_list.extend(lab[:seq_len])
                preds_list.extend(pred[:seq_len])
            elif action == "pred":
                labels_list.append(lab[:seq_len])
                preds_list.append(pred[:seq_len])
            else:
                # 默认是train
                labels_list.extend(lab[:seq_len])
                preds_list.extend(pred[:seq_len])

        return labels_list, preds_list