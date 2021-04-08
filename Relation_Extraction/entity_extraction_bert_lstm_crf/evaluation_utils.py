import sys
sys.path.append('./Base_Line')
sys.path.append('../../Base_Line')
from base_evaluation import Base_Evaluation
from tensorflow.contrib.crf import viterbi_decode

class Evaluation_Utils(Base_Evaluation):
    def __init__(self):
        self.name = "evaluation_utils"

    # 获取真实序列、标签长度。
    def make_mask(self, logits_, labels_, mask_list, is_CRF=False, transition_params_=None,action="train"):
        pred_list = []
        label_list = []
        sentence_legth = [sum(i) for i in mask_list]
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

    def extract_entity(self,words, res_dict):
        entity_dict = {}
        entity_list = []
        for key, values in res_dict.items():
            entity_dict[key] = []
            for value in values:
                entity_dict[key].append("".join(words[value[0]:value[1]]))
                entity_list.append([key, "".join(words[value[0]:value[1]]), value[0], value[1]])
        return entity_dict, entity_list