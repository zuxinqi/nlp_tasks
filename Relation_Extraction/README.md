# 关系抽取算法
### 说明
关系抽取算法，包含联合抽取、管道抽取、标签式抽取这三种模型架构。    
联合抽取：CasRel、multihead_joint_extraction  
管道抽取：R-bert_relation_recognition(关系分类)、attention_lstm_relation_recognition(关系分类)、entity_extraction_bert_lstm_crf(实体抽取)  
标签式抽取：tagging_scheme_joint_extraction
### 文件结构
```
Relation_Extraction
├── CasRel
├── multihead_joint_extraction
├── R-bert_relation_recognition
├── attention_lstm_relation_recognition
├── attention_lstm_relation_recognition_for_single_sentence
├── tagging_scheme_joint_extraction
├── entity_extraction_bert_lstm_crf
└── 数据准备
```
### 结果对比

| 模型 | f1 | 
| ---- | ---- | 
| CasRel | 0.7456 | 
| multihead_joint_extraction | 0.731 | 
| R-bert_relation_recognition | 0.7426| 
| attention_lstm_relation_recognition | 0.7405|
| tagging_scheme_joint_extraction | 0.738|

attention_lstm两个模型，结构类似，因此只拿一个做结果对比  
此f1值为三元组的整体f1，评价指标设定的较为严格，目前结果只是针对示例数据集进行了简单调参的结果，实际应用还需自行调整超参。