# 序列标注算法
### 说明
包含四种基础算法，其中bert_lstm_crf和bert_mrc效果较好，但是由于bert_mrc的时间复杂度较高，实用性较差。
### 文件结构
```
Sequence_Labeling
├── crf_suite
├── lstm_crf
├── biaffine
├── simple_lexicon
├── bert_lstm_crf
├── bert_mrc
└── 数据准备
```
### 结果对比

| 模型 | loss | Macro F1 | 
| ---- | ---- | ---- | 
| crf_suite | None | 0.73|  
| lstm_crf | 5.879 | 0.6983|  
| bert_lstm_crf | 7.123| 0.825 | 
| bert_mrc | 0.2608| 0.8283 |   
  
bert_mrc和其它的实体抽取模型的模型大体结构不一致，loss做的sigmiod交叉熵，因此loss与其他模型有较大差距  
目前结果只是针对示例数据集进行了简单调参的结果，实际应用还需自行调整超参。
