# bert_whitening 文本相似度匹配
## 参数设置
1. DEFAULT
    - uerdict_path 自定义词典的路径
    - stopwords_path 停用词词典的路径
    - tokenizer_name 分词器的名字
2. DATA_PROCESS
    - file_path 文件路径
    - save_path 保存初始化pickle文件路径
    - vocab_file bert词表的路径
3. MODEL
    - max_seq_length 句子最大长度
    - bert_config_file bert配置文件的路径
    - init_checkpoint bert保存数据文件的路径
    - model_save_path 模型保存路径
每次训练必给的参数为:
* uerdict_path 自定义词典的路径（这里需要是空的）
* stopwords_path 停用词词典的路径（这里需要是空的）
* file_path 文件路径
* save_path 保存初始化pickle文件路径
## 模型说明
使用预训练好的bert的第一层和最后一层作为文档特征，用余弦相似度计算文档与文档之间的相似度。