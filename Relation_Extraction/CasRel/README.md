# CasRel 关系抽取
## 参数设置
1. DEFAULT
    - uerdict_path 自定义词典的路径
    - stopwords_path 停用词词典的路径
    - tokenizer_name 分词器的名字
2. DATA_PROCESS
    - file_path 文件路径
    - save_path 保存初始化pickle文件路径
    - vocab_file 词表路径
3. MODEL
    - max_seq_length 句子最大长度
    - bert_config_file bert配置文件路径
    - init_checkpoint bert模型参数文件路径
    - learning_rate 学习率
    - num_train_epochs 训练轮数
    - batch_size 每批次样本数量
    
每次训练必给的参数为:
* uerdict_path 自定义词典的路径（这里需要是空的）
* stopwords_path 停用词词典的路径（这里需要是空的）
* file_path 文件路径
* save_path 保存初始化pickle文件路径
* max_seq_length 句子最大长度
## 模型架构
![alt Casrel](./img/Casrel.png)
## 模型说明
此模型为指针式联合抽取模型,使用bert预训练模型,2020年的sorted模型.
## 论文地址
https://arxiv.org/pdf/1909.03227.pdf