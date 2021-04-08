# Bert+Lstm+crf 实体抽取
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
    - is_training 是否开启训练
    - bert_config_file bert配置文件路径
    - init_checkpoint bert模型参数文件路径
    - dropout_rate dropout值
    - lstm_size lstm神经元的数量
    - cell RNN神经元选择
    - cell_nums RNN层数
    - learning_rate 学习率
    - num_train_epochs 训练轮数
    - batch_size 每批次样本数量
    - shuffle 每次训练是否随机打乱数据
    - dropout_rate dropout值
    - display_per_step 每多少步展示一次训练集效果
    - evaluation_per_step 每多少步展示一次验证集效果
    - require_improvement 有多少步没有提升，将停止训练  
    
每次训练必给的参数为:
* uerdict_path 自定义词典的路径（这里需要是空的）
* stopwords_path 停用词词典的路径（这里需要是空的）
* file_path 文件路径
* save_path 保存初始化pickle文件路径
* max_seq_length 句子最大长度
## 模型架构
![alt_bert_lstm_crf](./img/Bert_lstm_crf.png)
## 模型说明
此模型为bert+lstm+crf的实体抽取模型，使用谷歌预训练Bert模型，训练速度较慢，效果预期较好