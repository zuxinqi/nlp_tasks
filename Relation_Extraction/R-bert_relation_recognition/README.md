# R-bert 关系识别
## 参数设置
1. DEFAULT
    - uerdict_path 自定义词典的路径
    - stopwords_path 停用词词典的路径
    - tokenizer_name 分词器的名字
2. DATA_PROCESS
    - file_path 文件路径
    - save_path 保存初始化pickle文件路径
    - vocab_file 词表路径
    - max_seq_length 句子最大长度
3. MODEL
    - max_seq_length 句子最大长度
    - is_training 是否开启训练
    - bert_config_file bert配置文件路径
    - init_checkpoint bert模型参数文件路径
    - fc_output_dim 全连接层神经元个数
    - dropout_rate dropout值
    - learning_rate 学习率
    - num_train_epochs 训练轮数
    - batch_size 每批次样本数量
    - shuffle 每次训练是否随机打乱数据
    - warmup_proportion 慢热学习的速率
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
![alt r_bert](./img/r_bert.png)
## 模型说明
此模型为通过bert预训练模型来识别句子中实体的关系，通过句子embedding和实体embedding的融合来预测实体关系，此模型为pipline模式的关系抽取模型中的关系分类部分，需要搭配实体抽取模型一同使用。
## 论文地址
https://arxiv.org/pdf/1905.08284.pdf