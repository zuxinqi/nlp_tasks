# Simple+Lexicon+Lstm+crf 实体抽取
## 参数设置
1. DEFAULT
    - uerdict_path 自定义词典的路径
    - stopwords_path 停用词词典的路径
    - tokenizer_name 分词器的名字
2. DATA_PROCESS
    - file_path 文件路径
    - save_path 保存初始化pickle文件路径
    - char_emb 字embedding路径
    - bichar_emb bigram字embedding路径
    - gaz_file 词embedding路径
    - train_file 训练集文件路径
    - dev_file 验证集文件路径
    - test_file 测试集文件路径
3. MODEL
    - max_seq_length 句子最大长度
    - is_training 是否开启训练
    - update_embedding 是否更新词向量
    - hidden_size 隐含层神经元个数
    - cell_nums lstm的层数
    - use_cnn 是否后接cnn
    - kernel_nums cnn卷积核数量
    - kernel_size cnn卷积核大小
    - clip 梯度裁剪的范围
    - learning_rate 学习率
    - use_decay_learning_rate 是否使用学习率衰减
    - use_l2_regularization 是否使用l2正则
    - num_train_epochs 训练轮数
    - batch_size 每批次样本数量
    - shuffle 每次训练是否随机打乱数据
    - dropout_rate dropout值
    - display_per_step 每多少步展示一次训练集效果
    - evaluation_per_step 每多少步展示一次验证集效果
    - require_improvement 有多少步没有提升，将停止训练  
    
每次训练必给的参数为:
* uerdict_path 自定义词典的路径
* stopwords_path 停用词词典的路径（这里需要是空的）
* file_path 文件路径
* save_path 保存初始化pickle文件路径
* max_seq_length 句子最大长度
## 模型架构
![alt simple_lexicon](./img/simple-lexicon.png)
## 模型说明
此模型为Simple+Lexicon+lstm+crf的实体抽取模型，在字为基础粒度的基础上增加了词信息，词信息为当前字所在的BMES位置的加权和(或平均)，相对原始lstm+crf模型有了较大提升。
## 论文地址
https://arxiv.org/pdf/1908.05969.pdf