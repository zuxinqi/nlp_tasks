import modeling
import optimization
import tensorflow as tf

def fclayer(x,input_dim, output_dim,dropout_rate):
    with tf.variable_scope("tahn_dense", reuse=tf.AUTO_REUSE) as scope:
        x = tf.nn.dropout(x, keep_prob=dropout_rate)
        w1 = tf.get_variable('w', shape=[input_dim,output_dim], initializer=tf.random_normal_initializer())
        b = tf.get_variable('b', shape=[output_dim], initializer=tf.random_normal_initializer())
        res = tf.nn.tanh(tf.matmul(x,w1)+b)
    return res

def fclayer_pool(x,input_dim, output_dim,dropout_rate):
    with tf.variable_scope("tahn_dense_pool", reuse=tf.AUTO_REUSE) as scope:
        x = tf.nn.dropout(x, keep_prob=dropout_rate)
        w1 = tf.get_variable('w1', shape=[input_dim,output_dim], initializer=tf.random_normal_initializer())
        b = tf.get_variable('b1', shape=[output_dim], initializer=tf.random_normal_initializer())
        res = tf.nn.tanh(tf.matmul(x,w1)+b)
    return res


alpha=[[1], [1.5], [1.2], [1],[1], [2], [2], [1.2],[1]]

def focal_loss(logits, labels, alpha, epsilon = 1.e-7,gamma=2.0, multi_dim = False):
        '''
        :param logits:  [batch_size, n_class]
        :param labels: [batch_size]  not one-hot !!!
        :return: -alpha*(1-y)^r * log(y)
        它是在哪实现 1- y 的？ 通过gather选择的就是1-p,而不是通过计算实现的；
        logits soft max之后是多个类别的概率，也就是二分类时候的1-P和P；多分类的时候不是1-p了；

        怎么把alpha的权重加上去？
        通过gather把alpha选择后变成batch长度，同时达到了选择和维度变换的目的

        是否需要对logits转换后的概率值进行限制？
        需要的，避免极端情况的影响

        针对输入是 (N，P，C )和  (N，P)怎么处理？
        先把他转换为和常规的一样形状，（N*P，C） 和 （N*P,）

        bug:
        ValueError: Cannot convert an unknown Dimension to a Tensor: ?
        因为输入的尺寸有时是未知的，导致了该bug,如果batchsize是确定的，可以直接修改为batchsize

        '''


        if multi_dim:
            logits = tf.reshape(logits, [-1, logits.shape[2]])
            labels = tf.reshape(labels, [-1])

        # (Class ,1)
        alpha = tf.constant(alpha, dtype=tf.float32)

        labels = tf.cast(labels, dtype=tf.int32)
        logits = tf.cast(logits, tf.float32)
        # (N,Class) > N*Class
        softmax = tf.reshape(tf.nn.softmax(logits), [-1])  # [batch_size * n_class]
        # (N,) > (N,) ,但是数值变换了，变成了每个label在N*Class中的位置
        labels_shift = tf.range(0, tf.shape(logits)[0]) * logits.shape[1] + labels
        #labels_shift = tf.range(0, batch_size*32) * logits.shape[1] + labels
        # (N*Class,) > (N,)
        prob = tf.gather(softmax, labels_shift)
        # 预防预测概率值为0的情况  ; (N,)
        prob = tf.clip_by_value(prob, epsilon, 1. - epsilon)
        # (Class ,1) > (N,)
        alpha_choice = tf.gather(alpha, labels)
        # (N,) > (N,)
        weight = tf.pow(tf.subtract(1., prob), gamma)
        weight = tf.multiply(alpha_choice, weight)
        # (N,) > 1
        loss = -tf.reduce_mean(tf.multiply(weight, tf.log(prob)))
        return loss

def make_graph(param):
    graph = tf.Graph()
    with graph.as_default():
        input_ids = tf.placeholder(tf.int32, shape=[None, param.max_seq_length], name="input_ids")
        label_ids = tf.placeholder(tf.int32, shape=[None,], name="label_ids")
        input_mask = tf.placeholder(tf.float32, shape=[None, param.max_seq_length], name="input_mask")
        e1_mask = tf.placeholder(tf.float32, shape=[None, param.max_seq_length], name="input_mask")
        e2_mask = tf.placeholder(tf.float32, shape=[None, param.max_seq_length], name="input_mask")
        segment_ids = tf.placeholder(tf.int32, shape=[None, param.max_seq_length], name="segment_ids")

        bert_model = modeling.BertModel(
            config=param.bert_config,
            is_training=param.is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False
        )
        bert_seq_output = bert_model.get_sequence_output()
        bert_pooled_output = bert_model.get_pooled_output()

        e1_mask_ = tf.expand_dims(e1_mask, 1)
        e1_mask_sum_vector = tf.reduce_sum(e1_mask_, 2)
        e1_h = tf.matmul(e1_mask_, bert_seq_output)
        e1_h =  tf.squeeze(e1_h, 1)/e1_mask_sum_vector

        e2_mask_ = tf.expand_dims(e2_mask, 1)
        e2_mask_sum_vector = tf.reduce_sum(e2_mask_, 2)
        e2_h = tf.matmul(e2_mask_, bert_seq_output)
        e2_h = tf.squeeze(e2_h, 1) / e2_mask_sum_vector

        e1_h = fclayer(e1_h, 768, param.fc_output_dim, param.dropout_rate)
        e2_h = fclayer(e2_h, 768, param.fc_output_dim, param.dropout_rate)

        # bert_pooled_output = tf.nn.dropout(bert_pooled_output, keep_prob=dropout_rate)
        bert_pooled_output = fclayer_pool(bert_pooled_output, 768, param.fc_output_dim, param.dropout_rate)

        concat_h = tf.concat([bert_pooled_output,e1_h,e2_h],axis=-1)
        logits = tf.layers.dense(concat_h, param.num_labels)
        # logits = tf.layers.dense(bert_pooled_output, num_labels)
        pred = tf.argmax(tf.nn.softmax(logits,1),1)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label_ids))
        # loss = focal_loss(logits, label_ids, alpha)
        tvars = tf.trainable_variables()
        if param.init_checkpoint:
            (assignment_map, initialized_variable_names) = \
                modeling.get_assignment_map_from_checkpoint(tvars,
                                                            param.init_checkpoint)
            tf.train.init_from_checkpoint(param.init_checkpoint, assignment_map)
        global_step1 = tf.Variable(0, name="global_step1", trainable=False)
        global_add = global_step1.assign_add(1)
        train_op = optimization.create_optimizer(
            loss, param.learning_rate, param.num_train_steps, param.num_warmup_steps, False)

        # global_step = tf.train.create_global_step(graph)
        # optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # grads_and_vars = optim.compute_gradients(loss)
        # # 对梯度gradients进行裁剪，保证在[-params.clip, params.clip]之间。
        # grads_and_vars_clip = [[tf.clip_by_value(g, -clip, clip), v] for g, v in grads_and_vars]
        # train_op = optim.apply_gradients(grads_and_vars_clip, global_step=global_step)

        return graph,input_ids,label_ids,input_mask,e1_mask,e2_mask,segment_ids,pred,loss,train_op,global_add


