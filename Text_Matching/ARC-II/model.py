import tensorflow as tf

def make_conv1d(x,filters,kernel_size):
    with tf.variable_scope("conv1d",reuse=tf.AUTO_REUSE):  # 命名空间
        conv1 = tf.layers.conv1d(x, filters, kernel_size, activation=tf.nn.relu,
                                 padding='same',
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 bias_initializer=tf.truncated_normal_initializer(stddev=0.02))
        # conv1 = tf.nn.dropout(conv1, dropout_pl)
        # pool_out = tf.layers.max_pooling1d(conv1, max_seq_length, 1)
        # pool_out = tf.squeeze(pool_out,axis=1)
        # output = tf.layers.dense(inputs=pool_out, units=hidden_units, activation=tf.nn.relu)
        # output = tf.layers.dense(inputs=output, units=128, activation=tf.nn.relu)
    return conv1

def MatchingLayer(x1,x2,matching_type="dot",_normalize=True):
    """
    The computation logic of MatchingLayer.
    :param inputs: two input tensors.
    """
    if matching_type == 'dot':
        if _normalize:
            x1 = tf.math.l2_normalize(x1, axis=2)
            x2 = tf.math.l2_normalize(x2, axis=2)
    return tf.expand_dims(tf.einsum('abd,acd->abc', x1, x2), 3)

def make_graph(param):
    # 定义计算图
    graph = tf.Graph()
    with graph.as_default():
        # shape[batch_size, sentences]
        left_input = tf.placeholder(tf.int32, shape=[None, param.max_seq_length], name="left_input")
        # shape[batch_size, sentences]
        right_input = tf.placeholder(tf.int32, shape=[None, param.max_seq_length], name="right_input")

        # shape[batch_size, sentences, labels]
        labels = tf.placeholder(tf.int32, shape=[None, ], name="labels")

        # dropout keep_prob
        dropout_pl = tf.placeholder(dtype=tf.float32, shape=(), name="dropout")

        with tf.variable_scope("embeddings"):  # 命名空间
            _word_embeddings = tf.Variable(param.embedding_mat,  # shape[len_words,300]
                                           dtype=tf.float32,
                                           trainable=param.update_embedding,  # 嵌入层是否可以训练
                                           name="embedding_matrix")
            left_embeddings = tf.nn.embedding_lookup(params=_word_embeddings, ids=left_input, name="left_embeddings")
            right_embeddings = tf.nn.embedding_lookup(params=_word_embeddings, ids=right_input, name="right_embeddings")

            left_embeddings = tf.nn.dropout(left_embeddings, dropout_pl)
            right_embeddings = tf.nn.dropout(right_embeddings, dropout_pl)

        conv_1d_left = make_conv1d(left_embeddings, 100, 3)
        conv_1d_right = make_conv1d(right_embeddings, 100, 3)
        embed_cross = MatchingLayer(conv_1d_left, conv_1d_right)

        with tf.variable_scope("conv-2d"):
            embed_cross = tf.layers.conv2d(embed_cross,filters=64,kernel_size=3,padding="same",activation=tf.nn.relu)
            embed_cross = tf.layers.max_pooling2d(embed_cross,pool_size=2,strides=1,padding="same")
            embed_cross = tf.layers.conv2d(embed_cross, filters=32, kernel_size=3, padding="same",
                                           activation=tf.nn.relu)
            embed_cross = tf.layers.max_pooling2d(embed_cross, pool_size=2,strides=1,padding="same")
            # embed_cross = tf.layers.conv2d(embed_cross, filters=32, kernel_size=3, padding="same",
            #                                activation=tf.nn.relu)
            # embed_cross = tf.layers.max_pooling2d(embed_cross, pool_size=2,strides=1,padding="same")

            out = tf.reshape(embed_cross,(-1,32*param.max_seq_length*param.max_seq_length))
            out = tf.nn.dropout(out, dropout_pl)




        with tf.variable_scope("Similarity_calculation_layer"):
            out = tf.layers.dense(out, param.hidden_units,activation=tf.nn.relu)
            logits = tf.layers.dense(out, 2)
        # 计算损失
        with tf.variable_scope("loss"):
            # logits__ = tf.nn.softmax(logits)
            # loss = (-0.25 * tf.reduce_sum(labels[:, 0] * tf.log(logits__[:, 0]))
            #         - 0.75 * tf.reduce_sum(labels[:, 1] * tf.log(logits__[:, 1]))
            #         ) / tf.cast(tf.shape(labels)[0], tf.float32)



            labels_ = tf.one_hot(labels, 2)
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels_)
            loss = tf.reduce_mean(losses)

        # 选择优化器
        with tf.variable_scope("train_step"):
            # 优化函数
            global_step = tf.train.get_or_create_global_step()

            # learning_rate = tf.constant(value=learning_rate, shape=[], dtype=tf.float32)
            # learning_rate = tf.train.polynomial_decay(
            #     learning_rate,
            #     global_step,
            #     num_train_steps,
            #     end_learning_rate=0.0,
            #     power=1.0,
            #     cycle=False)

            optimizer = tf.train.AdamOptimizer(param.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
            # compute gradient
            gradients = optimizer.compute_gradients(loss)
            # apply gradient clipping to prevent gradient explosion
            capped_gradients = [(tf.clip_by_norm(grad, 5), var) for grad, var in gradients if grad is not None]

            train_op = optimizer.apply_gradients(capped_gradients, global_step=global_step)



        # 准确率/f1/p/r计算
        with tf.variable_scope("evaluation"):
            # true = tf.cast(labels_, tf.int32)  # 真实序列的值
            # labels_softmax = tf.nn.softmax(logits)
            # labels_softmax_ = tf.argmax(labels_softmax, axis=-1)
            true = tf.cast(tf.argmax(labels_, axis=-1), tf.float32)  # 真实序列的值
            labels_softmax = tf.nn.softmax(logits)
            labels_softmax_ = tf.argmax(labels_softmax, axis=-1)
            pred = tf.cast(labels_softmax_, tf.float32)  # 预测序列的值
            accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, true), tf.float32))

            # pred_ = tf.expand_dims(pred, 1)

            # epsilon = 1e-7
            # cm = tf.contrib.metrics.confusion_matrix(true, pred, num_classes=2)
            # precision = cm[1][1] / tf.reduce_sum(tf.transpose(cm)[1])
            # recall = cm[1][1] / tf.reduce_sum(cm[1])
            # fbeta_score = (2 * precision * recall / (precision + recall + epsilon))
    return graph,left_input,right_input,labels,dropout_pl,loss,train_op,true,pred,accuracy,global_step