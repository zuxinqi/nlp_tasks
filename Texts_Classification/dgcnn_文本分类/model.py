import tensorflow as tf


def conv1d_block(X, filters,kernel_size,dilation_rate=1, padding='same', scope='conv1d_block'):
    """
    gated dilation conv1d layer
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # filters = X.get_shape().as_list()[-1]
        glu = tf.layers.conv1d(X, filters, kernel_size,
                               dilation_rate=dilation_rate,
                               padding=padding,
                              )
        glu = tf.sigmoid(glu)
        glu = tf.nn.dropout(glu, 0.7)
        # glu = tf.sigmoid(tf.layers.dropout(glu, rate=0.1, training=is_train))
        conv = tf.layers.conv1d(X, filters, kernel_size,
                               dilation_rate=dilation_rate,
                               padding=padding,
                              )
        gated_conv = tf.multiply(conv, glu)
        gated_x = tf.multiply(X, 1 - glu)
        outputs = tf.add(gated_x, gated_conv)

        # mask
        outputs = tf.where(tf.equal(X, 0), X, outputs)

        # outputs = tf.layers.dropout(outputs, rate=0.25, training=self.is_train)
        # outputs = tf.layers.batch_normalization(outputs, training=True)
        return outputs

def make_graph(param):
    # 定义计算图
    graph = tf.Graph()
    with graph.as_default():
        global_step = tf.train.create_global_step(graph)
        # 定义占位符
        input_x = tf.placeholder(dtype=tf.int32, shape=[None, param.max_seq_length], name="input_data")
        input_y = tf.placeholder(dtype=tf.int32, shape=[None, ], name="label")
        dropout_keep_prob = tf.placeholder(dtype=tf.float32, name="dropout_keep_prob")

        with tf.variable_scope("embedding", dtype=tf.float32):
            embedding = tf.Variable(param.embedding_mat,
                                    dtype=tf.float32,
                                    trainable=param.update_embedding,
                                    name="_word_embeddings")
            embedded_wrods = tf.nn.embedding_lookup(embedding, input_x)
            embedded_wrods = tf.nn.dropout(embedded_wrods, dropout_keep_prob)

        # merged = tf.layers.conv1d(embedded_wrods, 300, 1, padding='same')

        # merged = conv1d_block(embedded_wrods, 200, 1, dilation_rate=1, padding='same', scope='conv1d_block_merge')

        merged = conv1d_block(embedded_wrods, 200, 3, dilation_rate=1, padding='same', scope='conv1d_block1')
        merged = conv1d_block(merged, 200, 3, dilation_rate=2, padding='same', scope='conv1d_block2')
        merged = conv1d_block(merged, 200, 3, dilation_rate=4, padding='same', scope='conv1d_block')

        # print(2333,"pool_concat", pool_concat.shape)
        pool_concat_flat = tf.reshape(merged, shape=(-1, param.max_seq_length*200))
        # p_drop = tf.nn.dropout(pool_concat_flat, dropout_keep_prob)
        # layer1 = tf.layers.dense(p_drop, param.hidden_num, activation=tf.nn.relu)

        # layer2 = tf.layers.dense(p_drop, 512, activation=tf.nn.relu)
        # h_drop = tf.nn.dropout(layer2, dropout_keep_prob)
        # layer1 = tf.layers.dense(h_drop, 256, activation=tf.nn.relu)

        logits_ = tf.layers.dense(pool_concat_flat, param.num_classes)
        # MSE
        # loss = tf.losses.mean_squared_error(predictions=pred, labels=input_y)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_, labels=input_y))
        if param.use_l2_regularization:
            tvars = tf.trainable_variables()
            reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4), tvars)
            loss = loss + reg
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_, labels=input_y))
        # print("########",sequence_output)
        # train_op = tf.train.AdamOptimizer(lr).minimize(loss)

        learning_rate = tf.constant(value=param.learning_rate, shape=[], dtype=tf.float32)
        if param.use_decay_learning_rate:
            learning_rate = tf.train.polynomial_decay(
                learning_rate,
                global_step,
                param.num_train_steps,
                end_learning_rate=0.0,
                power=1.0,
                cycle=False)

        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
        # compute gradient
        gradients = optimizer.compute_gradients(loss)
        # apply gradient clipping to prevent gradient explosion
        capped_gradients = [(tf.clip_by_norm(grad, 5), var) for grad, var in gradients if grad is not None]
        # capped_gradients = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in gradients if grad is not None]
        # update the
        train_op = optimizer.apply_gradients(capped_gradients, global_step=global_step)


        pred = tf.argmax(tf.nn.softmax(logits_), 1, name="pred")
        true_ = tf.cast(input_y, dtype=tf.int64)
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(pred, true_)
            acc = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
    return graph,input_x, input_y, dropout_keep_prob, train_op,loss,pred,true_,acc, global_step