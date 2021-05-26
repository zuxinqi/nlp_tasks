import tensorflow as tf

def CNN_block(param,conv3):
    with tf.name_scope("pool_1"):
        pool = tf.pad(conv3, paddings=[[0, 0], [0, 1], [0, 0], [0, 0]])
        pool = tf.nn.max_pool(pool, [1, 3, 1, 1], strides=[1, 2, 1, 1], padding='VALID')

    with tf.name_scope("conv3_2"):
        conv3 = tf.layers.conv2d(pool, param.num_filters, param.kernel_size,
                                 padding="same", activation=tf.nn.relu)
        conv3 = tf.layers.batch_normalization(conv3)

    with tf.name_scope("conv3_3"):
        conv3 = tf.layers.conv2d(conv3, param.num_filters, param.kernel_size,
                                 padding="same", activation=tf.nn.relu)
        conv3 = tf.layers.batch_normalization(conv3)

    # resdul
    conv3 = conv3 + pool
    return conv3

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
            embedding_inputs = tf.expand_dims(embedded_wrods, axis=-1)  # [None,seq,embedding,1]
            # region_embedding  # [batch,seq-3+1,1,250]
            region_embedding = tf.layers.conv2d(embedding_inputs, param.num_filters,
                                                [param.kernel_size, param.embedding_dim])
            pre_activation = tf.nn.relu(region_embedding, name='preactivation')


        with tf.name_scope("conv3_0"):
            conv3 = tf.layers.conv2d(pre_activation, param.num_filters, param.kernel_size,
                                     padding="same", activation=tf.nn.relu)
            conv3 = tf.layers.batch_normalization(conv3)

        with tf.name_scope("conv3_1"):
            conv3 = tf.layers.conv2d(conv3, param.num_filters, param.kernel_size,
                                     padding="same", activation=tf.nn.relu)
            conv3 = tf.layers.batch_normalization(conv3)

        # resdul
        conv3 = conv3 + region_embedding

        while conv3.shape[-3]>2:
            conv3 = CNN_block(param, conv3)

        if conv3.shape[-3] == 2:
            conv3 = tf.reshape(conv3, (-1, 2*param.num_filters))
        else:
            conv3 = tf.reshape(conv3,(-1,param.num_filters))

        logits_ = tf.layers.dense(conv3, param.num_classes)
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