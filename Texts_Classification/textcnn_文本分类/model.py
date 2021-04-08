import tensorflow as tf

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


        all_layer_outputs = []
        for i, filter_size_j in enumerate(param.filter_size):
            with tf.name_scope("conv-maxpool-word-%s" % filter_size_j):
                conv1 = tf.layers.conv1d(embedded_wrods, param.filters, filter_size_j, activation=tf.nn.relu,
                                         dilation_rate=1,
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                         bias_initializer=tf.truncated_normal_initializer(stddev=0.02))
                # bn = tf.layers.batch_normalization(conv1,training=is_training)
                # bn_output = tf.nn.relu(bn)
                # print("conv:",conv1.shape)
                pool_out = tf.layers.max_pooling1d(conv1, param.max_seq_length - filter_size_j + 1, 1)
                # pool_out = k_max_pooling(conv1,k)
                # print("k_max_pooling",pool_out.shape)
                all_layer_outputs.append(pool_out)

        pool_concat = tf.concat(all_layer_outputs, -1)
        # print(2333,"pool_concat", pool_concat.shape)
        pool_concat_flat = tf.reshape(pool_concat, shape=(-1, 1 * param.filters * len(param.filter_size)))
        p_drop = tf.nn.dropout(pool_concat_flat, dropout_keep_prob)
        layer1 = tf.layers.dense(p_drop, param.hidden_num, activation=tf.nn.relu)

        # layer2 = tf.layers.dense(p_drop, 512, activation=tf.nn.relu)
        # h_drop = tf.nn.dropout(layer2, dropout_keep_prob)
        # layer1 = tf.layers.dense(h_drop, 256, activation=tf.nn.relu)

        logits_ = tf.layers.dense(layer1, param.num_classes)
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