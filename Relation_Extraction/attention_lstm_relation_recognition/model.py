import tensorflow as tf

def make_graph(param):
    # 定义计算图
    graph = tf.Graph()
    with graph.as_default():
        global_step = tf.train.create_global_step(graph)
        # 定义占位符
        input_word = tf.placeholder(dtype=tf.int32, shape=[None, param.max_seq_length], name='input_word')
        input_pos1 = tf.placeholder(dtype=tf.int32, shape=[None, param.max_seq_length], name='input_pos1')
        input_pos2 = tf.placeholder(dtype=tf.int32, shape=[None, param.max_seq_length], name='input_pos2')
        input_y = tf.placeholder(dtype=tf.float32, shape=[None, param.num_classes], name='input_y')
        total_shape = tf.placeholder(dtype=tf.int32, shape=[param.batch_size + 1], name='total_shape')
        total_num = total_shape[-1]

        word_embedding = tf.Variable(param.embedding_mat,
                                           dtype=tf.float32,
                                           trainable=param.update_embedding,
                                           name="word_embeddings")
        # pos1_embedding = tf.Variable([param.pos_num, param.pos_size],
        #                                    dtype=tf.float32,
        #                                    name="pos1_embedding")
        # pos2_embedding = tf.Variable([param.pos_num, param.pos_size],
        #                              dtype=tf.float32,
        #                              name="pos2_embedding")
        pos1_embedding = tf.get_variable('pos1_embedding', [param.pos_num, param.pos_size])
        pos2_embedding = tf.get_variable('pos2_embedding', [param.pos_num, param.pos_size])
        attention_w = tf.get_variable('attention_omega', [param.gru_size, 1])
        sen_a = tf.get_variable('attention_A', [param.gru_size])
        sen_r = tf.get_variable('query_r', [param.gru_size, 1])
        relation_embedding = tf.get_variable('relation_embedding', [param.num_classes, param.gru_size])
        sen_d = tf.get_variable('bias_d', [param.num_classes])

        gru_cell_forward = tf.contrib.rnn.GRUCell(param.gru_size)
        gru_cell_backward = tf.contrib.rnn.GRUCell(param.gru_size)
        if param.is_training and param.dropout_rate < 1:
            gru_cell_forward = tf.contrib.rnn.DropoutWrapper(gru_cell_forward, output_keep_prob=param.dropout_rate)
            gru_cell_backward = tf.contrib.rnn.DropoutWrapper(gru_cell_backward, output_keep_prob=param.dropout_rate)

        cell_forward = tf.contrib.rnn.MultiRNNCell([gru_cell_forward] * param.num_layers)
        cell_backward = tf.contrib.rnn.MultiRNNCell([gru_cell_backward] * param.num_layers)

        sen_repre = []
        sen_alpha = []
        sen_s = []
        sen_out = []
        prob = []
        predictions = []
        loss = []
        accuracy = []
        total_loss = 0.0
        # embedding layer
        inputs_forward = tf.concat(axis=2, values=[tf.nn.embedding_lookup(word_embedding, input_word),
                                                   tf.nn.embedding_lookup(pos1_embedding, input_pos1),
                                                   tf.nn.embedding_lookup(pos2_embedding, input_pos2)])
        # 自己写的
        (output_forward, output_backward), states = tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward,
                                                                                    inputs_forward,
                                                                                    dtype=tf.float32)
        # word-level attention layer
        output_h = tf.add(output_forward, output_backward)
        attention_r = tf.reshape(tf.matmul(tf.reshape(tf.nn.softmax(
            tf.reshape(tf.matmul(tf.reshape(tf.tanh(output_h), [total_num * param.max_seq_length, param.gru_size]), attention_w),
                       [total_num, param.max_seq_length])), [total_num, 1, param.max_seq_length]), output_h), [total_num, param.gru_size])

        # sentence-level attention layer
        for i in range(param.batch_size):

            sen_repre.append(tf.tanh(attention_r[total_shape[i]:total_shape[i + 1]]))
            batch_size = total_shape[i + 1] - total_shape[i]

            sen_alpha.append(
                tf.reshape(tf.nn.softmax(tf.reshape(tf.matmul(tf.multiply(sen_repre[i], sen_a), sen_r), [batch_size])),
                           [1, batch_size]))

            sen_s.append(tf.reshape(tf.matmul(sen_alpha[i], sen_repre[i]), [param.gru_size, 1]))
            sen_out.append(tf.add(tf.reshape(tf.matmul(relation_embedding, sen_s[i]), [param.num_classes]), sen_d))

            prob.append(tf.nn.softmax(sen_out[i]))

            with tf.name_scope("output"):
                predictions.append(tf.argmax(prob[i], 0, name="predictions"))

            with tf.name_scope("loss"):
                loss.append(
                    tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=sen_out[i], labels=input_y[i])))
                if i == 0:
                    total_loss = loss[i]
                else:
                    total_loss += loss[i]

            # tf.summary.scalar('loss',self.total_loss)
            # tf.scalar_summary(['loss'],[self.total_loss])
            with tf.name_scope("accuracy"):
                accuracy.append(
                    tf.reduce_mean(tf.cast(tf.equal(predictions[i], tf.argmax(input_y[i], 0)), "float"),
                                   name="accuracy"))


        true_ = tf.argmax(input_y, -1)
        correct_predictions = tf.equal(predictions, true_)
        acc = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")


        # regularization
        l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                                                              weights_list=tf.trainable_variables())
        final_loss = total_loss + l2_loss

        optimizer = tf.train.AdamOptimizer(learning_rate=param.learning_rate)

        train_op = optimizer.minimize(final_loss, global_step=global_step)
    return graph,input_word,input_pos1,input_pos2,input_y,total_shape,predictions,true_,accuracy,acc,final_loss,train_op,global_step
