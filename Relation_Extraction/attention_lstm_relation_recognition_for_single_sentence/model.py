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
        input_batch_size = tf.placeholder(dtype=tf.int32, name='batch_size')


        word_embedding = tf.Variable(param.embedding_mat,
                                           dtype=tf.float32,
                                           trainable=param.update_embedding,
                                           name="word_embeddings")

        pos1_embedding = tf.get_variable('pos1_embedding', [param.pos_num, param.pos_size])
        pos2_embedding = tf.get_variable('pos2_embedding', [param.pos_num, param.pos_size])
        attention_w = tf.get_variable('attention_omega', [param.gru_size, 1])

        gru_cell_forward = tf.contrib.rnn.GRUCell(param.gru_size)
        gru_cell_backward = tf.contrib.rnn.GRUCell(param.gru_size)
        if param.is_training and param.dropout_rate < 1:
            gru_cell_forward = tf.contrib.rnn.DropoutWrapper(gru_cell_forward, output_keep_prob=param.dropout_rate)
            gru_cell_backward = tf.contrib.rnn.DropoutWrapper(gru_cell_backward, output_keep_prob=param.dropout_rate)

        cell_forward = tf.contrib.rnn.MultiRNNCell([gru_cell_forward] * param.num_layers)
        cell_backward = tf.contrib.rnn.MultiRNNCell([gru_cell_backward] * param.num_layers)

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
            tf.reshape(tf.matmul(tf.reshape(tf.tanh(output_h), [input_batch_size * param.max_seq_length, param.gru_size]), attention_w),
                       [input_batch_size, param.max_seq_length])), [input_batch_size, 1, param.max_seq_length]), output_h), [input_batch_size, param.gru_size])

        logits = tf.layers.dense(attention_r,param.num_classes)
        total_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=input_y))

        predictions = tf.argmax(tf.nn.softmax(logits, 1), 1)
        true_ = tf.argmax(input_y, -1)
        correct_predictions = tf.equal(predictions, true_)
        acc = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")


        # regularization
        l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                                                              weights_list=tf.trainable_variables())
        final_loss = total_loss + l2_loss

        optimizer = tf.train.AdamOptimizer(learning_rate=param.learning_rate)

        train_op = optimizer.minimize(final_loss, global_step=global_step)
    return graph,input_word,input_pos1,input_pos2,input_y,input_batch_size,predictions,true_,acc,final_loss,train_op,global_step
