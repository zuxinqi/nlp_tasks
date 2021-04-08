import tensorflow as tf
from tensorflow.contrib import rnn


def make_graph(param):
    # 定义计算图
    graph = tf.Graph()
    with graph.as_default():
        # 定义占位符
        input_x = tf.placeholder(dtype=tf.int32, shape=[None, param.max_seq_length], name="input_data")
        input_y = tf.placeholder(dtype=tf.int32, shape=[None, ], name="label")
        input_seq = tf.placeholder(dtype=tf.int32, shape=[None, ], name="label")
        dropout_keep_prob = tf.placeholder(dtype=tf.float32, name="dropout_keep_prob")

        with tf.variable_scope("embedding", dtype=tf.float32):
            embedding = tf.Variable(param.embedding_mat,
                                    dtype=tf.float32,
                                    trainable=param.update_embedding,
                                    name="_word_embeddings")
            # 利用词频计算新的词嵌入矩阵
            # normWordEmbedding = normalize(embedding, vocab_freqs)
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            embedded_wrods = tf.nn.embedding_lookup(embedding, input_x)
            embedded_wrods = tf.nn.dropout(embedded_wrods, dropout_keep_prob)

        with tf.name_scope('rnn'):
            cell_fw = [rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(param.lstm_hidden_size), output_keep_prob=dropout_keep_prob) for
                       _ in
                       range(param.cell_nums)]  # [0 for _ in range(10)] [0,0,0]
            cell_bw = [rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(param.lstm_hidden_size), output_keep_prob=dropout_keep_prob) for
                       _ in
                       range(param.cell_nums)]  # [0 for _ in range(10)] [0,0,0]
            rnn_cell_fw = tf.nn.rnn_cell.MultiRNNCell(cell_fw)
            rnn_cell_bw = tf.nn.rnn_cell.MultiRNNCell(cell_bw)

            outputs, states = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, embedded_wrods,
                                                              sequence_length=input_seq, dtype=tf.float32)


            all_layer_outputs = tf.concat([states[0][-1].h, states[1][-1].h], axis=-1)

        p_drop = tf.nn.dropout(all_layer_outputs, dropout_keep_prob)
        layer1 = tf.layers.dense(p_drop, param.hidden_num, activation=tf.nn.relu)
        # h_drop1 = tf.nn.dropout(layer1, dropout_keep_prob)

        # layer2 = tf.layers.dense(layer1, 1024, activation=tf.nn.relu)
        # h_drop = tf.nn.dropout(layer2, dropout_keep_prob)
        # layer3 = tf.layers.dense(h_drop, 512, activation=None)
        logits_ = tf.layers.dense(layer1, param.num_classes)

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_, labels=input_y))

        if param.use_l2_regularization:
            tvars = tf.trainable_variables()
            reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4), tvars)
            loss = loss + reg

        global_step = tf.train.create_global_step(graph)
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

        train_op = optimizer.apply_gradients(capped_gradients, global_step=global_step)
        pred = tf.argmax(tf.nn.softmax(logits_), 1, name="pred")
        true_ = tf.cast(input_y, dtype=tf.int64)
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(pred, true_)
            acc = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        return graph, input_x, input_y, input_seq, dropout_keep_prob, train_op, loss, pred, true_, acc, global_step