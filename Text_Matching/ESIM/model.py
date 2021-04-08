import tensorflow as tf
from tensorflow.contrib import rnn

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

        with tf.variable_scope("cell_by_one_layer_bi-lstm"):
            # 词1层bi-lstm
            # cell_fw = tf.nn.rnn_cell.LSTMCell(embedding_hidden_size)
            cell_fw = rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(param.embedding_hidden_size, name='cell_fw_1'),
                                         output_keep_prob=dropout_pl)
            # cell_bw = tf.nn.rnn_cell.LSTMCell(embedding_hidden_size)
            cell_bw = rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(param.embedding_hidden_size, name='cell_bw_1'),
                                         output_keep_prob=dropout_pl)
            (left_output_fw_seq, left_output_bw_seq), left_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                                                    left_embeddings,
                                                                                                    dtype=tf.float32)
            left_result = tf.concat([left_output_fw_seq, left_output_bw_seq], axis=-1)
        with tf.variable_scope("right_one_layer_bi-lstm"):
            (right_output_fw_seq, right_output_bw_seq), right_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                                                       right_embeddings,
                                                                                                       dtype=tf.float32)
            right_result = tf.concat([right_output_fw_seq, right_output_bw_seq], axis=-1)
        with tf.variable_scope("attention"):
            attention_weights = tf.matmul(left_result, tf.transpose(right_result, [0, 2, 1]))
            attentionsoft_a = tf.nn.softmax(attention_weights)
            attentionsoft_b = tf.nn.softmax(tf.transpose(attention_weights, [0, 2, 1]))
            # attentionsoft_a [batch_size,max_time,max_time]
            # right_result [batch_size,max_time,hidden_size]
            left_hat = tf.matmul(attentionsoft_a, right_result)
            right_hat = tf.matmul(attentionsoft_b, left_result)
        with tf.variable_scope("compute"):
            left_diff = tf.subtract(left_result, left_hat)
            left_mul = tf.multiply(left_result, left_hat)

            right_diff = tf.subtract(right_result, right_hat)
            right_mul = tf.multiply(right_result, right_hat)

            m_left = tf.concat([left_result, left_hat, left_diff, left_mul], axis=2)
            m_right = tf.concat([right_result, right_hat, right_diff, right_mul], axis=2)

        with tf.variable_scope("bi-lstm"):
            # cell_fw = tf.nn.rnn_cell.LSTMCell(context_hidden_size)
            cell_fw = rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(param.context_hidden_size, name='cell_fw_2'),
                               output_keep_prob=dropout_pl)
            # cell_bw = tf.nn.rnn_cell.LSTMCell(context_hidden_size)
            cell_bw = rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(param.context_hidden_size, name='cell_bw_2'),
                                         output_keep_prob=dropout_pl)

            (left_output_fw, left_output_bw), left_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, m_left,
                                                                                            dtype=tf.float32)
            left_output = tf.concat([left_output_fw, left_output_bw], axis=-1)
            v_left_avg = tf.reduce_mean(left_output, axis=1)
            v_left_max = tf.reduce_max(left_output, axis=1)

            # cell_fw = tf.nn.rnn_cell.LSTMCell(context_hidden_size)
            cell_fw = rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(param.context_hidden_size, name='cell_fw_3'),
                                         output_keep_prob=dropout_pl)
            # cell_bw = tf.nn.rnn_cell.LSTMCell(context_hidden_size)
            cell_bw = rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(param.context_hidden_size, name='cell_bw_3'),
                                         output_keep_prob=dropout_pl)

            (right_output_fw, right_output_bw), right_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                                               m_right,
                                                                                               dtype=tf.float32)
            right_output = tf.concat([right_output_fw, right_output_bw], axis=-1)
            v_right_avg = tf.reduce_mean(right_output, axis=1)
            v_right_max = tf.reduce_max(right_output, axis=1)

            # v_left = tf.concat([v_left_avg, v_left_max], axis=1)
            # v_right = tf.concat([v_right_avg, v_right_max], axis=1)
            v_concat = tf.concat([v_left_avg, v_left_max, v_right_avg, v_right_max], axis=1)

        with tf.variable_scope("classification"):
            output = tf.layers.dense(inputs=v_concat, units=param.hidden_units, activation=tf.nn.relu)
            output = tf.layers.dense(inputs=output, units=128, activation=tf.nn.relu)
            logits = tf.layers.dense(inputs=output, units=2)

        # 计算损失
        with tf.variable_scope("loss"):
            # logits__ = tf.nn.softmax(logits)
            # loss = (-0.25 * tf.reduce_sum(labels[:, 0] * tf.log(logits__[:, 0]))
            #         - 0.75 * tf.reduce_sum(labels[:, 1] * tf.log(logits__[:, 1]))
            #         ) / tf.cast(tf.shape(labels)[0], tf.float32)


            # losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)

            # labels_ = tf.expand_dims(labels,1)
            # losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels_)
            # loss = tf.reduce_mean(losses)

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

            true = tf.cast(tf.argmax(labels_, axis=-1), tf.float32)  # 真实序列的值
            labels_softmax = tf.nn.softmax(logits)
            labels_softmax_ = tf.argmax(labels_softmax, axis=-1)
            pred = tf.cast(labels_softmax_, tf.float32)  # 预测序列的值
            accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, true), tf.float32))

    return graph,left_input,right_input,labels,dropout_pl,loss,train_op,true,pred,accuracy,global_step
