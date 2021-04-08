import tensorflow as tf
from tensorflow.contrib import rnn


def attention(inputs, hidden_dim, name):
    with tf.variable_scope(name):
        # 采用general形式计算权重
        hidden_vec = tf.layers.dense(inputs, hidden_dim * 2, activation=tf.nn.tanh, name='w_hidden')
        u_context = tf.Variable(tf.truncated_normal([hidden_dim * 2]), name='u_context')
        alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(hidden_vec, u_context),
                                            axis=2, keep_dims=True), dim=1)
        # 对隐藏状态进行加权
        attention_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)

    return attention_output

def make_graph(param):
    # 定义计算图
    graph = tf.Graph()
    with graph.as_default():
        input_ids = tf.placeholder(tf.int32, [None, param.max_seq_length], name='input_x')
        input_y = tf.placeholder(tf.int32, [None], name='input_y')
        dropout_keep_prob = tf.placeholder(dtype=tf.float32, name="dropout_keep_prob")
        with tf.name_scope("embedding"):
            input_x = tf.split(input_ids, param.num_sentences, axis=1)
            # input_x shape:[num_sentences,None,max_seq_length/num_sentences]
            input_x = tf.stack(input_x, axis=1)
            # input_x shape:[None,num_sentences,max_seq_length/num_sentences]
            embedding = tf.Variable(param.embedding_mat,
                                    dtype=tf.float32,
                                    trainable=param.update_embedding,
                                    name="_word_embeddings")

            embedding_inputs = tf.nn.embedding_lookup(embedding, input_x)
            # [None,num_sentences,max_seq_length/num_sentences,embed_size]
            sentence_len = int(param.max_seq_length / param.num_sentences)
            embedding_inputs_reshaped = tf.reshape(embedding_inputs, shape=[-1, sentence_len, param.embedding_dim])
            # [None*num_sentences,sentence_length,embed_size]
            embedding_inputs_reshaped = tf.cast(embedding_inputs_reshaped,tf.float32)
            embedding_inputs_reshaped = tf.nn.dropout(embedding_inputs_reshaped, dropout_keep_prob)


        # 词汇层
        with tf.name_scope("word_encoder"):
            cell_fw = [rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(param.hidden_size,name='cell_fw_1'), output_keep_prob=dropout_keep_prob) for
                       _ in
                       range(param.cell_nums)]  # [0 for _ in range(10)] [0,0,0]
            cell_bw = [rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(param.hidden_size,name='cell_fw_2'), output_keep_prob=dropout_keep_prob) for
                       _ in
                       range(param.cell_nums)]  # [0 for _ in range(10)] [0,0,0]
            # cell_bw = [tf.nn.rnn_cell.LSTMCell(hidden_size) for _ in range(cell_nums)]
            rnn_cell_fw = tf.nn.rnn_cell.MultiRNNCell(cell_fw)
            rnn_cell_bw = tf.nn.rnn_cell.MultiRNNCell(cell_bw)
            # tf.nn.bidirectional_dynamic_rnn 返回值：A tuple (outputs, output_states) where:
            # outputs: A tuple (output_fw, output_bw) containing the forward and the backward rnn output `Tensor`.
            # 输入输出的格式为：[batch_size, max_time, depth].
            # tf.nn.static_bidirectional_rnn()
            embedding_inputs_reshaped = tf.cast(embedding_inputs_reshaped,tf.float32)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, embedding_inputs_reshaped,
                                                              dtype=tf.float32)

            word_hidden_state = tf.concat(outputs, 2)
        with tf.name_scope("word_attention"):
            # [batch_size*num_sentences, hidden_size * 2]
            sentence_vec = attention(word_hidden_state, param.hidden_size,"word_attention")

        # 句子层
        with tf.name_scope("sentence_encoder"):
            # [batch_size,num_sentences,hidden_size*2]
            sentence_vec = tf.reshape(sentence_vec, shape=[-1, param.num_sentences, param.hidden_size * 2])
            # output_fw, output_bw = self.bidirectional_rnn(sentence_vec, "sentence_encoder")
            # # [batch_size*num_sentences,sentence_length,hidden_size * 2]
            # sentence_hidden_state = tf.concat((output_fw, output_bw), 2)

            cell_fw = [rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(param.hidden_size,name='cell_fw_3'), output_keep_prob=dropout_keep_prob) for
                       _ in
                       range(param.cell_nums)]  # [0 for _ in range(10)] [0,0,0]
            cell_bw = [rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(param.hidden_size,name='cell_fw_4'), output_keep_prob=dropout_keep_prob) for
                       _ in
                       range(param.cell_nums)]  # [0 for _ in range(10)] [0,0,0]
            # cell_bw = [tf.nn.rnn_cell.LSTMCell(hidden_size) for _ in range(cell_nums)]
            rnn_cell_fw = tf.nn.rnn_cell.MultiRNNCell(cell_fw)
            rnn_cell_bw = tf.nn.rnn_cell.MultiRNNCell(cell_bw)
            # tf.nn.bidirectional_dynamic_rnn 返回值：A tuple (outputs, output_states) where:
            # outputs: A tuple (output_fw, output_bw) containing the forward and the backward rnn output `Tensor`.
            # 输入输出的格式为：[batch_size, max_time, depth].
            # tf.nn.static_bidirectional_rnn()
            sentence_vec = tf.cast(sentence_vec,tf.float32)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, sentence_vec,
                                                              dtype=tf.float32)

            sentence_hidden_state = tf.concat(outputs, 2)

        with tf.name_scope("sentence_attention"):
            # [batch_size, hidden_size * 2]
            doc_vec = attention(sentence_hidden_state, param.hidden_size, "sentence_attention")

        # Add dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(doc_vec, dropout_keep_prob)

        # 输出层
        with tf.name_scope("output"):
            # 分类器
            logits = tf.layers.dense(h_drop, param.num_classes, name='fc2')
            y_pred = tf.argmax(tf.nn.softmax(logits), 1, name="pred")  # 预测类别

        # 损失函数
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=input_y))
        if param.use_l2_regularization:
            tvars = tf.trainable_variables()
            reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4), tvars)
            loss = loss + reg

        # 优化函数
        global_step = tf.train.get_or_create_global_step()


        learning_rate = tf.constant(value=param.learning_rate, shape=[], dtype=tf.float32)
        if param.use_decay_learning_rate:
            learning_rate = tf.train.polynomial_decay(
                learning_rate,
                global_step,
                param.num_train_steps,
                end_learning_rate=0.0,
                power=1.0,
                cycle=False)



        # 原文的优化方法
        # learning_rate = tf.train.exponential_decay(learning_rate, global_step,
        #                                            learning_decay_steps, learning_decay_rate,
        #                                            staircase=True)
        #
        # optimizer = tf.train.AdamOptimizer(learning_rate)
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # optim = slim.learning.create_train_op(total_loss=loss, optimizer=optimizer, update_ops=update_ops)
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
        # compute gradient
        gradients = optimizer.compute_gradients(loss)
        # apply gradient clipping to prevent gradient explosion
        capped_gradients = [(tf.clip_by_norm(grad, 5), var) for grad, var in gradients if grad is not None]

        train_op = optimizer.apply_gradients(capped_gradients, global_step=global_step)

        true_ = tf.cast(input_y, dtype=tf.int64)
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(y_pred, true_)
            acc = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    return graph,input_ids,input_y,dropout_keep_prob,y_pred,loss,true_,train_op,acc,global_step