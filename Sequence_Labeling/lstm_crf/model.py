import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.crf import crf_log_likelihood

def make_graph(param):
    graph = tf.Graph()
    with graph.as_default():
        word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        sequence_lengths = tf.placeholder(tf.int32, shape=[None, ], name="sequence_lengths")
        dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(param.embedding_mat,
                                           dtype=tf.float32,
                                           trainable=param.update_embedding,
                                           name="_word_embeddings")
            # word_embeddings的shape是[None, None,param.embedding_dim]
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                     ids=word_ids,
                                                     name="word_embeddings")
            word_embeddings = tf.nn.dropout(word_embeddings, dropout_pl)
        with tf.variable_scope("fb-lstm"):
            cell_fw = [rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(param.hidden_size, name='cell_fw_1'),
                                output_keep_prob=dropout_pl) for _ in range(param.cell_nums)]

            cell_bw = [rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(param.hidden_size, name='cell_fw_2'),
                                output_keep_prob=dropout_pl) for _ in range(param.cell_nums)]

            rnn_cell_fw = tf.nn.rnn_cell.MultiRNNCell(cell_fw)
            rnn_cell_bw = tf.nn.rnn_cell.MultiRNNCell(cell_bw)
            (output_fw_seq, output_bw_seq), states = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw,
                                                                                     word_embeddings,
                                                                                     sequence_length=sequence_lengths,
                                                                                     dtype=tf.float32)
            # output的shape是[None, None, params.hidden_size*2]
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, dropout_pl)
        if param.use_cnn:
            output = tf.layers.conv1d(output, param.kernel_nums, param.kernel_size, strides=1, padding='same',
                             activation=tf.nn.relu)
            output = tf.nn.dropout(output, dropout_pl)
        with tf.variable_scope("classification"):
            # logits的shape是[None, None, params.num_tags]
            logits = tf.layers.dense(output, param.num_classes)
        with tf.variable_scope("loss"):
            log_likelihood, transition_params = crf_log_likelihood(inputs=logits,
                                                                   tag_indices=labels,
                                                                   sequence_lengths=sequence_lengths)
            loss = -tf.reduce_mean(log_likelihood)

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


        optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
        grads_and_vars = optim.compute_gradients(loss)
        # 对梯度gradients进行裁剪，保证在[-params.clip, params.clip]之间。
        grads_and_vars_clip = [[tf.clip_by_value(g, -param.clip, param.clip), v] for g, v in grads_and_vars]
        train_op = optim.apply_gradients(grads_and_vars_clip, global_step=global_step)
    return graph,word_ids,labels,sequence_lengths,dropout_pl,logits,transition_params,loss,train_op,global_step