# import modeling
# import optimization
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood


def broadcasting(left, right):

    left = tf.transpose(left, perm=[1, 0, 2])
    # 384 16 32
    left = tf.expand_dims(left, 3)
    # 384 16 32 1
    right = tf.transpose(right, perm=[0, 2, 1])
    # 16 32 384
    right = tf.expand_dims(right, 0)
    # 1 16 32 384
    B = left + right
    # 384 16 32 384
    B = tf.transpose(B, perm=[1, 0, 3, 2])
    # 16 384 384 32

    return B

def broadcasting2(left, right,max_seq_length):
    left = tf.expand_dims(left, 2)
    left = tf.tile(left, [1, 1, max_seq_length, 1])
    right = tf.expand_dims(right, 1)
    right = tf.tile(right, [1, max_seq_length, 1, 1])
    span_matrix = tf.concat([left, right], axis=-1)
    # left = tf.transpose(left, perm=[1, 0, 2])
    # # 384 16 32
    # left = tf.expand_dims(left, 3)
    # # 384 16 32 1
    # right = tf.transpose(right, perm=[0, 2, 1])
    # # 16 32 384
    # right = tf.expand_dims(right, 0)
    # # 1 16 32 384
    # B = left + right
    # # 384 16 32 384
    # B = tf.transpose(B, perm=[1, 0, 3, 2])
    # # 16 384 384 32

    return span_matrix

def getHeadSelectionScores(lstm_out,hidden_size_lstm,label_embeddings_size,hidden_size_n1, rel_num_tags,rel_activation,rel_dropout=1):

    u_a = tf.get_variable("u_a", [(hidden_size_lstm * 2)+label_embeddings_size, hidden_size_n1])  # [128 32]
    w_a = tf.get_variable("w_a", [(hidden_size_lstm * 2)+label_embeddings_size, hidden_size_n1])  # [128 32]
    v = tf.get_variable("v", [hidden_size_n1, rel_num_tags])  # [32,1] or [32,4]
    b_s = tf.get_variable("b_s", [hidden_size_n1])

    left = tf.einsum('aij,jk->aik', lstm_out, u_a)  # [16 348 64] * #[64 32] = [16 348 32]
    right = tf.einsum('aij,jk->aik', lstm_out, w_a)  # [16 348 64] * #[64 32] = [16 348 32]

    outer_sum = broadcasting(left, right)  # [16 348 348 32]

    outer_sum_bias = outer_sum + b_s

    if rel_activation == "tanh":
        output = tf.tanh(outer_sum_bias)
    elif rel_activation == "relu":
        output = tf.nn.relu(outer_sum_bias)
    else:
        # 默认
        output = tf.nn.relu(outer_sum_bias)


    output = tf.nn.dropout(output, keep_prob=rel_dropout)

    g = tf.einsum('aijk,kp->aijp', output, v)

    g = tf.reshape(g, [tf.shape(g)[0], tf.shape(g)[1], tf.shape(g)[2], rel_num_tags])

    return g

def make_graph(param):
    graph = tf.Graph()
    with graph.as_default():
        input_ids = tf.placeholder(tf.int32, shape=[None, param.max_seq_length], name="input_ids")
        sequence_lengths = tf.placeholder(tf.int32, shape=[None, ], name="seqlen")
        ner_label_ids = tf.placeholder(tf.int32, shape=[None, param.max_seq_length], name="ner_label_ids")
        rel_label_ids = tf.placeholder(tf.float32, shape=[None, param.max_seq_length,param.max_seq_length,param.rel_num_tags], name="rel_label_ids")
        is_train = tf.placeholder(tf.int32)
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(param.embedding_mat,
                                           dtype=tf.float32,
                                           trainable=param.update_embedding,
                                           name="_word_embeddings")
            # word_embeddings的shape是[None, None,params.embedding_dim]
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                     ids=input_ids,
                                                     name="word_embeddings")
            word_embeddings = tf.nn.dropout(word_embeddings, param.embedding_dropout)
        input_rnn = word_embeddings
        for i in range(param.num_lstm_layers):
            if param.use_lstm_dropout and i > 0:
                input_rnn = tf.nn.dropout(input_rnn, keep_prob=param.lstm_dropout)
                # input_rnn = tf.Print(input_rnn, [dropout_lstm_keep], 'lstm:  ', summarize=1000)

            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(param.hidden_size_lstm)
            # Backward direction cell
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(param.hidden_size_lstm)

            lstm_out, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=lstm_fw_cell,
                cell_bw=lstm_bw_cell,
                inputs=input_rnn,
                sequence_length=sequence_lengths,
                dtype=tf.float32, scope='BiLSTM' + str(i))

            input_rnn = tf.concat(lstm_out, 2)

            lstm_output = input_rnn
        if param.use_lstm_dropout:
            lstm_output = tf.nn.dropout(lstm_output, keep_prob=param.lstm_dropout)

        ner_input = lstm_output
        with tf.variable_scope("ner_loss"):
            # logits的shape是[None, None, params.num_tags]
            ner_logits = tf.layers.dense(ner_input, param.ner_num_tags)
            log_likelihood, transition_params = crf_log_likelihood(inputs=ner_logits,
                                                                   tag_indices=ner_label_ids,
                                                                   sequence_lengths=sequence_lengths)
            ner_loss = -tf.reduce_mean(log_likelihood)

            ner_pred, viterbi_score = tf.contrib.crf.crf_decode(
                ner_logits, transition_params, sequence_lengths)

        rel_input = lstm_output
        if param.label_embeddings_size > 0:
            label_matrix = tf.get_variable(name="label_embeddings", dtype=tf.float32,
                                            shape=[param.ner_num_tags, param.label_embeddings_size])
            labels = tf.cond(is_train > 0, lambda: ner_label_ids, lambda: ner_pred)
            label_embeddings = tf.nn.embedding_lookup(label_matrix, labels)
            rel_input = tf.concat([lstm_output, label_embeddings], axis=2)

        with tf.variable_scope("rel_loss"):
            rel_logits = getHeadSelectionScores(rel_input, param.hidden_size_lstm, param.label_embeddings_size,param.hidden_size_n1, param.rel_num_tags, param.rel_activation,
                                   param.rel_dropout)
            rel_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=rel_logits, labels=rel_label_ids)
            probas = tf.nn.sigmoid(rel_logits)
            rel_predict = tf.round(probas)

        final_mask = tf.cast(tf.sequence_mask(sequence_lengths, maxlen=param.max_seq_length), tf.int32)
        span_start_mask = tf.expand_dims(final_mask, 2)
        span_start_mask = tf.tile(span_start_mask, [1, 1, param.max_seq_length])
        span_start_mask = tf.equal(span_start_mask, 1)
        span_end_mask = tf.expand_dims(final_mask, 1)
        span_end_mask = tf.tile(span_end_mask, [1, param.max_seq_length, 1])
        span_end_mask = tf.equal(span_end_mask, 1)
        match_label_mask = span_start_mask & span_end_mask
        match_label_mask = tf.cast(match_label_mask, dtype=tf.float32)
        match_label_mask = tf.expand_dims(match_label_mask, 3)
        match_label_mask = tf.tile(match_label_mask, [1, 1, 1, param.rel_num_tags])

        rel_loss = rel_loss * match_label_mask
        rel_loss = tf.reduce_mean(rel_loss)

        total_loss = ner_loss + rel_loss

        global_step = tf.train.create_global_step(graph)
        optim = tf.train.AdamOptimizer(learning_rate=param.learning_rate)
        grads_and_vars = optim.compute_gradients(total_loss)
        # 对梯度gradients进行裁剪，保证在[-params.clip, params.clip]之间。
        grads_and_vars_clip = [[tf.clip_by_value(g, -param.clip, param.clip), v] for g, v in grads_and_vars]
        train_op = optim.apply_gradients(grads_and_vars_clip, global_step=global_step)

    return graph,input_ids,sequence_lengths,ner_label_ids,rel_label_ids,is_train,ner_pred,rel_predict,total_loss,train_op,global_step

