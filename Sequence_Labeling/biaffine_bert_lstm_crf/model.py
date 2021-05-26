import util
import modeling
import tensorflow as tf
from tensorflow.contrib import rnn


def lstm_contextualize(text_emb, text_len, lstm_dropout,hidden_size,cell_nums):
    num_sentences = tf.shape(text_emb)[0]

    current_inputs = text_emb  # [num_sentences, max_sentence_length, emb]

    for layer in range(cell_nums):
        with tf.variable_scope("layer_{}".format(layer), reuse=tf.AUTO_REUSE):
            with tf.variable_scope("fw_cell"):
                cell_fw = util.CustomLSTMCell(hidden_size, num_sentences, lstm_dropout)
            with tf.variable_scope("bw_cell"):
                cell_bw = util.CustomLSTMCell(hidden_size, num_sentences, lstm_dropout)
            state_fw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_fw.initial_state.c, [num_sentences, 1]),
                                                     tf.tile(cell_fw.initial_state.h, [num_sentences, 1]))
            state_bw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_bw.initial_state.c, [num_sentences, 1]),
                                                     tf.tile(cell_bw.initial_state.h, [num_sentences, 1]))

            (fw_outputs, bw_outputs), ((_, fw_final_state), (_, bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=current_inputs,
                sequence_length=text_len,
                initial_state_fw=state_fw,
                initial_state_bw=state_bw)

            text_outputs = tf.concat([fw_outputs, bw_outputs], 2)  # [num_sentences, max_sentence_length, emb]
            text_outputs = tf.nn.dropout(text_outputs, lstm_dropout)
            if layer > 0:
                highway_gates = tf.sigmoid(
                    util.projection(text_outputs,
                                    util.shape(text_outputs, 2)))  # [num_sentences, max_sentence_length, emb]
                text_outputs = highway_gates * text_outputs + (1 - highway_gates) * current_inputs
            current_inputs = text_outputs

    return text_outputs

def bilstm(word_embeddings, sequence_lengths, dropout_pl,hidden_size,cell_nums):
    with tf.variable_scope("fb-lstm"):
        cell_fw = [rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(hidden_size, name='cell_fw_1'),
                                      output_keep_prob=dropout_pl) for _ in range(cell_nums)]

        cell_bw = [rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(hidden_size, name='cell_fw_2'),
                                      output_keep_prob=dropout_pl) for _ in range(cell_nums)]

        rnn_cell_fw = tf.nn.rnn_cell.MultiRNNCell(cell_fw)
        rnn_cell_bw = tf.nn.rnn_cell.MultiRNNCell(cell_bw)
        (output_fw_seq, output_bw_seq), states = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw,
                                                                                 word_embeddings,
                                                                                 sequence_length=sequence_lengths,
                                                                                 dtype=tf.float32)
        # output的shape是[None, None, params.hidden_size*2]
        output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
        output = tf.nn.dropout(output, dropout_pl)
    return output

def make_graph(param):
    graph = tf.Graph()
    with graph.as_default():
        word_ids = tf.placeholder(tf.int32, shape=[None, param.max_seq_length], name="word_ids")
        input_mask = tf.placeholder(tf.float32, shape=[None, param.max_seq_length], name="input_mask")
        segment_ids = tf.placeholder(tf.int32, shape=[None, param.max_seq_length], name="segment_ids")
        sequence_lengths = tf.placeholder(tf.int32, shape=[None, ], name="sequence_lengths")
        labels = tf.placeholder(tf.int32, shape=[None,], name="labels")
        dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")

        text_len_mask = tf.sequence_mask(sequence_lengths, maxlen=param.max_seq_length)
        candidate_scores_mask = tf.logical_and(tf.expand_dims(text_len_mask, [1]), tf.expand_dims(text_len_mask, [
            2]))  # [num_sentence, max_sentence_length,max_sentence_length]
        candidate_scores_mask = tf.linalg.band_part(candidate_scores_mask, 0, param.max_seq_length)
        flattened_candidate_scores_mask = tf.reshape(candidate_scores_mask,
                                                     [-1])  # [num_sentence * max_sentence_length * max_sentence_length]

        bert_model = modeling.BertModel(
            config=param.bert_config,
            is_training=param.is_training,
            input_ids=word_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False
        )
        output = bert_model.get_sequence_output()
        output = tf.nn.dropout(output, dropout_pl)
        if param.use_lstm:
            output = bilstm(output, sequence_lengths, dropout_pl, param.lstm_hidden_size, param.cell_nums)
        else:
            output = lstm_contextualize(output, sequence_lengths, dropout_pl, param.lstm_hidden_size, param.cell_nums)

        # if param.use_cnn:
        #     output = tf.layers.conv1d(output, param.kernel_nums, param.kernel_size, strides=1, padding='same',
        #                      activation=tf.nn.relu)
        #     output = tf.nn.dropout(output, dropout_pl)

        with tf.variable_scope("candidate_starts_ffnn"):
            candidate_starts_emb = util.projection(output, param.fc_hidden_size)  # [num_sentences, max_sentences_length,emb]
        with tf.variable_scope("candidate_ends_ffnn"):
            candidate_ends_emb = util.projection(output, param.fc_hidden_size)  # [num_sentences, max_sentences_length, emb]

        candidate_ner_scores_ = util.bilinear_classifier(candidate_starts_emb, candidate_ends_emb, dropout_pl,
                                                        output_size=param.num_classes)  # [num_sentence, max_sentence_length,max_sentence_length,types+1]
        candidate_ner_scores = tf.boolean_mask(tf.reshape(candidate_ner_scores_, [-1, param.num_classes]),
                                               flattened_candidate_scores_mask)

        pred = tf.argmax(tf.nn.softmax(candidate_ner_scores_,axis=-1),axis=-1)
        candidate_scores_mask_ = tf.cast(candidate_scores_mask,dtype=tf.int64)
        pred = pred * candidate_scores_mask_

        # labels = labels * candidate_scores_mask_
        labels_masked = tf.boolean_mask(labels, flattened_candidate_scores_mask)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_masked, logits=candidate_ner_scores)
        loss = tf.reduce_sum(loss)

        # with tf.variable_scope("classification"):
        #     # logits的shape是[None, None, params.num_tags]
        #     logits = tf.layers.dense(output, param.num_classes)
        # with tf.variable_scope("loss"):
        #     log_likelihood, transition_params = crf_log_likelihood(inputs=logits,
        #                                                            tag_indices=labels,
        #                                                            sequence_lengths=sequence_lengths)
        #     loss = -tf.reduce_mean(log_likelihood)

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


        with tf.variable_scope('opt'):
            global_step1 = tf.Variable(0, name="global_step1", trainable=False)
            global_add = global_step1.assign_add(1)
            params_of_bert = []
            params_of_other = []
            for var in tf.trainable_variables():
                vname = var.name
                if vname.startswith("bert"):
                    params_of_bert.append(var)
                else:
                    params_of_other.append(var)
            opt1 = tf.train.AdamOptimizer(learning_rate)
            opt2 = tf.train.AdamOptimizer(1e-3)
            gradients_bert = tf.gradients(loss, params_of_bert)
            gradients_other = tf.gradients(loss, params_of_other)
            gradients_bert_clipped, norm_bert = tf.clip_by_global_norm(gradients_bert, 5.0)
            gradients_other_clipped, norm_other = tf.clip_by_global_norm(gradients_other, 5.0)
            train_op_bert = opt1.apply_gradients(zip(gradients_bert_clipped, params_of_bert))
            train_op_other = opt2.apply_gradients(zip(gradients_other_clipped, params_of_other))
        tvars = tf.trainable_variables()
        if param.init_checkpoint:
            (assignment_map, initialized_variable_names) = \
                modeling.get_assignment_map_from_checkpoint(tvars,
                                                        param.init_checkpoint)
            tf.train.init_from_checkpoint(param.init_checkpoint, assignment_map)
    return graph,word_ids,input_mask,segment_ids,labels,sequence_lengths,dropout_pl,pred,loss,(train_op_bert, train_op_other),global_add