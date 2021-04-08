# import modeling
# import optimization
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
import modeling
import optimization


def make_graph(param):
    graph = tf.Graph()
    with graph.as_default():
        input_ids = tf.placeholder(tf.int32, shape=[None, param.max_seq_length], name="input_ids")
        sequence_lengths = tf.placeholder(tf.int32, shape=[None, ], name="seqlen")
        ner_label_ids = tf.placeholder(tf.float32, shape=[None, param.max_seq_length,param.ner_num_tags], name="ner_label_ids")
        input_mask = tf.placeholder(tf.float32, shape=[None, param.max_seq_length], name="input_mask")
        segment_ids = tf.placeholder(tf.int32, shape=[None, param.max_seq_length], name="segment_ids")

        bert_model = modeling.BertModel(
            config=param.bert_config,
            is_training=param.is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False
        )
        bert_seq_output = bert_model.get_sequence_output()

        bert_seq_output = tf.nn.dropout(bert_seq_output, param.embedding_dropout)
        if param.use_lstm:
            with tf.variable_scope("fb-lstm"):
                cell_fw = [tf.nn.rnn_cell.LSTMCell(param.hidden_size) for _ in range(param.cell_nums)]
                cell_bw = [tf.nn.rnn_cell.LSTMCell(param.hidden_size) for _ in range(param.cell_nums)]
                rnn_cell_fw = tf.nn.rnn_cell.MultiRNNCell(cell_fw)
                rnn_cell_bw = tf.nn.rnn_cell.MultiRNNCell(cell_bw)
                (output_fw_seq, output_bw_seq), states = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw,
                                                                                         bert_seq_output,
                                                                                         sequence_length=sequence_lengths,
                                                                                         dtype=tf.float32)
                # output的shape是[None, None, params.hidden_size*2]
                output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
        if param.use_lstm_dropout:
            lstm_output = tf.nn.dropout(output, keep_prob=param.lstm_dropout)

        ner_input = lstm_output
        with tf.variable_scope("ner_loss"):
            # logits的shape是[None, None, params.num_tags]
            ner_input = tf.layers.dense(ner_input, param.hidden_size,activation=tf.nn.tanh)
            ner_logits = tf.layers.dense(ner_input, param.ner_num_tags)
            ner_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=ner_label_ids,logits=ner_logits)


            ner_mask = tf.expand_dims(input_mask,2)
            ner_mask = tf.tile(ner_mask, [1, 1, param.ner_num_tags])
            ner_loss = ner_loss * ner_mask
            ner_loss = tf.reduce_mean(ner_loss)


        ner_pred = tf.round(tf.nn.sigmoid(ner_logits))
        tvars = tf.trainable_variables()
        if param.init_checkpoint:
            (assignment_map, initialized_variable_names) = \
                modeling.get_assignment_map_from_checkpoint(tvars,
                                                            param.init_checkpoint)
            tf.train.init_from_checkpoint(param.init_checkpoint, assignment_map)


        train_op = optimization.create_optimizer(
            ner_loss, param.learning_rate, param.num_train_steps, param.num_warmup_steps, False)
        global_step1 = tf.Variable(0, name="global_step1", trainable=False)
        global_add = global_step1.assign_add(1)
        # global_step = tf.train.create_global_step(graph)
        # optim = tf.train.AdamOptimizer(learning_rate=lr)
        # train_op = optim.minimize(total_loss,global_step=global_step)

        # grads_and_vars = optim.compute_gradients(total_loss)
        # # 对梯度gradients进行裁剪，保证在[-params.clip, params.clip]之间。
        # grads_and_vars_clip = [[tf.clip_by_value(g, -clip, clip), v] for g, v in grads_and_vars]
        # train_op = optim.apply_gradients(grads_and_vars_clip, global_step=global_step)

        return graph,input_ids,input_mask,segment_ids,sequence_lengths,ner_label_ids,ner_loss,ner_pred,train_op,global_add

