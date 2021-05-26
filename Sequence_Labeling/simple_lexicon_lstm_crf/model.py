import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.crf import crf_log_likelihood

def make_graph(param):
    graph = tf.Graph()
    with graph.as_default():
        word_inputs = tf.placeholder(tf.int32, shape=[None, None], name="word_inputs")
        biword_inputs = tf.placeholder(tf.int32, shape=[None, None], name="biword_inputs")
        layer_gaz = tf.placeholder(tf.int32, shape=[None, None,4,None], name="layer_gaz")
        gaz_count = tf.placeholder(tf.int32, shape=[None, None,4,None], name="gaz_count")
        gaz_chars = tf.placeholder(tf.int32, shape=[None, None,4,None,None], name="gaz_chars")
        gaz_mask = tf.placeholder(tf.int32, shape=[None, None, 4, None], name="gaz_mask")
        gazchar_mask = tf.placeholder(tf.int32, shape=[None, None, 4, None, None], name="gazchar_mask")
        labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        sequence_lengths = tf.placeholder(tf.int32, shape=[None, ], name="sequence_lengths")
        dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")

        with tf.variable_scope("embeddings"):
            _word_embeddings = tf.Variable(param.pretrain_word_embedding,
                                           dtype=tf.float32,
                                           trainable=param.update_embedding,
                                           name="_word_embeddings")
            _biword_embeddings = tf.Variable(param.pretrain_biword_embedding,
                                           dtype=tf.float32,
                                           trainable=param.update_embedding,
                                           name="_biword_embeddings")
            _gaz_embeddings = tf.Variable(param.pretrain_gaz_embedding,
                                           dtype=tf.float32,
                                           trainable=param.update_embedding,
                                           name="_gaz_embeddings")

        with tf.variable_scope("features"):
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                     ids=word_inputs,
                                                     name="word_embeddings")
            if param.use_biword:
                biword_embs = tf.nn.embedding_lookup(params=_biword_embeddings,
                                                     ids=biword_inputs,
                                                     name="biword_embeddings")
                word_embeddings = tf.concat([word_embeddings, biword_embs], axis=-1)

            if param.use_char:
                gazchar_embeds = tf.nn.embedding_lookup(params=_word_embeddings,
                                                     ids=gaz_chars,
                                                     name="gazchar_embeddings")
                gazchar_mask_ = tf.abs(gazchar_mask-1)
                gazchar_mask_ = tf.cast(gazchar_mask_, tf.float32)
                gazchar_mask_ = tf.expand_dims(gazchar_mask_,-1)

                gazchar_embeds = gazchar_embeds * gazchar_mask_

                gaz_charnum = tf.reduce_sum(gazchar_mask,axis=-1,keepdims=True)
                gaz_charnum = tf.cast(gaz_charnum,tf.float32)
                gaz_charnum = gaz_charnum + 1

                gazchar_embeds =tf.reduce_sum(gazchar_embeds,axis=-2)
                # 这里改变名字了，变成gaz_embeds
                gaz_embeds = gazchar_embeds/gaz_charnum
            else:
                gaz_embeds = tf.nn.embedding_lookup(params=_gaz_embeddings,
                                                     ids=layer_gaz,
                                                     name="gaz_embeddings")
                gaz_mask_ = tf.abs(gaz_mask - 1)
                gaz_mask_ = tf.cast(gaz_mask_, tf.float32)
                gaz_mask__ = tf.expand_dims(gaz_mask_,-1)
                gaz_embeds = gaz_embeds * gaz_mask__

            if param.use_count:
                count_sum = tf.reduce_sum(gaz_count,axis=3,keepdims=True)
                count_sum = tf.reduce_sum(count_sum,axis=2,keepdims=True)
                count_sum = count_sum+1

                weights = gaz_count/count_sum
                weights = weights * 4
                weights = tf.cast(weights, tf.float32)
                weights = tf.expand_dims(weights, -1)

                gaz_embeds = gaz_embeds * weights

                gaz_embeds = tf.reduce_sum(gaz_embeds, axis=3)
            else:
                gaz_num = tf.reduce_sum(gaz_mask_,axis=-1,keepdims=True)
                # gaz_num = tf.cast(gaz_num,tf.float32)
                # gaz_num = gaz_num +1
                gaz_embeds = tf.reduce_sum(gaz_embeds, axis=-2)
                # 这里改变名字了，变成gaz_embeds
                gaz_embeds = gaz_embeds / gaz_num

                # gaz_embeds = tf.reduce_mean(gaz_embeds, axis=-2)

            # b = word_embeddings.get_shape()[0]
            # s = word_embeddings.get_shape()[1]
            # gaz_embeds = tf.reshape(gaz_embeds,(b,s,4*50))
            g0, g1, g2, g3 = tf.split(gaz_embeds, num_or_size_splits=4, axis=-2)

            g0 = tf.squeeze(g0,axis=-2)
            g1 = tf.squeeze(g1,axis=-2)
            g2 = tf.squeeze(g2,axis=-2)
            g3 = tf.squeeze(g3,axis=-2)
            gaz_embeds_ = tf.concat([g0,g1,g2,g3],axis=-1)
            all_word_embeddings = tf.concat([word_embeddings, gaz_embeds_], axis=-1)

            # 过一个dropout
            all_word_embeddings = tf.nn.dropout(all_word_embeddings, dropout_pl)
        with tf.variable_scope("fb-lstm"):
            cell_fw = [rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(param.hidden_size, name='cell_fw_1'),
                                output_keep_prob=dropout_pl) for _ in range(param.cell_nums)]

            cell_bw = [rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(param.hidden_size, name='cell_fw_2'),
                                output_keep_prob=dropout_pl) for _ in range(param.cell_nums)]

            rnn_cell_fw = tf.nn.rnn_cell.MultiRNNCell(cell_fw)
            rnn_cell_bw = tf.nn.rnn_cell.MultiRNNCell(cell_bw)
            (output_fw_seq, output_bw_seq), states = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw,
                                                                                     all_word_embeddings,
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
            # loss = -log_likelihood / tf.cast(sequence_lengths, tf.float32)

            loss = tf.reduce_mean(-log_likelihood)
        preds, scores = tf.contrib.crf.crf_decode(logits, transition_params, sequence_lengths)

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
        train_op = optim.minimize(loss,global_step=global_step)
        # grads_and_vars = optim.compute_gradients(loss)
        # # 对梯度gradients进行裁剪，保证在[-params.clip, params.clip]之间。
        # grads_and_vars_clip = [[tf.clip_by_value(g, -param.clip, param.clip), v] for g, v in grads_and_vars]
        # train_op = optim.apply_gradients(grads_and_vars_clip, global_step=global_step)
    return graph,word_inputs,biword_inputs,layer_gaz,gaz_count,gaz_chars,gaz_mask,gazchar_mask,labels,sequence_lengths,dropout_pl,loss,preds,train_op,global_step
