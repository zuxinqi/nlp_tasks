import tensorflow as tf
import modeling
import optimization


def make_graph(param):
    # 定义计算图
    graph = tf.Graph()
    with graph.as_default():
        input_ids = tf.placeholder(tf.int32, shape=[None, param.max_seq_length], name="input_ids")
        input_mask = tf.placeholder(tf.float32, shape=[None, param.max_seq_length], name="input_mask")
        segment_ids = tf.placeholder(tf.int32, shape=[None, param.max_seq_length], name="segment_ids")
        labels = tf.placeholder(tf.int32, shape=[None, ], name="labels")
        dropout_pl = tf.placeholder(dtype=tf.float32, shape=(), name="dropout")


        bert_model = modeling.BertModel(
            config=param.bert_config,
            is_training=param.is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False
        )
        bert_pooled_output = bert_model.get_pooled_output()
        bert_pooled_output = tf.nn.dropout(bert_pooled_output, dropout_pl)

        logits = tf.layers.dense(bert_pooled_output, 2)

        labels_ = tf.one_hot(labels, 2)
        losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels_)
        loss = tf.reduce_mean(losses)

        tvars = tf.trainable_variables()
        if param.init_checkpoint:
            (assignment_map, initialized_variable_names) = \
                modeling.get_assignment_map_from_checkpoint(tvars,
                                                            param.init_checkpoint)
            tf.train.init_from_checkpoint(param.init_checkpoint, assignment_map)

        global_step1 = tf.Variable(0, name="global_step1", trainable=False)
        global_add = global_step1.assign_add(1)
        train_op = optimization.create_optimizer(
            loss, param.learning_rate, param.num_train_steps, param.num_warmup_steps, False)

        # 准确率/f1/p/r计算
        with tf.variable_scope("evaluation"):
            true = tf.cast(tf.argmax(labels_, axis=-1), tf.float32)  # 真实序列的值
            labels_softmax = tf.nn.softmax(logits)
            labels_softmax_ = tf.argmax(labels_softmax, axis=-1)
            pred = tf.cast(labels_softmax_, tf.float32)  # 预测序列的值
            accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, true), tf.float32))

    return graph, input_ids, input_mask, segment_ids, labels, dropout_pl, loss, train_op, true, pred, accuracy, global_add
