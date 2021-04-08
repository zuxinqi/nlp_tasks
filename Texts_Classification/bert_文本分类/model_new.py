import tensorflow as tf
import modeling
import optimization

def make_graph(param):
    graph = tf.Graph()
    with graph.as_default():
        input_ids = tf.placeholder(tf.int32, shape=[None, param.max_seq_length], name="input_ids")
        input_mask = tf.placeholder(tf.float32, shape=[None, param.max_seq_length], name="input_mask")
        segment_ids = tf.placeholder(tf.int32, shape=[None, param.max_seq_length], name="segment_ids")
        label_ids = tf.placeholder(tf.int32, shape=[None, ], name="label_ids")
        dropout_rate = tf.placeholder(tf.float32, name="dropout_rate")

        model = modeling.BertModel(
            config=param.bert_config,
            is_training=param.is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False)

        output_layer = model.get_pooled_output()
        if param.is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=dropout_rate)

        with tf.variable_scope("loss"):
            logits = tf.layers.dense(output_layer, len(param.tag2id))
            one_hot_labels = tf.one_hot(label_ids, depth=len(param.tag2id), dtype=tf.float32)
            per_example_loss = tf.nn.softmax_cross_entropy_with_logits_v2(one_hot_labels, logits)
            total_loss = tf.reduce_mean(per_example_loss)
            if param.use_l2_regularization:
                tvars_for_l2 = tf.trainable_variables()
                reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4), tvars_for_l2)
                total_loss = total_loss + reg

        with tf.name_scope("accuracy"):
            probabilities = tf.nn.softmax(logits, axis=-1)
            pred = tf.argmax(probabilities, axis=1)
            true_ = tf.cast(label_ids, dtype=tf.int64)
            correct_predictions = tf.equal(pred, true_)
            acc = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        tvars = tf.trainable_variables()
        if param.init_checkpoint:
            (assignment_map, initialized_variable_names) = \
                modeling.get_assignment_map_from_checkpoint(tvars,
                                                            param.init_checkpoint)
            tf.train.init_from_checkpoint(param.init_checkpoint, assignment_map)
        global_step1 = tf.Variable(0, name="global_step1", trainable=False)
        global_add = global_step1.assign_add(1)
        train_op = optimization.create_optimizer(
            total_loss, param.learning_rate, param.num_train_steps, param.num_warmup_steps, False)
    return graph, input_ids,input_mask,segment_ids, label_ids,\
           dropout_rate,train_op,total_loss,logits,pred,true_,acc,global_add
