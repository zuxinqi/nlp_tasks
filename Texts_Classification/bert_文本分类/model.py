import tensorflow as tf
import modeling
import optimization


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, dropout_rate,use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=dropout_rate)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        pred = tf.argmax(probabilities, axis=1)
        true_ = tf.cast(labels, dtype=tf.int64)
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(pred, true_)
            acc = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        return (loss, per_example_loss, logits, pred, true_, acc)


def make_graph(param):
    graph = tf.Graph()
    with graph.as_default():
        input_ids = tf.placeholder(tf.int32, shape=[None, param.max_seq_length], name="input_ids")
        input_mask = tf.placeholder(tf.float32, shape=[None, param.max_seq_length], name="input_mask")
        segment_ids = tf.placeholder(tf.int32, shape=[None, param.max_seq_length], name="segment_ids")
        label_ids = tf.placeholder(tf.int32, shape=[None, ], name="label_ids")
        dropout_rate = tf.placeholder(tf.float32, name="dropout_rate")
        (total_loss, per_example_loss, logits, pred, true_, acc) = create_model(
            param.bert_config, param.is_training, input_ids, input_mask, segment_ids, label_ids,
            len(param.tag2id), dropout_rate, False)
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
