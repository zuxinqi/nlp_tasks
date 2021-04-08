import tensorflow as tf
import modeling
import optimization


def focal_loss(logits,labels,mask,num_labels,one_hot=True,lambda_param=1.5):
    probs = tf.nn.softmax(logits,axis=-1)
    pos_probs = probs[:,:,1]
    prob_label_pos = tf.where(tf.equal(labels,1),pos_probs,tf.ones_like(pos_probs))
    prob_label_neg = tf.where(tf.equal(labels,0),pos_probs,tf.zeros_like(pos_probs))
    loss = tf.pow(1. - prob_label_pos,lambda_param)*tf.log(prob_label_pos + 1e-7) + \
           tf.pow(prob_label_neg,lambda_param)*tf.log(1. - prob_label_neg + 1e-7)
    loss = -loss * tf.cast(mask,tf.float32)
    loss = tf.reduce_sum(loss,axis=-1,keepdims=True)
    # loss = loss/tf.cast(tf.reduce_sum(mask,axis=-1),tf.float32)
    loss = tf.reduce_mean(loss)
    return loss

def make_graph(param):
    graph = tf.Graph()
    with graph.as_default():
        input_ids = tf.placeholder(tf.int32, shape=[None, param.max_seq_length], name="input_ids")
        start_labels = tf.placeholder(tf.int32, shape=[None, param.max_seq_length], name="start_labels")
        end_labels = tf.placeholder(tf.int32, shape=[None, param.max_seq_length], name="end_labels")
        token_type_ids_list = tf.placeholder(tf.int32, shape=[None, param.max_seq_length], name="token_type_ids_list")
        query_len_list = tf.placeholder(tf.int32, shape=[None,], name="query_len_list")
        text_length_list = tf.placeholder(tf.int32, shape=[None,], name="text_length_list")
        bert_model = modeling.BertModel(
            config=param.bert_config,
            is_training=param.is_training,
            input_ids=input_ids,
            text_length=text_length_list,
            max_seq_length = param.max_seq_length,
            token_type_ids=token_type_ids_list,
            use_one_hot_embeddings=False
        )
        bert_seq_output = bert_model.get_sequence_output()

        # 添加dropout
        bert_seq_output = tf.nn.dropout(bert_seq_output, param.dropout_rate)

        # bert_project = tf.layers.dense(bert_seq_output, self.hidden_units, activation=tf.nn.relu)
        # bert_project = tf.layers.dropout(bert_project, rate=self.dropout_rate, training=is_training)

        start_logits = tf.layers.dense(bert_seq_output, param.num_classes)
        end_logits = tf.layers.dense(bert_seq_output, param.num_classes)
        query_span_mask = tf.cast(tf.sequence_mask(query_len_list), tf.int32)
        total_seq_mask = tf.cast(tf.sequence_mask(text_length_list,maxlen=param.max_seq_length), tf.int32)
        query_span_mask = query_span_mask * -1
        query_len_max = tf.shape(query_span_mask)[1]
        left_query_len_max = tf.shape(total_seq_mask)[1] - query_len_max
        zero_mask_left_span = tf.zeros((tf.shape(query_span_mask)[0], left_query_len_max), dtype=tf.int32)
        final_mask = tf.concat((query_span_mask, zero_mask_left_span), axis=-1)
        final_mask = final_mask + total_seq_mask

        # final_mask_max = tf.shape(final_mask)[1]
        # final_left_query_len_max = tf.shape(input_ids)[1] - final_mask_max
        # final_zero_mask_left_span = tf.zeros((tf.shape(final_mask)[0], final_left_query_len_max), dtype=tf.int32)
        # new_final_mask = tf.concat((final_mask, final_zero_mask_left_span), axis=-1)

        predict_start_ids = tf.argmax(start_logits, axis=-1, name="pred_start_ids")
        predict_end_ids = tf.argmax(end_logits, axis=-1, name="pred_end_ids")
        # one_hot_labels = tf.one_hot(labels, depth=self.num_labels, dtype=tf.float32)
        # start_loss = ce_loss(start_logits,start_labels,final_mask,self.num_labels,True)
        # end_loss = ce_loss(end_logits,end_labels,final_mask,self.num_labels,True)

        # focal loss
        start_loss = focal_loss(start_logits, start_labels, final_mask, param.num_classes, True)
        end_loss = focal_loss(end_logits, end_labels, final_mask, param.num_classes, True)

        total_loss = start_loss + end_loss

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

    return graph,input_ids, start_labels, end_labels, token_type_ids_list, query_len_list, \
           text_length_list, predict_start_ids, predict_end_ids,\
           global_add, total_loss, train_op

