import tensorflow as tf
import modeling
import optimization
from models import create_model



def make_graph(param):
    graph = tf.Graph()
    with graph.as_default():
        input_ids = tf.placeholder(tf.int32, shape=[None, param.max_seq_length], name="input_ids")
        label_ids = tf.placeholder(tf.int32, shape=[None, param.max_seq_length], name="label_ids")
        input_mask = tf.placeholder(tf.float32, shape=[None, param.max_seq_length], name="input_mask")
        segment_ids = tf.placeholder(tf.int32, shape=[None, param.max_seq_length], name="segment_ids")
        total_loss, logits, trans, pred_ids = create_model(
            param.bert_config, param.is_training, input_ids, input_mask, segment_ids, label_ids,
            len(param.tag2id) + 1, False, param.dropout_rate, param.lstm_size, param.cell, param.cell_nums)
        tvars = tf.trainable_variables()
        if param.init_checkpoint:
            (assignment_map, initialized_variable_names) = \
                modeling.get_assignment_map_from_checkpoint(tvars,
                                                            param.init_checkpoint)
            tf.train.init_from_checkpoint(param.init_checkpoint, assignment_map)

        # num_train_steps = len(train_examples) *1.0 / args.batch_size * args.num_train_epochs)
        # num_warmup_steps = int(num_train_steps * args.warmup_proportion)
        global_step1 = tf.Variable(0, name="global_step1", trainable=False)
        global_add = global_step1.assign_add(1)
        train_op = optimization.create_optimizer(
            total_loss, param.learning_rate, param.num_train_steps, param.num_warmup_steps, False)

        return graph,input_ids,  input_mask, segment_ids, label_ids, total_loss, logits, trans, pred_ids, train_op, global_add