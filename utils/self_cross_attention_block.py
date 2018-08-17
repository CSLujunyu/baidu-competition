import tensorflow as tf

import utils.layers as layers
import utils.operations as op

def self_cross_attention_block(
        config,
        Hu,
        every_turn_len,
        Hr,
        response_len
        ):
    """

    :param config:
    :param Hu: shape = (batch_size, max_turn_num, sentence_len, emb_size)
    :param every_turn_len: shape = (batch_size, max_turn_num )
    :param Hr: shape = (batch_size, sentence_len, emb_size)
    :param response_len: shape = (batch_size)
    :return:
    """

    if config['is_positional'] and config['stack_num'] > 0:
        with tf.variable_scope('positional',reuse=tf.AUTO_REUSE):
            Hr = op.positional_encoding_vector(Hr, max_timescale=10)
    Hr_stack = [Hr]

    for index in range(config['stack_num']):
        with tf.variable_scope('self_stack_' + str(index),reuse=tf.AUTO_REUSE):
            # Hr.shape = (batch_size, max_turn_len, emb_size)
            Hr = layers.block(
                Hr, Hr, Hr,
                Q_lengths=response_len, K_lengths=response_len)
            Hr_stack.append(Hr)

    # context part
    # a list of length max_turn_num, every element is a tensor with shape [batch, max_turn_len, emb_size]
    list_turn_t = tf.unstack(Hu, axis=1)
    list_turn_length = tf.unstack(every_turn_len, axis=1)

    sim_turns = []
    # for every Hu calculate matching vector
    for Hu, t_turn_length in zip(list_turn_t, list_turn_length):
        if config['is_positional'] and config['stack_num'] > 0:
            with tf.variable_scope('positional',reuse=tf.AUTO_REUSE):
                Hu = op.positional_encoding_vector(Hu, max_timescale=10)
        Hu_stack = [Hu]

        for index in range(config['stack_num']):
            with tf.variable_scope('self_stack_' + str(index),reuse=tf.AUTO_REUSE):
                Hu = layers.block(
                    Hu, Hu, Hu,
                    Q_lengths=t_turn_length, K_lengths=t_turn_length)

                Hu_stack.append(Hu)

        r_a_t_stack = []
        t_a_r_stack = []
        for index in range(config['stack_num'] + 1):

            with tf.variable_scope('t_attend_r_' + str(index),reuse=tf.AUTO_REUSE):
                try:
                    t_a_r = layers.block(
                        Hu_stack[index], Hr_stack[index], Hr_stack[index],
                        Q_lengths=t_turn_length, K_lengths=response_len)
                except ValueError:
                    tf.get_variable_scope().reuse_variables()
                    t_a_r = layers.block(
                        Hu_stack[index], Hr_stack[index], Hr_stack[index],
                        Q_lengths=t_turn_length, K_lengths=response_len)

            with tf.variable_scope('r_attend_t_' + str(index),reuse=tf.AUTO_REUSE):
                try:
                    r_a_t = layers.block(
                        Hr_stack[index], Hu_stack[index], Hu_stack[index],
                        Q_lengths=response_len, K_lengths=t_turn_length)
                except ValueError:
                    tf.get_variable_scope().reuse_variables()
                    r_a_t = layers.block(
                        Hr_stack[index], Hu_stack[index], Hu_stack[index],
                        Q_lengths=response_len, K_lengths=t_turn_length)

            t_a_r_stack.append(t_a_r)
            r_a_t_stack.append(r_a_t)

        t_a_r_stack.extend(Hu_stack)
        r_a_t_stack.extend(Hr_stack)

        t_a_r = tf.stack(t_a_r_stack, axis=-1)
        r_a_t = tf.stack(r_a_t_stack, axis=-1)

        # calculate similarity matrix
        with tf.variable_scope('similarity',reuse=tf.AUTO_REUSE):
            # sim shape [batch, max_turn_len, max_turn_len, 2*stack_num+1]
            # divide sqrt(200) to prevent gradient explosion
            sim = tf.einsum('biks,bjks->bijs', t_a_r, r_a_t) / tf.sqrt(200.0)

        sim_turns.append(sim)

    # cnn and aggregation
    sim = tf.stack(sim_turns, axis=1)
    print('sim shape: %s' % sim.shape)
    with tf.variable_scope('cnn_aggregation',reuse=tf.AUTO_REUSE):
        final_info = layers.CNN_3d(sim, 32, 16)

    with tf.variable_scope('linear', reuse=tf.AUTO_REUSE):
        W = tf.get_variable(
            name='weights',
            shape=[final_info.shape[-1], 1],
            initializer=tf.orthogonal_initializer())
        bias = tf.get_variable(
            name='bias',
            shape=[1],
            initializer=tf.zeros_initializer())

    logits = tf.reshape(tf.matmul(final_info, W) + bias, [-1])


    return logits


