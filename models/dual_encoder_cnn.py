import tensorflow as tf


def dual_encoder_CNN_model(
        config,
        context_embed,
        context_len,
        candidate_embed,
        candidate_len
        ):
    """

    :param config:
    :param context_embed: shape = (batch_size, max_turn_num, max_turn_len, embed_dim)
    :param context_len: shape = (batch_size, max_turn_num )
    :param candidate_embed: shape = (batch_size, options_num, max_turn_len, embed_dim)
    :param candidate_len: shape = (batch_size, options_num)
    :return:
    """

    with tf.variable_scope('context_conv',reuse=tf.AUTO_REUSE):
        # context_feature.shape = (batch_size, max_turn_num, max_turn_len, conv_filter_num)
        context_feature = tf.layers.conv2d(inputs=context_embed,
                                           filters=config['conv_filter_num'],
                                           kernel_size=[1,3],
                                           padding='same',
                                           activation=tf.nn.relu)
    with tf.variable_scope('candidate_conv',reuse=tf.AUTO_REUSE):
        # candidate_embed.shape = (batch_size, options_num, max_turn_len, conv_filter_num)
        candidate_feature = tf.layers.conv2d(inputs=candidate_embed,
                                           filters=config['conv_filter_num'],
                                           kernel_size=[1,3],
                                           padding='same',
                                           activation=tf.nn.relu)


    with tf.variable_scope('aggregation'):
        # M.shape = (max_turn_len, conv_filter_num)
        M = tf.get_variable("M", shape=context_feature.shape[-2:],
                            initializer=tf.truncated_normal_initializer())

        # c_M.shape = (batch_size, max_turn_num, max_turn_len, conv_filter_num)
        c_M = tf.einsum('bijk,jk->bijk', context_feature, M)

        # c_r.shape = (batch_size, max_turn_num, options_num)
        c_r = tf.einsum('btij,boij->bto', c_M, candidate_feature)

        # logits.shape = (batch_size, options_num)
        logits = tf.reduce_sum(c_r, axis=1)

        # probs.shape = (batch_size, options_num)
        probs = tf.nn.softmax(logits)


    return probs, logits
