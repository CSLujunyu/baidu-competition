import tensorflow as tf

def attention(
        Q, K, V,
        Q_lengths, K_lengths,
        attention_type='dot',
        is_mask=True, mask_value=-2 ** 32 + 1,
        block_type = 'self',
        drop_prob=None):
    """

    :param Q: shape = (batch_size, parallel_num_q, max_turn_len, emb_size)
    :param K: shape = (batch_size, parallel_num_k, max_turn_len, emb_size)
    :param V: shape = (batch_size, parallel_num_v, max_turn_len, emb_size)
    :param Q_lengths: shape = (batch_size, parallel_num_q)
    :param K_lengths:
    :param attention_type:
    :param is_mask:
    :param mask_value:
    :param drop_prob:
    :return:
    """

    assert attention_type in ('dot', 'bilinear')
    if attention_type == 'dot':
        assert Q.shape[-1] == K.shape[-1]

    max_row_length = Q.shape[2]
    max_col_length = K.shape[2]

    ### self attention ###
    if block_type == 'self':
        if attention_type == 'dot':
            # logits.shape = (batch_size, parallel_num, max_turn_len, max_turn_len)
            logits = tf.einsum('bijk,bimk->bijm', Q, K)

        if is_mask:
            # bool, [batch_size, parallel_num_q, max_turn_len]
            mask = tf.sequence_mask(Q_lengths, max_row_length)
            mask = tf.cast(tf.expand_dims(mask, -1), tf.float32)
            # mask.shape = (batch_size, parallel_num_q, max_turn_len, max_turn_len)
            mask = tf.einsum('bmik,bmjk->bmij', mask, mask)
            logits = mask * logits + (1 - mask) * mask_value

        # apply softmax to the last dimension
        # attention.shape = (batch_size, parallel_num_q, max_turn_len, max_turn_len)
        attention = tf.nn.softmax(logits)

        if drop_prob is not None:
            print('use attention drop')
            attention = tf.nn.dropout(attention, drop_prob)
        res = tf.einsum('bmij,bmjk->bmik', attention, V)

    ### cross attention ###

    elif block_type == 'cross':
        # Q.shape = (batch_size, parallel_num_q, max_turn_len, emb_size) , K.shape = (batch_size, parallel_num_k, max_turn_len, emb_size)
        if attention_type == 'dot':
            # logits.shape = (batch_size, parallel_num_q, parallel_num_k, max_turn_len, max_turn_len)
            logits = tf.einsum('bijk,bmnk->bimjn', Q, K)

        if is_mask:
            # Q_mask.shape = (batch_size, parallel_num_q, max_turn_len)
            Q_mask = tf.sequence_mask(Q_lengths, max_row_length)
            # K_mask.shape = (batch_size, parallel_num_k, max_turn_len)
            K_mask = tf.sequence_mask(K_lengths, max_col_length)
            Q_mask = tf.cast(tf.expand_dims(Q_mask, -1), tf.float32)
            K_mask = tf.cast(tf.expand_dims(K_mask, -1), tf.float32)
            # mask.shape = (batch_size, parallel_num_q, parallel_num_k, max_turn_len, max_turn_len)
            mask = tf.einsum('bijk,bmnk->bimjn', Q_mask, K_mask)
            logits = mask * logits + (1 - mask) * mask_value

        # apply softmax to the last dimension
        # attention.shape = (batch_size, parallel_num_q, parallel_num_k, max_turn_len, max_turn_len)
        attention = tf.nn.softmax(logits)

        if drop_prob is not None:
            print('use attention drop')
            attention = tf.nn.dropout(attention, drop_prob)
        # V.shape = (batch_size, parallel_num_v, max_turn_len, emb_size)
        # res.shape = (batch_size, parallel_num_q, parallel_num_v, max_turn_len, emb_size)
        # TODO:check einsum
        res = tf.einsum('bijkm,bjmn->bijkn', attention, V)
    else:
        res = None
    return res


def FFN(x,block_type ='self'):
    """

    :param x: shape = (batch_size, parallel_num, max_turn_len, emb_size)
    :return:
    """
    with tf.variable_scope('FFN_1', reuse=tf.AUTO_REUSE):
        W = tf.get_variable(
            name='weights',
            shape=[x.shape[-1], x.shape[-1]],
            dtype=tf.float32,
            initializer=tf.orthogonal_initializer())
        bias = tf.get_variable(
            name='bias',
            shape=[1],
            dtype=tf.float32,
            initializer=tf.zeros_initializer())
        if block_type == 'self':
            y = tf.nn.relu(tf.einsum('bmik,kj->bmij', x, W) + bias)
        else:
            y = tf.nn.relu(tf.einsum('bmnik,kj->bmnij', x, W) + bias)
    with tf.variable_scope('FFN_2', reuse=tf.AUTO_REUSE):
        W = tf.get_variable(
            name='weights',
            shape=[x.shape[-1], x.shape[-1]],
            dtype=tf.float32,
            initializer=tf.orthogonal_initializer())
        bias = tf.get_variable(
            name='bias',
            shape=[1],
            dtype=tf.float32,
            initializer=tf.zeros_initializer())
        if block_type == 'self':
            z = tf.einsum('bmik,kj->bmij', y, W) + bias
        else:
            z = tf.einsum('bmnik,kj->bmnij', y, W) + bias
    return z

def layer_norm_debug(x, axis = None, epsilon=1e-6):
    '''Add layer normalization.

    Args:
        x: a tensor
        axis: the dimensions to normalize

    Returns:
        a tensor the same shape as x.

    Raises:
    '''
    if axis is None:
        axis = [-1]
    shape = [x.shape[i] for i in axis]

    scale = tf.get_variable(
        name='scale',
        shape=shape,
        dtype=tf.float32,
        initializer=tf.ones_initializer())
    bias = tf.get_variable(
        name='bias',
        shape=shape,
        dtype=tf.float32,
        initializer=tf.zeros_initializer())

    mean = tf.reduce_mean(x, axis=axis, keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=axis, keepdims=True)
    norm = (x-mean) * tf.rsqrt(variance + epsilon)
    return scale * norm + bias

def CNN_3d(x, out_channels_0, out_channels_1, add_relu=True):
    '''Add a 3d convlution layer with relu and max pooling layer.

    Args:
        x: a tensor with shape [batch, in_depth, in_height, in_width, in_channels]
        out_channels: a number
        filter_size: a number
        pooling_size: a number

    Returns:
        a flattened tensor with shape [batch, num_features]

    Raises:
    '''
    in_channels = x.shape[-1]
    weights_0 = tf.get_variable(
        name='filter_0',
        shape=[3, 3, 3, in_channels, out_channels_0],
        dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.01, 0.01))
    bias_0 = tf.get_variable(
        name='bias_0',
        shape=[out_channels_0],
        dtype=tf.float32,
        initializer=tf.zeros_initializer())

    conv_0 = tf.nn.conv3d(x, weights_0, strides=[1, 1, 1, 1, 1], padding="SAME")
    # print('conv_0 shape: %s' %conv_0.shape)
    conv_0 = conv_0 + bias_0

    if add_relu:
        conv_0 = tf.nn.elu(conv_0)

    pooling_0 = tf.nn.max_pool3d(
        conv_0,
        ksize=[1, 3, 3, 3, 1],
        strides=[1, 3, 3, 3, 1],
        padding="SAME")
    # print('pooling_0 shape: %s' %pooling_0.shape)

    #layer_1
    weights_1 = tf.get_variable(
        name='filter_1',
        shape=[3, 3, 3, out_channels_0, out_channels_1],
        dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.01, 0.01))
    bias_1 = tf.get_variable(
        name='bias_1',
        shape=[out_channels_1],
        dtype=tf.float32,
        initializer=tf.zeros_initializer())

    conv_1 = tf.nn.conv3d(pooling_0, weights_1, strides=[1, 1, 1, 1, 1], padding="SAME")
    # print('conv_1 shape: %s' %conv_1.shape)
    conv_1 = conv_1 + bias_1

    if add_relu:
        conv_1 = tf.nn.elu(conv_1)

    pooling_1 = tf.nn.max_pool3d(
        conv_1,
        ksize=[1, 3, 3, 3, 1],
        strides=[1, 3, 3, 3, 1],
        padding="SAME")
    # print('pooling_1 shape: %s' %pooling_1.shape)

    return tf.contrib.layers.flatten(pooling_1)