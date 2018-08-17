import tensorflow as tf
import utils.parallel_operations as pop

def self_block(
        Q, K, V,
        Q_lengths, K_lengths,
        is_layer_norm=True,
        is_mask=True, mask_value=-2 ** 32 + 1,
        drop_prob=None):
    """
    :param Q: shape = (batch_size, parallel_num, max_turn_len, emb_size)
    :param K: shape = (batch_size, parallel_num, max_turn_len, emb_size)
    :param V: shape = (batch_size, parallel_num, max_turn_len, emb_size)
    :param Q_lengths: shape = (batch_size, parallel_num)
    :param K_lengths: shape = (batch_size, parallel_num)
    :param attention_type:
    :param parallel_num:
    :param is_layer_norm:
    :param is_mask:
    :param mask_value:
    :param drop_prob:
    :return:
    """
    # att.shape = (batch_size, parallel_num, max_turn_len, emb_size)
    att = pop.attention(Q, K, V,
                    Q_lengths, K_lengths,
                    attention_type='dot',
                    is_mask=is_mask, mask_value=mask_value,
                    drop_prob=drop_prob,
                    block_type='self')
    if is_layer_norm:
        with tf.variable_scope('attention_layer_norm', reuse=tf.AUTO_REUSE):
            # y.shape = (batch_size, parallel_num, max_turn_len, emb_size)
            y = pop.layer_norm_debug(Q + att)
    else:
        y = Q + att

    z = pop.FFN(y,block_type='self')
    if is_layer_norm:
        with tf.variable_scope('FFN_layer_norm', reuse=tf.AUTO_REUSE):
            w = pop.layer_norm_debug(y + z)
    else:
        w = y + z
    # w.shape = (batch_size, max_turn_len, emb_size)
    return w

def cross_block(
        Q, K, V,
        Q_lengths, K_lengths,
        is_layer_norm=True,
        is_mask=True, mask_value=-2 ** 32 + 1,
        drop_prob=None):
    """
    :param Q: shape = (batch_size, parallel_num_q, max_turn_len, emb_size)
    :param K: shape = (batch_size, parallel_num_k, max_turn_len, emb_size)
    :param V: shape = (batch_size, parallel_num_v, max_turn_len, emb_size)
    :param Q_lengths: shape = (batch_size, parallel_num)
    :param K_lengths: shape = (batch_size, parallel_num)
    :param is_layer_norm:
    :param is_mask:
    :param mask_value:
    :param drop_prob:
    :return:
    """
    # att.shape = (batch_size, parallel_num_q, parallel_num_v, max_turn_len, emb_size)
    att = pop.attention(Q, K, V,
                    Q_lengths, K_lengths,
                    attention_type='dot',
                    is_mask=is_mask, mask_value=mask_value,
                    drop_prob=drop_prob,
                    block_type='cross')
    if is_layer_norm:
        with tf.variable_scope('attention_layer_norm', reuse=tf.AUTO_REUSE):
            # different from self block
            # Q.shape = (batch_size, parallel_num_q, max_turn_len, emb_size)
            # y.shape = (batch_size, parallel_num_v, max_turn_len, emb_size)
            Q = tf.tile(tf.expand_dims(Q,axis=2),multiples=[1,1,K.shape[1],1,1])
            y = pop.layer_norm_debug(Q + att)
    else:
        y = Q + att

    z = pop.FFN(y,block_type='cross')
    if is_layer_norm:
        with tf.variable_scope('FFN_layer_norm', reuse=tf.AUTO_REUSE):
            w = pop.layer_norm_debug(y + z)
    else:
        w = y + z
    # w.shape = (batch_size, parallel_num_q, parallel_num_v, max_turn_len, emb_size)
    return w
