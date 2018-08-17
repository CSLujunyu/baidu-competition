import utils.parallel_attention_block as pab
import utils.parallel_operations as pop
import tensorflow as tf

def self_cross_attention(
        config,
        context_embedded,
        context_len,
        candidate_embedded,
        candidate_len
        ):
    """

    :param config:
    :param context_embedded: shape = (batch_size, max_turn_num, max_turn_len, emb_size)
    :param context_len: shape = (batch_size, max_turn_num )
    :param candidate_embedded: shape = (batch_size, options_num, max_turn_len, emb_size)
    :param candidate_len: shape = (batch_size, options_num)
    :return:
    """

    # feature is a list of tensors which shape is (batch_size, max_turn_num, options_num, max_turn_len, max_turn_len)
    feature = [tf.einsum('bimn,bjmn->bij',context_embedded,candidate_embedded)]
    C_stack = [context_embedded]
    R_stack = [candidate_embedded]
    CR_stack = []
    RC_stack = []
    self_C = context_embedded
    self_R = candidate_embedded

    for i in range(config['stack_num']):
            with tf.variable_scope('self_stack_'+str(i), reuse=tf.AUTO_REUSE):
                # self_C.shape = (batch_size, max_turn_num, max_turn_len, emb_size)
                self_C = pab.self_block(Q=self_C, K=self_C, V=self_C, Q_lengths=context_len, K_lengths=context_len)
                # self_R.shape = (batch_size, options_num, max_turn_len, emb_size)
                self_R = pab.self_block(Q=self_R, K= self_R, V=self_R, Q_lengths=candidate_len, K_lengths=candidate_len)
            C_stack.append(self_C)
            R_stack.append(self_R)

            with tf.variable_scope('C_at_R_stack_'+str(i),tf.AUTO_REUSE):
                # cross_CR.shape = (batch_size, max_turn_num, options_num, max_turn_len, emb_size)
                cross_CR = pab.cross_block(Q=C_stack[i], K=R_stack[i], V=R_stack[i], Q_lengths=context_len, K_lengths=candidate_len)
            with tf.variable_scope('R_at_C_stack_',str(i),reuse=tf.AUTO_REUSE):
                # cross_RC.shape = (batch_size, options_num, max_turn_num, max_turn_len, emb_size)
                cross_RC = pab.cross_block(Q=R_stack[i], K=C_stack[i], V=C_stack[i], Q_lengths=candidate_len, K_lengths=context_len)
            CR_stack.append(cross_CR)
            RC_stack.append(cross_RC)

    CR_stack.append(pab.cross_block(Q=C_stack[-1], K=R_stack[-1], V=R_stack[-1], Q_lengths=context_len, K_lengths=candidate_len))
    RC_stack.append(pab.cross_block(Q=R_stack[-1], K=C_stack[-1], V=C_stack[-1], Q_lengths=candidate_len, K_lengths=context_len))
    # self_feature.shape = (batch_size, options_num, max_turn_num, max_turn_len, max_turn_len, stack_num)
    self_F = tf.einsum('bijks,bmnks->bimjns',tf.stack(R_stack,axis=-1),tf.stack(C_stack,axis=-1)) / tf.sqrt(200.0)
    # cross_feature.shape = (batch_size, options_num, max_turn_num, max_turn_len, max_turn_len, stack_num)
    cross_F = tf.einsum('bijkls,bjizls->bijkzs', tf.stack(RC_stack,axis=-1), tf.stack(CR_stack,axis=-1)) / tf.sqrt(200.0)

    # feature.shape = (batch_size * options_num, max_turn_num, max_turn_len, max_turn_len, stack_num)
    feature = tf.reshape(tf.concat([self_F,cross_F],axis=-1), shape=[-1, self_F.shape[2], self_F.shape[3], self_F.shape[4], self_F.shape[5]+cross_F.shape[5]])
    with tf.variable_scope('cnn_aggregation'):
        final_info = pop.CNN_3d(feature,32,16)
    with tf.variable_scope('linear'):
        W = tf.get_variable(
            name='weights',
            shape=[final_info.shape[-1], 1],
            initializer=tf.orthogonal_initializer())
        bias = tf.get_variable(
            name='bias',
            shape=[1],
            initializer=tf.zeros_initializer())

        logits = tf.reshape(tf.matmul(final_info, W) + bias, [-1,self_F.shape[1]])

    probs = tf.nn.softmax(logits)

    return probs, logits
