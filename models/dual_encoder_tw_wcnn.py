import tensorflow as tf
from utils.layers import CNN_3d

def dual_encoder_wcnn_model(
        graph,
        config,
        context_embedded,
        context_len,
        turns_num,
        utterances_embedded,
        utterances_len
        ):
    """

    :param config:
    :param context_embedded: shape = (batch_size, max_turn_num, sentence_len, embed_dim)
    :param context_len: shape = (batch_size, max_turn_num )
    :param turns_num: shape = (batch_size,)
    :param utterances_embedded: shape = (batch_size, option, sentence_len, embed_dim)
    :param utterances_len: shape = (batch_size, options)
    :return:
    """

    check = []

    # context_embedded.shape = (max_turn_num, batch_size, sentence_len, embed_dim)
    context_embedded = tf.transpose(context_embedded, perm=[1, 0, 2, 3])
    # context_len.shape = (max_turn_num, batch_size)
    context_len = tf.transpose(context_len, perm=[1, 0])



    # Build the Context Encoder RNN
    with tf.variable_scope("encoder-context"):
        # We use an LSTM Cell
        fw_cell_context = tf.nn.rnn_cell.GRUCell(
            config['rnn_dim'],
            kernel_initializer=tf.orthogonal_initializer
        )

        bw_cell_context = tf.nn.rnn_cell.GRUCell(
            config['rnn_dim'],
            kernel_initializer=tf.orthogonal_initializer
        )

        # Run context through the RNN
        # context_encoded_outputs.shape = (batch_size, sentence_len, lstm_cell)
        # context_encoded.shape = (batch_size, lstm_cell)
        all_turn_output = []
        for i in range(config['max_turn_num']):
            context_encoded_outputs, context_encoded = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_cell_context,
                cell_bw=bw_cell_context,
                inputs=context_embedded[i],
                sequence_length=context_len[i],
                dtype=tf.float32)
            all_turn_output.append(tf.concat(context_encoded_outputs, axis=-1))
        # all_turn_output.shape = (batch_size, max_turn_num, max_turn_len, lstm_cell*2)
        all_turn_output = tf.stack(all_turn_output, axis = 1)

        # regularization
        reg_context = tf.contrib.layers.l2_regularizer(config['reg_rate'])(graph.get_tensor_by_name(
            'encoder-context/bidirectional_rnn/fw/gru_cell/candidate/kernel:0')) + tf.contrib.layers.l2_regularizer(
            config['reg_rate'])(graph.get_tensor_by_name('encoder-context/bidirectional_rnn/bw/gru_cell/candidate/kernel:0'))

    # Build the Utterance Encoder RNN
    with tf.variable_scope("encoder-candidate"):
        # We use an LSTM Cell
        fw_cell_utterance = tf.nn.rnn_cell.GRUCell(
            config['rnn_dim'],
            kernel_initializer=tf.orthogonal_initializer
        )
        bw_cell_utterance = tf.nn.rnn_cell.GRUCell(
            config['rnn_dim'],
            kernel_initializer=tf.orthogonal_initializer
        )
        # Run all utterances through the RNN batch by batch
        all_candidate_output = []
        for i in range(config['batch_size']):
            # utterances_embedded[:,i].shape = (options, sentence_len, embed_dim)
            # temp_outputs.shape = (options, sentence_len, lstm_cell)
            # temp_states.shape = (options, lstm_cell)
            # temp_states[1].shape = (options, lstm_cell) , 0 is cell state, 1 is hidden state
            temp_outputs, temp_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_cell_utterance,
                cell_bw=bw_cell_utterance,
                inputs=utterances_embedded[i],
                sequence_length=utterances_len[i],
                dtype=tf.float32)
            all_candidate_output.append(tf.concat(temp_outputs, axis=-1))

        # all_candidate_output.shape = (batch_size, options, max_turn_len, lstm_cell*2)
        all_candidate_output = tf.stack(all_candidate_output, axis= 0)

        reg_candidate = tf.contrib.layers.l2_regularizer(config['reg_rate'])(graph.get_tensor_by_name(
            'encoder-candidate/bidirectional_rnn/fw/gru_cell/candidate/kernel:0')) + tf.contrib.layers.l2_regularizer(
            config['reg_rate'])(graph.get_tensor_by_name('encoder-candidate/bidirectional_rnn/bw/gru_cell/candidate/kernel:0'))

    ## word attention
    ## w_attention.shape = (batch_size, max_turn_num, max_turn_len, max_turn_len, 1, options_num)
    with tf.variable_scope('w_CNN', reuse=tf.AUTO_REUSE):
        w_attention = tf.expand_dims(tf.einsum('bijk,bmnk->bijnm',all_turn_output,all_candidate_output), axis=-2)
        check.append(w_attention)
        w_attention = tf.unstack(w_attention, axis=-1)
        w_attention_fino = []
        for t in w_attention:
            s = CNN_3d(t,10,5)
            W = tf.get_variable(
                name='weights',
                shape=[s.shape[-1], 1],
                initializer=tf.orthogonal_initializer())
            bias = tf.get_variable(
                name='bias',
                shape=[1],
                initializer=tf.zeros_initializer())

            s = tf.reshape(tf.matmul(s, W) + bias, [-1])
            w_attention_fino.append(s)


    with tf.variable_scope("prediction"):

        # Dot product between generated response and actual response
        # c * r logits.shape = (batch_size, options)
        # w_attention_fino.shape = (batch_size, options_num)
        logits = tf.stack(w_attention_fino, axis=-1)
        check.append(logits)

        # Apply sigmoid to convert logits to probabilities
        # probs.shape = (batch_size, options)
        probs = tf.nn.softmax(logits)


    return probs, logits, reg_context + reg_candidate, check
