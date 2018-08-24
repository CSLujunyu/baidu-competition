import tensorflow as tf

def dual_encoder_model(
        graph,
        config,
        context_embedded,
        context_num,
        context_len,
        utterances_embedded,
        utterances_len
        ):
    """

    :param config:
    :param context_embedded: shape = (batch_size, max_turn_num, sentence_len, embed_dim)
    :param context_len: shape = (batch_size, max_turn_num )
    :param utterances_embedded: shape = (batch_size, option, sentence_len, embed_dim)
    :param utterances_len: shape = (batch_size, options)
    :return:
    """

    # context_embedded.shape = (max_turn_num, batch_size, sentence_len, embed_dim)
    context_embedded = tf.transpose(context_embedded, perm=[1, 0, 2, 3])
    # context_len.shape = (max_turn_num, batch_size)
    context_len = tf.transpose(context_len, perm=[1, 0])



    # Build the Context Encoder RNN
    with tf.variable_scope("encoder-context") as vs:
        # We use an LSTM Cell
        fw_cell_context = tf.nn.rnn_cell.LSTMCell(
            config['rnn_dim'],
            forget_bias=2.0,
            use_peepholes=True,
            state_is_tuple=True)

        bw_cell_context = tf.nn.rnn_cell.LSTMCell(
            config['rnn_dim'],
            forget_bias=2.0,
            use_peepholes=True,
            state_is_tuple=True)

        # Run context through the RNN
        # context_encoded_outputs.shape = (batch_size, sentence_len, lstm_cell)
        # context_encoded.shape = (batch_size, lstm_cell)
        all_turn_encoded = []
        for i in range(config['max_turn_num']):
            context_encoded_outputs, context_encoded = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_cell_context,
                cell_bw=bw_cell_context,
                inputs=context_embedded[i],
                sequence_length=context_len[i],
                dtype=tf.float32)
            context_encoded = tf.concat([context_encoded[0][1],context_encoded[1][1]], axis=1)
            all_turn_encoded.append(context_encoded)
        # all_turn_encoded.shape = (batch_size, max_turn_len, lstm_cell*2)
        all_turn_encoded = tf.stack(all_turn_encoded, axis= 1)

        # regularization
        reg_context = tf.contrib.layers.l2_regularizer(config['reg_rate'])(graph.get_tensor_by_name(
            'encoder-context/bidirectional_rnn/fw/lstm_cell/kernel:0')) + tf.contrib.layers.l2_regularizer(
            config['reg_rate'])(graph.get_tensor_by_name('encoder-context/bidirectional_rnn/bw/lstm_cell/kernel:0'))

    # Build the Utterance Encoder RNN
    with tf.variable_scope("encoder-candidate") as vs:
        # We use an LSTM Cell
        fw_cell_utterance = tf.nn.rnn_cell.LSTMCell(
            config['rnn_dim'],
            forget_bias=2.0,
            use_peepholes=True,
            state_is_tuple=True)
        bw_cell_utterance = tf.nn.rnn_cell.LSTMCell(
            config['rnn_dim'],
            forget_bias=2.0,
            use_peepholes=True,
            state_is_tuple=True)
        # Run all utterances through the RNN batch by batch
        all_candidate_encoded = []
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
            temp_states = tf.concat([temp_states[0][1],temp_states[1][1]], axis=1)
            all_candidate_encoded.append(temp_states) # since it's a tuple, use the hidden states

        # all_candidate_encoded.shape = (batch_size, options, lstm_cell*2)
        all_candidate_encoded = tf.stack(all_candidate_encoded, axis=0)

        reg_candidate = tf.contrib.layers.l2_regularizer(config['reg_rate'])(graph.get_tensor_by_name(
            'encoder-candidate/bidirectional_rnn/fw/lstm_cell/kernel:0')) + tf.contrib.layers.l2_regularizer(
            config['reg_rate'])(graph.get_tensor_by_name('encoder-candidate/bidirectional_rnn/bw/lstm_cell/kernel:0'))

    with tf.variable_scope("prediction"):

        # all_candidate_encoded.shape = (batch_size, lstm_cell*2, options )
        all_candidate_encoded = tf.transpose(all_candidate_encoded, perm=[0, 2, 1])

        # Dot product between generated response and actual response
        # c * r logits.shape = (batch_size, max_turn_num, options)
        logits = tf.matmul(all_turn_encoded, all_candidate_encoded)

        # # logits.shape = (batch_size, options)
        logits = tf.reduce_sum(logits, axis= 1)

        # Apply sigmoid to convert logits to probabilities
        # probs.shape = (batch_size, options)
        probs = tf.nn.softmax(logits)




    return probs, logits, reg_context + reg_candidate
