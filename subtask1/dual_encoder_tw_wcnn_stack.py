import tensorflow as tf
from subtask1.layers import CNN_3d_3bn

def dual_encoder_wcnn_stack_model(
        graph,
        config,
        context_embedded,
        context_len,
        turns_num,
        utterances_embedded,
        utterances_len,
        keep_rate,
        is_training
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
    # all_turn_output.shape = (batch_size, max_turn_num, max_turn_len, n_layers, emb_size*2)
    all_turn_output = Context_Encoder(graph, config, context_embedded, context_len)
    # all_candidate_output.shape = (batch_size, options_num, max_turn_len, n_layers, emb_size*2)
    all_candidate_output = Utterance_Encoder(graph, config, utterances_embedded, utterances_len)

    ## word attention
    ## w_attention.shape = (batch_size, max_turn_num, max_turn_len, max_turn_len, n_layers, options_num)
    with tf.variable_scope('w_CNN', reuse=tf.AUTO_REUSE):
        w_attention = tf.einsum('bijlk,bmnlk->bijnlm',all_turn_output,all_candidate_output) / tf.sqrt(float(config['emb_size']))
        check.append(w_attention)
        w_attention = tf.unstack(w_attention, axis=-1)
        w_attention_fino = []
        for t in w_attention:
            s = CNN_3d_3bn(t,config['cnn_channel'], kernel_size=config['kernel_size'], is_training=is_training)
            w_attention_fino.append(s)

        # Dot product between generated response and actual response
        # c * r logits.shape = (batch_size, options)
        # w_attention_fino.shape = (batch_size, options_num)
        logits = tf.concat(w_attention_fino, axis=-1)
        check.append(logits)

        # Apply sigmoid to convert logits to probabilities
        # probs.shape = (batch_size, options)
        probs = tf.nn.softmax(logits)


    return probs, logits, 0 , check

def Context_Encoder(graph, config, context_embedded, context_len):

    context_embedded = tf.reshape(context_embedded, shape=[-1,config['max_turn_len'],config['emb_size']])
    context_len = tf.reshape(context_len,shape=[-1])

    # Build the Context Encoder RNN
    all_turn_output = [context_embedded]

    for n in range(config['n_layers']):
        with tf.variable_scope("encoder-context-" + str(n),reuse=tf.AUTO_REUSE):
            # We use an LSTM Cell
            cell_fw = tf.nn.rnn_cell.GRUCell(
                config['rnn_dim']/2,
                kernel_initializer=tf.orthogonal_initializer()
            )
            cell_bw = tf.nn.rnn_cell.GRUCell(
                config['rnn_dim']/2,
                kernel_initializer=tf.orthogonal_initializer()
            )
            # Run context through the RNN
            # context_encoded_outputs.shape = (batch_size, sentence_len, lstm_cell)
            # context_encoded.shape = (batch_size, lstm_cell)
            context_encoded_outputs, context_encoded = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs= all_turn_output[-1],
                sequence_length=context_len,
                dtype=tf.float32
            )
            all_turn_output.append(tf.concat(context_encoded_outputs, axis=-1))
    output_shape = [config['batch_size'],config['max_turn_num'],
                    config['max_turn_len'],config['n_layers']+1,
                    config['emb_size']]
    all_turn_output = tf.reshape(tf.stack(all_turn_output, axis=-2), shape=output_shape)

    return all_turn_output

def Utterance_Encoder(graph, config, utterances_embedded, utterances_len):


    # Build the Candidate Encoder RNN
    utterances_embedded = tf.reshape(utterances_embedded, shape=[-1, config['max_turn_len'], config['emb_size']])
    utterances_len = tf.reshape(utterances_len, shape=[-1])
    # Build the Candidate Encoder RNN
    all_candidate_output = [utterances_embedded]

    for n in range(config['n_layers']):
        with tf.variable_scope("encoder-candidate-" + str(n), reuse=tf.AUTO_REUSE):
            # We use an GRU Cell
            cell_fw = tf.nn.rnn_cell.GRUCell(
                config['rnn_dim']/2,
                kernel_initializer=tf.orthogonal_initializer()
            )
            cell_bw = tf.nn.rnn_cell.GRUCell(
                config['rnn_dim']/2,
                kernel_initializer=tf.orthogonal_initializer()
            )
            # Run candidate through the RNN
            # candidate_encoded_outputs.shape = (batch_size, sentence_len, lstm_cell)
            # candidate_encoded.shape = (batch_size, lstm_cell)
            candidate_encoded_outputs, candidate_encoded = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs= all_candidate_output[-1],
                sequence_length=utterances_len,
                dtype=tf.float32
            )
            all_candidate_output.append(tf.concat(candidate_encoded_outputs, axis=-1))

    output_shape = [config['batch_size'], config['options_num'],
             config['max_turn_len'], config['n_layers'] + 1,
             config['emb_size']]
    all_candidate_output = tf.reshape(tf.stack(all_candidate_output, axis=-2), shape=output_shape)

    return all_candidate_output