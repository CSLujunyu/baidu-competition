import tensorflow as tf
from utils.self_cross_attention_block import self_cross_attention_block

def self_cross_attention(
        config,
        context_embedded,
        context_len,
        utterances_embedded,
        utterances_len
        ):
    """

    :param config:
    :param context_embedded: shape = (batch_size, max_turn_num, sentence_len, embed_dim)
    :param context_len: shape = (batch_size, max_turn_num )
    :param utterances_embedded: shape = (batch_size, options_num, sentence_len, embed_dim)
    :param utterances_len: shape = (batch_size, options_num)
    :return:
    """

    logits = []
    for op in range(config['options_num']):
        print('option number:',op)
        s = self_cross_attention_block(config,context_embedded,context_len,utterances_embedded[:,op],utterances_len[:,op])
        logits.append(s)

    # # logits.shape = (batch_size, options_num)
    logits = tf.stack(logits, axis=1)

    # Apply sigmoid to convert logits to probabilities
    # probs.shape = (batch_size, options_num)
    probs = tf.nn.softmax(logits)


    return probs, logits
