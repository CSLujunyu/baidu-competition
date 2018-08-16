import tensorflow as tf
from models.dual_encoder import dual_encoder_model

class Net(object):
    def __init__(self, conf):
        self._graph = tf.Graph()
        self._conf = conf


    def build_graph(self):
        with self._graph.as_default():
            if self._conf['rand_seed'] is not None:
                rand_seed = self._conf['rand_seed']
                tf.set_random_seed(rand_seed)
                print('set tf random seed: %s' %self._conf['rand_seed'])


            #define placehloders
            self.turns = tf.placeholder(
                tf.int32,
                shape=[None, self._conf['max_turn_num'], self._conf['max_turn_len']])

            self.turn_num = tf.placeholder(
                tf.int32,
                shape=[None])

            self.turn_len = tf.placeholder(
                tf.int32,
                shape=[None, self._conf['max_turn_num']])
    
            self.response = tf.placeholder(
                tf.int32, 
                shape=[None, self._conf['options_num'], self._conf['max_turn_len']])

            self.response_len = tf.placeholder(
                tf.int32, 
                shape=[None, self._conf['options_num']])

            self.label = tf.placeholder(
                tf.int32,
                shape=[None])

            self.table = tf.placeholder(
                tf.float32,
                shape=[self._conf['vocab_size'], self._conf['emb_size']]
            )

            # convert table to variable
            self.table_v = tf.Variable(self.table, name='table')

            # C.shape = (batch_size, max_turn_num, max_turn_len, embed_dim)
            self.turns_embed = tf.nn.embedding_lookup(self.table_v, self.turns)
            # R.shape = (batch_size, option_num，max_turn_len, embed_dim)
            self.response_embed = tf.nn.embedding_lookup(self.table_v, self.response)

            # Dual encoder model
            self.probs, self.de_logits = dual_encoder_model(self._conf, self.turns_embed, self.turn_len, self.response_embed, self.response_len)

            # Calculate cross-entropy loss
            self.de_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.de_logits, labels=self.label), name='de_loss')

            self.opt = tf.train.AdamOptimizer(self._conf['learning_rate']).minimize(self.de_loss)

            self.saver = tf.train.Saver(max_to_keep=self._conf["max_to_keep"])
    
        return self._graph

