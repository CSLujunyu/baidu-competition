import tensorflow as tf
from subtask3.layers import hinge_loss


class Net(object):
    def __init__(self, conf):
        self._graph = tf.Graph()
        self._conf = conf
        if conf['Model'] == 'WCNN_S_A':
            from models.dual_encoder_tw_wcnn_stack_beifen import dual_encoder_wcnn_stack_model as model
        elif conf['Model'] == 'WCNN_S_BN':
            from subtask4.dual_encoder_tw_wcnn_stack import dual_encoder_wcnn_stack_model as model
        else:
            model = None
        self.Model = model


    def build_graph(self):
        with self._graph.as_default():
            if self._conf['rand_seed'] is not None:
                rand_seed = self._conf['rand_seed']
                tf.set_random_seed(rand_seed)
                print('set tf random seed: %s' %self._conf['rand_seed'])


            #define placehloders
            self.turns = tf.placeholder(
                tf.int32,
                shape=[self._conf['batch_size'], self._conf['max_turn_num'], self._conf['max_turn_len']])

            self.turn_num = tf.placeholder(
                tf.int32,
                shape=[self._conf['batch_size']])

            self.turn_len = tf.placeholder(
                tf.int32,
                shape=[self._conf['batch_size'], self._conf['max_turn_num']])
    
            self.response = tf.placeholder(
                tf.int32, 
                shape=[self._conf['batch_size'], self._conf['options_num'], self._conf['max_turn_len']])

            self.response_len = tf.placeholder(
                tf.int32, 
                shape=[self._conf['batch_size'], self._conf['options_num']])

            self.label = tf.placeholder(
                tf.float32,
                shape=[self._conf['batch_size'], self._conf['options_num']])

            self.table = tf.placeholder(
                tf.float32,
                shape=[self._conf['vocab_size'], self._conf['emb_size']])

            self.keep_rate = tf.placeholder(
                tf.float32,
                shape=[])

            self.is_training = tf.placeholder(
                tf.bool,
                shape=[])

            # convert table to variable
            with tf.device('/cpu:0'):
                self.table_v = tf.Variable(self.table, name='table')

            # C.shape = (batch_size, max_turn_num, max_turn_len, embed_dim)
            self.turns_embed = tf.nn.embedding_lookup(self.table_v, self.turns)
            # R.shape = (batch_size, option_num，max_turn_len, embed_dim)
            self.response_embed = tf.nn.embedding_lookup(self.table_v, self.response)

            # Dual encoder model
            self.probs, self.de_logits, self.reg, self.check = self.Model(self._graph, self._conf, self.turns_embed,
                                                                          self.turn_len, self.turn_num ,
                                                                          self.response_embed, self.response_len,
                                                                          self.keep_rate,self.is_training
                                                                          )

            # Calculate cross-entropy loss
            tv = tf.trainable_variables()
            for v in tv:
                print(v)
            regularization_cost = 0
            # for v in tv:
            #     if v.name.startswith('w_CNN') and v.name.find('bias') == -1:
            #         regularization_cost += tf.nn.l2_loss(v)
            self.de_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.de_logits,
                                                                                  labels=self.label),
                                          name='de_loss') +  self._conf['reg_rate'] * regularization_cost


            self.opt = tf.train.AdamOptimizer(self._conf['learning_rate']).minimize(self.de_loss)

            self.saver = tf.train.Saver(max_to_keep=self._conf["max_to_keep"])

        return self._graph


