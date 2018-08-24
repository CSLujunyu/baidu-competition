import tensorflow as tf

class Net(object):
    def __init__(self, conf):
        self._graph = tf.Graph()
        self._conf = conf
        if conf['Model'] == 'BiLSTM':
            from ELMo_models.dual_encoder import dual_encoder_model as model
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
                tf.float32,
                shape=[self._conf['batch_size'], self._conf['max_turn_num'], self._conf['elmo_layer'], self._conf['max_turn_len'], self._conf['emb_size']])

            self.turn_num = tf.placeholder(
                tf.int32,
                shape=[self._conf['batch_size']])

            self.turn_len = tf.placeholder(
                tf.int32,
                shape=[self._conf['batch_size'], self._conf['max_turn_num']])
    
            self.response = tf.placeholder(
                tf.float32,
                shape=[self._conf['batch_size'], self._conf['options_num'], self._conf['elmo_layer'], self._conf['max_turn_len'], self._conf['emb_size']])

            self.response_len = tf.placeholder(
                tf.int32, 
                shape=[self._conf['batch_size'], self._conf['options_num']])

            self.label = tf.placeholder(
                tf.int32,
                shape=[self._conf['batch_size']])


            # ELMo task embedding
            with tf.variable_scope('elmo_task',reuse=tf.AUTO_REUSE):
                s_task = tf.get_variable(name='s_task',shape=[self._conf['elmo_layer']],dtype=tf.float32,initializer=tf.random_normal_initializer)
                r_task = tf.get_variable(name='r_task',shape=[],dtype=tf.float32,initializer=tf.random_normal_initializer)
                s_task = tf.nn.softmax(s_task)
                turns_emb = r_task * tf.einsum('bijmn,j->bimn', self.turns, s_task)
                response_emb = r_task * tf.einsum('bijmn,j->bimn', self.response, s_task)


            # Dual encoder model
            self.probs, self.de_logits, self.reg = self.Model(self._graph, self._conf, turns_emb, self.turn_num, self.turn_len, response_emb, self.response_len)

            # Calculate cross-entropy loss
            self.de_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.de_logits, labels=self.label), name='de_loss') + self.reg

            self.opt = tf.train.AdamOptimizer(self._conf['learning_rate']).minimize(self.de_loss)

            self.saver = tf.train.Saver(max_to_keep=self._conf["max_to_keep"])
    
        return self._graph

