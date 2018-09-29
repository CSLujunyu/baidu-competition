import sys
import os
import time

import pickle
import tensorflow as tf
import numpy as np

import utils.evaluation as eva
from utils.data_generator import DataGenerator


def test(conf, _model):

    if not os.path.exists(conf['save_path']):
        os.makedirs(conf['save_path'])

    # config display
    print('configurations: %s' % conf)

    # Data Generate
    dg = DataGenerator(conf)
    print('Test data size: ', dg.test_data_size)

    # refine conf
    test_batch_num = int(dg.test_data_size / conf["batch_size"])

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ' : Build graph')
    _graph = _model.build_graph()
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ' : Build graph sucess')

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(graph=_graph, config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init, feed_dict={_model.table: dg.table})
        if conf["init_model"]:
            model_path = tf.train.latest_checkpoint(conf["init_model"])
            _model.saver.restore(sess, model_path)
            print("sucess init %s" % conf["init_model"])

        test_score_file_path = conf['save_path'] + 'test_score.txt'
        test_score_file = open(test_score_file_path, 'w')
        # caculate test score
        for batch_index in range(test_batch_num):
            print(batch_index)
            turns, turn_num, turn_len, response, response_len, label = dg.test_data_generator(
                batch_index)
            feed = {
                _model.turns: turns,
                _model.turn_num: turn_num,
                _model.turn_len: turn_len,
                _model.response: response,
                _model.response_len: response_len,
                _model.label: label,
                _model.keep_rate: 1.0
            }

            scores, check, keep = sess.run([_model.de_logits, _model.check, _model.keep_rate],
                                           feed_dict=feed)

            for i in range(conf["batch_size"]):
                for j in range(conf['options_num']):
                    if j == label[i]:
                        lab = 1
                    else:
                        lab = 0
                    test_score_file.write(
                        str(scores[i][j]) + '\t' +
                        str(lab) + '\n')
        test_score_file.close()

        # write evaluation result
        test_result = eva.evaluate(test_score_file_path)
        test_result_file_path = conf["save_path"] + "test_result.txt"
        with open(test_result_file_path, 'w') as out_file:
            for p_at in test_result:
                out_file.write(str(p_at) + '\n')
        print('finish test evaluation')
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
