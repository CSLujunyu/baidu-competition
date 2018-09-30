import sys
import os
import time

import pickle
import tensorflow as tf
import numpy as np

import subtask4.evaluation as eva
from subtask4.data_generator import DataGenerator


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
            test_data = dg.test_data_generator(batch_index)
            feed = {
                _model.turns: test_data['turns'],
                _model.turn_num: test_data['turn_num'],
                _model.turn_len: test_data['turn_len'],
                _model.response: test_data['response'],
                _model.response_len: test_data['response_len'],
                _model.keep_rate: 1.0
            }

            scores= sess.run(_model.de_logits,feed_dict=feed)

            for i in range(conf["batch_size"]):
                for j in range(conf['options_num']):
                    test_score_file.write(
                        str(test_data['example_id'][i]) + '\t' +
                        str(test_data['candidate_id'][i][j]) + '\t' +
                        str(scores[i][j]) +'\n')
        test_score_file.close()

        # write evaluation result
        # test_result = eva.evaluate(test_score_file_path)
        # test_result_file_path = conf["save_path"] + "test_result.txt"
        # with open(test_result_file_path, 'w') as out_file:
        #     for p_at in test_result:
        #         out_file.write(str(p_at) + '\n')
        # print('finish test evaluation')
        # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
