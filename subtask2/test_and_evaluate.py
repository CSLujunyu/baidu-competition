import os
import time

import tensorflow as tf
import numpy as np

import subtask2.test_evaluation as eva
from subtask2.data_generator import DataGenerator
from subtask2.operations import sort


def test(conf, _model):
    
    if conf['rand_seed'] is not None:
        np.random.seed(conf['rand_seed'])

    if not os.path.exists(conf['save_path']):
        os.makedirs(conf['save_path'])

    # config display
    print('configurations: %s' % conf)

    # Data Generate
    dg = DataGenerator(conf)
    print('Dev data size: ', dg.dev_data_size)
    print('Test data size: ', dg.test_data_size)
    print('Response pool size: ', dg.response_pool_size)

    # refine conf
    val_batch_num = int(dg.dev_data_size / conf["batch_size"])

    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),' : Build graph')
    _graph = _model.build_graph()
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), ' : Build graph sucess')


    with tf.Session(graph=_graph) as sess:
        init = tf.global_variables_initializer()
        sess.run(init,feed_dict={_model.table:dg.table})
        if conf["init_model"]:
            model_path = tf.train.latest_checkpoint(conf["init_model"])
            _model.saver.restore(sess, model_path)
            print("sucess init %s" %conf["init_model"])

        average_loss = 0.0
        batch_index = 0
        step = 0
        best_result = [0, 0, 0, 0, 0, 0, 0]

        dev_score_file_path = conf['save_path'] + 'score'
        dev_score_file = open(dev_score_file_path, 'w')
        # caculate dev score
        for batch_index in range(val_batch_num):
            data = dg.dev_data_generator(batch_index)
            feed = {
                _model.turns: data['turns'],
                _model.turn_num: data['turn_num'],
                _model.turn_len: data['turn_len'],
                _model.label: data['y'],
                _model.keep_rate: 1.0,
                _model.is_training: False
            }

            r_score = []
            r_label = []
            print(time.strftime(' %Y-%m-%d %H:%M:%S', time.localtime(time.time())), '  Evaluate batch %d'%batch_index)
            for batch_option_id in range(conf['options_batch_num']):
                response_data = dg.response_pool_generator(batch_option_id)

                feed.update({_model.response:np.tile(response_data['response'],(conf['batch_size'],1,1)),
                             _model.response_len:np.tile(response_data['response_len'],(conf['batch_size'],1))})
                curr_logits = sess.run( _model.de_logits, feed_dict = feed)
                r_score.append(curr_logits)
                r_label.append(response_data['candidate_id'])

            ## r_score.shape = (batch_size, 120000)
            r_score = np.concatenate(r_score, axis=1)
            ## r_label.shape = (120000)
            r_label = np.concatenate(r_label, axis=0)

            for i in range(conf["batch_size"]):
                scores = sort(zip(r_score[i, :], r_label))
                for j in range(conf['options_num']):
                    dev_score_file.write(
                        str(data['example_id'][i]) + '\t' +
                        str(scores[j][1]) + '\t' +
                        str(scores[j][0]) + '\t' +
                        str(data['label'][i]) + '\n')

        dev_score_file.close()

        #write evaluation result
        dev_result = eva.evaluate(dev_score_file_path)
        dev_result_file_path = conf["save_path"] + "result"
        with open(dev_result_file_path, 'w') as out_file:
            for p_at in dev_result:
                out_file.write(str(p_at) + '\n')
        print('finish dev evaluation')
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

        if dev_result[-1]> best_result[-1]:
            best_result = dev_result
            print('best result: ',best_result)
                    
                

