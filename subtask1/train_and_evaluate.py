import os
import time

import tensorflow as tf
import numpy as np

import subtask1.evaluation as eva
from subtask1.data_generator import DataGenerator


def train(conf, _model):
    
    if conf['rand_seed'] is not None:
        np.random.seed(conf['rand_seed'])

    if not os.path.exists(conf['save_path']):
        os.makedirs(conf['save_path'])

    # config display
    print('configurations: %s' % conf)

    # Data Generate
    dg = DataGenerator(conf)
    print('Train data size: ', dg.train_data_size)
    print('Dev data size: ', dg.dev_data_size)
    print('Test data size: ', dg.test_data_size)

    # refine conf
    train_batch_num = int(dg.train_data_size / conf["batch_size"])
    val_batch_num = int(dg.dev_data_size / conf["batch_size"])

    conf["train_steps"] = conf["num_scan_data"] * train_batch_num
    conf["save_step"] = int(max(1, train_batch_num / 10))
    conf["print_step"] = int(max(1, train_batch_num / 100))

    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),' : Build graph')
    _graph = _model.build_graph()
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), ' : Build graph sucess')


    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(graph=_graph, config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init,feed_dict={_model.table:dg.table})
        if conf["init_model"]:
            _model.saver.restore(sess, conf["init_model"])
            print("sucess init %s" %conf["init_model"])

        average_loss = 0.0
        batch_index = 0
        step = 0
        best_result = [0, 0, 0, 0, 0, 0, 0]

        for step_i in range(conf["num_scan_data"]):
            for batch_index in range(train_batch_num):
                data = dg.train_data_generator(batch_index)
                feed = {
                    _model.turns: data['turns'],
                    _model.turn_num: data['turn_num'],
                    _model.turn_len: data['turn_len'],
                    _model.response: data['response'],
                    _model.response_len: data['response_len'],
                    _model.label: data['label'],
                    _model.keep_rate:conf['drop_rate'],
                    _model.is_training:True
                }
                batch_index = (batch_index + 1) % train_batch_num

                _, curr_loss, check = sess.run([_model.opt, _model.de_loss, _model.check], feed_dict = feed)


                average_loss += curr_loss

                step += 1

                if step % conf["print_step"] == 0 and step > 0:
                    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),
                          " processed: [" + str(step * 1.0 / train_batch_num) +
                          "] loss: [" + str(average_loss / conf["print_step"]) + "]")
                    average_loss = 0


                if step % conf["save_step"] == 0 and step > 0:
                    index = step / conf['save_step']
                    dev_score_file_path = conf['save_path'] + 'score.' + str(index)
                    dev_score_file = open(dev_score_file_path, 'w')
                    print(time.strftime(' %Y-%m-%d %H:%M:%S',time.localtime(time.time())), '  Save step: %s' %index)

                    # caculate dev score
                    for batch_index in range(val_batch_num):
                        data = dg.dev_data_generator(batch_index)
                        feed = {
                            _model.turns: data['turns'],
                            _model.turn_num: data['turn_num'],
                            _model.turn_len: data['turn_len'],
                            _model.response: data['response'],
                            _model.response_len: data['response_len'],
                            _model.label: data['label'],
                            _model.keep_rate: 1.0,
                            _model.is_training:False
                        }

                        scores = sess.run(_model.de_logits, feed_dict = feed)

                        for i in range(conf["batch_size"]):
                            for j in range(conf['options_num']):
                                dev_score_file.write(
                                    str(data['example_id'][i]) + '\t' +
                                    str(data['candidate_id'][i][j]) + '\t' +
                                    str(scores[i][j]) + '\t' +
                                    str(data['label'][i][j]) + '\n')
                    dev_score_file.close()

                    #write evaluation result
                    dev_result = eva.evaluate(dev_score_file_path)
                    dev_result_file_path = conf["save_path"] + "result." + str(index)
                    with open(dev_result_file_path, 'w') as out_file:
                        for p_at in dev_result:
                            out_file.write(str(p_at) + '\n')
                    print('finish dev evaluation')
                    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

                    if dev_result[-1]> best_result[-1]:
                        best_result = dev_result
                        print('best result: ',best_result)
                        _save_path = _model.saver.save(sess, conf["save_path"] + "model.ckpt." + str(step / conf["save_step"]))
                        print("succ saving model in " + _save_path)
                        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
                    
                

