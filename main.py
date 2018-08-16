import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import models.net as net
import bin.train_and_evaluate as train

# configure

ubuntu_conf = {
    'train_context':'/hdd/lujunyu/dataset/DSTC7_track1/baseline_DAM/subtask1/u_train_context.pkl',
    'dev_context':'/hdd/lujunyu/dataset/DSTC7_track1/baseline_DAM/subtask1/u_dev_context.pkl',
    'train_response':'/hdd/lujunyu/dataset/DSTC7_track1/baseline_DAM/subtask1/u_train_respone.pkl',
    'dev_response':'/hdd/lujunyu/dataset/DSTC7_track1/baseline_DAM/subtask1/u_dev_respone.pkl',
    'process_train_data':'/hdd/lujunyu/dataset/DSTC7_track1/baseline_DAM/subtask1/u_train_processs.pkl',
    'process_dev_data':'/hdd/lujunyu/dataset/DSTC7_track1/baseline_DAM/subtask1/u_dev_process.pkl',


    "save_path": "./output/conv/temp/",
    "word_emb_init": "/hdd/lujunyu/dataset/DAM_data/DSTC7_1/embed4data_official",


    "init_model": None, #should be set for test

    "rand_seed": None,

    "drop_dense": None,
    "drop_attention": None,

    "is_mask": True,
    "is_layer_norm": True,
    "is_positional": False,

    "stack_num": 5,
    "attention_type": "dot",

    "learning_rate": 1e-3,
    "vocab_size": 108290,    #434513 for DAM  ， 128205 for dstc
    "emb_size": 300,
    "batch_size": 100, #200 for test

    "max_turn_num": 10,
    "max_turn_len": 60,

    "max_to_keep": 1,
    "num_scan_data": 15,
    "_EOS_": 36, #1455 for DSTC7, 28270 for DAM_source, #1 for douban data  , 6 for advising
    "final_n_class": 1,

    "rnn_dim":256,
    'options_num':100,
    'conv_filter_num':50
}


model = net.Net(ubuntu_conf)
train.train(ubuntu_conf, model)


#test and evaluation, init_model in conf should be set
#test.test(conf, model)

