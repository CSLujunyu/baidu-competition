import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import models.net as net
import bin.train_and_evaluate as train

# configure

ubuntu_conf = {
    'train_context':'/hdd/lujunyu/dataset/DSTC7_track1/DAM/s1/train_ubuntu_1_context.pkl',
    'train_response':'/hdd/lujunyu/dataset/DSTC7_track1/DAM/s1/train_ubuntu_1_response.pkl',
    'dev_context': '/hdd/lujunyu/dataset/DSTC7_track1/DAM/s1/dev_ubuntu_1_context.pkl',
    'dev_response': '/hdd/lujunyu/dataset/DSTC7_track1/DAM/s1/dev_ubuntu_1_response.pkl',
    'test_context':'/hdd/lujunyu/dataset/DSTC7_track1/DAM/s1/catch_test/test_ubuntu_1_context.pkl',
    'test_response':'/hdd/lujunyu/dataset/DSTC7_track1/DAM/s1/catch_test/test_ubuntu_1_response.pkl',

    'process_train_data':'/hdd/lujunyu/dataset/DSTC7_track1/DAM/s1/DE_process_train.pkl',
    'process_dev_data':'/hdd/lujunyu/dataset/DSTC7_track1/DAM/s1/DE_process_dev.pkl',
    'process_test_data': '/hdd/lujunyu/dataset/DSTC7_track1/DAM/s1/catch_test/DE_process_test.pkl',

    "save_path": "./output/CNN/stack/",
    "word_emb_init": "/hdd/lujunyu/dataset/DSTC7_track1/DAM/s1/embed4data.pkl",


    "init_model": None, #should be set for test

    "rand_seed": None,
    "is_mask": True,
    "is_layer_norm": True,
    "is_positional": False,

    "stack_num": 3,
    "attention_type": "dot",

    "learning_rate": 1e-3,
    "reg_rate":3e-5,
    "drop_rate": 0.3,
    "vocab_size": 111695,    #434513 for DAM  ， 128205 for dstc
    "emb_size": 300,
    "batch_size": 10, #200 for test

    "max_turn_num": 9,
    "max_turn_len": 50,

    "max_to_keep": 1,
    "num_scan_data": 5,
    "_EOS_": 37, #1455 for DSTC7, 28270 for DAM_source, #1 for douban data  , 6 for advising
    "final_n_class": 1,

    "rnn_dim":300,
    'options_num':100,
    'conv_filter_num':50,

    'n_layers':3,

    'Model':'WCNN_S', ## [BiLSTM,GRU, BiLSTM_tw,CNN, LSTM_ATTENTION, DAM, DAM_p]
}


model = net.Net(ubuntu_conf)
train.train(ubuntu_conf, model)


#test and evaluation, init_model in conf should be set
#test.test(conf, model)

