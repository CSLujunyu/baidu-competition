import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import subtask4.net as net
import subtask4.train_and_evaluate as train

# configure

ubuntu_conf = {
    'train_context':'/hdd/lujunyu/dataset/DSTC7_track1/DAM/s4/15_60/train_ubuntu_4_context.pkl',
    'train_response':'/hdd/lujunyu/dataset/DSTC7_track1/DAM/s4/15_60/train_ubuntu_4_response.pkl',
    'dev_context': '/hdd/lujunyu/dataset/DSTC7_track1/DAM/s4/15_60/dev_ubuntu_4_context.pkl',
    'dev_response': '/hdd/lujunyu/dataset/DSTC7_track1/DAM/s4/15_60/dev_ubuntu_4_response.pkl',
    'test_context':'/hdd/lujunyu/dataset/DSTC7_track1/DAM/s4/15_60/test_ubuntu_4_context.pkl',
    'test_response':'/hdd/lujunyu/dataset/DSTC7_track1/DAM/s4/15_60/test_ubuntu_4_response.pkl',

    'process_train_data': '/hdd/lujunyu/dataset/DSTC7_track1/DAM/s4/DE_process_train.pkl',
    'process_dev_data': '/hdd/lujunyu/dataset/DSTC7_track1/DAM/s4/DE_process_dev.pkl',
    'process_test_data': '/hdd/lujunyu/dataset/DSTC7_track1/DAM/s4/DE_process_test.pkl',

    # 'train_context':'/hdd/lujunyu/dataset/DSTC7_track1/DAM/s4/15_60/small_train_context.pkl',
    # 'train_response':'/hdd/lujunyu/dataset/DSTC7_track1/DAM/s4/15_60/small_train_response.pkl',
    # 'dev_context': '/hdd/lujunyu/dataset/DSTC7_track1/DAM/s4/15_60/small_dev_context.pkl',
    # 'dev_response': '/hdd/lujunyu/dataset/DSTC7_track1/DAM/s4/15_60/small_dev_response.pkl',
    # 'test_context':'/hdd/lujunyu/dataset/DSTC7_track1/DAM/s4/15_60/small_test_context.pkl',
    # 'test_response':'/hdd/lujunyu/dataset/DSTC7_track1/DAM/s4/15_60/small_test_response.pkl',

    # 'process_train_data':'/hdd/lujunyu/dataset/DSTC7_track1/DAM/s4/small_DE_process_train.pkl',
    # 'process_dev_data':'/hdd/lujunyu/dataset/DSTC7_track1/DAM/s4/small_DE_process_dev.pkl',
    # 'process_test_data': '/hdd/lujunyu/dataset/DSTC7_track1/DAM/s4/small_DE_process_test.pkl',

    "save_path": "./output/CNN/s4/stack/",
    "word_emb_init": "/hdd/lujunyu/dataset/DSTC7_track1/DAM/s4/15_60/embed4data.pkl",


    "init_model": None, #should be set for test

    "rand_seed": 2,
    "is_mask": True,
    "is_layer_norm": True,
    "is_positional": False,

    "stack_num": 3,
    "attention_type": "dot",

    "learning_rate": 1e-3,
    "reg_rate":3e-5,
    "drop_rate": 0.3,
    "vocab_size": 111518,    #434513 for DAM  ， 128205 for dstc
    "emb_size": 300,
    "batch_size": 10, #200 for test

    "max_turn_num": 9,
    "max_turn_len": 50,

    "max_to_keep": 1,
    "num_scan_data": 5,
    "_EOS_": 38, #1455 for DSTC7, 28270 for DAM_source, #1 for douban data  , 6 for advising
    "final_n_class": 1,

    "rnn_dim":300,
    'options_num':100,
    'conv_filter_num':50,

    'n_layers':3,
    'cnn_channel':[12,24,48],
    'kernel_size':[
        [3,3,3],
        [3,3,3],
        [3,3,3],
        [3,3,3]
    ],
    'FC_size':[1024],

    'Model':'WCNN_S_BN', ## [BiLSTM,GRU, BiLSTM_tw,CNN, LSTM_ATTENTION, DAM, DAM_p]
}


model = net.Net(ubuntu_conf)
train.train(ubuntu_conf, model)


#test and evaluation, init_model in conf should be set
#test.test(conf, model)

