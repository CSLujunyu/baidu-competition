import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
sys.path.append('/home/lujunyu/repository/DSTC7/official-baseline/')

import subtask2.net as net
import subtask2.train_and_evaluate as train
import subtask2.test_and_evaluate as test

# configure
raw_data_path = '/hdd/lujunyu/dataset/DSTC7_track1/subtask2/'
model_data_path = '/hdd/lujunyu/dataset/DSTC7_track1/model_data/Ubuntu/s2/'
model_path = '/hdd/lujunyu/model/DSTC7/Ubuntu/s2/'

ubuntu_conf = {
    "train_context_path": os.path.join(model_data_path, 'train_context.pkl'),
    "dev_context_path": os.path.join(model_data_path, 'dev_context.pkl'),
    "test_context_path": os.path.join(model_data_path, 'test_context.pkl'),
    "train_response_path": os.path.join(model_data_path, 'train_response.pkl'),
    "dev_response_path": os.path.join(model_data_path, 'dev_response.pkl'),
    "response_pool":os.path.join(model_data_path,'response_pool.pkl'),

    'process_train_data': os.path.join(model_data_path,'DE_process_train.pkl'),
    'process_dev_data': os.path.join(model_data_path,'DE_process_dev.pkl'),
    'process_test_data': os.path.join(model_data_path,'DE_process_test.pkl'),
    'process_pool': os.path.join(model_data_path,'DE_process_pool.pkl'),

    "save_path": os.path.join(model_path,'model_1/'),
    "word_emb_init": os.path.join(model_data_path,'embed4data.pkl'),


    "init_model": None, #should be set for test

    "rand_seed": 1,
    "is_mask": True,
    "is_layer_norm": True,
    "is_positional": False,

    "stack_num": 3,
    "attention_type": "dot",

    "learning_rate": 1e-3,
    "reg_rate":3e-5,
    "drop_rate": 0.3,
    "vocab_size": 61003,    #434513 for DAM  ， 128205 for dstc
    "emb_size": 300,
    "batch_size": 10, #200 for test

    "max_turn_num": 9,
    "max_turn_len": 50,

    "max_to_keep": 1,
    "num_scan_data": 5,
    "_EOS_": 1, #1455 for DSTC7, 28270 for DAM_source, #1 for douban data  , 6 for advising
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

    'Model':'task2', ## [BiLSTM,GRU, BiLSTM_tw,CNN, LSTM_ATTENTION, DAM, DAM_p]
}


model = net.Net(ubuntu_conf)
train.train(ubuntu_conf, model)


#test and evaluation, init_model in conf should be set
#test.test(conf, model)

