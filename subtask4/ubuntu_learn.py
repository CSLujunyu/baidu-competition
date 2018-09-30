import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
sys.path.append('/home/lujunyu/repository/DSTC7/official-baseline/')

import subtask4.net as net
import subtask4.train_and_evaluate as train


# configure
raw_data_path = '/hdd/lujunyu/dataset/DSTC7_track1/subtask4/'
model_data_path = '/hdd/lujunyu/dataset/DSTC7_track1/model_data/Ubuntu/s4/'
model_path = '/hdd/lujunyu/model/DSTC7/Ubuntu/s4/'

ubuntu_conf = {
    "train_context_path":os.path.join(model_data_path,'train_context.pkl'),
    "dev_context_path":os.path.join(model_data_path,'dev_context.pkl'),
    "test_context_path":os.path.join(model_data_path,'test_context.pkl'),
    "train_response_path":os.path.join(model_data_path,'train_response.pkl'),
    "dev_response_path":os.path.join(model_data_path,'dev_response.pkl'),
    "test_response_path":os.path.join(model_data_path,'test_response.pkl'),

    'process_train_data': os.path.join(model_data_path,'DE_process_train.pkl'),
    'process_dev_data': os.path.join(model_data_path,'DE_process_dev.pkl'),
    'process_test_data': os.path.join(model_data_path,'DE_process_test.pkl'),

    "save_path": os.path.join(model_path,'model_1/'),
    "word_emb_init": os.path.join(model_data_path,'embed4data.pkl'),


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

