import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import sys
sys.path.append('/home/lujunyu/repository/DSTC7/official-baseline/')

import ELMo_models.net as net
import ELMo_models.train_and_evaluate as train

# configure

ubuntu_conf = {
    'train_data':'/hdd/lujunyu/dataset/DSTC7_track1/baseline_DAM/subtask1/elmo_train_data.pkl',
    'dev_data':'/hdd/lujunyu/dataset/DSTC7_track1/baseline_DAM/subtask1/elmo_dev_data.pkl',
    'process_train_data':'/hdd/lujunyu/dataset/DSTC7_track1/baseline_DAM/subtask1/u_train_processs_elmo.pkl',
    'process_dev_data':'/hdd/lujunyu/dataset/DSTC7_track1/baseline_DAM/subtask1/u_dev_process_elmo.pkl',

    'elmo_option_file':'/hdd/lujunyu/dataset/ELMo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json',
    'elmo_weight_file':'/hdd/lujunyu/dataset/ELMo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5',


    "save_path": "./output/elmo/temp/",


    "init_model": None, #should be set for test

    "rand_seed": None,

    "drop_dense": None,
    "drop_attention": None,

    "is_mask": True,
    "is_layer_norm": True,
    "is_positional": False,

    "stack_num": 3,
    "elmo_layer": 3,
    "attention_type": "dot",

    "learning_rate": 1e-3,
    "reg_rate":1e-5,
    "vocab_size": 108290,    #434513 for DAM  ， 128205 for dstc
    "emb_size": 1024,
    "batch_size": 10, #200 for test

    "max_turn_num": 15,
    "max_turn_len": 60,

    "max_to_keep": 1,
    "num_scan_data": 5,
    "_EOS_": 36, #1455 for DSTC7, 28270 for DAM_source, #1 for douban data  , 6 for advising
    "final_n_class": 1,

    "rnn_dim":256,
    'options_num':100,
    'conv_filter_num':50,

    'Model':'BiLSTM', ## [BiLSTM, CNN, LSTM_ATTENTION, DAM, DAM_p]
    'num_gpus':2
}


model = net.Net(ubuntu_conf)
train.train(ubuntu_conf, model)


#test and evaluation, init_model in conf should be set
#test.test(conf, model)

