from utils.data_generator import DataGenerator
import utils.evaluation as eva

# configure

ubuntu_conf = {
    "data_path": "/hdd/lujunyu/dataset/DAM_data/DSTC7_1/official_prepocessing_small.pkl",
    "save_path": "/home/lujunyu/repository/DSTC7/official-baseline/output/product_prob/temp/",
    "word_emb_init": "/hdd/lujunyu/dataset/DAM_data/DSTC7_1/embed4data_official",

    'train_context':'/hdd/lujunyu/dataset/DSTC7_track1/baseline_DAM/subtask1/u_train_context_small.pkl',
    'dev_context':'/hdd/lujunyu/dataset/DSTC7_track1/baseline_DAM/subtask1/u_dev_context_small.pkl',
    'train_response':'/hdd/lujunyu/dataset/DSTC7_track1/baseline_DAM/subtask1/u_train_respone_small.pkl',
    'dev_response':'/hdd/lujunyu/dataset/DSTC7_track1/baseline_DAM/subtask1/u_dev_respone_small.pkl',

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
    "batch_size": 3, #200 for test

    "max_turn_num": 10,
    "max_turn_len": 60,

    "max_to_keep": 1,
    "num_scan_data": 5,
    "_EOS_": 36, #1455 for DSTC7, 28270 for DAM_source, #1 for douban data  , 6 for advising
    "final_n_class": 1,

    "rnn_dim":256,
    'options_num':100

}


# dg = DataGenerator(ubuntu_conf)
# example_id, turns, turn_num, turn_len, response, response_len, label = dg.train_data_generator(0)

index = 29
dev_score_file_path = ubuntu_conf['save_path'] + 'dev_score.' + str(index) + '.0'
dev_result = eva.evaluate(dev_score_file_path)
for p_at in dev_result:
    print(p_at)
print('finish dev evaluation')
