import numpy as np
import pickle
import os
import time
from allennlp.commands.elmo import ElmoEmbedder

class DataGenerator():
    def __init__(self, configs):
        self.configs = configs

        self.elmo = ElmoEmbedder(options_file=self.configs['elmo_option_file'], weight_file=self.configs['elmo_weight_file'],cuda_device=0)

        self.train_c_r, self.train_label = self.load_train_data()
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ' : Finish Loading Training Data...')

        self.dev_c_r, self.dev_label = self.load_dev_data()
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ' : Finish Loading Dev Data...')

        self.train_data_size = len(self.train_label)
        print('Train set size: ', self.train_data_size)
        self.dev_data_size = len(self.dev_label)
        print('Dev set size: ', self.dev_data_size)


    def train_data_generator(self,batch_num):

        train_size = self.train_data_size
        start = batch_num * self.configs['batch_size'] % train_size
        end = (batch_num * self.configs['batch_size'] + self.configs['batch_size']) % train_size

        # shuffle data at the beginning of every epoch
        if batch_num == 0:
            self.train_c_r, self.train_label, _ = self.unison_shuffled_copies(self.train_c_r,self.train_label)
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ' : Finish Shuffling Data...')

        if start < end:
            batches_label = self.train_label[start:end]
            batches_c_r = self.train_c_r[start:end]
        else:
            batches_label = self.train_label[train_size - self.configs['batch_size']:train_size]
            batches_c_r = self.train_c_r[train_size - self.configs['batch_size']:train_size]

        turns, turn_num, turn_len, response, response_len, label = self.batch2placeholder(batches_c_r, batches_label)

        return turns, turn_num, turn_len, response, response_len, label

    def dev_data_generator(self, batch_num):
        """
           This function return training/validation/test data for classifier. batch_num*batch_size is start point of the batch.
           :param batch_size: int. the size of each batch
           :return: [[[float32,],],]. [[[wordembedding]element,]batch,]
           """

        dev_size = self.dev_data_size
        start = batch_num * self.configs['batch_size'] % dev_size
        end = (batch_num * self.configs['batch_size'] + self.configs['batch_size']) % dev_size
        if start < end:
            batches_label = self.dev_label[start:end]
            batches_c_r = self.dev_c_r[start:end]
        else:
            batches_label = self.dev_label[start:]
            batches_c_r = self.dev_c_r[start:]

        turns, turn_num, turn_len, response, response_len, label = self.batch2placeholder(batches_c_r,
                                                                                          batches_label)

        return turns, turn_num, turn_len, response, response_len, label


    def batch2placeholder(self, batches_c_r, batches_label):

        tmp = list(zip(*batches_c_r))
        example_id_c_r, turns, turn_num, turn_len, candidate, candidate_len = tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5]

        tmp = list(zip(*batches_label))
        example_id_y, label = tmp[0], tmp[1]

        assert example_id_c_r  == example_id_y

        # shuffle respone order in one example
        candidate, candidate_len, label = self.shuffle_response(candidate, candidate_len, label)

        # generate elmo embedding
        turns, turn_len, candidate = self.elmo_emb(turns, turn_len, candidate)


        return turns, turn_num, turn_len, candidate, candidate_len, label

    def unison_shuffled_copies(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p], p

    def shuffle_response(self,response, response_len, label):
        """
        responses contain ground truth id
        :param response: (batch_size, options_num, max_turn_len)
        :param response_len: (batch_size, options_num)
        :param label: (batch_size)
        :return:
        """
        tmp_response = np.zeros_like(response)
        tmp_response_len = np.zeros_like(response_len)
        tmp_label = np.zeros_like(label)
        for i in range(len(response)):
            tmp_response[i], tmp_response_len[i], shuffle_id = self.unison_shuffled_copies(np.array(response[i]), np.array(response_len[i]))
            tmp_label[i] = np.argwhere(shuffle_id == label[i])

        return tmp_response, tmp_response_len, tmp_label

    def get_context_response(self, data):
        """

        :param data:
        :param eos_idx:
        :param max_turn_num:
        :param max_turn_len:
        :return: array of tuple, tuple:(sent_list, example_turn_num, example_turn_len)
        """

        saver = []

        for c in range(data.shape[0]):
            turn_num = data['turn_num'][c]
            turn_len = data['turn_len'][c]
            c_s = data['context'][c]
            if len(c_s) > self.configs['max_turn_num']:
                c_s = c_s[-self.configs['max_turn_num']:]
                turn_num = self.configs['max_turn_num']
                turn_len = turn_len[-self.configs['max_turn_num']:]

            r_s = data['candidate'][c]

            res = np.array([data['id'][c], c_s, turn_num, turn_len, r_s, data['candidate_len'][c]],dtype=object)
            saver.append(res)

        return np.array(saver)




    def get_label(self, data):

        saver = []
        for e in range(data.shape[0]):
            res = np.array([data['id'][e], 0],dtype=object)
            saver.append(res)

        return np.array(saver)


    def load_train_data(self):

        if os.path.exists(self.configs['process_train_data']) and os.path.getsize(self.configs['process_train_data']) > 0:
            with open(self.configs['process_train_data'],'rb') as f:
                train_c_r, train_label = pickle.load(f)
        else:
            with open(self.configs['train_data'], 'rb') as f:
                train_data = pickle.load(f)

            train_c_r = self.get_context_response(train_data)
            train_label = self.get_label(train_data)

            with open(self.configs['process_train_data'], 'wb') as f:
                pickle.dump((train_c_r, train_label), f)


        return train_c_r, train_label

    def load_dev_data(self):

        if os.path.exists(self.configs['process_dev_data']) and os.path.getsize(self.configs['process_dev_data']) > 0:
            with open(self.configs['process_dev_data'],'rb') as f:
                dev_c_r, dev_label = pickle.load(f)
        else:
            with open(self.configs['dev_data'], 'rb') as f:
                dev_data = pickle.load(f)

            dev_c_r = self.get_context_response(dev_data)
            dev_label = self.get_label(dev_data)

            with open(self.configs['process_dev_data'], 'wb') as f:
                pickle.dump((dev_c_r, dev_label), f)


        return dev_c_r, dev_label

    def elmo_emb(self, turns, turn_len, candidate):

        _turns = []
        _candidate = []
        _turns_len = []
        for idx in range(self.configs['batch_size']):
            turns_emb = self.elmo.embed_batch(turns[idx])
            candidate_emb = self.elmo.embed_batch(candidate[idx])
            pad_len = np.zeros(shape=[self.configs['max_turn_num']])
            pad_len[:len(turn_len[idx])] = turn_len[idx]
            _turns_len.append(pad_len)

            # Padding turns embedding
            turns_emb_pad = []
            for i, emb in enumerate(turns_emb):
                pad_emb = np.zeros(shape=[self.configs['elmo_layer'], self.configs['max_turn_len'], self.configs['emb_size']],
                                   dtype=np.float32)
                pad_emb[:emb.shape[0],:emb.shape[1], :emb.shape[2]] = emb
                turns_emb_pad.append(pad_emb)

            turns_emb_pad = np.array(turns_emb_pad)
            turns_pad = np.zeros(shape=[self.configs['max_turn_num'],self.configs['elmo_layer'], self.configs['max_turn_len'], self.configs['emb_size']],
                                   dtype=np.float32)
            turns_pad[:turns_emb_pad.shape[0],:,:,:] = turns_emb_pad
            # Padding candidate embedding
            candidate_emb_pad = []
            for emb in candidate_emb:
                pad_emb = np.zeros(shape=[self.configs['elmo_layer'], self.configs['max_turn_len'], self.configs['emb_size']],
                                   dtype=np.float32)
                pad_emb[:emb.shape[0], :emb.shape[1], :emb.shape[2]] = emb
                candidate_emb_pad.append(pad_emb)
            candidate_emb_pad = np.array(candidate_emb_pad)

            _turns.append(turns_pad)
            _candidate.append(candidate_emb_pad)

        _turns = np.array(_turns)
        _candidate = np.array(_candidate)
        _turns_len = np.array(_turns_len)


        return _turns, _turns_len, _candidate




