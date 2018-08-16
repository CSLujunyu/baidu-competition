import numpy as np
import pandas as pd
import gensim
from nltk.corpus import stopwords
import string
import pickle
import os
import time

class DataGenerator():
    def __init__(self, configs):
        self.configs = configs
        self.train_context, self.train_respone, self.train_label = self.load_train_data()
        print('Finish Loading Training Data: ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

        self.dev_context, self.dev_respone, self.dev_label = self.load_dev_data()
        print('Finish Loading Dev Data:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

        self.table = self.table_generator()
        print('Finish Loading Table:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

        self.train_data_size = len(self.train_label)
        self.dev_data_size = len(self.dev_label)


    def train_data_generator(self,batch_num):

        train_size = self.train_data_size
        start = batch_num * self.configs['batch_size'] % train_size
        end = (batch_num * self.configs['batch_size'] + self.configs['batch_size']) % train_size

        # shuffle data at the beginning of every epoch
        if batch_num == 0:
            self.train_context, self.train_respone, self.train_label = self.unison_shuffled_copies(self.train_context,
                                                                                               self.train_respone,
                                                                                               self.train_label)
            print('Finish Shuffling Data:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

        if start < end:
            batches_label = self.train_label[start:end]
            batches_context = self.train_context[start:end]
            batches_response = self.train_respone[start:end]
        else:
            batches_label = self.train_label[train_size - self.configs['batch_size']:train_size]
            batches_context = self.train_context[train_size - self.configs['batch_size']:train_size]
            batches_response = self.train_respone[train_size - self.configs['batch_size']:train_size]

        example_id, turns, turn_num, turn_len, response, response_len, label = self.batch2placeholder(batches_context, batches_response, batches_label)

        return example_id, turns, turn_num, turn_len, response, response_len, label

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
            batches_context = self.dev_context[start:end]
            batches_response = self.dev_respone[start:end]
        else:
            batches_label = self.dev_label[dev_size - self.configs['batch_size']:dev_size]
            batches_context = self.dev_context[dev_size - self.configs['batch_size']:dev_size]
            batches_response = self.dev_respone[dev_size - self.configs['batch_size']:dev_size]

        example_id, turns, turn_num, turn_len, response, response_len, label = self.batch2placeholder(batches_context,
                                                                                                      batches_response,
                                                                                                      batches_label)

        return example_id, turns, turn_num, turn_len, response, response_len, label


    def table_generator(self):

        if self.configs['word_emb_init'] is not None:
            with open(self.configs['word_emb_init'], 'rb') as f:
                self._word_embedding_init = pickle.load(f, encoding='latin1')
        else:
            self._word_embedding_init = np.random.random(size=[self.configs['vocab_size'], self.configs['emb_size']])

        return self._word_embedding_init


    def batch2placeholder(self, batches_context, batches_response, batches_label):

        tmp = list(zip(*batches_context))
        example_id, turns, turn_num, turn_len = tmp[0], tmp[1], tmp[2], tmp[3]

        tmp = list(zip(*batches_response))
        example_id, response, response_len = tmp[0], tmp[1], tmp[2]

        tmp = list(zip(*batches_label))
        example_id, label = tmp[0], tmp[1]


        return example_id, turns, turn_num, turn_len, response, response_len, label

    def unison_shuffled_copies(self, a, b, c):
        assert len(a) == len(b) == len(c)
        p = np.random.permutation(len(a))
        return a[p], b[p], c[p]



    def get_context(self, context):
        """

        :param data:
        :param eos_idx:
        :param max_turn_num:
        :param max_turn_len:
        :return: array of tuple, tuple:(sent_list, example_turn_num, example_turn_len)
        """

        eos_idx = self.configs['_EOS_']
        max_turn_num = self.configs['max_turn_num']
        max_turn_len = self.configs['max_turn_len']

        saver = []

        for c in range(context.shape[0]):

            example_turn_len = []

            # spilt to sentence and padding 0 to every sentence
            sent_list = []
            tmp = []
            num = 0
            for word in context['cc'][c]:
                if word != eos_idx:
                    num += 1
                    tmp.append(word)
                else:
                    if num >= max_turn_len:
                        tmp = tmp[0:max_turn_len]
                        example_turn_len.append(max_turn_len)
                    else:
                        pad = [0] * (max_turn_len - num)
                        example_turn_len.append(num)
                        tmp += pad
                    sent_list.append(np.array(tmp))
                    tmp = []
                    num = 0

            # padding zero vector to normalize turn num
            pad_sent = np.array([0] * max_turn_len)
            if len(sent_list) < max_turn_num:
                example_turn_num = len(sent_list)
                for i in range(max_turn_num - len(sent_list)):
                    sent_list.append(pad_sent)
                    example_turn_len.append(0)
            else:
                example_turn_num = max_turn_num
                sent_list = sent_list[-max_turn_num:]
                example_turn_len = example_turn_len[-max_turn_num:]

            res = np.array([context['example-id'][c], np.array(sent_list), example_turn_num, np.array(example_turn_len)],dtype=object)
            saver.append(res)

        return np.array(saver)


    def get_response(self, response):
        """

        :param PATH:
        :return: array of tuple, tuple:(sent, example_response_len)
        """

        max_respone_len = self.configs['max_turn_len']
        options_num = self.configs['options_num']
        saver = []

        for e in range(int(response.shape[0] / options_num)):
            example_data  = response[e*options_num:(e+1)*options_num]
            example_data = example_data.reset_index(drop=True)
            example_sent = []
            example_response_len = []
            for r in range(options_num):
                options_response_len = 0
                options_sent = []
                for word in example_data['rr'][r]:
                    if options_response_len >= max_respone_len:
                        break
                    else:
                        options_sent.append(word)
                        options_response_len += 1

                if len(options_sent) < max_respone_len:
                    pad = [0] * (max_respone_len - len(options_sent))
                    options_sent += pad

                example_sent.append(np.array(options_sent))
                example_response_len.append(options_response_len)

            res = np.array([response['example-id'][e*options_num], np.array(example_sent), np.array(example_response_len)],dtype=object)
            saver.append(res)

        return np.array(saver)

    def get_label(self, data):

        options_num = self.configs['options_num']
        saver = []
        for e in range(int(data.shape[0] / options_num)):
            res = np.array([data['example-id'][e*options_num], 0],dtype=object)
            saver.append(res)

        return np.array(saver)

    def load_train_data(self):

        if os.path.exists(self.configs['process_train_data']) and os.path.getsize(self.configs['process_train_data']) > 0:
            with open(self.configs['process_train_data'],'rb') as f:
                train_context, train_response, train_label = pickle.load(f)
        else:
            with open(self.configs['train_context'], 'rb') as f:
                context  = pickle.load(f)

            with open(self.configs['train_response'], 'rb') as f:
                response  = pickle.load(f)

            train_context = self.get_context(context)
            train_response = self.get_response(response)
            train_label = self.get_label(response)

            with open(self.configs['process_train_data'], 'wb') as f:
                pickle.dump((train_context, train_response, train_label), f)


        return train_context, train_response, train_label

    def load_dev_data(self):

        if os.path.exists(self.configs['process_dev_data']) and os.path.getsize(self.configs['process_dev_data']) > 0:
            with open(self.configs['process_dev_data'],'rb') as f:
                dev_context, dev_response, dev_label = pickle.load(f)
        else:

            with open(self.configs['dev_context'], 'rb') as f:
                context  = pickle.load(f)

            with open(self.configs['dev_response'], 'rb') as f:
                response  = pickle.load(f)

            dev_context = self.get_context(context)
            dev_response = self.get_response(response)
            dev_label = self.get_label(response)

            with open(self.configs['process_dev_data'], 'wb') as f:
                pickle.dump((dev_context, dev_response, dev_label), f)


        return dev_context, dev_response, dev_label

