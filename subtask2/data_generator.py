import numpy as np
import pickle
import os
import time

class DataGenerator():
    def __init__(self, configs):
        self.configs = configs
        self.train_context = self.load_train_data()
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ' : Finish Loading Training Data')

        self.dev_context = self.load_dev_data()
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ' : Finish Loading Dev Data')

        self.test_context = self.load_test_data()
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ' : Finish Loading Test Data')

        self.response_pool = self.load_response_pool()
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ' : Finish Loading Response Pool')

        self.table = self.table_generator()
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ' : Finish Loading Table')

        self.train_data_size = len(self.train_context)
        self.dev_data_size = len(self.dev_context)
        self.test_data_size = len(self.test_context)
        self.response_pool_size = len(self.response_pool)


    def train_data_generator(self,batch_num):

        train_size = self.train_data_size
        start = batch_num * self.configs['batch_size'] % train_size
        end = (batch_num * self.configs['batch_size'] + self.configs['batch_size']) % train_size

        # shuffle data at the beginning of every epoch
        if batch_num == 0:
            self.train_context = self.shuffled(self.train_context)
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ' : Finish Shuffling Data:')

        if start < end:
            batches_context = self.train_context[start:end]
        else:
            batches_context = self.train_context[train_size - self.configs['batch_size']:train_size]

        e_id, turns, turn_num, turn_len, y, label = self.batch2placeholder(batches_context)

        data = {'example_id':e_id, 'turns':turns, 'turn_num':turn_num, 'turn_len':turn_len, 'y':y, 'label':label}

        return data

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
            batches_context = self.dev_context[start:end]
        else:
            batches_context = self.dev_context[start:]

        e_id, turns, turn_num, turn_len, y, label = self.batch2placeholder(batches_context)

        data = {'example_id': e_id, 'turns': turns, 'turn_num': turn_num, 'turn_len': turn_len,  'y':y, 'label': label}

        return data

    def test_data_generator(self, batch_num):
        """
           This function return training/validation/test data for classifier. batch_num*batch_size is start point of the batch.
           :param batch_size: int. the size of each batch
           :return: [[[float32,],],]. [[[wordembedding]element,]batch,]
           """

        start = batch_num * self.configs['batch_size']
        end = (batch_num * self.configs['batch_size'] + self.configs['batch_size'])
        if start < end:
            batches_context = self.test_context[start:end]
        else:
            batches_context = self.test_context[start:]

        e_id, turns, turn_num, turn_len, y, label = self.batch2placeholder(batches_context)

        data = {'example_id': e_id, 'turns': turns, 'turn_num': turn_num, 'turn_len': turn_len, 'y':y, 'label': label}

        return data

    def response_pool_generator(self, batch_num):

        tmp = list(zip(*self.response_pool[batch_num*100:(batch_num+1)*100]))
        id, response, response_len = tmp[0], tmp[1], tmp[2]

        data = {'candidate_id':id, 'response':response, 'response_len':response_len}

        return data


    def table_generator(self):

        if self.configs['word_emb_init'] is not None:
            with open(self.configs['word_emb_init'], 'rb') as f:
                self._word_embedding_init = pickle.load(f, encoding='latin1')
        else:
            self._word_embedding_init = np.random.random(size=[self.configs['vocab_size'], self.configs['emb_size']])

        return self._word_embedding_init


    def batch2placeholder(self, batches_context):

        tmp = list(zip(*batches_context))
        example_id_c, turns, turn_num, turn_len, y, label = tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5]

        return example_id_c, turns, turn_num, turn_len, y, label

    def shuffled(self, a):
        p = np.random.permutation(len(a))
        return a[p]

    def unison_shuffled(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p], p

    def unison_shuffled_copies(self, a, b, c):
        assert len(a) == len(b) == len(c)
        p = np.random.permutation(len(a))
        return a[p], b[p], c[p], p


    def get_context(self, context, type='test'):
        """

        :param data:
        :param eos_idx:
        :param max_turn_num:
        :param max_turn_len:
        :return: array of tuple, tuple:(sent_list, example_turn_num, example_turn_len)
        """

        assert type in ['train', 'dev', 'test']

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
            for word in context['c'][c]:
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

            if type != 'test':
                res = np.array([context['example-id'][c], np.array(sent_list), example_turn_num, np.array(example_turn_len),
                                context['y'][c],context['label'][c]],dtype=object)
            else:
                res = np.array(
                    [context['example-id'][c], np.array(sent_list), example_turn_num, np.array(example_turn_len),
                     None,None], dtype=object)
            saver.append(res)

        return np.array(saver)


    def get_response(self, response):
        """

        :param PATH:
        :return: array of tuple, tuple:(sent, example_response_len)
        """

        max_respone_len = self.configs['max_turn_len']
        saver = []

        for i in range(response.shape[0]):
            if len(response['r'][i]) < max_respone_len:
                sent_len = len(response['r'][i])
                sent = response['r'][i] + [0] * (max_respone_len-sent_len)
            else:
                sent_len = max_respone_len
                sent = response['r'][i][:sent_len]

            res = np.array([response['candidate-id'][i], sent, sent_len],dtype=object)
            saver.append(res)

        return np.array(saver)

    def load_train_data(self):

        if os.path.exists(self.configs['process_train_data']) and os.path.getsize(self.configs['process_train_data']) > 0:
            with open(self.configs['process_train_data'],'rb') as f:
                train_context = pickle.load(f)
        else:
            with open(self.configs['train_context'], 'rb') as f:
                context  = pickle.load(f)

            train_context = self.get_context(context,'train')

            with open(self.configs['process_train_data'], 'wb') as f:
                pickle.dump(train_context, f)


        return train_context

    def load_dev_data(self):

        if os.path.exists(self.configs['process_dev_data']) and os.path.getsize(self.configs['process_dev_data']) > 0:
            with open(self.configs['process_dev_data'],'rb') as f:
                dev_context = pickle.load(f)
        else:

            with open(self.configs['dev_context'], 'rb') as f:
                context  = pickle.load(f)

            dev_context = self.get_context(context,'dev')

            with open(self.configs['process_dev_data'], 'wb') as f:
                pickle.dump(dev_context, f)


        return dev_context

    def load_test_data(self):

        if os.path.exists(self.configs['process_test_data']) and os.path.getsize(self.configs['process_test_data']) > 0:
            with open(self.configs['process_test_data'],'rb') as f:
                test_context = pickle.load(f)
        else:

            with open(self.configs['test_context'], 'rb') as f:
                context  = pickle.load(f)

            test_context = self.get_context(context,'test')

            with open(self.configs['process_test_data'], 'wb') as f:
                pickle.dump(test_context, f)

        return test_context

    def load_response_pool(self):

        if os.path.exists(self.configs['process_pool']) and os.path.getsize(self.configs['process_pool']) > 0:
            with open(self.configs['process_pool'],'rb') as f:
                response_pool = pickle.load(f)
        else:

            with open(self.configs['response_pool'], 'rb') as f:
                response_pool  = pickle.load(f)

            response_pool = self.get_response(response_pool)

            with open(self.configs['process_pool'], 'wb') as f:
                pickle.dump(response_pool, f)

        return response_pool

