import numpy as np
import pickle
import os
import time

class DataGenerator():
    def __init__(self, configs):
        self.configs = configs
        self.train_context, self.train_respone = self.load_train_data()
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ' : Finish Loading Training Data')

        self.dev_context, self.dev_respone = self.load_dev_data()
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ' : Finish Loading Dev Data')

        self.test_context, self.test_respone = self.load_test_data()
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ' : Finish Loading Test Data')

        self.table = self.table_generator()
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ' : Finish Loading Table')

        self.train_data_size = len(self.train_context)
        self.dev_data_size = len(self.dev_context)
        self.test_data_size = len(self.test_context)


    def train_data_generator(self,batch_num):

        train_size = self.train_data_size
        start = batch_num * self.configs['batch_size'] % train_size
        end = (batch_num * self.configs['batch_size'] + self.configs['batch_size']) % train_size

        # shuffle data at the beginning of every epoch
        if batch_num == 0:
            self.train_context, self.train_respone, _ = self.unison_shuffled(self.train_context,
                                                                             self.train_respone)
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), ' : Finish Shuffling Data:')

        if start < end:
            batches_context = self.train_context[start:end]
            batches_response = self.train_respone[start:end]
        else:
            batches_context = self.train_context[train_size - self.configs['batch_size']:train_size]
            batches_response = self.train_respone[train_size - self.configs['batch_size']:train_size]

        e_id, turns, turn_num, turn_len, r_id, response, response_len, label = self.batch2placeholder(batches_context,
                                                                                                      batches_response)

        data = {'example_id':e_id, 'turns':turns, 'turn_num':turn_num, 'turn_len':turn_len, 'candidate_id':r_id,
                'response':response, 'response_len':response_len, 'label':label}

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
            batches_response = self.dev_respone[start:end]
        else:
            batches_context = self.dev_context[start:]
            batches_response = self.dev_respone[start:]


        e_id, turns, turn_num, turn_len, r_id, response, response_len, label = self.batch2placeholder(batches_context,
                                                                                                      batches_response)

        data = {'example_id': e_id, 'turns': turns, 'turn_num': turn_num, 'turn_len': turn_len, 'candidate_id': r_id,
                'response': response, 'response_len': response_len, 'label': label}

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
            batches_response = self.test_respone[start:end]
        else:
            batches_context = self.test_context[start:]
            batches_response = self.test_respone[start:]

        e_id, turns, turn_num, turn_len, r_id, response, response_len, label = self.batch2placeholder(batches_context,
                                                                                                      batches_response)

        data = {'example_id': e_id, 'turns': turns, 'turn_num': turn_num, 'turn_len': turn_len, 'candidate_id': r_id,
                'response': response, 'response_len': response_len, 'label': label}

        return data


    def table_generator(self):

        if self.configs['word_emb_init'] is not None:
            with open(self.configs['word_emb_init'], 'rb') as f:
                self._word_embedding_init = pickle.load(f, encoding='latin1')
        else:
            self._word_embedding_init = np.random.random(size=[self.configs['vocab_size'], self.configs['emb_size']])

        return self._word_embedding_init


    def batch2placeholder(self, batches_context, batches_response):

        tmp = list(zip(*batches_context))
        example_id_c, turns, turn_num, turn_len = tmp[0], tmp[1], tmp[2], tmp[3]

        tmp = list(zip(*batches_response))
        example_id_r, candidat_id, response, response_len, label = tmp[0], tmp[1], tmp[2], tmp[3], tmp[4]

        assert example_id_c == example_id_r

        # shuffle respone order in one example
        candidat_id, response, response_len, label = self.shuffle_response(candidat_id, response, response_len, label)

        return example_id_c, turns, turn_num, turn_len, candidat_id, response, response_len, label

    def unison_shuffled(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p], p

    def unison_shuffled_copies(self, a, b, c, d):
        assert len(a) == len(b) == len(c) == len(d)
        p = np.random.permutation(len(a))
        return a[p], b[p], c[p], d[p], p

    def shuffle_response(self,candidate_id, response, response_len, label):
        """
        responses contain ground truth id
        :param response: (batch_size, options_num, max_turn_len)
        :param response_len: (batch_size, options_num)
        :param label: (batch_size, options_num)
        :return:
        """
        candidate_id, response, response_len, label = list(candidate_id), list(response), list(response_len), list(label)
        for i in range(len(response)):
            candidate_id[i], response[i], response_len[i], label[i], _ = self.unison_shuffled_copies(
                candidate_id[i],response[i],response_len[i],label[i])

        return candidate_id, response, response_len, label


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
                    sent_list.append(np.array(tmp,dtype=np.int32))
                    tmp = []
                    num = 0

            # padding zero vector to normalize turn num
            pad_sent = np.array([0] * max_turn_len,dtype=np.int32)
            if len(sent_list) < max_turn_num:
                example_turn_num = len(sent_list)
                for i in range(max_turn_num - len(sent_list)):
                    sent_list.append(pad_sent)
                    example_turn_len.append(0)
            else:
                example_turn_num = max_turn_num
                sent_list = sent_list[-max_turn_num:]
                example_turn_len = example_turn_len[-max_turn_num:]

            res = np.array([context['example-id'][c], np.array(sent_list,dtype=np.int32), example_turn_num,
                            np.array(example_turn_len,dtype=np.int32)],dtype=object)
            saver.append(res)

        return np.array(saver)


    def get_response(self, response, flag='test'):
        """

        :param PATH:
        :return: array of tuple, tuple:(sent, example_response_len)
        """

        assert flag in ['test','train','dev']

        max_respone_len = self.configs['max_turn_len']
        options_num = self.configs['options_num']
        saver = []

        for e in range(int(response.shape[0] / options_num)):
            example_data  = response[e*options_num:(e+1)*options_num]

            assert len(set(example_data['example-id'])) == 1

            example_data = example_data.reset_index(drop=True)
            example_sent = []
            example_response_len = []
            for r in range(options_num):
                options_response_len = 0
                options_sent = []
                for word in example_data['r'][r]:
                    if options_response_len >= max_respone_len:
                        break
                    else:
                        options_sent.append(word)
                        options_response_len += 1

                if len(options_sent) < max_respone_len:
                    pad = [0] * (max_respone_len - len(options_sent))
                    options_sent += pad

                example_sent.append(np.array(options_sent,dtype=np.int32))
                example_response_len.append(options_response_len)

            if flag == 'test':
                res = np.array([response['example-id'][e * options_num], np.array(example_data['candidate-id']),
                                np.array(example_sent,dtype=np.int32), np.array(example_response_len,dtype=np.int32),
                                np.array([None] * len(example_data))],
                               dtype=object)
            else:
                res = np.array([response['example-id'][e * options_num], np.array(example_data['candidate-id']),
                                np.array(example_sent,dtype=np.int32), np.array(example_response_len,dtype=np.int32),
                                np.array(example_data['y'])],
                               dtype=object)
            saver.append(res)

        return np.array(saver)

    def load_train_data(self):

        if os.path.exists(self.configs['process_train_data']) and os.path.getsize(self.configs['process_train_data']) > 0:
            with open(self.configs['process_train_data'],'rb') as f:
                train_context, train_response = pickle.load(f)
        else:
            with open(self.configs['train_context_path'], 'rb') as f:
                context  = pickle.load(f)

            with open(self.configs['train_response_path'], 'rb') as f:
                response  = pickle.load(f)

            train_context = self.get_context(context)
            train_response = self.get_response(response,'train')

            with open(self.configs['process_train_data'], 'wb') as f:
                pickle.dump((train_context, train_response), f)


        return train_context, train_response

    def load_dev_data(self):

        if os.path.exists(self.configs['process_dev_data']) and os.path.getsize(self.configs['process_dev_data']) > 0:
            with open(self.configs['process_dev_data'],'rb') as f:
                dev_context, dev_response = pickle.load(f)
        else:

            with open(self.configs['dev_context_path'], 'rb') as f:
                context  = pickle.load(f)

            with open(self.configs['dev_response_path'], 'rb') as f:
                response  = pickle.load(f)

            dev_context = self.get_context(context)
            dev_response = self.get_response(response,'dev')

            with open(self.configs['process_dev_data'], 'wb') as f:
                pickle.dump((dev_context, dev_response), f)


        return dev_context, dev_response

    def load_test_data(self):

        if os.path.exists(self.configs['process_test_data']) and os.path.getsize(self.configs['process_test_data']) > 0:
            with open(self.configs['process_test_data'],'rb') as f:
                test_context, test_response = pickle.load(f)
        else:

            with open(self.configs['test_context_path'], 'rb') as f:
                context  = pickle.load(f)

            with open(self.configs['test_response_path'], 'rb') as f:
                response  = pickle.load(f)

            test_context = self.get_context(context)
            test_response = self.get_response(response)

            with open(self.configs['process_test_data'], 'wb') as f:
                pickle.dump((test_context, test_response), f)


        return test_context, test_response

