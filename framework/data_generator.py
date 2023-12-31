import numpy as np
import h5py, os, pickle, torch
import time
from framework.utilities import calculate_scalar, scale
import framework.config as config



class DataGenerator_MLGL(object):
    def __init__(self, batch_size, seed=42, normalization=False):

        ############### data split #######################################################

        self.batch_size = batch_size
        self.random_state = np.random.RandomState(seed)
        self.validate_random_state = np.random.RandomState(0)

        # Load data
        load_time = time.time()
        data_path = os.path.join(os.getcwd(), 'Dataset')
        file_path = os.path.join(data_path, 'training.pickle')
        print('using: ', file_path)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        self.train_audio_ids, self.train_rates, self.train_event_label = \
            data['audio_ids'], data['rates'], data['event_label']
        self.train_x = data['x']

        file_path = os.path.join(data_path, 'validation.pickle')
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        self.val_audio_ids, self.val_rates, self.val_event_label = \
            data['audio_ids'], data['rates'], data['event_label']
        self.val_x = data['x']

        file_path = os.path.join(data_path, 'test.pickle')
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        self.test_audio_ids, self.test_rates, self.test_event_label = \
            data['audio_ids'], data['rates'], data['event_label']
        self.test_x = data['x']

        ################################### map and sort labels ######################################
        # print(self.train_event_label.shape)
        # (2200, 24)
        self.train_event_label, self.train_coarse_level_subject_labels = self.sort_map_labels(
            self.train_event_label)

        self.val_event_label, self.val_coarse_level_subject_labels = self.sort_map_labels(
            self.val_event_label)

        self.test_event_label, self.test_coarse_level_subject_labels = self.sort_map_labels(
            self.test_event_label)

        ##############################################################################################

        source_dir = os.path.join(data_path, '1_blank_event24_semantic7_rate1_ed64')
        with open(os.path.join(source_dir, '0.pickle'), 'rb') as tf:
            self.one_graph_pkl_24_7_1 = pickle.load(tf)

        source_dir = os.path.join(data_path, '1_blank_event24_1_ed64')
        with open(os.path.join(source_dir, '0.pickle'), 'rb') as tf:
            self.one_graph_pkl_24_1 = pickle.load(tf)

        source_dir = os.path.join(data_path, '1_blank_event24_7_ed64')
        with open(os.path.join(source_dir, '0.pickle'), 'rb') as tf:
            self.one_graph_pkl_24_7 = pickle.load(tf)

        source_dir = os.path.join(data_path, '1_blank_event24_ed64')
        with open(os.path.join(source_dir, '0.pickle'), 'rb') as tf:
            self.one_graph_pkl_24 = pickle.load(tf)

        source_dir = os.path.join(data_path, '1_blank_event7_ed64')
        with open(os.path.join(source_dir, '0.pickle'), 'rb') as tf:
            self.one_graph_pkl_7 = pickle.load(tf)

        source_dir = os.path.join(data_path, '1_blank_event7_1_ed64')
        with open(os.path.join(source_dir, '0.pickle'), 'rb') as tf:
            self.one_graph_pkl_7_1 = pickle.load(tf)

        print('Loading data time: {:.3f} s'.format(time.time() - load_time))

        print('Split development data to {} training {} '
              'validation data and {} test data. '.format(len(self.train_audio_ids),
                                         len(self.val_audio_ids),
                                         len(self.test_audio_ids)))

        # Split development data to 2200 training and 245 validation data and 445 test data.
        # Number of 445 audios in testing

        self.normal = normalization
        if self.normal:
            (self.mean, self.std) = calculate_scalar(self.train_x)

    def sort_map_labels(self, source_labels):
        sort_object_event_labels = []
        coarse_level_subject_labels = []

        for i in range(len(source_labels)):
            row = source_labels[i]
            new_row = np.zeros_like(row)

            new_coarse_subject_row = np.zeros(len(config.subject_labels))
            for num, each in enumerate(list(row)):
                if each:
                    new_row[config.source_to_sort_indices[num]] = 1

            if sum(new_row[:7]):
                new_coarse_subject_row[0] = 1
            if sum(new_row[7:9]):  # 7, 8
                new_coarse_subject_row[1] = 1
            if sum(new_row[9:11]):
                new_coarse_subject_row[2] = 1
            if sum(new_row[11:16]):
                new_coarse_subject_row[3] = 1
            if sum(new_row[16:18]):
                new_coarse_subject_row[4] = 1
            if sum(new_row[18:20]):
                new_coarse_subject_row[5] = 1
            if sum(new_row[20:24]):
                new_coarse_subject_row[6] = 1

            sort_object_event_labels.append(new_row)
            coarse_level_subject_labels.append(new_coarse_subject_row)

        sort_object_event_labels = np.stack(sort_object_event_labels)
        coarse_level_subject_labels = np.stack(coarse_level_subject_labels)
        return sort_object_event_labels, coarse_level_subject_labels

    def generate_train(self):
        audios_num = len(self.train_audio_ids)
        audio_indexes = [i for i in range(audios_num)]

        self.random_state.shuffle(audio_indexes)

        iteration = 0
        pointer = 0

        while True:
            if pointer >= audios_num:
                pointer = 0
                self.random_state.shuffle(audio_indexes)

            # Get batch indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
            pointer += self.batch_size

            iteration += 1

            # load big graph
            batch_graph_24 = [self.one_graph_pkl_24 for j in range(self.batch_size)]
            batch_graph_7 = [self.one_graph_pkl_7 for j in range(self.batch_size)]
            batch_graph_24_1 = [self.one_graph_pkl_24_1 for j in range(self.batch_size)]
            batch_graph_7_1 = [self.one_graph_pkl_7_1 for j in range(self.batch_size)]
            batch_graph_24_7 = [self.one_graph_pkl_24_7 for j in range(self.batch_size)]
            batch_graph_24_7_1 = [self.one_graph_pkl_24_7_1 for j in range(self.batch_size)]

            batch_x = self.train_x[batch_audio_indexes]
            batch_y = self.train_rates[batch_audio_indexes]
            batch_y_event = self.train_event_label[batch_audio_indexes]
            batch_y_semantic7 = self.train_coarse_level_subject_labels[batch_audio_indexes]

            if self.normal:
                batch_x = self.transform(batch_x)

            yield batch_x, batch_y, batch_y_event, batch_y_semantic7, batch_graph_24, batch_graph_7, batch_graph_24_1, \
                      batch_graph_7_1, batch_graph_24_7, batch_graph_24_7_1


    def generate_validate(self, data_type, max_iteration=None):
        audios_num = len(self.val_audio_ids)
        audio_indexes = [i for i in range(audios_num)]

        self.validate_random_state.shuffle(audio_indexes)

        print('Number of {} audios in {}'.format(len(audio_indexes), data_type))

        iteration = 0
        pointer = 0

        while True:
            if iteration == max_iteration:
                break
            if pointer >= audios_num:
                break

            batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
            pointer += self.batch_size

            iteration += 1

            # load big graph
            batch_graph_24 = [self.one_graph_pkl_24 for j in range(self.batch_size)]
            batch_graph_7 = [self.one_graph_pkl_7 for j in range(self.batch_size)]
            batch_graph_24_1 = [self.one_graph_pkl_24_1 for j in range(self.batch_size)]
            batch_graph_7_1 = [self.one_graph_pkl_7_1 for j in range(self.batch_size)]
            batch_graph_24_7 = [self.one_graph_pkl_24_7 for j in range(self.batch_size)]
            batch_graph_24_7_1 = [self.one_graph_pkl_24_7_1 for j in range(self.batch_size)]

            batch_x = self.val_x[batch_audio_indexes]
            batch_y = self.val_rates[batch_audio_indexes]
            batch_y_event = self.val_event_label[batch_audio_indexes]
            batch_y_semantic7 = self.val_coarse_level_subject_labels[batch_audio_indexes]

            if self.normal:
                batch_x = self.transform(batch_x)

            # print(batch_y.shape)
            # print(batch_y_event.shape)
            # print(batch_y_semantic7.shape)

            yield batch_x, batch_y, batch_y_event, batch_y_semantic7, batch_graph_24, batch_graph_7, batch_graph_24_1, \
                  batch_graph_7_1, batch_graph_24_7, batch_graph_24_7_1


    def generate_test(self, data_type, max_iteration=None):
        audios_num = len(self.test_audio_ids)
        audio_indexes = [i for i in range(audios_num)]

        print('Number of {} audios in {}'.format(len(audio_indexes), data_type))

        iteration = 0
        pointer = 0

        while True:
            if iteration == max_iteration:
                break
            if pointer >= audios_num:
                break

            batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
            pointer += self.batch_size

            iteration += 1

            # load big graph
            batch_graph_24 = [self.one_graph_pkl_24 for j in range(self.batch_size)]
            batch_graph_7 = [self.one_graph_pkl_7 for j in range(self.batch_size)]
            batch_graph_24_1 = [self.one_graph_pkl_24_1 for j in range(self.batch_size)]
            batch_graph_7_1 = [self.one_graph_pkl_7_1 for j in range(self.batch_size)]
            batch_graph_24_7 = [self.one_graph_pkl_24_7 for j in range(self.batch_size)]
            batch_graph_24_7_1 = [self.one_graph_pkl_24_7_1 for j in range(self.batch_size)]

            batch_x = self.test_x[batch_audio_indexes]
            batch_y = self.test_rates[batch_audio_indexes]
            batch_y_event = self.test_event_label[batch_audio_indexes]
            batch_y_semantic7 = self.test_coarse_level_subject_labels[batch_audio_indexes]

            if self.normal:
                batch_x = self.transform(batch_x)

            yield batch_x, batch_y, batch_y_event, batch_y_semantic7, batch_graph_24, batch_graph_7, batch_graph_24_1, \
                  batch_graph_7_1, batch_graph_24_7, batch_graph_24_7_1


    def transform(self, x):
        """Transform data.

        Args:
          x: (batch_x, seq_len, freq_bins) | (seq_len, freq_bins)

        Returns:
          Transformed data.
        """

        return scale(x, self.mean, self.std)




