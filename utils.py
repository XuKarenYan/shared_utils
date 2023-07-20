import numpy as np
from scipy.signal import butter, resample
from scipy import signal
import ast
import tqdm
import random
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import h5py
from collections import Counter
import pandas as pd




def model_namer(sql_conn, train_on_server, table='EEGNet'):
    '''Generates a unique two word name based on the inbuilt unix dictionary. 
    
    Parameters
    ----------
    sql_conn: sqlite3.Connection
    train_on_server: bool
    table: string
        Name of the model structure.

    Returns
    -------
    new_name: string
        Generated two-word model name.
    '''
    
    # get previously used names
    if train_on_server == True:
        used_names = pd.read_sql(f'SELECT * FROM {table}', sql_conn)
        used_names = list(used_names.name)
    
    # import word list from unix install
    with open('/usr/share/dict/words') as f:
        words = f.read().splitlines()
    words = [word.lower() for word in words if "'" not in word]
    
    if train_on_server == True:
        unique_name = False
        while not unique_name:
            new_name = random.choice(words) + '_' + random.choice(words)
            new_name = new_name + '_' + table
            if new_name not in used_names:
                unique_name = True
    else:                                                                   # cannot check duplicate names for local training
        new_name = random.choice(words) + '_' + random.choice(words)

    return new_name



def decide_kind(data_name):
    '''Decide which kind of dataset it is based on the information in the data_name.

    Parameters
    ----------
    data_name: string
        The name of the data. Should be in format: YYYY-MM-DD_SUBJECT_KIND_SESSIONID_NOTES.

    Returns
    -------
    kind: string
        Either 'OL' or 'CL'.
    '''
    
    if 'OL' in data_name:
        kind = 'OL'
    elif 'CL' in data_name:
        kind = 'CL'
    else:
        raise ValueError(f'The dataset {data_name} doesn\'t indicate whether it is OL or CL. Please check it.')
    
    return kind



def read_config(yaml_name):
    '''Read in the information from the yaml setting file where stores which data to use.

    Parameters
    ----------
    yaml_name: str
        The yaml file name. Example: settings.yaml

    Returns
    -------
    config: dict
        A dict of information in the assigned yaml file.
    '''

    yaml_file = open(yaml_name, 'r')
    config = yaml.safe_load(yaml_file)
    yaml_file.close()
    return config



def read_data_file_to_dict(filename, return_dict=True):
    '''Read in the information in .bin file into a dict.

    Parameters
    ----------
    filename: str
        Path to .bin file.
    return_dict: bool
        Whether return a dictionary format or an array format.

    Returns
    -------
    If return dict and it's eeg data:
        eeg_data: dict
            'eegbuffersignal': 2-d array with shape (n_samples, n_electrodes)
                Raw data collected with sampling rate as 1000 Hz. Already applied bandpass filter: 4 - 90 Hz.
            'databuffer': 2-d array with shape (n_samples, n_electrodes)
                Filtered raw data with sampling rate as 1000 Hz. Already applied bandpass filter: 4 - 40 Hz.
            'task_step': 1-d array with shape (n_samples,)
                Record the sample indices in the task data that each eeg data corresponds to. Each element is a number in [0, n_task_samples].
                E.g.: (array([  252,   253,   255, ..., 75038, 75039, 75040], dtype=int32), array([ 2, 15, 43, ..., 20, 21, 20]))
            'time_ns': 1-d array with shape (n_samples,)
                The absolute time in nanoseconds.
            'name': str, 'eeg'
            'dtypes': list, ['66<f4', '66<f4', '<i4', '<i8']

    If return dict and it's task data:
        task_data: dict
            'state_task': 1-d array with shape (n_task_samples,)
                State we set, like [-1,  0,  1,  2,  3,  4]. Sampling rate is 50 Hz.
            'decoder_output': 2-d array with shape (n_task_samples, n_class)
                For OL, it's all zero.
            'decoded_pos': 2-d array with shape (n_task_samples, 2)
                For OL, it's all zero.
            'target_pos': 2-d array with shape (n_task_samples, 2)
                Record the target position.
                    [[-0.85  0.  ]
                     [ 0.   -0.85]
                     [ 0.    0.85]
                     [ 0.85  0.  ]]
            'eeg_step': 1-d array with shape (n_task_samples,)
                Record the sample indices in the eeg data that each task data corresponds to. Each element is a number in [0, n_samples].
            'time_ns': 1-d array with shape (n_task_samples,)
                The absolute time in nanoseconds.
            'name': str, 'task'
            'dtypes': list, ['|i1', '5<f4', '2<f4', '2<f4', '<i4', '<i8']

    If not return dict and it's eeg data:
        data: 1-d array with shape (n_samples,)
            Each row is a numpy.void with length 4, coresponding to 'eegbuffersignal' (in shape (n_electrodes,)), 'databuffer', 'task_step', 'time_ns'.

    If not return dict and it's task data:
        data: 1-d array with shape (n_task_samples,)
            Each row is a numpy.void with length 6, coresponding to 'state_task', 'decoder_output', 'decoded_pos', 'target_pos', 'eeg_step', 'time_ns'.
    '''

    with open(filename, 'rb') as openfile:
        name = openfile.readline().decode('utf-8').strip()
        keys = openfile.readline().decode('utf-8').strip()
        dtypes = openfile.readline().decode('utf-8').strip()
        shapes = None

        if len(dtypes.split('$')) == 2:             # shapes can be indicated with a $ to separate.
            dtypes, shapes = dtypes.split('$')
            dtypes = dtypes.strip()
            shapes = ast.literal_eval(shapes.strip())
        
        keys = keys.split(',')
        dtypes = dtypes.split(',')
        if shapes is None:
            data = np.fromfile(openfile, dtype=[item for item in zip(keys, dtypes)])
        else:
            data = np.fromfile(openfile, dtype=[item for item in zip(keys, dtypes, shapes)])
        if not return_dict:
            return data
        data_dict = {key: data[key] for key in keys}
        data_dict['name'] = name
        data_dict['dtypes'] = dtypes
    return data_dict



class DataPreprocessor:
    '''This preprocessor handles the general preprocessing work, including dropping channels, several filtering, and normalization.

    Example
    -------
    preprocessor = DataPreprocessor(config_dict)
    eeg_data['databuffer'] = preprocessor.preprocess(eeg_data['databuffer'])
    '''

    def __init__(self, config):
        '''Initializes DataPreprocessor with given arguments from the config file.

        Parameters
        ----------
        config: dict
            Configurations from the yaml file.
        '''

        self.eeg_cap_type = config['eeg_cap_type']
        self.ch_to_drop = config['ch_to_drop']
        self.apply_bandpass = config['bandpass_filter']['apply']
        self.lowcut = config['bandpass_filter']['lowcut']
        self.highcut = config['bandpass_filter']['highcut']
        self.order = config['bandpass_filter']['order']
        self.sf = config['sampling_frequency']

    def get_electrode_position(self):
        '''Get electrode names and grid coordinates for different cap types.

        Returns
        -------
        ch_names: list of string
        coords: list of integer pairs (list)
            Position indices of each electrode in the grid.
        '''

        if self.eeg_cap_type == 'gel64':
            ch_names = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
                        'M1', 'T7', 'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6',
                        'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'O2', 'EOG',
                        'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6',
                        'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CP4',
                        'P5', 'P1', 'P2', 'P6', 'PO5', 'PO3', 'PO4', 'PO6',
                        'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'Oz',
                        'TRGR', 'COUNT']
            coords = [[0, 4], [0, 5], [0, 6], [2, 1], [2, 3], [2, 5], [2, 7], [2, 9],
                      [3, 2], [3, 4], [3, 6], [3, 8], [4, 0], [4, 1], [4, 3], [4, 5], [4, 7], [4, 9], [4, 10], 
                      [5, 2], [5, 4], [5, 6], [5, 8], [6, 1], [6, 3], [6, 5], [6, 7], [6, 9],
                      [7, 5], [8, 4], [8, 6], [0, 0], [1, 1], [1, 3], [1, 7], [1, 9],
                      [2, 2], [2, 4], [2, 6], [2, 8], [3, 3], [3, 5], [3, 7],
                      [4, 2], [4, 4], [4, 6], [4, 8], [5, 3], [5, 7],
                      [6, 2], [6, 4], [6, 6], [6, 8],
                      [7, 2], [7, 3], [7, 7], [7, 8],
                      [3, 1], [3, 9], [5, 1], [5, 9], [7, 1], [7, 9], [8, 5],
                      [4, 8], [8, 5]]
        elif self.eeg_cap_type == 'dry64':
            ch_names = ['0Z', '1Z', '2Z', '3Z', '4Z',
                        '1L', '1R', '1LB', '1RB', 
                        '2L', '2R', '3L', '3R', '4L', '4R', 
                        '1LC', '1RC', '2LB', '2RB', '1LA', '1RA', '1LD', '1RD', '2LC', '2RC', 
                        '3LB', '3RB', '3LC', '3RC', '2LD', '2RD', '3RD',
                        '3LD', '9Z', '8Z', '7Z', '6Z', '5Z', 
                        '10L', '10R', '9L', '9R', '8L', '8R', '7L', '7R', '6L', '6R', '5L', '5R', 
                        '4LD', '4RD', '5LC', '5RC', '5LB', '5RB', 
                        '3LA', '3RA', '2LA', '2RA', '4LC', '4RC', '4LB', '4RB',
                        'TRGR', 'COUNT']
            coords = [[0, 5], [2, 5], [4, 5], [6, 5], [8, 5],
                      [1, 4], [1, 6], [5, 2], [5, 8],
                      [3, 4], [3, 6], [5, 4], [5, 6], [7, 4], [7, 6],
                      [5, 1], [5, 9], [7, 2], [7, 8], [8, 3], [8, 7], [6, 0], [6, 10], [7, 1], [7, 9],
                      [9, 2], [9, 8], [9, 1], [9, 9], [8, 0], [8, 10], [10, 10],
                      [10, 0], [18, 5], [16, 5], [14, 5], [12, 5], [10, 5],
                      [19, 4], [19, 6], [17, 4], [17, 6], [15, 4], [15, 6], [13, 4], [13, 6], [11, 4], [11, 6], [9, 4], [9, 6],
                      [12, 0], [12, 10], [13, 1], [13, 9], [13, 2], [13, 8],
                      [12, 3], [12, 7], [10, 3], [10, 7], [11, 1], [11, 9], [11, 2], [11, 8],
                      [19, 10], [19, 0]]
        elif self.eeg_cap_type == 'saline64':
            ch_names = ['1Z', '2Z', '3Z', '4Z', '6Z', '7Z', '8Z', '9Z',
                        '1L', '2L', '3L', '4L', '5L', '6L', '7L', '8L', '9L', '10L', '11L',
                        '1R', '2R', '3R', '4R', '5R', '6R', '7R', '8R', '9R', '10R', '11R',
                        '1LA', '2LA', '3LA', '1LB', '2LB', '3LB', '4LB', '5LB', '1LC', '2LC', '3LC', '4LC', '5LC',
                        '1LD', '2LD', '3LD', '4LD',
                        '1RA', '2RA', '3RA', '1RB', '2RB', '3RB', '4RB', '5RB', '1RC', '2RC', '3RC', '4RC', '5RC',
                        '1RD', '2RD', '3RD', '4RD',
                        'TRGR', 'COUNT']
            coords = [[2, 5], [4, 5], [6, 5], [8, 5], [12, 5], [14, 5], [16, 5], [18, 5],
                      [1, 4], [3, 4], [5, 4], [7, 4], [9, 4], [11, 4], [13, 4], [15, 4], [17, 4], [19, 4], [21, 4],
                      [1, 6], [3, 6], [5, 6], [7, 6], [9, 6], [11, 6], [13, 6], [15, 6], [17, 6], [19, 6], [21, 6],
                      [8, 3], [10, 3], [12, 3], [5, 2], [7, 2], [9, 2], [11, 2], [13, 2], [5, 1], [7, 1], [9, 1], [11, 1], [13, 1],
                      [6, 0], [8, 0], [10, 0], [12, 0],
                      [8, 7], [10, 7], [12, 7], [5, 8], [7, 8], [9, 8], [11, 8], [13, 8], [5, 9], [7, 9], [9, 9], [11, 9], [13, 9],
                      [6, 10], [8, 10], [10, 10], [12, 10],
                      [10,  5], [0, 5]]
        else:
            raise ValueError('eeg_cap_type must be one of "gel64", "dry64", "saline64".')
        return ch_names, coords

    def bandpass_channels(self, data):
        '''Apply butter bandpass filter. Both high pass band and low pass band can be assigned.

        Parameters
        ----------
        data: 2-d array with shape (n_samples, n_electrodes)

        Returns
        -------
        data: 2-d array with shape (n_samples, n_electrodes)
        '''

        def butter_bandpass_filter(data, lowcut, highcut, sf, order):
            nyq = 0.5 * sf
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='band')
            y = signal.filtfilt(b, a, data)
            return y
        
        for electrode_ix in range(data.shape[1]):
            data[:, electrode_ix] = butter_bandpass_filter(data[:,electrode_ix], self.lowcut, self.highcut, self.sf, self.order)

        return data

    def laplacian_filtering(self, data):
        '''Apply laplacian filter to data with neighbor distance as 2 (next next one).
        
        Parameters
        ----------
        data: 2-d array with shape (n_samples, n_electrodes)

        Returns
        -------
        data: 2-d array with shape (n_samples, n_electrodes)
        '''

        # Get labels and coordinates of all channels for cap type
        ch_names, coords = self.get_electrode_position()
        GRIDSHAPE = (max([coord[1] for coord in coords])+1, max([coord[0] for coord in coords])+1)
        
        # Drop assigned channels
        ch_names, coords = zip(*[(ch_name, coord) for ch_name, coord in zip(ch_names, coords) if ch_name not in self.ch_to_drop])

        # Fill in each electrode (by order) into grid
        inds_grid = np.empty(GRIDSHAPE, dtype='int') * np.nan
        for i, ind in enumerate(coords):
            inds_grid[ind[1], ind[0]] = i

        # List neighboring electrodes for each electrode in four directions (with distance 2)
        neighbors = []
        for i, ind in enumerate(coords):
            iy, ix = ind
            neighbors_i = []
            if ix > 1 and ~np.isnan(inds_grid[ix-2, iy]):
                neighbors_i.append(int(inds_grid[ix-2, iy]))
            if ix < GRIDSHAPE[0]-2 and ~np.isnan(inds_grid[ix+2, iy]):
                neighbors_i.append(int(inds_grid[ix+2, iy]))
            if iy > 1 and ~np.isnan(inds_grid[ix, iy-2]):
                neighbors_i.append(int(inds_grid[ix, iy-2]))
            if iy < GRIDSHAPE[1]-2 and ~np.isnan(inds_grid[ix, iy+2]):
                neighbors_i.append(int(inds_grid[ix, iy+2]))
            neighbors.append(neighbors_i)

        # Create row for each electrode indicating all neighbors
        next_adjacency = np.zeros((len(ch_names), len(ch_names)))
        for i, neighbors_i in enumerate(neighbors):
            next_adjacency[i, neighbors_i] = 1
        D = len(ch_names)

        laplacian_next = np.eye(D) - (next_adjacency / np.maximum(np.sum(next_adjacency, axis=1), 1)).T
        print("Laplacian applied.")

        return data @ laplacian_next.T

    def normalize_channels(self, data):
        '''Normalize each channel to have mean 0 and variance 1.

        Parameters
        ----------
        data: 2-d array with shape (n_samples, n_electrodes)

        Returns
        -------
        data: 2-d array with shape (n_samples, n_electrodes)
        '''

        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        
        return data

    def preprocess(self, data):
        '''Manage the whole preprocessing procedure.

        Parameters
        ----------
        data: 2-d array with shape (n_samples, n_electrodes)
            Data from eeg_data['databuffer'].

        Returns
        -------
        data: 2-d array with shape (n_samples, n_electrodes)
        '''

        # Throw channels
        ch_names, _ = self.get_electrode_position()
        ch_index_to_drop = [ch_names.index(ch) for ch in self.ch_to_drop]
        data = np.delete(data, ch_index_to_drop, axis=1)

        # Apply filters
        if self.apply_bandpass:                     # bandpass filter
            data = self.bandpass_channels(data)
        data = self.laplacian_filtering(data)       # laplacian filter

        # Normalize channels
        data = self.normalize_channels(data)

        return data



class DatasetGenerator:
    '''This generator will cut the eeg signals into trials and produce trials that can be used in training.
       When creating an instance of this class, set the ms to drop from each trial as well as the window_length (the time window of the model).
    
    Example
    -------
    generator = DatasetGenerator(config)
    trials, labels = dataset_generator.generate_dataset(data_dicts)
    '''

    def __init__(self, config):
        '''Initializes DatasetGenerator with given arguments.

        Parameters
        ----------
        config: dict
            Configurations from the yaml file.
        '''
        
        self.dataset_operation = config['dataset_operation']
        self.first_ms_to_drop = config['first_ms_to_drop']
        self.window_length = config['window_length']
        self.omit_angles = config['omit_angles']

    def cut_into_trials(self, eeg_data, task_data, kind):
        '''Cut the eeg signals in shape (n_samples, n_electrodes) into trials according to the info from task_data.

        Parameters
        ----------
        eeg_data: dict
            Stores the info reading from eeg.bin file. See function read_data_file_to_dict for details.
        task_data: dict
            Stores the info reading from task.bin file. See function read_data_file_to_dict for details.
        kind: string
            Indicates which kind of dataset it is, either 'OL' or 'CL'.

        Returns
        -------
        trials: list of arrays with shape (n_electrodes, n_samples, 1)
            Trials here don't include the first and the last trials.
        labels: list of tuples where each tuple is (label of the trial, labels of each sample in this trial) with shape (int, an array with shape (n_samples, ))
            Corresponds to the labels of each sample in each trial.
        '''
        
        trials, labels = [], []
        # Find trials
        state_changes = np.flatnonzero(np.diff(task_data['state_task'].flatten())) + 1        # get the starting point of each trial, with the first trial dropped
        trial_labels = task_data['state_task'].flatten()[state_changes][:-1]                  # get the labels of each trial, drop the last 
        if kind == 'CL':
            trial_labels_in_ms = generateLabelWithRotation(task_data, self.omit_angles)

        # Partition trials with the first and the last trials dropped, and drop trials that are too short (than window_length) as well
        task_starts = state_changes[:-1]

        task_ends = state_changes[1:]
        eeg_starts = task_data['eeg_step'][task_starts]
        eeg_starts = np.where(eeg_starts==-1, 0, eeg_starts)
        eeg_ends = task_data['eeg_step'][task_ends]

        for idx in range(len(task_starts)):
            # drop trials that are too short for the model to train on
            if int(eeg_ends[idx] - eeg_starts[idx] - self.first_ms_to_drop) < self.window_length:
                continue

            # append data, label of each trial to respective lists
            trials.append(eeg_data['databuffer'][eeg_starts[idx]:eeg_ends[idx]])
            if kind == 'OL':
                labels.append((trial_labels[idx], np.array([trial_labels[idx]] * (eeg_ends[idx] - eeg_starts[idx]))))
            else:
                labels.append((trial_labels[idx], np.array(trial_labels_in_ms[eeg_starts[idx]:eeg_ends[idx]])))

        # Reshape trials into image-like 3-d data
        trials = [np.expand_dims(trial.transpose(1, 0), axis=-1) for trial in trials]

        return trials, labels
    
    def select_trials_and_relabel(self, trials, labels, index):
        '''Select the trials that we want to keep based on the information from the yaml file.
           Reassign the labels to avoid conflicts. New labels are the x in 'classx' in the yaml file.
        
        Parameters
        ----------
        trials: list of arrays with shape (n_electrodes, n_samples, 1)
        labels: list of tuples
        index: int
            Indicates which dataset it is in all the datasets we use.

        Returns
        -------
        filtered_trials: list of arrays with shape (n_electrodes, n_samples, 1)
            It only contains the trials that we want to keep according to labels.
        filtered_labels: list of tuples
        '''

        if self.dataset_operation['concatenate']:
            if self.dataset_operation['selected_labels']:           # use the data with selected labels
                filtered_trials = [trial for trial, label in zip(trials, labels) if label[0] in self.dataset_operation['selected_labels']]
                filtered_labels = [label for label in labels if label[0] in self.dataset_operation['selected_labels']]
            else:                                                   # use all data
                return trials, labels
        else:                                                       # select subset of data at trial level and change labels correspondingly
            mapping = {k: v[index] for k, v in self.dataset_operation['mapped_labels'].items()}
            filtered_trials = [trial for trial, label in zip(trials, labels) if label[0] in mapping.values()]
            mapping = {v: int(k[-1]) for k, v in mapping.items()}
            filtered_labels = [(mapping[label[0]], label[1]) for label in labels if label[0] in mapping]                                        # change the label of each trial
            filtered_labels = [(trial_labels[0], [mapping.get(label, -1) for label in trial_labels[1]]) for trial_labels in filtered_labels]    # change the label of each sample for each trial

        return filtered_trials, filtered_labels
    
    def generate_dataset(self, data_dicts):
        '''Apply cut_into_trials and select_trials_and_relabel for each data in data_dicts and combine trials for further training.
        
        Parameters
        ----------
        data_dicts: list of [eeg_data, task_data, kind] pairs

        Returns
        -------
        all_trials: list of arrays with shape (n_electrodes, n_samples, 1)
            Stores all trials from all data we want to use.
        all_labels: list of ints
            Stores reassigned labels.
        all_kinds: list of string
            Stores the kind of each trial.
        output_dim: int
            Count how many different classes in the dataset.
        '''

        all_trials, all_labels, all_kinds = [], [], []
        for i, (eeg_data, task_data, kind) in enumerate(data_dicts):
            trials, labels = self.cut_into_trials(eeg_data, task_data, kind)
            trials, labels = self.select_trials_and_relabel(trials, labels, i)
            kinds = len(labels) * [kind]
            all_trials.extend(trials)
            all_labels.extend(labels)
            all_kinds.extend(kinds)
        output_dim = len(set([label[0] for label in all_labels]))
        return all_trials, all_labels, all_kinds, output_dim



def partition_data(labels, num_folds):
    '''Partition the indices of the trials into the number of folds. Data of different labels are balanced among folds. NOT partitioning the data of the trials.
    
    Parameters
    ----------
    labels: list of tuples
        The labels of each trial.
    num_folds: int
        The number of folds we use for k-fold validation.

    Returns
    -------
    ids_folds: list of list
        It contains num_folds sublist. Each sublist contains the indices of this fold, the number of which is around 1 / num_folds.
    '''
    
    ids = np.arange(len(labels))
    labels = [label[0] for label in labels]
    label_set = list(set(labels))

    sub_ids = []
    for label in label_set:
        selected_ids = [l[0] for l in zip(ids, labels) if l[1] == label]
        np.random.shuffle(selected_ids)
        sub_ids_folds = np.array_split(selected_ids, num_folds)
        sub_ids.append(sub_ids_folds)

    ids_folds = [np.concatenate([subgroup[i] for subgroup in sub_ids]) for i in range(num_folds)]

    return ids_folds



def augment_data_to_file(trials, labels, kinds, ids_folds, h5_file, config):#TODO
    '''For each fold of data, augment the data to a 5x large dataset by adding 4 separate noises to each data window. Store the downsampled augmented data into a .h5 file.

    Parameters
    ----------
    trials: list of arrays with shape (n_electrodes, n_samples, 1)
    labels: list of ints
    ids_folds: list of list
        It contains num_folds sublist. Each sublist contains the indices of each fold, the number of which is around 1 / num_folds.
    h5_file: str
        The path to the .h5 data file.
    config: dict
        A dict of information in the assigned yaml file.

    Notes
    -----
    The augmented data will be stored in file named "data.h5". The augmented trials and augmented labels will be stored as an array with shape
    (n_trials, n_electrodes, n_samples, 1) and (n_trials,), respectively.
    '''

    window_length = config['window_length']
    stride = config['stride']
    new_samp_freq = config['new_sampling_frequency']
    num_noise = config['num_noise']

    window_size = int(new_samp_freq * window_length / 1000)
    portion = int(0.2 * window_length)
    labels_to_keep = set([label[0] for label in labels])
    
    file = h5py.File(h5_file, 'w')
    for fold, ids in enumerate(ids_folds):
        a_trials = [trials[j] for j in ids]
        a_labels = [labels[j] for j in ids]
        a_kinds = [kinds[j] for j in ids]

        n_electrodes = a_trials[0].shape[0]
        n_train_windows = np.sum([((data.shape[1] - window_length) // stride) + 1 for data in a_trials]) * (1+num_noise)
        dataset1 = file.create_dataset(str(fold)+'_trials', shape=(n_train_windows, n_electrodes, window_size, 1), dtype='float32')
        dataset2 = file.create_dataset(str(fold)+'_labels', shape=(n_train_windows,), dtype='int32')
        
        pbar = tqdm.tqdm(range(len(a_labels)))
        pbar.set_description("Augmenting fold " + str(fold))

        counter = 0
        dist = []

        for i in pbar:
            trial, label, kind = a_trials[i], a_labels[i], a_kinds[i]
            n_samples = trial.shape[1]

            # Slide a window on this trial
            window_start = 0
            window_end = window_start + window_length
            while (window_end <= n_samples):
                trial_window = trial[:, window_start:window_end, :]
                label_window = label[1][window_start:window_end]

                # new_label = label[0]
                if kind == 'OL':
                    new_label = label[0]
                else:
                    new_label, label_num = Counter(label_window).most_common(1)[0]
                    if label_num / len(label_window) < 0.8:
                        new_label = Counter(label_window[portion:]).most_common(1)[0][0]
                    if new_label not in labels_to_keep:
                        window_start += stride
                        window_end += stride
                        continue
                dist.append(new_label)

                # save the original data of this window
                dataset1[counter] = resample(trial_window, window_size, axis=1)
                dataset2[counter] = new_label
                counter += 1

                # generate 4 different noised data of this window and save
                for j in range(num_noise):
                    noise = np.max(trial_window) * np.random.uniform(-0.5, 0.5, trial_window.shape)
                    dataset1[counter] = resample(trial_window + noise, window_size, axis=1)
                    dataset2[counter] = new_label
                    counter += 1

                window_start += stride
                window_end += stride
            
        val_trials = file.create_dataset(str(fold)+'_val_trials', data=dataset1[::5])
        val_labels = file.create_dataset(str(fold)+'_val_labels', data=dataset2[::5])
        label_distribution = np.unique(dist, return_counts=True)

        print(f'Label distribution in fold {fold}:')
        print(np.unique(dist,return_counts=True))
    file.close()



def generateLabelWithRotation(task, omitAngles=10):
    '''Relabel the label for each sample in the close loop experiment according to the relative position between the cursor and the target. Then generate a list of labels for all EEG time steps.

    Parameters
    ----------
    task: dict
    omitAngles: int

    Returns
    -------
    label1ms: list
    '''

    labelAngles = {"left" : [135+omitAngles,-135-omitAngles],
                    "right" : [-45+omitAngles,45-omitAngles],
                    "up" : [45+omitAngles,135-omitAngles],
                    "down" : [-135+omitAngles,-45-omitAngles]}

    label2num = {"left" : 0,
                "right" : 1,
                "up" : 2,
                "down" : 3,
                "still" : 4}

    poses = task['decoded_pos']
    targets = task['target_pos']
    states = task['state_task']
    targetSize = (0.2,0.2) if 'target_size' not in task else task['target_size'][-100]      # no dataset has this key. 

    length = len(task['target_pos'])
    labels = np.full(length,-1) # this is where I store the labels
    dirs = targets - poses  # directions
    dirsAngles = np.arctan2(dirs[:,1],dirs[:,0]) * 180 / np.pi          # TODO: (x, y) in each dirs, switch to degree

    # 4 cardinal direction
    invalidFlag = np.geterr()["invalid"] # "warn"
    np.seterr(invalid='ignore') # ignore warning
    for label,angle in labelAngles.items():
        if label=="left":
            select = (dirsAngles > angle[0]) + (dirsAngles < angle[1])
        else:
            select = (dirsAngles > angle[0]) * (dirsAngles < angle[1])
        labels[select] = label2num[label]

    # still direction
    labels[states == 4] = label2num['still']

    # still when inside target
    posDirs = abs(dirs) 
    inTarget = (posDirs[:,0] <= targetSize[0]) * (posDirs[:,1] <= targetSize[1])
    labels[inTarget] = label2num['still']

    np.seterr(invalid = invalidFlag) # go back to original invalid flag: "warn"

    # stretch labels to 1ms 
    steps = task['eeg_step']
    label1ms = []
    for i in range(1,len(steps)):
        label1ms += [labels[i]] * (steps[i]-steps[i-1])
    label1ms = np.array(label1ms)

    return label1ms