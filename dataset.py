import numpy as np
import h5py
import tqdm
from scipy.signal import resample
import sys
try:
    # relative path needed if to be used as module
    from .preprocessor import DataPreprocessor
    from .utils import (read_data_file_to_dict, detect_artifact, decide_kind, 
                        read_config, generateLabelWithRotation, decideLabelWithRotation)
except ImportError:
    # absolute path selected if to ran as stand alone shared_utils
    from preprocessor import DataPreprocessor
    from utils import (read_data_file_to_dict, detect_artifact, decide_kind, 
                       read_config, generateLabelWithRotation, decideLabelWithRotation)


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
        self.omit_trials = config.get('omit_trials',{})
        self.selected_seconds = config.get('selected_seconds',{})

    def cut_into_trials(self, eeg_data, task_data, kind, index, for_mne=False):
        '''Cut the eeg signals in shape (n_samples, n_electrodes) into trials according to the info from task_data.

        Parameters
        ----------
        eeg_data: dict
            Stores the info reading from eeg.bin file. See function read_data_file_to_dict for details.
        task_data: dict
            Stores the info reading from task.bin file. See function read_data_file_to_dict for details.
        kind: string
            Indicates which kind of dataset it is, either 'OL' or 'CL'.
        index: int
            Indicates which dataset it is in all the datasets we use.
        for_mne: bool
            Only use it when doing visualization.

        Returns
        -------
        trials: list of arrays with shape (n_electrodes, n_samples, 1)
            Trials here don't include the first and the last trials.
            Trials here don't include omit trials
            Trials here only includes the selected length
        labels: list of tuples where each tuple is (label of the trial, labels of each sample in this trial) with shape (int, an array with shape (n_samples, ))
            Corresponds to the labels of each sample in each trial.
        '''
        
        trials, labels = [], []
        # Find trials
        state_changes = np.flatnonzero(np.diff(task_data['state_task'].flatten())) + 1        # get the starting point of each trial, with the first trial dropped
        trial_labels = task_data['state_task'].flatten()[state_changes][:-1]                  # get the labels of each trial, drop the last 
        if kind == 'CL':
            trial_labels_in_ms = generateLabelWithRotation(task_data, self.omit_angles)
        if for_mne:
            game_state_change = [sc-1 for sc in state_changes]
            game_states = task_data['game_state'].flatten()[game_state_change][:-1]

        # Partition trials with the first and the last trials dropped, and drop trials that are too short (than window_length) as well
        task_starts = state_changes[:-1]

        task_ends = state_changes[1:]
        eeg_starts = task_data['eeg_step'][task_starts]
        eeg_starts = np.where(eeg_starts==-1, 0, eeg_starts)
        eeg_ends = task_data['eeg_step'][task_ends]

        # prepare to filter selected_seconds (first print how many seconds is in data)
        start_times_by_trial = (eeg_data['time_ns'][eeg_starts] - eeg_data['time_ns'][0]) / (10**9)
        end_times_by_trial = (eeg_data['time_ns'][eeg_ends] - eeg_data['time_ns'][0]) / (10**9)
        print(f"dataset #{index}: {round(((task_data['time_ns'][-1]-task_data['time_ns'][0]) / (10**9)))} seconds")

        for idx in range(len(task_starts)):
            # drop trials that are too short for the model to train on
            if int(eeg_ends[idx] - eeg_starts[idx] - self.first_ms_to_drop) < self.window_length:
                if for_mne:
                    del game_states[idx]
                continue

            # omit trials in omit_trials
            if index in self.omit_trials and (idx+2) in self.omit_trials[index]:
                # recall trial1 in omit_trials refers to trial0 because of indexing
                # However, since first trial already dropped trials[0] is really referring to trial2 in omit_trials
                if for_mne:
                    del game_states[idx]
                continue

            # omit trials outside of selected_seconds
            if index in self.selected_seconds:
                time_frames = self.selected_seconds[index]
                withinTimeFrame = False
                for start_time, end_time in time_frames:
                    if end_time < 0: end_time = end_times_by_trial[-1]
                    # check if start and end of trial is within specified time frame
                    if start_time <= end_times_by_trial[idx] <= end_time and start_time <= start_times_by_trial[idx] <= end_time:
                        withinTimeFrame = True
                        break
                if not withinTimeFrame:
                    if for_mne:
                        del game_states[idx]
                    continue

            # append data, label of each trial to respective lists
            trials.append(eeg_data['databuffer'][eeg_starts[idx]:eeg_ends[idx]])
            if kind == 'OL':
                labels.append((trial_labels[idx], np.array([trial_labels[idx]] * (eeg_ends[idx] - eeg_starts[idx]))))
            else:
                labels.append((trial_labels[idx], np.array(trial_labels_in_ms[eeg_starts[idx]:eeg_ends[idx]])))

        # Reshape trials into image-like 3-d data
        trials = [np.expand_dims(trial.transpose(1, 0), axis=-1) for trial in trials]

        if for_mne:
            return trials, labels, game_states
        else:
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

        if not self.dataset_operation['relabel']:
            if self.dataset_operation['selected_labels']:           # use the data with selected labels
                filtered_trials = [trial for trial, label in zip(trials, labels) if label[0] in self.dataset_operation['selected_labels']]
                filtered_labels = [label for label in labels if label[0] in self.dataset_operation['selected_labels']]
            else: # use all data
                #Get rid of intertrial period (label -1)
                filtered_trials = [trial for trial, label in zip(trials, labels) if (label[0] != -1)]
                filtered_labels = [label for label in labels if (label[0] != -1)]
        else:                                                       # select subset of data at trial level and change labels correspondingly
            mapping = {k: v[index] for k, v in self.dataset_operation['mapped_labels'].items()}
            flattened_mapping = {} # handle case where v is a list
            for k,v in mapping.items():
                if type(v) is list:
                    for i in v: flattened_mapping[str(i)+k] = i
                else: flattened_mapping[k] = v
            mapping = flattened_mapping # no change if mapping contains no list of label for single dataset
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
        '''

        all_trials, all_labels, all_kinds = [], [], []
        for i, (eeg_data, task_data, kind) in enumerate(data_dicts):
            trials, labels = self.cut_into_trials(eeg_data, task_data, kind, i)
            trials, labels = self.select_trials_and_relabel(trials, labels, i)
            kinds = len(labels) * [kind]
            all_trials.extend(trials)
            all_labels.extend(labels)
            all_kinds.extend(kinds)
        return all_trials, all_labels, all_kinds


def partition_data(labels, num_folds):
    '''Partition the indices of the trials into the number of folds. Data of different labels are balanced among folds. NOT partitioning the data of the trials.
    
    Parameters
    ----------
    labels: list of tuples
        The labels of each trial and every tick in this trial.
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

def augment_data_to_file(trials, labels, kinds, ids_folds, h5_file, config):
    '''For each fold of data, augment the data to a 5x large dataset by adding 4 separate noises to each data window. Store the downsampled augmented data into a .h5 file.

    Parameters
    ----------
    trials: list of arrays with shape (n_electrodes, n_samples, 1)
    labels: list of tuples
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

    window_length = config['augmentation']['window_length']
    stride = config['augmentation']['stride']
    new_samp_freq = config['augmentation']['new_sampling_frequency']
    num_noise = config['augmentation']['num_noise']
    detect_artifacts = config['artifact_handling']['detect_artifacts']
    reject_std = config['artifact_handling']['reject_std']

    window_size = int(new_samp_freq * window_length / 1000)
    portion = int(0.2 * window_length)
    labels_to_keep = set([label[0] for label in labels])
    #If auto using all labels in dataset and rotation being applied, add still state to list of labels to keep
    if (not config['dataset_generator']['dataset_operation']['relabel']) and (not config['dataset_generator']['dataset_operation']['selected_labels']) and ('CL' in kinds):
        labels_to_keep.add(4)
    
    with h5py.File(h5_file, 'w') as file:
        for fold, ids in enumerate(ids_folds):
            a_trials = [trials[j] for j in ids]
            a_labels = [labels[j] for j in ids]
            a_kinds = [kinds[j] for j in ids]

            if len(a_trials) != 0:
                n_electrodes = a_trials[0].shape[0]
            
            pbar = tqdm.tqdm(range(len(a_labels)))
            pbar.set_description("Augmenting fold " + str(fold))

            clean_counter = 0
            artifacts_detected = 0
            trial_data = []
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

                    if kind == 'OL':
                        new_label = label[0]
                    else:
                        new_label = decideLabelWithRotation(label_window)
                        if (new_label not in labels_to_keep):
                            window_start += stride
                            window_end += stride
                            continue
                    
                    #If detecting artifacts, skip this window if artifact detected
                    if detect_artifacts:
                        #Reshape data to [samples, electrodes] for artifact detection
                        if detect_artifact(np.reshape(trial_window, [window_length, n_electrodes]), 
                                        reject_std):
                            artifacts_detected += 1
                            window_start += stride
                            window_end += stride
                            continue
                    
                    #Add label and data to lists
                    dist.append(new_label)
                    trial_data.append(resample(trial_window, window_size, axis=1))
                    clean_counter += 1  

                    # generate noised data of this window and save
                    for j in range(num_noise):
                        noise = np.max(trial_window) * np.random.uniform(-0.5, 0.5, trial_window.shape)
                        trial_data.append(resample(trial_window + noise, window_size, axis=1))
                        dist.append(new_label)

                    window_start += stride
                    window_end += stride

            #Convert lists to arrays and save to h5 file
            eeg_array = np.array(trial_data, dtype=np.float32)
            label_array = np.array(dist, dtype=np.int32)
            file.create_dataset(str(fold)+'_trials', data=eeg_array)
            file.create_dataset(str(fold)+'_labels', data=label_array)
            #Only take non augmented / noisy windows into validation data    
            file.create_dataset(str(fold)+'_val_trials', data=eeg_array[::num_noise+1])
            file.create_dataset(str(fold)+'_val_labels', data=label_array[::num_noise+1])
            
            label_distribution = np.unique(dist, return_counts=True)
            if clean_counter == 0: artifact_rejection_percent = 0.
            else: artifact_rejection_percent = artifacts_detected / (clean_counter + artifacts_detected)

            print(f'Label distribution in fold {fold}:')
            print(np.unique(dist,return_counts=True))
            print(f'Share of windows rejected as containing artifacts = {artifact_rejection_percent}')
        
            
def create_dataset(config, h5_path):
    '''creates dataset as h5 file according to yaml_file and stores it at path given by h5_path
    
    Parameters
    ----------
    config: dict
        Configurations from the yaml file.
    num_folds: int
        path on which the h5 dataset will be stored

    Returns
    -------
    None
    '''
    
    preprocessor = DataPreprocessor(config['data_preprocessor'])
    dataset_generator = DatasetGenerator(config['dataset_generator'])

    # Read in and preprocess the data
    data_dicts = []
    for i, data in enumerate(config['data_names']):
        if 'data_kinds' in config:
            kind = config['data_kinds'][i] # Used when treating CL datasets as OL
        else:
            kind = decide_kind(data)
        eeg_data = read_data_file_to_dict(config['data_dir'] + data + "/eeg.bin")
        task_data = read_data_file_to_dict(config['data_dir'] + data + "/task.bin")
        eeg_data['databuffer'] = preprocessor.preprocess(eeg_data['databuffer'])    # preprocess
        data_dicts.append([eeg_data, task_data, kind])                                    # store in data_dicts

    # Generate the dataset on trial level
    trials, labels, kinds = dataset_generator.generate_dataset(data_dicts)

    # Partition data by partitioning the indices
    ids_folds = partition_data(labels, config['partition']['num_folds'])

    # Augment dataset according to each fold
    augment_data_to_file(trials, labels, kinds, ids_folds, h5_path, config)


if __name__ == "__main__":

    # put in yaml file name as input (i.e config.yaml)
    yaml_path = sys.argv[1]
    config = read_config(yaml_path)

    create_dataset(config, h5_path="./sampleDataset.h5")
    print(f"created sample dataset: sampleDataset.h5")