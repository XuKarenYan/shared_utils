import numpy as np
from scipy.signal import resample
import sys
import tqdm
try:
    # relative path needed if to be used as module
    from .utils import (read_data_file_to_dict, detect_artifact, decide_kind, 
                        read_config, generateLabelWithRotation, decideLabelWithRotation)
    from .preprocessor import DataPreprocessor
except ImportError:
    # absolute path selected if to run as stand alone shared_utils
    from utils import (read_data_file_to_dict, detect_artifact, decide_kind, 
                       read_config, generateLabelWithRotation, decideLabelWithRotation)
    from preprocessor import DataPreprocessor

class PreprocessLikeClosedLoop:
    '''This class handles takes one or more data files and transforms them into 
    a shape where a model can evaluate them like raspy does, i.e., tick by tick.

    Uses a dictionary of configuration settings which is generally intended to 
    be created from a yaml file.

    Note that this Class can only operate on a single dataset at a time. Only the 
    first data name in config['data_names'] will be used

    Example
    -------
    preprocessor = PreprocessLikeClosedLoop(config_dict)
    eeg_trials, eeg_trial_labels = preprocessor.preprocess(eeg_data, task_data)
    '''

    def __init__(self, config):
        '''Initializes DataPreprocessor with given arguments from the config file.

        Note that it is currently only designed to function on a single data set at a time

        Parameters
        ----------
        config: dict
            Configurations from the yaml file.

        self.first_run is used to flag that the preprocess function has not yet 
        been run for online experiments.
        '''
        
        #This class is only designed to operate in online mode, overwrite that config setting if not
        config['data_preprocessor']['online_status'] = 'online'
        #Save config to pass on to other classes
        self.config = config
        self.omit_angles = config['dataset_generator']['omit_angles']
        self.eeg_cap_type = config['data_preprocessor']['eeg_cap_type']
        self.ch_to_drop = config['data_preprocessor']['ch_to_drop']
        self.online_status = config['data_preprocessor']['online_status']
        self.window_length = config['dataset_generator']['window_length']
        self.normalizer_type = config['data_preprocessor']['closed_loop_settings']['normalizer_type']
        self.labels_to_keep = config['data_preprocessor']['closed_loop_settings']['labels_to_keep']
        self.relabel_pairs = config['data_preprocessor']['closed_loop_settings']['relabel_pairs']
        self.detect_artifacts = config['artifact_handling']['detect_artifacts']
        self.reject_std = config['artifact_handling']['reject_std']
        self.initial_ticks = config['data_preprocessor']['closed_loop_settings']['initial_ticks']
        self.resample_rate = int(config['augmentation']['new_sampling_frequency'] * self.window_length / 1000)
        self.first_run = True
        self.data_dir = config['data_dir']
        self.data_name = config['data_names'][0]
        if len(config['data_names']) > 1: print(f'This class only designed to work on one dataset at a time. Proceeding using first data name provided., {self.data_name}')

    def generate_dataset(self):
        '''This function preprocesses data in the same way it happens online in a 
        closed loop experiment. In other words, it iterates over the data tick by 
        tick an processes it using only data available at that point in time.

        eeg_data and task_data should be dictionaries extracted from a data file 
        using read_data_file_to_dict'''

        #Generate eeg and task data dictionaries for test dataset
        data_path = self.data_dir + self.data_name
        eeg_data = read_data_file_to_dict(data_path + "/eeg.bin")
        task_data = read_data_file_to_dict(data_path + "/task.bin")
        
        #For closed loop experiments, generate label every ms of data
        if 'data_kinds' in self.config:
            kind = self.config['data_kinds'][0] # Used when treating CL datasets as OL
        else:
            kind = self.decide_kind(self.data_name)
        if kind == 'CL':
            trial_labels_in_ms = generateLabelWithRotation(task_data, self.omit_angles)

        #Instantiate preprocessor to drop channels, laplacian filter, and normalize data
        preprocessor = DataPreprocessor(self.config['data_preprocessor'])
        
        #Create empty lists to hold the data we will return from function
        eeg_trials = []  # will hold each trial of eeg data
        eeg_trial_labels = [] # will hold the label for each trial
        #Create counter to track how many artifact windows are detected
        artifact_counter = 0
        #Create list of bad labels we don't want to consider
        if kind == 'CL':
            all_labels = np.unique(trial_labels_in_ms)
        else:
            all_labels = np.unique(task_data['state_task'])
        #Add intertrial periods to list of bad labels
        bad_labels = [-1]
        #Add other labels we want to exclude to bad labels
        if self.labels_to_keep != ['all']:
            also_bad = [label for label in all_labels.tolist() if label not in self.labels_to_keep]
            bad_labels = bad_labels + also_bad


        #Iterate across the ticks of the test dataset and process them
        pbar = tqdm.tqdm(range(task_data['eeg_step'].shape[0]))
        pbar.set_description('iterating across all ticks to preprocess like closed loop')
        for tick in pbar:
            end_ix = task_data['eeg_step'][tick]

            #If not enough data to make prediction, move to next tick
            if end_ix < self.window_length:
                continue

            #If we have enough data in the databuffer, use it
            else:
                #Extract the latency period that decoder would see closed loop
                start_ix = end_ix - self.window_length
                data = eeg_data['databuffer'][start_ix:end_ix, :]

            #Drop channels and normalize the data
            data = preprocessor.preprocess(data)
            
            #Create flag for whether artifact detected in this tick
            artifact = False
            
            #Adjust counter; check if min ticks passed to start artifact detection
            #Detect artifacts function also adds data to normalizer if it is good
            self.initial_ticks -= 1
            if self.initial_ticks <= 0:
                #If detecting artifacts, do so
                if self.detect_artifacts:
                    artifact = detect_artifact(data,
                                            reject_std=self.reject_std)
            #If artifact detected, move on to next window
            if artifact:
                artifact_counter += 1
                continue
            
            #Else no artifact; add data to normalizer and get label
            else:
                #If passed initial ticks, include this data in running normalizer calcs of mean and std
                if self.initial_ticks <= 0:
                    preprocessor.normalizer.include(data)

                #Get label for this tick
                if kind == 'CL':
                    label = decideLabelWithRotation(trial_labels_in_ms[start_ix:end_ix])
                else:
                    label = task_data['state_task'][tick]

                #Check if label in bad_labels; if so move to next tick
                if label in bad_labels:
                    continue

                #If keeping only first 2 seconds of each trial, assume 20ms ticks and overwrite label to drop
                if kind == 'first2sec' and len(eeg_trial_labels) > 100:
                    if (label == eeg_trial_labels[-1]) and (len({ele[0] for ele in eeg_trial_labels[-100:]}) == 1):
                        continue
                
                #Relabel this window if needed
                if self.relabel_pairs != [None]:
                    for pair in self.relabel_pairs:
                        if label == pair[0]:
                            label = pair[1]

                #Downsample to final desired rate
                data = resample(data, self.resample_rate, axis=0)

                #Save data to our running lists
                eeg_trials.append(data)
                eeg_trial_labels.append(label)
    
        #Calculate share of ticks rejected as artifacts
        total_ticks = len(eeg_trials) + artifact_counter
        artifact_percent = artifact_counter / total_ticks
        
        #Convert data and labels to arrays
        eeg_trials = np.array(eeg_trials)
        eeg_trial_labels = np.array(eeg_trial_labels)

        #Return share of ticks rejected as artifacts, eeg_data, and labels
        print(f'Share of trials that were rejected as artifacts = {artifact_percent}')
        print(f'Label distribution: {np.unique(eeg_trial_labels, return_counts=True)}')
        return artifact_percent, eeg_trials, eeg_trial_labels