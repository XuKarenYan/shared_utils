
from scipy.signal import resample
import numpy as np
try:
    # relative path needed if to be used as module
    from .preprocessor import DataPreprocessor
    from .utils import read_data_file_to_dict, detect_artifact, decide_kind
    from .dataset import  generateLabelWithRotation, decideLabelWithRotation
except ImportError:
    # absolute path selected if to ran as stand alone shared_utils
    from preprocessor import DataPreprocessor
    from utils import read_data_file_to_dict, detect_artifact, decide_kind
    from dataset import generateLabelWithRotation, decideLabelWithRotation

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
        self.data_folder = config['data_dir']
        if len(config['data_names']) > 1:
            print(f"This Class can only operate on a single dataset at a time. Only {config['data_names'][0]} is being used")
        self.data_name = config['data_names'][0]
        self.kind = decide_kind(self.data_name)
        self.omit_angles = config['dataset_generator']['omit_angles']
        self.eeg_cap_type = config['data_preprocessor']['eeg_cap_type']
        self.ch_to_drop = config['data_preprocessor']['ch_to_drop']
        self.online_status = 'online'
        self.labels_to_keep = config['labeling']['labels_to_keep']
        self.relabel_pairs = config['labeling']['relabel_pairs']
        self.window_length = config['dataset_generator']['window_length']
        self.normalizer_type = config['data_preprocessor']['closed_loop_settings']['normalizer_type']
        self.detect_artifacts = config['artifact_handling']['detect_artifacts']
        self.reject_std = config['artifact_handling']['reject_std']
        self.initial_ticks = config['artifact_handling']['initial_ticks']
        self.resample_rate = int(config['augmentation']['new_sampling_frequency'] * self.window_length / 1000)
        self.first_run = True

    def generate_dataset(self):
        '''This function preprocesses data in the same way it happens online in a 
        closed loop experiment. In other words, it iterates over the data tick by 
        tick an processes it using only data available at that point in time.

        data_dir should be an absolute path to a folder within which which eeg and task data is stored.
        e.g., "/data/raspy/DATA_DIR"'''

        eeg_data = read_data_file_to_dict(self.data_folder + self.data_name + "/eeg.bin")
        task_data = read_data_file_to_dict(self.data_folder + self.data_name + "/task.bin")

        #Some data has shape XXXX, 1 instead of flat state_task - flatten if so
        if task_data['state_task'].ndim == 2:
            task_data['state_task'] = task_data['state_task'].flatten()
        #For closed loop experiments, generate label every ms of data
        if self.kind == 'CL':
            trial_labels_in_ms = generateLabelWithRotation(task_data, self.omit_angles)

        #Instantiate preprocessor to drop channels, laplacian filter, and normalize data
        preprocessor = DataPreprocessor(self.config['data_preprocessor'])
        
        #Create empty lists to hold the data we will return from function
        eeg_trials = []  # will hold each trial of eeg data
        eeg_trial_labels = [] # will hold the label for each trial
        #Create counter to track how many artifact windows are detected
        artifact_counter = 0
        #Create list of bad labels we don't want to consider
        if self.kind == 'CL':
            all_labels = np.unique(trial_labels_in_ms)
        else:
            all_labels = np.unique(task_data['state_task'])
        #Add intertrial periods to list of bad labels
        bad_labels = [-1]
        #Add other labels we want to exclude to bad labels
        if self.labels_to_keep != ['all']:
            also_bad = [label for label in all_labels.list() if label not in labels_to_keep]
            bad_labels = bad_labels + also_bad


        #Iterate across the ticks of the test dataset and process them
        print('iterating across all ticks to preprocess like closed loop')
        for tick in range(task_data['eeg_step'].shape[0]):
            end_ix = task_data['eeg_step'][tick]

            #If not enough data to make prediction, move to next tick
            if end_ix < self.window_length:
                continue

            #If we have enough data in the databuffer, use it
            else:
                #Extract the latency period - that decoder would see closed loop
                start_ix = end_ix - self.window_length
                data = eeg_data['databuffer'][start_ix:end_ix, :]

            #Normalize the data
            data = preprocessor.preprocess(data)

            #Create flag for whether artifact detected in this tick
            artifact = False
            
            #Adjust counter; check if min ticks passed to start artifact detection
            #Detect artifacts function also adds data to normalizer if it is good
            self.initial_ticks -= 1
            if self.initial_ticks <= 0 and self.detect_artifacts:
                artifact = detect_artifact(data,
                                           reject_std=self.reject_std)
            #If artifact detected, move on to next window
            if artifact:
                artifact_counter += 1
                continue
            
            #Else no artifact; add data to normalizer and get label
            else:
                preprocessor.normalizer.include(data)

                #Get label for this tick
                if self.kind == 'CL':
                    label = decideLabelWithRotation(trial_labels_in_ms[start_ix:end_ix])
                else:
                    label = task_data['state_task'][tick]

                #Check if label in bad_labels; if so move to next tick
                if label in bad_labels:
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
        return artifact_percent, eeg_trials, eeg_trial_labels