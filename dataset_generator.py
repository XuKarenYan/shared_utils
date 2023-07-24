import numpy as np

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