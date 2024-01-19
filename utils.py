import sys
import os
import numpy as np
from scipy import signal
import ast
import random
import yaml
import copy
from collections import Counter
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def model_namer(model_dir, model_arch_name):
    '''Generates a unique two word name based on the inbuilt unix dictionary. 
    
    Parameters
    ----------
    model_dir: string directory where model will be stored
    model_arch_name: string
        Name of the model architecture.

    Returns
    -------
    new_name: string
        Generated two-word model name with model architecture's name, e.g. wooden_jazz_EEGNet.
    '''

    # get previously used names
    existing_models = os.listdir(model_dir)
    used_names = [model.split('_')[0] + '_' + model.split('_')[1] for model in existing_models]
    
    # import word list from unix install
    with open('/usr/share/dict/words') as f:
        words = f.read().splitlines()
    words = [word.lower() for word in words if "'" not in word]
    
    unique_name = False
    while not unique_name:
        new_name = random.choice(words) + '_' + random.choice(words)
        if new_name not in used_names:
            unique_name = True

    new_name = new_name + '_' + model_arch_name

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


class ArtifactTuner():
    '''This object is designed to be called during online sessions in order to adjust the 
    standard deviation threshold and hit a targeted percentage of ticks which will 
    be identified as artifacts.'''
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.reject_std = self.config['reject_std']
        self.apply_adjustment = self.config['tune_rejection_threshold']['apply']
        self.ticks_till_adjustment = self.config['tune_rejection_threshold']['ticks_per_adjustment']
        self.percentile_goal = 100 - self.config['tune_rejection_threshold']['share_of_ticks_to_reject']
        self.std_dev_maxes = []

    def detect_artifact(self, eeg_data):
        '''This function is designed to be called every tick. It should be passed the results of the 
        artifact detector on this tick's data (i.e., either True or False for whether 
        this tick's data was flagged as an artifact).
        
        This function expects to be passed windows of eeg_data that are already normalized'''
        #Check if any element in the data array exceeds standard dev threshold
        artifact = np.any(np.fabs(eeg_data) > self.reject_std)
        #If adjusting threshold over time, do so
        if self.apply_adjustment:
            #Capture needed data to make threshold adjustment decision
            self.ticks_till_adjustment -= 1
            self.std_dev_maxes.append(np.max(eeg_data))
            #If ticks to adjustment is full, adjust reject_std going forward
            if self.ticks_till_adjustment == 0:
                #Find rejection threshold that would have resulted in desired rejection percent and use going forward
                maxes = np.array(self.std_dev_maxes)
                self.reject_std = np.percentile(a=maxes, q=self.percentile_goal)

                #Reset for next period of adjustment
                self.ticks_till_adjustment = copy.deepcopy(self.config['tune_rejection_threshold']['ticks_per_adjustment'])
        #Return whether to flag this tick's data as an artifact
        return artifact


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
    if 4 in states: labels[states == 4] = label2num['still']

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


def decideLabelWithRotation(label_window, 
                            preponderance=0.8,
                            final_portion=0.2):
    '''Evaluates a window of a closed loop experiment to determine what final 
    label should be assigned.
    
    Takes in a label_window which should be a list of labels for every ms which represents 
    the relative position of the cursor to the target in that ms.

    preponderance is the share of labels in the window that must be a single label for 
    the window to be assigned that label.

    final_portion is used when no label appears more than the preponderance share, 
    and instead assigns the most common label in the final_portion of the label_window
    
    Returns the single label that should be assigned to that window.'''
    portion = int((1 - final_portion) * len(label_window))

    new_label, label_num = Counter(label_window).most_common(1)[0]
    if label_num / len(label_window) < 0.8:
        new_label = Counter(label_window[portion:]).most_common(1)[0][0]
    return new_label


def create_confusion_matrix(pred_labels,true_labels,file_name=None):
    """
    pred_labesl : list of integer labels
    true_labels : list of integer labels
    file_name : None means don't save only display
    otherwise, give it a path to save "confusion-matrix-1.jpg"
    """
    
    # class labels
    class_labels = list(set(pred_labels))

    # creates confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    accuracy = cm / np.sum(cm, axis=1, keepdims=True)

    
    fig, ax = plt.subplots()
    # im = ax.imshow(cm, cmap='Blues')
    im = ax.imshow(accuracy, cmap='Blues', vmin = 0, vmax = 1) # Use accuracy values and set vmin/vmax

    # Add accuracy values to the plot
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({accuracy[i, j]:.2f})", ha='center', va='center')

    # Set axis labels and title
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.xticks(np.arange(len(class_labels)), class_labels)
    plt.yticks(np.arange(len(class_labels)), class_labels)
    # Add colorbar
    ax.figure.colorbar(im, ax=ax)

    # Display the plot
    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name)

    return fig