from shared_utils import utils
import os
import csv
import torch

def load_EEGNet_model(model_folder,
                      device,
                      output_dim,
                      fold=None):
    '''Imports a save pytorch model and loads it to device.
    
    model_folder should be a path to a folder which contains:

    1. a results.csv showing the validation accuracy of each fold of the model
    2. Folds of the model, which are saved torch state dicts with model weights
    3. A config yaml file which includes the parameters needed to reinstantiate the model. There must be only one yaml file in the model_folder
    4. An EEGNet.py file containing the model architecture

    device is the device to which the model should be loaded (cpu or gpu)

    if fold is specified, it will instantiate that fold. Otherwise will instantiate 
    fold with the highest validation accuracy as recorded in results.csv

    returns the instantiated model
    
    '''

    #Folder where function should look for definition of EEGNet object
    model_contents = os.listdir(model_folder)
    
    #Get paths to all needed files 
    yaml_file = [content for content in model_contents if '.yaml' in content][0]
    yaml_path = os.path.join(model_folder, yaml_file)
    EEGNet_path = os.path.join(model_folder, 'EEGNet.py')
    results_path = os.path.join(model_folder, 'results.csv')
    
    #Create config dictionary from model yamls
    config = utils.read_config(yaml_path)
    config_model = config['model']
    
    #Create model architecture
    spec = importlib.util.spec_from_file_location('EEGNet', EEGNet_path)
    Decoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(Decoder)
    DecoderClass = getattr(Decoder, Decoder.name)

    #Get number of electrodes to use
    ch_to_drop = config['data_preprocessor']['ch_to_drop']
    n_electrodes = 66 - len(ch_to_drop)
    
    #Load model
    model = DecoderClass(config_model, output_dim, n_electrodes)
    model.eval()
    #Models trained in Data Parallel, so set to match, but on single core to avoid errors
    model = torch.nn.DataParallel(model, device_ids=[0])

    #Get highest performing fold if not explicitly called
    if not fold:
        with open(results_path) as results:
            results_dict = csv.DictReader(results)
            highest_val_acc = 0
            best_fold = None
            for row in results_dict:
                if row['Validation Acc'] > highest_val_acc:
                    best_fold = row['Validation Fold']
    model_path = os.path.join(model_folder, best_fold)

    #Load existing params to model on either gpu or cpu
    model.load_state_dict(torch.load(model_path,
                                     map_location=device))
    
    return model