## Purpose:
to unify preprocessing and data creation methods among different model training (Offline EEGNet, Riemannian Decoder) and model testing (raspy) pipelines

Please run each of the python file to get a grasp of how they work.

* **dataset.py**: contains everything relevant to creating h5 dataset. main function is create_dataset(config,h5_path)
* **preprocessor.py**: contains everything that has to do with preprocessing a piece of EEG data
* **utils.py**: contains standalone random things that is helpful to data reading in general
