This repository includes the python code for the Automatica paper "Sparse Bayesian Deep Learning forDynamic System Identification". The code can be implemented in the deep learning framework, PyTorch. The details of the repository are as follows:


1) model.py builds the MLP and LSTM model for each benchmark.

2) algorithm.py includes the details about Bayesian approach. 

3) lib.py includes the necessary functions which will be used in algorithm.py and main file in each sub-folder.

4) Five sub-folders are included, where CED/, CT/ are two nonlinear systems representing Coupled Electric Drives and Cascaded Tanks, respectively; DRY/, GT/ and HEX/ are three linear systems representing Hairdryer, Glass Tube (GT) manufacturing and Heat exchanger, respectively.

Within each sub-folder, the main files for implementing an MLP and LSTM model are included, which can be run from the terminal as follows:

python .\MLP_DRY_main.py 5 5 1 1 2 

or

python .\LSTM_DRY_main.py 5 5 1 1 2 

The results will be saved in each folder automatically.




