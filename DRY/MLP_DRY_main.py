import scipy.io as scio
from tqdm import tqdm
import numpy as np
import torch
import copy
import sys
import os

sys.path.append('../')
import lib as lib
from model import MLPNetwork
from algorithm import BayesianAlgorithm
from torch.nn.utils import prune

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 9
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams["font.serif"] = 'times'

TS = 0.08 # sampling time
FOLDER_HEAD = './MLP/'
PROGRESS_BAR_WIDTH = 100
TORCH_MACHINE = torch.device('cpu')
PLOT = False

# Input Parameters:
indexLambda = float(sys.argv[1])    # Regularization parameter
lag = int(sys.argv[2])              # Chosen lags for regressors
onoffRegu = int(sys.argv[3])        # Regularization boolean
percentage = float(sys.argv[4])     # Percentage of data used
repeatExperiment = int(sys.argv[5]) # Number of time to repeat experiment
lagInput = lagOutput = lag

# Training settings
mlpLayer = 50                       # Number of hidden units per layer
bias = False                        # Choice of bias in the NN linear maps
onoffLambdaDecay = False            # Regularization decay term
epochs = 500                        # Number of epochs per training round
outerTimes = 6                      # Number of training rounds
lr = 1e-2                           # Learning rate
lambdaDecayEpochs = 1000            # Lambda decay each'lambdaDecayEpochs'
gammaThreshold = 1e-3               # Threshold for gamma pruning
weightThreshold = 1e-3              # Threshold for weight pruning
trainBatchSize = 300                # Choose batch size for each train

# Other relevant data parameter Setting
onoffDetrend = False                 # Detrend data (*-mean)
onoffNormalize = False               # Normalize data (*-mean)/std

# Methods Choice
if onoffRegu :
    folderReguSign = 'regu'
    onoffRegularization = {'group_in': True, 'group_out': True,
                           'l1': False, 'l2': False}
else:
    folderReguSign = 'noregu'
    outerTimes = 1  # Number of training rounds overriden
    onoffRegularization = {'group_in': False, 'group_out': False,
                           'l1': False, 'l2': False}

# Lambda Initialization
lambdaStack = []
lambdaOrg = {'lambda_in': 0.1 * np.array([1., 1., 1., 1., 1.]),
             'lambda_out': 0.1 * np.array([1., 1., 1., 1., 1.]),
             'lambda_l1': 0. * np.array([0., 0., 0., 0., 0, 0.]),
             'lambda_l2': 0. * np.array([0., 0., 0., 0., 0., 0.])
             }
lambdaTmp = copy.deepcopy(lambdaOrg)
coefficient = torch.pow(torch.tensor([0.1]),
                        torch.tensor([indexLambda]).float(), out=None)
lambdaTmp['lambda_in'] *= coefficient.detach().numpy()
lambdaTmp['lambda_out'] *= coefficient.detach().numpy()

# Import training and testing data
loadData = scio.loadmat('DRY_Data/dry.mat')
uEst = torch.tensor(loadData['ue'], device=TORCH_MACHINE).float()
yEst = torch.tensor(loadData['ye'], device=TORCH_MACHINE).float()
uVal = torch.tensor(loadData['uv'], device=TORCH_MACHINE).float()
yVal = torch.tensor(loadData['yv'], device=TORCH_MACHINE).float()

# Take the percentage of data
uEst, yEst = lib.select_train_data(uEst, yEst, percentage)

# Normalize Data
uEstData, yEstData, stdY, meanY, stdU, meanU = lib.clean_data(uEst, yEst,
                                                              onoffDetrend,
                                                              onoffNormalize)
uValData, yValData = lib.data_normalize(uVal, yVal, meanU, stdU, meanY, stdY)

# Generate data according to lags
uEstDataPro, yEstDataPro = lib.generate_prediction_data(uEstData, yEstData,
                                                        lagInput, lagOutput)
uValDataPro, yValDataPro = lib.generate_prediction_data(uValData, yValData,
                                                        lagInput, lagOutput)

# Batchify training data
batches = lib.batchify(uEstDataPro, yEstDataPro, trainBatchSize)

# NN Structure
layers = [uEstDataPro.shape[1], mlpLayer, yEstDataPro.shape[1]]
layersN = len(layers)-2


# Loop over number of experiments
for repIndex in range(repeatExperiment):
    minimalLoss = 1000

    # Initialize model and training objects
    model = MLPNetwork(layers, activation=None, bias=bias,
                        save_device=TORCH_MACHINE).to(TORCH_MACHINE)
    lambda_base = copy.deepcopy(lambdaTmp)
    algorithm = BayesianAlgorithm(model, lambda_base)
    optimizer = torch.optim.Adam(params=algorithm.model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           epochs * outerTimes)

    # Folder and filename
    folder = FOLDER_HEAD + 'DRY_results_layers_' + str(layersN) + '_units_' + \
             str(mlpLayer) + '/' + 'percentage_' + str(percentage) + '/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    if onoffRegu:
        filename = folder + 'lag_' + str(lag) + '_lambda_' \
                   + str(indexLambda+1) + '_repeat_' + str(repIndex+1)
    else:
        filename = folder + 'lag_' + str(lag) + '_repeat_' + str(repIndex+1)
    print(filename)

    # Initialize mask for Weight and Uncertainty Pruning
    masks = {}
    for name in model.names:
        mask = torch.ones(model.weights[name].size(),
                           dtype=torch.bool, device=TORCH_MACHINE)
        masks[name] = mask

    # Initialize for plots
    if PLOT:
        fig = plt.figure()
        plt.ion()
        plt.show()
    epochRecord = 0
    predLosses, trainLosses = [], []
    simTotalValLosses, predTotalValLosses = [], []

    # LOOP
    for outer in range(outerTimes):

        # Initialize progress bar
        outerStr = 'TRAINING ROUND ' + str(outer + 1) + '/' + str(
            outerTimes)
        epochBar = tqdm(range(epochs), position=0, leave=False,
                        total=epochs, desc=outerStr)

        # TRAIN
        for epoch in epochBar:
            onoffHessian = False
            epochRecord += 1
            if epoch == epochs - 1 and onoffRegu:
                onoffHessian = True

            trainLoss, predLoss, hessianAuto = lib.train(model, algorithm,
                                                         optimizer,
                                                         batches,
                                                         onoffRegularization,
                                                         onoffHessian)

            # Plot training progress
            predLosses = np.append(predLosses, predLoss.cpu())
            trainLosses = np.append(trainLosses, trainLoss.cpu())
            if PLOT:
                plt.axis([0, outerTimes * epochs,
                          0,
                          max(np.concatenate((predLosses, trainLosses)))])
                its = np.arange(epochRecord)
                predPlot, = plt.plot(its, predLosses, 'blue')
                trainPlot, = plt.plot(its, trainLosses, 'orange')
                plt.gca().grid(True, 'major', 'x')
                plt.gca().grid(True, 'both', 'y')
                plt.gca().set_title(
                    'Training Progress (Round: ' + str(outer + 1)
                    + ' | Epoch: ' + str(epoch + 1) + ')',
                    fontweight="bold")
                plt.gca().legend((predPlot, trainPlot),
                                 ('Prediction Loss (MSE = {:2.4f})'.format(
                                     predLoss),
                                  'Training Loss (MSE = {:2.4f})'.format(
                                      trainLoss)),
                                 loc='upper right')
                plt.gca().set_ylabel('MSE')
                plt.gca().set_xlabel(
                    'iteration [' + str(epochRecord) + ']')
                fig.canvas.draw_idle()
                fig.canvas.start_event_loop(0.01)

            # Update progress bar
            epochBar.update(1)

            # Lambda decay
            if onoffLambdaDecay and (epochRecord % lambdaDecayEpochs == 0):
                for key, value in lambda_base.items():
                    lambda_base[key] *= 0.1
                algorithm.lambda_base = lambda_base

            # Scheduler step
            scheduler.step()

        # Update algorithm parameters based on hessian
        if (outer < (outerTimes - 1)) and onoffRegu:
            algorithm.update(hessianAuto, onoffRegularization)

        # Prune
        for name, module in model.named_modules():
            if "linear" in name:
                maskGamma = torch.abs(algorithm.sum_gamma[name]) > gammaThreshold
                maskWeight = torch.abs(model.weights[name]) > weightThreshold
                mask = maskGamma & maskWeight
                masks[name] = mask
                model.weights[name].data = model.weights[name].masked_fill(
                    ~mask, 0)
                prune.custom_from_mask(module, "weight", mask)

        # TEST
        print('TRAINING MSE      :', end='')
        lossEstPred, predEst = lib.validate(model, uEstDataPro,
                                            yEstDataPro, stdY, meanY)
        print(' {:2.15f}'.format(lossEstPred), end='')

        # Validation with validation data
        print('    |    VALIDATION MSE      :', end='')
        predVal = yVal.clone()
        lossValPred, predVal[lag:] = lib.validate(model, uValDataPro,
                                                  yValDataPro, stdY, meanY)
        print(' {:2.15f}     '.format(lossValPred), end='')

        lossValSim, simVal = lib.validate_sim(model, uValData, yValData,
                                              stdY, meanY, lagInput, lagOutput)
        print('{:2.15f}'.format(lossValSim), end='')

        # Append losses
        lossTotalPred = lossValPred
        lossTotal = lossValSim

        predTotalValLosses.append(lossTotalPred)
        simTotalValLosses.append(lossTotal)

        # Save the model every outer if regu
        pruned_model = copy.deepcopy(model)
        for name, module in pruned_model.named_modules():
            if "linear" in name:
                prune.remove(module, "weight")
        modelDict = {'model_state_dict': pruned_model.state_dict(),
                     'lambdaTmp': lambdaTmp,
                     'masks': masks,
                     'hessianAuto': hessianAuto,
                     'gamma': algorithm.gamma,
                     'omega': algorithm.omega,
                     'alpha': algorithm.alpha,
                     'sum_gamma': algorithm.sum_gamma
                     }
        if onoffRegu:
            torch.save(modelDict, filename + '_outer_' + str(outer + 1)
                       + '_model.tar')

        # Save the model with smallest validation error
        if lossTotal < minimalLoss:
            print('     Best model saved.')
            minimalLoss = lossTotal

            # Set convregu/noregu/bestregu sign
            if not onoffRegu:
                sign = '_noregu'
                torch.save(modelDict, filename + sign + '_model.tar')
            elif outer == 0:
                sign = '_convregu' # Model already saved in outers
            else:
                sign = '_bestregu'
                modelDict['outer'] = outer + 1
                torch.save(modelDict, filename + sign + '_model.tar')

            # Save data for outers
            scio.savemat(filename + sign + '_validation.mat',
                         {
                             'realEst': yEst.cpu().numpy(),
                             'predEst': predEst.cpu().numpy(),
                             'realVal': yVal.cpu().numpy(),
                             'predVal': predVal.cpu().numpy(),
                             'simVal': simVal.cpu().numpy(),
                         })

        else:
            print('')
    # Save training progress
    if onoffRegu:
        scio.savemat(filename + '_training_hyperparameters.mat', {
            'epochs': epochs,
            'trainBatchSize': trainBatchSize,
            'outer_times': outerTimes,
            'lr': lr,
            'onoffLambdaDecay': onoffLambdaDecay,
            'lambda_decay_epochs': lambdaDecayEpochs,
            'gamma_threshold': gammaThreshold,
            'weight_threshold': weightThreshold,
        })
        scio.savemat(filename + '_training_results.mat', {
            'predTotalValLosses': predTotalValLosses,
            'simTotalValLosses': simTotalValLosses,
            'predLosses': predLosses,
            'trainLosses': trainLosses
        })
    # Turn interactive PLOT off
    if PLOT:
        plt.ioff()