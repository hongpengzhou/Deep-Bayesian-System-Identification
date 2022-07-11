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

TS = 0.02 # sampling time
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
bias = True                         # Choice of bias in the NN linear maps
onoffLambdaDecay = False            # Regularization decay term
outerTimes = 10                      # Number of training rounds
epochs = 200                        # Number of epochs per training round
lr = 1e-2                           # Learning rate
lambdaDecayEpochs = 600             # Lambda decays each'lambdaDecayEpochs'
gammaThreshold = 1e-3               # Threshold for gamma pruning
weightThreshold = 1e-3              # Threshold for weight pruning
trainBatchSize = 290                # Choose batch size for train

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
lambdaOrg = {'lambda_in': 0.1 * np.array([1., 1., 1., 1., 1., 1., 1., 1., 1.]),
             'lambda_out': 0.1 * np.array([1., 1., 1., 1., 1., 1., 1., 1., 1.]),
             'lambda_l1': 0. * np.array([0., 0., 0., 0., 0, 0., 0., 0., 0.]),
             'lambda_l2': 0. * np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
             }
lambdaTmp = copy.deepcopy(lambdaOrg)
coefficient = torch.pow(torch.tensor([0.1]),
                        torch.tensor([indexLambda]).float(), out=None)
lambdaTmp['lambda_in'] *= coefficient.detach().numpy()
lambdaTmp['lambda_out'] *= coefficient.detach().numpy()

# Import training and testing data
loadData = scio.loadmat('./CED_Data/DATAUNIF')
u1 = torch.tensor(loadData['u11'], device=TORCH_MACHINE).float()
y1 = torch.tensor(loadData['z11'], device=TORCH_MACHINE).float()
u2 = torch.tensor(loadData['u12'], device=TORCH_MACHINE).float()
y2 = torch.tensor(loadData['z12'], device=TORCH_MACHINE).float()

# Take the percentage of data
N = 300
uEst1, yEst1, uVal1, yVal1 = lib.select_train_test_data(u1, y1, N)
uEst2, yEst2, uVal2, yVal2 = lib.select_train_test_data(u2, y2, N)

uEst1, yEst1 = lib.select_train_data(uEst1, yEst1, percentage)
uEst2, yEst2 = lib.select_train_data(uEst2, yEst2, percentage)

# Normalize Data
uEstData1, yEstData1, stdY, meanY, stdU, meanU = lib.clean_data(uEst1, yEst1,
                                                                onoffDetrend,
                                                                onoffNormalize)
uEstData2, yEstData2 = lib.data_normalize(uEst2, yEst2, meanU, stdU,
                                               meanY, stdY)
uValData1, yValData1 = lib.data_normalize(uVal1, yVal1, meanU, stdU,
                                               meanY, stdY)
uValData2, yValData2 = lib.data_normalize(uVal2, yVal2, meanU, stdU,
                                               meanY, stdY)

# Generate data according to lags
uEstDataPro1, yEstDataPro1 = lib.generate_prediction_data(uEstData1, yEstData1,
                                                          lagInput, lagOutput)
uEstDataPro2, yEstDataPro2 = lib.generate_prediction_data(uEstData2, yEstData2,
                                                          lagInput, lagOutput)
uEstDataPro = torch.cat((uEstDataPro1,uEstDataPro2), dim=0)
yEstDataPro = torch.cat((yEstDataPro1,yEstDataPro2), dim=0)

uValDataPro1, yValDataPro1 = lib.generate_prediction_data(uValData1, yValData1,
                                                          lagInput, lagOutput)
uValDataPro2, yValDataPro2 = lib.generate_prediction_data(uValData2, yValData2,
                                                          lagInput, lagOutput)

# Batchify training data
batches = lib.batchify(uEstDataPro, yEstDataPro, trainBatchSize)

# NN Structure
layers = [uEstDataPro1.shape[1], mlpLayer, mlpLayer, yEstDataPro1.shape[1]]
layersN = len(layers)-2

# Loop over number of experiments
for repIndex in range(repeatExperiment):

    minimalLoss = 1000

    # Initialize model and training objects
    model = MLPNetwork(layers, activation=torch.relu,
                        save_device=TORCH_MACHINE, bias=bias).to(TORCH_MACHINE)
    lambda_base = copy.deepcopy(lambdaTmp)
    algorithm = BayesianAlgorithm(model, lambda_base)
    optimizer = torch.optim.Adam(params=algorithm.model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           epochs * outerTimes)

    # Folder and filename
    folder = FOLDER_HEAD + 'CED_results_layers_' + str(layersN) + '_units_' + \
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
        mask = torch.ones(model.weights[name].size(), dtype=torch.bool,
                           device=TORCH_MACHINE)
        masks[name] = mask

    # Initialize for plots
    if PLOT:
        fig = plt.figure()
        plt.ion()
        plt.show()
    epochRecord = 0
    predLosses, trainLosses = [], []
    simTotalValLosses, predTotalValLosses = [], []

    for outer in range(outerTimes):

        # Initialize progress bar
        outerStr = 'TRAINING ROUND '+ str(outer+1) + '/' + str(outerTimes)
        epochBar = tqdm(range(epochs), position=0, leave=False,
                        total=epochs, desc=outerStr)

        # TRAIN
        for epoch in epochBar:
            onoffHessian = False
            epochRecord += 1
            if epoch == epochs - 1 and onoffRegu:
                onoffHessian = True

            trainLoss, predLoss, hessianAuto = lib.train(model, algorithm,
                                                         optimizer, batches,
                                                         onoffRegularization,
                                                         onoffHessian)

            # Plot training progress
            predLosses = np.append(predLosses, predLoss.cpu())
            trainLosses = np.append(trainLosses, trainLoss.cpu())
            if PLOT:
                plt.axis([0, outerTimes * epochs,
                          0, max(np.concatenate((predLosses, trainLosses)))])
                its = np.arange(epochRecord)
                predPlot, = plt.plot(its, predLosses, 'blue')
                trainPlot, = plt.plot(its, trainLosses, 'orange')
                plt.gca().grid(True, 'major', 'x')
                plt.gca().grid(True, 'both', 'y')
                plt.gca().set_title('Training Progress ( Round: ' + str(outer+1)
                                    + ' | Epoch: ' + str(epoch+1) + ')',
                             fontweight="bold")
                plt.gca().legend((predPlot, trainPlot),
                            ('Prediction Loss (MSE = {:2.4f})'.format(predLoss),
                             'Training Loss (MSE = {:2.4f})'.format(trainLoss)),
                            loc='upper right')
                plt.gca().set_ylabel('MSE')
                plt.gca().set_xlabel('iteration [' + str(epochRecord) + ']')
                fig.canvas.draw_idle()
                fig.canvas.start_event_loop(0.01)

            # Update progress bar
            epochBar.update(1)

            # Lambda Decay
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
        print('TRAINING MSE      : ', end='')
        lossEstPred1, predEst1 = lib.validate(model, uEstDataPro1,
                                              yEstDataPro1, stdY, meanY)
        print('{:2.15f} '.format(lossEstPred1), end='')
        lossEstPred2, predEst2 = lib.validate(model, uEstDataPro2,
                                              yEstDataPro2, stdY, meanY)
        print('{:2.15f} '.format(lossEstPred2), end='')

        # Validation with validation data
        print('  | VALIDATION MSE      :', end='')
        predVal1 = yVal1.clone()
        lossValPred1, predVal1[lag:] = lib.validate(model, uValDataPro1,
                                                    yValDataPro1, stdY, meanY)
        print('{:2.15f} '.format(lossValPred1), end='')
        predVal2 = yVal2.clone()
        lossValPred2, predVal2[lag:] = lib.validate(model, uValDataPro2,
                                                    yValDataPro2, stdY, meanY)
        print('{:2.15f} | '.format(lossValPred2), end='')

        lossValSim1, simVal1 = lib.validate_sim(model, uValData1, yValData1,
                                                stdY, meanY,
                                                lagInput, lagOutput)
        print('{:2.15f} '.format(lossValSim1), end='')
        lossValSim2, simVal2 = lib.validate_sim(model, uValData2, yValData2,
                                                stdY, meanY,
                                                lagInput, lagOutput)
        print('{:2.15f} '.format(lossValSim2), end='')

        # Append losses
        lossTotalPred = torch.sqrt(torch.nn.functional.mse_loss(
            torch.cat((predVal1, predVal2), 0),
            torch.cat((yVal1.cpu().float(),
                       yVal2.cpu().float()), 0))).item()

        lossTotal = torch.sqrt(torch.nn.functional.mse_loss(
            torch.cat((simVal1, simVal2), 0),
            torch.cat((yVal1.cpu().float(),
                       yVal2.cpu().float()), 0))).item()

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
            elif outer == 0:
                sign = '_convregu'
            else:
                sign = '_bestregu'

            # Save data for outers
            scio.savemat(filename + sign + '_validation.mat',
                         {
                             'realEst1': yEst1.cpu().numpy(),
                             'realEst2': yEst2.cpu().numpy(),
                             'predEst1': predEst1.cpu().numpy(),
                             'predEst2': predEst2.cpu().numpy(),
                             'realVal1': yVal1.cpu().numpy(),
                             'realVal2': yVal2.cpu().numpy(),
                             'predVal1': predVal1.cpu().numpy(),
                             'predVal2': predVal2.cpu().numpy(),
                             'simVal1': simVal1.cpu().numpy(),
                             'simVal2': simVal2.cpu().numpy(),
                         })
            if outer > 0:
                modelDict['outer'] = outer + 1
                torch.save(modelDict, filename + sign + '_model.tar')

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