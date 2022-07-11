import torch
import numpy as np
import scipy.io as scio
import matplotlib
import matplotlib.pyplot as plt
import os
if not os.path.exists("./figures"):
    os.makedirs("./figures")

import sys
sys.path.append('../')
import lib as lib
from model import MLPNetwork, RNNNetwork

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams["savefig.format"] = 'pdf'

TORCH_MACHINE = torch.device('cpu')
TS = 0.02 # sampling time

# Universal parameters
lag = 10                            # Chosen lags for regressors
percentage = float(1)               # Percentage of data used
lag_input = lag_output = lag

onoff_detrend = False                 # Detrend data (*-mean)
onoff_normalize = False               # Normalize data (*-mean)/std

# Import training and testing data
load_data = scio.loadmat('./CED_Data/DATAUNIF')
u1 = torch.tensor(load_data['u11'], device=TORCH_MACHINE).float()
y1 = torch.tensor(load_data['z11'], device=TORCH_MACHINE).float()
u2 = torch.tensor(load_data['u12'], device=TORCH_MACHINE).float()
y2 = torch.tensor(load_data['z12'], device=TORCH_MACHINE).float()

# Take the percentage of data
N = 300
u_est1, y_est1, u_val1, y_val1 = lib.select_train_test_data(u1, y1, N)
u_est2, y_est2, u_val2, y_val2 = lib.select_train_test_data(u2, y2, N)

u_est1, y_est1 = lib.select_train_data(u_est1, y_est1, percentage)
u_est2, y_est2 = lib.select_train_data(u_est2, y_est2, percentage)

# Normalize Data
u_est_data1, y_est_data1, std_y, mean_y, std_u, mean_u = lib.clean_data(
    u_est1, y_est1, onoff_detrend, onoff_normalize)
u_est_data2, y_est_data2 = lib.data_normalize(u_est2, y_est2, mean_u, std_u,
                                               mean_y, std_y)
u_val_data1, y_val_data1 = lib.data_normalize(u_val1, y_val1, mean_u, std_u,
                                               mean_y, std_y)
u_val_data2, y_val_data2 = lib.data_normalize(u_val2, y_val2, mean_u, std_u,
                                               mean_y, std_y)

# Generate data according to lags
u_est_data_pro1, y_est_data_pro1 = lib.generate_prediction_data(u_est_data1, y_est_data1,
                                                          lag_input, lag_output)
u_est_data_pro2, y_est_data_pro2 = lib.generate_prediction_data(u_est_data2, y_est_data2,
                                                          lag_input, lag_output)
u_val_data_pro1, y_val_data_pro1 = lib.generate_prediction_data(u_val_data1, y_val_data1,
                                                          lag_input, lag_output)
u_val_data_pro2, y_val_data_pro2 = lib.generate_prediction_data(u_val_data2, y_val_data2,
                                                          lag_input, lag_output)

# Time
time = np.arange(u1.shape[0]) * TS
val_time = np.arange(N, N+ y_val1.shape[0]) * TS

############################### IMPORT MLP ###################################

index_lambda = float(3.5)            # Regularization parameter
rep_index = 1                        # Number of time to repeat experiment
layers = 2                         # Number of hidden layers
mlp_layer = 50                       # Number of hidden units per layer
outer_times = 10                     # Number of identification cycles
bias = True
activation = torch.relu

FOLDER_HEAD = './MLP/'
folder = FOLDER_HEAD + 'CED_results_layers_' + str(layers) + '_units_' + \
         str(mlp_layer) + '/' + 'percentage_' + str(percentage) + '/'

noregu_results = folder  +'lag_' + str(lag) + '_repeat_' + str(rep_index) + \
                 '_noregu_validation.mat'
convregu_results = folder  +'lag_' + str(lag) + '_lambda_' + str(index_lambda) + \
           '_repeat_' + str(rep_index) + '_convregu_validation.mat'
regu_results = folder  +'lag_' + str(lag) + '_lambda_' + str(index_lambda) + \
           '_repeat_' + str(rep_index) + '_bestregu_validation.mat'
model_filename = folder  +'lag_' + str(lag) + '_lambda_' + str(index_lambda) + \
           '_repeat_' + str(rep_index) + '_bestregu_model.tar'
training_progress = folder  +'lag_' + str(lag) + '_lambda_' + str(index_lambda) + \
           '_repeat_' + str(rep_index) + '_training_results.mat'

# Import files
regu_data = scio.loadmat(regu_results)
y_val1 = torch.tensor(regu_data['realVal1'])
y_val2 = torch.tensor(regu_data['realVal2'])
mlp_pred_val1 = torch.tensor(regu_data['predVal1'])
mlp_pred_val2 = torch.tensor(regu_data['predVal2'])
mlp_sim_val1 = torch.tensor(regu_data['simVal1'])
mlp_sim_val2 = torch.tensor(regu_data['simVal2'])

noregu_data = scio.loadmat(noregu_results)
noregu_sim_val1 = torch.tensor(noregu_data['simVal1'])
noregu_sim_val2 = torch.tensor(noregu_data['simVal2'])

convregu_data = scio.loadmat(convregu_results)
convregu_sim_val1 = torch.tensor(convregu_data['simVal1'])
convregu_sim_val2 = torch.tensor(convregu_data['simVal2'])

train_data = scio.loadmat(training_progress)
pred_losses = np.ndarray.flatten(train_data['predLosses'])
train_losses = np.ndarray.flatten(train_data['trainLosses'])
pred_vals = np.ndarray.flatten(train_data['predTotalValLosses'])
sim_vals = np.ndarray.flatten(train_data['simTotalValLosses'])

# Compute RMSE
mlp_loss_pred1 = torch.sqrt((torch.nn.functional.mse_loss(mlp_pred_val1, y_val1)))
mlp_loss_pred2 = torch.sqrt((torch.nn.functional.mse_loss(mlp_pred_val2, y_val2)))
mlp_loss_sim1 = torch.sqrt((torch.nn.functional.mse_loss(mlp_sim_val1, y_val1)))
mlp_loss_sim2 = torch.sqrt((torch.nn.functional.mse_loss(mlp_sim_val2, y_val2)))
convregu_mlp_loss_sim1 = torch.sqrt((torch.nn.functional.mse_loss(convregu_sim_val1, y_val1)))
convregu_mlp_loss_sim2 = torch.sqrt((torch.nn.functional.mse_loss(convregu_sim_val2, y_val2)))
noregu_mlp_loss_sim1 = torch.sqrt((torch.nn.functional.mse_loss(noregu_sim_val1, y_val1)))
noregu_mlp_loss_sim2 = torch.sqrt((torch.nn.functional.mse_loss(noregu_sim_val2, y_val2)))

############################ Load best model #################################

# Load Model and Bayesian parameters
layers = [u_est_data_pro1.shape[1], mlp_layer, mlp_layer, y_est_data_pro1.shape[1]]
best_mlp_param = torch.load(model_filename)
mlp_model = MLPNetwork(layers, activation=activation, bias=bias,
                       save_device=TORCH_MACHINE).to(TORCH_MACHINE)

mlp_model.load_state_dict(best_mlp_param["model_state_dict"])
hessian_auto = best_mlp_param["hessianAuto"]
masks = best_mlp_param["masks"]
sum_gamma = best_mlp_param["sum_gamma"]

############################## Plot Input Data ################################

fig_data0 = plt.figure(figsize=[7, 4])
ax = fig_data0.add_subplot(1, 1, 1)
u1_est = u1.clone()
u1_val = u1.clone()
u1_est[N:] = np.nan
u1_val[:N]= np.nan
plot_min = np.floor(np.min(u1.cpu().numpy()))
plot_max = np.ceil(np.max(u1.cpu().numpy()))
ax.plot(time, u1_est.cpu().numpy(), color='tab:blue',
                      label=r'Estimation Input')
ax.plot(time, u1_val.cpu().numpy(), color='tab:orange',
                      label=r'Validation Input')
ax.vlines(N*TS, plot_min, plot_max, color='gold', linewidth=2)
ax.axis([0, time[-1], plot_min, plot_max])
ax.set_xticks(np.arange(0,time[-1],1))
ax.grid(True, 'major', 'x')
# ax.set_title('First UNIF Benchmark Dataset for CED.')
ax.set_ylabel(r'drives input [V]')
ax.set_xlabel(r'time [s]')
ax.legend(loc='upper right')
fig_data0.subplots_adjust(left=0.15, bottom=0.12, right=0.9,
                          top=0.9, wspace=0.3, hspace=0.2)
plt.savefig("./figures/ced_input_data1.pdf")

fig_data1 = plt.figure(figsize=[7, 4])
ax = fig_data1.add_subplot(1, 1, 1)
u2_est = u2.clone()
u2_val = u2.clone()
u2_est[N:] = np.nan
u2_val[:N]= np.nan
plot_min = np.floor(np.min(u2.cpu().numpy()))
plot_max = np.ceil(np.max(u2.cpu().numpy()))
ax.plot(time, u2_est.cpu().numpy(), color='tab:blue',
                      label=r'Estimation Input')
ax.plot(time, u2_val.cpu().numpy(), color='tab:orange',
                      label=r'Validation Input')
ax.vlines(N*TS, plot_min, plot_max, color='gold', linewidth=2)
ax.axis([0, time[-1], plot_min, plot_max])
ax.set_xticks(np.arange(0,time[-1],1))
ax.grid(True, 'major', 'x')
# ax.set_title('Second UNIF Benchmark Dataset for CED.')
ax.set_ylabel(r'drives input [V]')
ax.set_xlabel(r'time [s]')
ax.legend(loc='upper right')
fig_data1.subplots_adjust(left=0.15, bottom=0.12, right=0.9,
                          top=0.9, wspace=0.3, hspace=0.2)
plt.savefig("./figures/ced_input_data2.pdf")

##################### Plot Prediction Validation I ###########################
pred_val_mean1 = y_val1[:,0].clone()
pred_val_std1 = torch.zeros(pred_val_mean1.size())
pred_val_mean1[lag:], pred_val_std1[lag:] = lib.predict_distribution(
    mlp_model, u_val_data_pro1, y_val_data_pro1, masks, sum_gamma, hessian_auto,
    std_y, mean_y, repeat=100000)
minimalLossPred1 = torch.sqrt((torch.nn.functional.mse_loss(pred_val_mean1, y_val1[:,0])))

fig_mlp0 = plt.figure(figsize=[7, 4])
ax = fig_mlp0.add_subplot(1, 1, 1)
bayes_str = 'Bayesian Model (RMSE = {:2.4f}'.format(minimalLossPred1)
true_pred1 = ax.plot(val_time, y_val1.cpu().numpy(), color='tab:blue', label=r'True Output')
pred_regu1 = ax.plot(val_time, pred_val_mean1.cpu().numpy(), '-',
                     color='tab:orange', label=bayes_str + ')')
ax.fill_between(val_time, pred_val_mean1.cpu().numpy() - 2*pred_val_std1.cpu().numpy(),
                pred_val_mean1.cpu().numpy() + 2*pred_val_std1.cpu().numpy(),
                color='tab:orange', alpha=0.2)
plot_min = -0.5
plot_max = 2
ax.axis([val_time[0], val_time[-1], plot_min, plot_max])
ax.grid(True, 'major', 'x')
ax.grid(True, 'both', 'y')
# ax.set_title('One Step Ahead prediction results of first dataset.')
ax.set_ylabel(r'amplitude [V]')
ax.set_xlabel(r'time [s]')
ax.legend(loc='upper right')
fig_mlp0.subplots_adjust(left=0.15, bottom=0.12, right=0.9,
                    top=0.9, wspace=0.3, hspace=0.2)
name = 'lag_' + str(lag) + '_lambda_' + str(index_lambda)
plt.savefig("./figures/ced_mlp_pred_results1_" + name + ".pdf")

#################### Plot Prediction Validation II ##########################
pred_val_mean2 = y_val2[:,0].clone()
pred_val_std2 = torch.zeros(pred_val_mean2.size())
pred_val_mean2[lag:], pred_val_std2[lag:] = lib.predict_distribution(
    mlp_model, u_val_data_pro2, y_val_data_pro2, masks, sum_gamma, hessian_auto,
    std_y, mean_y, repeat=100000)
mlp_loss_pred2 = torch.sqrt((torch.nn.functional.mse_loss(pred_val_mean2, y_val2[:,0])))

fig_mlp1 = plt.figure(figsize=[7, 4])
ax = fig_mlp1.add_subplot(1, 1, 1)
bayes_str = 'Bayesian Model (RMSE = {:2.4f}'.format(mlp_loss_pred2)
true_pred2 = ax.plot(val_time, y_val2.cpu().numpy(), color='tab:blue', label=r'True Output')
pred_regu2 = ax.plot(val_time, pred_val_mean2.cpu().numpy(), '-',
                     color='tab:orange', label=bayes_str + ')')
ax.fill_between(val_time, pred_val_mean2.cpu().numpy() - 2*pred_val_std2.cpu().numpy(),
                pred_val_mean2.cpu().numpy() + 2*pred_val_std2.cpu().numpy(),
                color='tab:orange', alpha=0.2)
plot_min = -0.5
plot_max = 3
ax.axis([val_time[0], val_time[-1], plot_min, plot_max])
ax.grid(True, 'major', 'x')
ax.grid(True, 'both', 'y')
# ax.set_title('Free-Run Simulation results of second dataset.')
ax.set_ylabel(r'amplitude [V]')
ax.set_xlabel(r'time [s]')
ax.legend()
fig_mlp1.subplots_adjust(left=0.15, bottom=0.12, right=0.9,
                    top=0.9, wspace=0.3, hspace=0.2)
name = 'lag_' + str(lag) + '_lambda_' + str(index_lambda)
plt.savefig("./figures/ced_mlp_pred_results2_" + name + ".pdf")

##################### Plot Simulation Validation I ###########################
sim_val_mean1, sim_val_std1= lib.simulate_distribution(
    mlp_model, u_val_data1, y_val_data1, masks, sum_gamma, hessian_auto,
    std_y, mean_y, lag_input, lag_output, repeat=10000)
mlp_loss_sim1 = torch.sqrt((torch.nn.functional.mse_loss(sim_val_mean1, y_val1[:,0])))

fig_mlp2 = plt.figure(figsize=[7, 4])
ax = fig_mlp2.add_subplot(1, 1, 1)
bayes_str = 'Bayesian Model (RMSE = {:2.4f}'.format(mlp_loss_sim1)
true_sim1 = ax.plot(val_time, y_val1.cpu().numpy(), color='tab:blue', label=r'True Output')
sim_regu1 = ax.plot(val_time, sim_val_mean1.cpu().numpy(), '-',
                    color='tab:orange', label=bayes_str + ')')
ax.fill_between(val_time, sim_val_mean1.cpu().numpy() - sim_val_std1.cpu().numpy(),
                sim_val_mean1.cpu().numpy() + sim_val_std1.cpu().numpy(),
                color='tab:orange', alpha=0.2)
plot_min = np.floor(np.nanmin(sim_val_mean1.cpu().numpy() - sim_val_std1.cpu().numpy()))
plot_max = np.ceil(np.nanmax(sim_val_mean1.cpu().numpy() + sim_val_std1.cpu().numpy()))
ax.axis([val_time[0], val_time[-1], plot_min, plot_max])
ax.grid(True, 'major', 'x')
ax.grid(True, 'both', 'y')
ax.set_title('Free-Run Simulation results of first dataset.')
ax.set_ylabel(r'amplitude [V]')
ax.set_xlabel(r'time [s]')
ax.legend()
fig_mlp2.subplots_adjust(left=0.15, bottom=0.12, right=0.9,
                    top=0.9, wspace=0.3, hspace=0.2)
name = 'lag_' + str(lag) + '_lambda_' + str(index_lambda)
plt.savefig("./figures/ced_mlp_sim_results1_" + name + ".pdf")

##################### Plot Simulation Validation II ##########################
sim_val_mean2, sim_val_std2 = lib.simulate_distribution(
    mlp_model, u_val_data2, y_val_data2, masks, sum_gamma, hessian_auto,
    std_y, mean_y, lag_input, lag_output, repeat=10000)
mlp_loss_sim2 = torch.sqrt((torch.nn.functional.mse_loss(sim_val_mean2, y_val2[:,0])))

fig_mlp3 = plt.figure(figsize=[7, 4])
ax = fig_mlp3.add_subplot(1, 1, 1)
bayes_str = 'Bayesian Model (RMSE = {:2.4f}'.format(mlp_loss_sim2)
true_sim2 = ax.plot(val_time, y_val2.cpu().numpy(), color='tab:blue', label=r'True Output')
sim_regu2 = ax.plot(val_time, sim_val_mean2.cpu().numpy(), '-',
                    color='tab:orange', label=bayes_str + ')')
ax.fill_between(val_time, sim_val_mean2.cpu().numpy() - sim_val_std2.cpu().numpy(),
                sim_val_mean2.cpu().numpy() + sim_val_std2.cpu().numpy(),
                color='tab:orange', alpha=0.2)
plot_min = np.floor(np.nanmin(sim_val_mean2.cpu().numpy() - sim_val_std2.cpu().numpy()))
plot_max = np.ceil(np.nanmax(sim_val_mean2.cpu().numpy() + sim_val_std2.cpu().numpy()))
ax.axis([val_time[0], val_time[-1], plot_min, plot_max])
ax.grid(True, 'major', 'x')
ax.grid(True, 'both', 'y')
ax.set_title('Free-Run Simulation results of second dataset.')
ax.set_ylabel(r'amplitude [V]')
ax.set_xlabel(r'time [s]')
ax.legend()
fig_mlp3.subplots_adjust(left=0.15, bottom=0.12, right=0.9,
                    top=0.9, wspace=0.3, hspace=0.2)
name = 'lag_' + str(lag) + '_lambda_' + str(index_lambda)
plt.savefig("./figures/ced_mlp_sim_results2_" + name + ".pdf")

######################## Plot Training Progress #############################
outers = np.arange(outer_times+1)
epochs = int(len(train_losses) / outer_times)
its = np.arange(len(train_losses))

models_sparsity = [0.0]
for outer in np.arange(outer_times):
    model_file = folder  +'lag_' + str(lag) + '_lambda_' + str(index_lambda) +\
                 '_repeat_' + str(rep_index) + '_outer_' + \
                 str(outer + 1) +'_model.tar'
    model_info = torch.load(model_file)
    outer_model = MLPNetwork(layers, activation=activation, bias=bias,
                       save_device=TORCH_MACHINE).to(TORCH_MACHINE)

    outer_model.load_state_dict(model_info["model_state_dict"])
    models_sparsity.append(1-lib.model_overall_sparsity(outer_model))
chosen_outer = best_mlp_param['outer']

fig_mlp4 = plt.figure(figsize=[7, 4])
ax = fig_mlp4.add_subplot(1, 1, 1)
ax.axis([0, len(pred_losses), 0, 0.4])
ax.set_xticks(np.arange(0, len(train_losses), epochs))
ax2 = ax.twinx()
train_prog = ax.plot(its, train_losses, color ='tab:green',
                     label=r'Prediction Loss + regu')
sparsity_prog = ax2.plot(outers*epochs, models_sparsity,'*', color='tab:blue',
                         linewidth=2, markersize=8, label=r'Model Sparsity')
chosen_model = ax.vlines(chosen_outer*epochs, 0, 1, color='tab:red',
                          linestyle='--', linewidth=3, label = r'Chosen Model')
ax2.set_yticks(np.linspace(0, 1, len(ax.get_yticks())))
ax.grid(True, 'major', 'x')
ax.grid(True, 'both', 'y')
# ax.set_title(r'Training Progress and Model Sparsity')
lns = train_prog + sparsity_prog
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='center right')
ax.set_xlabel(r'iteration [.]')
ax.set_ylabel(r'loss function [MSE]')
ax2.set_ylabel(r'model sparsity [\%]')
fig_mlp4.tight_layout()
fig_mlp4.subplots_adjust(left=0.12, bottom=0.12, right=0.88,
                    top=0.9, wspace=0.3, hspace=0.2)
name = 'lag_' + str(lag) + '_lambda_' + str(index_lambda)
plt.savefig("./figures/ced_mlp_training_progress_" + name + ".pdf")

######################### Plot Validation Progress ##########################
fig_mlp5 = plt.figure(figsize=[7, 4])
ax = fig_mlp5.add_subplot(1, 1, 1)
ax.axis([0, len(pred_vals), 0.05, 0.18])
ax.set_xticks(np.arange(0, outer_times+1, 1))
pred_prog = ax.plot(np.arange(outer_times) + 1, pred_vals, 'x',
                    color = 'tab:blue', linewidth=2,
                    label=r'One Step Ahead Prediction RMSE')
sim_prog = ax.plot(np.arange(outer_times) + 1, sim_vals, 'x',
                   color = 'tab:orange', linewidth=2,
                   label=r'Free Run Simulation RMSE')
chosen_model = ax.vlines(chosen_outer, 0, 1, color='tab:red',
                          linestyle='--', linewidth=3, label = r'Chosen Model')
ax.grid(True, 'major', 'x')
ax.grid(True, 'both', 'y')
# ax.set_title(r'Prediction and Simulation validation errors per ID cycle')
lns = pred_prog + sim_prog
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='center left')
ax.set_xlabel(r'identification cycle [.]')
ax.set_ylabel(r'validation RMSE')
fig_mlp5.tight_layout()
fig_mlp5.subplots_adjust(left=0.12, bottom=0.12, right=0.88,
                    top=0.9, wspace=0.3, hspace=0.2)
name = 'lag_' + str(lag) + '_lambda_' + str(index_lambda)
plt.savefig("./figures/ced_mlp_validation_progress_" + name + ".pdf")

######################### Plot Sparsity MLP #################################
fig_mlp6 = plt.figure(figsize=[7, 4])
gs = fig_mlp6.add_gridspec(1, 5)
ax = fig_mlp6.add_subplot(gs[0,0:2])
ax.spy(np.abs(mlp_model.weights['linear_0'].detach().cpu().numpy().transpose()),
       markersize=4)
ax.tick_params(axis='both', which='both', left=False, right=False,
               bottom=False, top=False, labelbottom=False, labelsize=8)
ax.set_title('First MLP Layer', y=0, pad=-33, verticalalignment="top")
ax.set_ylabel('Inputs z')
ax = fig_mlp6.add_subplot(gs[0,2:4])
ax.spy(np.abs(mlp_model.weights['linear_1'].detach().cpu().numpy().transpose()),
       markersize=4)
ax.tick_params(axis='both', which='both', left=False, right=False,
               bottom=False, top=False, labelbottom=False, labelsize=8)
ax.set_title('Second MLP Layer', y=0, pad=-15, verticalalignment="top")
ax = fig_mlp6.add_subplot(gs[0,4])
ax.spy(np.abs(mlp_model.weights['linear_2'].detach().cpu().numpy().transpose()),
       markersize=4)
ax.tick_params(axis='both', which='both', left=False, right=False,
               bottom=False, top=False, labelbottom=False, labeltop=False,
               labelsize=8)
ax.set_title('Output Layer', y=0, pad=-15, verticalalignment="top")
fig_mlp6.subplots_adjust(top=0.78, bottom=0.2, wspace=1)
# fig_mlp6.suptitle('Sparsity plots of the estimated MLP Network')
name = 'lag_' + str(lag) + '_lambda_' + str(index_lambda)
plt.savefig("./figures/ced_mlp_sparse_model_" + name + ".pdf")

##############################################################################
##############################################################################
############################### IMPORT LSTM ##################################

index_lambda = float(3.5)            # Regularization parameter
rep_index = 15                       # Number of time to repeat experiment
layers = 1                           # Number of hidden layers
lstm_layer = 10                      # Number of hidden units per layer
outer_times = 10                     # Number of identification cycles
bias = True

FOLDER_HEAD = './LSTM/'
folder = FOLDER_HEAD + 'CED_results_layers_' + str(layers) + '_units_' + \
         str(lstm_layer) + '/' + 'percentage_' + str(percentage) + '/'

noregu_results = folder  +'lag_' + str(lag) + '_repeat_' + str(rep_index) + \
                 '_noregu_validation.mat'
convregu_results = folder  +'lag_' + str(lag) + '_lambda_' + str(index_lambda) + \
           '_repeat_' + str(rep_index) + '_convregu_validation.mat'
regu_results = folder  +'lag_' + str(lag) + '_lambda_' + str(index_lambda) + \
           '_repeat_' + str(rep_index) + '_bestregu_validation.mat'
model_filename = folder  +'lag_' + str(lag) + '_lambda_' + str(index_lambda) + \
           '_repeat_' + str(rep_index) + '_bestregu_model.tar'
training_progress = folder  +'lag_' + str(lag) + '_lambda_' + str(index_lambda) + \
           '_repeat_' + str(rep_index) + '_training_results.mat'

# Import files
regu_data = scio.loadmat(regu_results)
y_val1 = torch.tensor(regu_data['realVal1'])
y_val2 = torch.tensor(regu_data['realVal2'])
lstm_pred_val1 = torch.tensor(regu_data['predVal1'])
lstm_pred_val2 = torch.tensor(regu_data['predVal2'])
lstm_sim_val1 = torch.tensor(regu_data['simVal1'])
lstm_sim_val2 = torch.tensor(regu_data['simVal2'])

noregu_data = scio.loadmat(noregu_results)
noregu_sim_val1 = torch.tensor(noregu_data['simVal1'])
noregu_sim_val2 = torch.tensor(noregu_data['simVal2'])

convregu_data = scio.loadmat(convregu_results)
convregu_sim_val1 = torch.tensor(convregu_data['simVal1'])
convregu_sim_val2 = torch.tensor(convregu_data['simVal2'])

train_data = scio.loadmat(training_progress)
pred_losses = np.ndarray.flatten(train_data['predLosses'])
train_losses = np.ndarray.flatten(train_data['trainLosses'])
pred_vals = np.ndarray.flatten(train_data['predTotalValLosses'])
sim_vals = np.ndarray.flatten(train_data['simTotalValLosses'])

# Compute RMSE
lstm_loss_pred1 = torch.sqrt((torch.nn.functional.mse_loss(lstm_pred_val1, y_val1)))
lstm_loss_pred2 = torch.sqrt((torch.nn.functional.mse_loss(lstm_pred_val2, y_val2)))
lstm_loss_sim1 = torch.sqrt((torch.nn.functional.mse_loss(lstm_sim_val1, y_val1)))
lstm_loss_sim2 = torch.sqrt((torch.nn.functional.mse_loss(lstm_sim_val2, y_val2)))
convregu_lstm_loss_sim1 = torch.sqrt((torch.nn.functional.mse_loss(convregu_sim_val1, y_val1)))
convregu_lstm_loss_sim2 = torch.sqrt((torch.nn.functional.mse_loss(convregu_sim_val2, y_val2)))
noregu_lstm_loss_sim1 = torch.sqrt((torch.nn.functional.mse_loss(noregu_sim_val1, y_val1)))
noregu_lstm_loss_sim2 = torch.sqrt((torch.nn.functional.mse_loss(noregu_sim_val2, y_val2)))

###################### Load data and best model #############################

# Load Model and Bayesian parameters
layers = [u_est_data_pro1.shape[1], lstm_layer, y_est_data_pro1.shape[1]]
best_lstm_param = torch.load(model_filename)
lstm_model = RNNNetwork(layers, bias=bias, save_device=TORCH_MACHINE).to(TORCH_MACHINE)

lstm_model.load_state_dict(best_lstm_param["model_state_dict"])
hessian_auto = best_lstm_param["hessianAuto"]
masks = best_lstm_param["masks"]
sum_gamma = best_lstm_param["sum_gamma"]

##################### Plot Prediction Validation I ###########################
pred_val_mean1 = y_val1[:,0].clone()
pred_val_std1 = torch.zeros(pred_val_mean1.size())

pred_val_mean1[lag:], pred_val_std1[lag:] = lib.predict_distribution(
    lstm_model, u_val_data_pro1, y_val_data_pro1, masks, sum_gamma, hessian_auto,
    std_y, mean_y, repeat=10000)
lstm_loss_pred1 = torch.sqrt((torch.nn.functional.mse_loss(pred_val_mean1, y_val1[:,0])))

fig_lstm0 = plt.figure(figsize=[7, 4])
ax = fig_lstm0.add_subplot(1, 1, 1)
bayes_str = 'Bayesian Model (RMSE = {:2.4f}'.format(lstm_loss_pred1)
true_pred1 = ax.plot(val_time, y_val1.cpu().numpy(), color='tab:blue', label=r'True Output')
pred_regu1 = ax.plot(val_time, pred_val_mean1.cpu().numpy(), '-',
                     color='tab:orange', label=bayes_str + ')')
ax.fill_between(val_time, pred_val_mean1.cpu().numpy() - 2*pred_val_std1.cpu().numpy(),
                pred_val_mean1.cpu().numpy() + 2*pred_val_std1.cpu().numpy(),
                color='tab:orange', alpha=0.2)
plot_min = -0.5
plot_max = 2
ax.axis([val_time[0], val_time[-1], plot_min, plot_max])
ax.grid(True, 'major', 'x')
ax.grid(True, 'both', 'y')
# ax.set_title('One Step Ahead prediction results of first dataset.')
ax.set_ylabel(r'amplitude [V]')
ax.set_xlabel(r'time [s]')
ax.legend(loc='upper right')
fig_lstm0.subplots_adjust(left=0.15, bottom=0.12, right=0.9,
                    top=0.9, wspace=0.3, hspace=0.2)
name = 'lag_' + str(lag) + '_lambda_' + str(index_lambda)
plt.savefig("./figures/ced_lstm_pred_results1_" + name + ".pdf")

##################### Plot Prediction Validation II ##########################
pred_val_mean2 = y_val2[:,0].clone()
pred_val_std2 = torch.zeros(pred_val_mean2.size())
pred_val_mean2[lag:], pred_val_std2[lag:] = lib.predict_distribution(
    lstm_model, u_val_data_pro2, y_val_data_pro2, masks, sum_gamma, hessian_auto,
    std_y, mean_y, repeat=10000)
lstm_loss_pred2 = torch.sqrt((torch.nn.functional.mse_loss(pred_val_mean2, y_val2[:,0])))

fig_lstm1 = plt.figure(figsize=[7, 4])
ax = fig_lstm1.add_subplot(1, 1, 1)
bayes_str = 'Bayesian Model (RMSE = {:2.4f}'.format(lstm_loss_pred2)
true_pred2 = ax.plot(val_time, y_val2.cpu().numpy(), color='tab:blue', label=r'True Output')
pred_regu2 = ax.plot(val_time, pred_val_mean2.cpu().numpy(), '-',
                     color='tab:orange', label=bayes_str + ')')
ax.fill_between(val_time, pred_val_mean2.cpu().numpy() - 2*pred_val_std2.cpu().numpy(),
                pred_val_mean2.cpu().numpy() + 2*pred_val_std2.cpu().numpy(),
                color='tab:orange', alpha=0.2)
plot_min = -0.5
plot_max = 3
ax.axis([val_time[0], val_time[-1], plot_min, plot_max])
ax.grid(True, 'major', 'x')
ax.grid(True, 'both', 'y')
# ax.set_title('One Step Ahead prediction results of second dataset.')
ax.set_ylabel(r'amplitude [V]')
ax.set_xlabel(r'time [s]')
ax.legend()
fig_lstm1.subplots_adjust(left=0.15, bottom=0.12, right=0.9,
                    top=0.9, wspace=0.3, hspace=0.2)
name = 'lag_' + str(lag) + '_lambda_' + str(index_lambda)
plt.savefig("./figures/ced_lstm_pred_results2_" + name + ".pdf")

##################### Plot Simulation Validation I ###########################
sim_val_mean1, sim_val_std1= lib.simulate_distribution(
    lstm_model, u_val_data1, y_val_data1, masks, sum_gamma, hessian_auto,
    std_y, mean_y, lag_input, lag_output, repeat=1000)
lstm_loss_sim1 = torch.sqrt((torch.nn.functional.mse_loss(sim_val_mean1, y_val1[:,0])))

fig_lstm2 = plt.figure(figsize=[7, 4])
ax = fig_lstm2.add_subplot(1, 1, 1)
bayes_str = 'Bayesian Model (RMSE = {:2.4f}'.format(lstm_loss_sim1)
true_sim1 = ax.plot(val_time, y_val1.cpu().numpy(), color='tab:blue', label=r'True Output')
sim_regu1 = ax.plot(val_time, sim_val_mean1.cpu().numpy(), '-',
                    color='tab:orange', label=bayes_str + ')')
ax.fill_between(val_time, sim_val_mean1.cpu().numpy() - sim_val_std1.cpu().numpy(),
                sim_val_mean1.cpu().numpy() + sim_val_std1.cpu().numpy(),
                color='tab:orange', alpha=0.2)
plot_min = np.floor(np.nanmin(sim_val_mean1.cpu().numpy() - sim_val_std1.cpu().numpy()))
plot_max = np.ceil(np.nanmax(sim_val_mean1.cpu().numpy() + sim_val_std1.cpu().numpy()))
ax.axis([val_time[0], val_time[-1], plot_min, plot_max])
ax.grid(True, 'major', 'x')
ax.grid(True, 'both', 'y')
ax.set_title('Free-Run Simulation results of first dataset.')
ax.set_ylabel(r'amplitude [V]')
ax.set_xlabel(r'time [s]')
ax.legend()
fig_lstm2.subplots_adjust(left=0.15, bottom=0.12, right=0.9,
                    top=0.9, wspace=0.3, hspace=0.2)
name = 'lag_' + str(lag) + '_lambda_' + str(index_lambda)
plt.savefig("./figures/ced_lstm_sim_results1_" + name + ".pdf")

##################### Plot Simulation Validation II ##########################
sim_val_mean2, sim_val_std2 = lib.simulate_distribution(
    lstm_model, u_val_data2, y_val_data2, masks, sum_gamma, hessian_auto,
    std_y, mean_y, lag_input, lag_output, repeat=1000)
lstm_loss_sim2 = torch.sqrt((torch.nn.functional.mse_loss(sim_val_mean2, y_val2[:,0])))

fig_lstm3 = plt.figure(figsize=[7, 4])
ax = fig_lstm3.add_subplot(1, 1, 1)
bayes_str = 'Bayesian Model (RMSE = {:2.4f}'.format(lstm_loss_sim2)
true_sim2 = ax.plot(val_time, y_val2.cpu().numpy(), color='tab:blue', label=r'True Output')
sim_regu2 = ax.plot(val_time, sim_val_mean2.cpu().numpy(), '-',
                    color='tab:orange', label=bayes_str + ')')
ax.fill_between(val_time, sim_val_mean2.cpu().numpy() - sim_val_std2.cpu().numpy(),
                sim_val_mean2.cpu().numpy() + sim_val_std2.cpu().numpy(),
                color='tab:orange', alpha=0.2)
plot_min = np.floor(np.nanmin(sim_val_mean2.cpu().numpy() - sim_val_std2.cpu().numpy()))
plot_max = np.ceil(np.nanmax(sim_val_mean2.cpu().numpy() + sim_val_std2.cpu().numpy()))
ax.axis([val_time[0], val_time[-1], plot_min, plot_max])
ax.grid(True, 'major', 'x')
ax.grid(True, 'both', 'y')
ax.set_title('Free-Run Simulation results of second dataset.')
ax.set_ylabel(r'amplitude [V]')
ax.set_xlabel(r'time [s]')
ax.legend()
fig_lstm3.subplots_adjust(left=0.15, bottom=0.12, right=0.9,
                    top=0.9, wspace=0.3, hspace=0.2)
name = 'lag_' + str(lag) + '_lambda_' + str(index_lambda)
plt.savefig("./figures/ced_lstm_sim_results2_" + name + ".pdf")

##################### Plot Training Progress #############################
outers = np.arange(outer_times+1)
epochs = int(len(train_losses) / outer_times)
its = np.arange(len(train_losses))

models_sparsity = [0.0]
for outer in np.arange(outer_times):
    model_file = folder  +'lag_' + str(lag) + '_lambda_' + str(index_lambda) +\
                 '_repeat_' + str(rep_index) + '_outer_' + \
                 str(outer + 1) +'_model.tar'
    model_info = torch.load(model_file)
    outer_model = RNNNetwork(layers, bias=bias,
                       save_device=TORCH_MACHINE).to(TORCH_MACHINE)

    outer_model.load_state_dict(model_info["model_state_dict"])
    models_sparsity.append(1-lib.model_overall_sparsity(outer_model))
chosen_outer = best_lstm_param['outer']

fig_lstm4 = plt.figure(figsize=[7, 4])
ax = fig_lstm4.add_subplot(1, 1, 1)
ax.axis([0, len(pred_losses), 0, 0.5])
ax.set_xticks(np.arange(0, len(train_losses), epochs))
ax2 = ax.twinx()
train_prog = ax.plot(its, train_losses, color ='tab:green',
                     label=r'Prediction Loss + regu')
sparsity_prog = ax2.plot(outers*epochs, models_sparsity,'*', color='tab:blue',
                         linewidth=2, markersize=8, label=r'Model Sparsity')
chosen_model = ax.vlines(chosen_outer*epochs, 0, 1, color='tab:red',
                          linestyle='--', linewidth=3, label = r'Chosen Model')
ax2.set_yticks(np.linspace(0, 1, len(ax.get_yticks())))
ax.grid(True, 'major', 'x')
ax.grid(True, 'both', 'y')
# ax.set_title(r'Training Progress and Model Sparsity')
lns = train_prog + sparsity_prog
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='upper center')
ax.set_xlabel(r'iteration [.]')
ax.set_ylabel(r'loss function [MSE]')
ax2.set_ylabel(r'model sparsity [\%]')
fig_lstm4.tight_layout()
fig_lstm4.subplots_adjust(left=0.12, bottom=0.12, right=0.88,
                    top=0.9, wspace=0.3, hspace=0.2)
name = 'lag_' + str(lag) + '_lambda_' + str(index_lambda)
plt.savefig("./figures/ced_lstm_training_progress_" + name + ".pdf")

########################## Plot Validation Progress ##########################
fig_lstm5 = plt.figure(figsize=[7, 4])
ax = fig_lstm5.add_subplot(1, 1, 1)
ax.axis([0, len(pred_vals), 0.02, 0.18])
ax.set_xticks(np.arange(0, outer_times+1, 1))
pred_prog = ax.plot(np.arange(outer_times) + 1, pred_vals, 'x',
                    color = 'tab:blue', linewidth=2,
                    label=r'One Step Ahead Prediction RMSE')
sim_prog = ax.plot(np.arange(outer_times) + 1, sim_vals, 'x',
                   color = 'tab:orange', linewidth=2,
                   label=r'Free Run Simulation RMSE')
chosen_model = ax.vlines(chosen_outer, 0, 1, color='tab:red',
                          linestyle='--', linewidth=3, label = r'Chosen Model')
ax.grid(True, 'major', 'x')
ax.grid(True, 'both', 'y')
# ax.set_title(r'Prediction and Simulation validation errors per ID cycle')
lns = pred_prog + sim_prog
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='upper center')
ax.set_xlabel(r'identification cycle [.]')
ax.set_ylabel(r'validation RMSE')
fig_lstm5.tight_layout()
fig_lstm5.subplots_adjust(left=0.12, bottom=0.12, right=0.88,
                    top=0.9, wspace=0.3, hspace=0.2)
name = 'lag_' + str(lag) + '_lambda_' + str(index_lambda)
plt.savefig("./figures/ced_lstm_validation_progress_" + name + ".pdf")

######################### Plot Sparsity LSTM #################################
fig_lstm6 = plt.figure(figsize=[7, 4])
gs = fig_lstm6.add_gridspec(4, 5)
ax = fig_lstm6.add_subplot(gs[0:2,0])
ax.spy(np.abs(lstm_model.weights['lstm_0.linear_ii'].detach().numpy().transpose()),
       markersize=4)
ax.tick_params(axis='both', which='both', left=False, right=False,
               bottom=False, top=False, labelbottom=False, labelsize=8)
ax.set_title(r'$W_{ii}$', y=0, pad=-15, verticalalignment="top")
ax.set_ylabel('Inputs z')
ax = fig_lstm6.add_subplot(gs[0:2,1])
ax.spy(np.abs(lstm_model.weights['lstm_0.linear_ij'].detach().numpy().transpose()),
markersize=4)
ax.tick_params(axis='both', which='both', left=False, right=False,
               bottom=False, top=False, labelbottom=False, labelsize=8)
ax.set_title(r'$W_{ij}$', y=0, pad=-15, verticalalignment="top")
ax.tick_params(axis='both', which='both', left=False, right=False,
               bottom=False, top=False, labelbottom=False)
ax = fig_lstm6.add_subplot(gs[0:2,2])
ax.spy(np.abs(lstm_model.weights['lstm_0.linear_if'].detach().numpy().transpose()),
       markersize=4)
ax.tick_params(axis='both', which='both', left=False, right=False,
               bottom=False, top=False, labelbottom=False, labelsize=8)
ax.set_title(r'$W_{if}$', y=0, pad=-15, verticalalignment="top")
ax = fig_lstm6.add_subplot(gs[0:2,3])
ax.spy(np.abs(lstm_model.weights['lstm_0.linear_io'].detach().numpy().transpose()),
       markersize=4)
ax.tick_params(axis='both', which='both', left=False, right=False,
               bottom=False, top=False, labelbottom=False, labelsize=8)
ax.set_title(r'$W_{io}$', y=0, pad=-15, verticalalignment="top")
ax = fig_lstm6.add_subplot(gs[2:4,0])
ax.spy(np.abs(lstm_model.weights['lstm_0.linear_hi'].detach().numpy().transpose()),
       markersize=4)
ax.tick_params(axis='both', which='both', left=False, right=False,
               bottom=False, top=False, labelbottom=False, labelsize=8)
ax.set_title(r'$W_{hi}$', y=0, pad=-15, verticalalignment="top")
ax = fig_lstm6.add_subplot(gs[2:4,1])
ax.spy(np.abs(lstm_model.weights['lstm_0.linear_hj'].detach().numpy().transpose()),
       markersize=4)
ax.tick_params(axis='both', which='both', left=False, right=False,
               bottom=False, top=False, labelbottom=False, labelsize=8)
ax.set_title(r'$W_{hj}$', y=0, pad=-15, verticalalignment="top")
ax = fig_lstm6.add_subplot(gs[2:4,2])
ax.spy(np.abs(lstm_model.weights['lstm_0.linear_hf'].detach().numpy().transpose()),
       markersize=4)
ax.tick_params(axis='both', which='both', left=False, right=False,
               bottom=False, top=False, labelbottom=False, labelsize=8)
ax.set_title(r'$W_{hf}$', y=0, pad=-15, verticalalignment="top")
ax = fig_lstm6.add_subplot(gs[2:4,3])
ax.spy(np.abs(lstm_model.weights['lstm_0.linear_ho'].detach().numpy().transpose()),
       markersize=4)
ax.tick_params(axis='both', which='both', left=False, right=False,
               bottom=False, top=False, labelbottom=False, labelsize=8)
ax.set_title(r'$W_{ho}$', y=0, pad=-15, verticalalignment="top")
ax = fig_lstm6.add_subplot(gs[1:3,4])
ax.spy(np.abs(lstm_model.weights['linear_f'].detach().numpy().transpose()),
       markersize=4)
ax.tick_params(axis='both', which='both', left=False, right=False,
               bottom=False, top=False, labelbottom=False, labeltop=False, labelsize=8)
ax.set_title(r'$W_{f}$', y=0, pad=-15, verticalalignment="top")
fig_lstm6.subplots_adjust(left=0.08, bottom=0.05, right=0.98, top=0.85,
                          wspace=1, hspace=0.8)
# fig_lstm6.suptitle('Sparsity plots of the estimated LSTM Network')
name = 'lag_' + str(lag) + '_lambda_' + str(index_lambda)
plt.savefig("./figures/lstm_sparse_model_" + name + ".pdf")

######################### Plot Simulation Comparison #########################
fig_sim1 = plt.figure(figsize=[7, 4])
ax = fig_sim1.add_subplot(1, 1, 1)
bayes_mlp_str = 'Bayesian MLP Model (RMSE$ = {:2.4f}'.format(mlp_loss_sim1)
bayes_lstm_str = 'Bayesian LSTM Model (RMSE$ = {:2.4f}'.format(lstm_loss_sim1)

ax.plot(val_time, y_val1.cpu().numpy(), color='tab:red', label=r'True Output')
ax.plot(val_time, lstm_sim_val1.cpu().numpy(),
        color='tab:green', label=bayes_lstm_str + '$)')
ax.plot(val_time, mlp_sim_val1.cpu().numpy(),
        color='tab:blue', label=bayes_mlp_str + '$)')

plot_min = -0.5
plot_max = 2
ax.axis([val_time[0], val_time[-1], plot_min, plot_max])
ax.grid(True, 'major', 'x')
ax.grid(True, 'both', 'y')
# ax.set_title('Simulation Comparison of first validation CED dataset.')
ax.set_ylabel(r'amplitude [V]')
ax.set_xlabel(r'time [s]')
ax.legend(loc='upper right')
fig_sim1.subplots_adjust(left=0.15, bottom=0.12, right=0.9,
                    top=0.9, wspace=0.3, hspace=0.2)
name = 'lag_' + str(lag) + '_lambda_' + str(index_lambda)
plt.savefig("./figures/ced_sim_comparison1_" + name + ".pdf")

fig_sim2 = plt.figure(figsize=[7, 4])
ax = fig_sim2.add_subplot(1, 1, 1)
bayes_mlp_str = 'Bayesian MLP Model (RMSE$ = {:2.4f}'.format(mlp_loss_sim2)
bayes_lstm_str = 'Bayesian LSTM Model (RMSE$ = {:2.4f}'.format(lstm_loss_sim2)

ax.plot(val_time, y_val2.cpu().numpy(), color='tab:red', label=r'True Output')
ax.plot(val_time, lstm_sim_val2.cpu().numpy(),
        color='tab:green', label=bayes_lstm_str + '$)')
ax.plot(val_time, mlp_sim_val2.cpu().numpy(),
        color='tab:blue', label=bayes_mlp_str + '$)')
plot_min = -0.5
plot_max = 3
ax.axis([val_time[0], val_time[-1], plot_min, plot_max])
ax.grid(True, 'major', 'x')
ax.grid(True, 'both', 'y')
# ax.set_title('Simulation Comparison of second validation CED dataset.')
ax.set_ylabel(r'amplitude [V]')
ax.set_xlabel(r'time [s]')
ax.legend(loc='upper right')
fig_sim2.subplots_adjust(left=0.15, bottom=0.12, right=0.9,
                    top=0.9, wspace=0.3, hspace=0.2)
name = 'lag_' + str(lag) + '_lambda_' + str(index_lambda)
plt.savefig("./figures/ced_sim_comparison2_" + name + ".pdf")