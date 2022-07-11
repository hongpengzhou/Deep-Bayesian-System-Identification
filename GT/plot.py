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
TS = 4.0 # sampling time

# Universal parameters
lag = 5                             # Chosen lags for regressors
percentage = float(1)               # Percentage of data used
lag_input = lag_output = lag

onoff_detrend = False                 # Detrend data (*-mean)
onoff_normalize = False               # Normalize data (*-mean)/std

# Import training and testing data
loadData = scio.loadmat('GT_Data/gt_ds.mat')
u_est = torch.tensor(loadData['udse'], device=TORCH_MACHINE).float()
y_est = torch.tensor(loadData['ydse'], device=TORCH_MACHINE).float()
u_val = torch.tensor(loadData['udsv'], device=TORCH_MACHINE).float()
y_val = torch.tensor(loadData['ydsv'], device=TORCH_MACHINE).float()

# Take the percentage of data
u_est, y_est = lib.select_train_data(u_est, y_est, percentage)

# Normalize Data
u_est_data, y_est_data,  std_y, mean_y, std_u, mean_u = lib.clean_data(u_est, y_est,
                                                              onoff_detrend,
                                                              onoff_normalize)
u_val_data, y_val_data = lib.data_normalize(u_val, y_val, mean_u, std_u,
                                            mean_y, std_y)

# Generate data according to lags
u_est_data_pro, y_est_data_pro = lib.generate_prediction_data(u_est_data, y_est_data,
                                                        lag_input, lag_output)
u_val_data_pro, y_val_data_pro = lib.generate_prediction_data(u_val_data, y_val_data,
                                                        lag_input, lag_output)

# Time
time = np.arange(u_est.shape[0]) * TS
val_time = np.arange(u_val.shape[0]) * TS

############################### IMPORT MLP ###################################

index_lambda = float(3)              # Regularization parameter
rep_index = 1                        # Number of time to repeat experiment
layers = 1                           # Number of hidden layers
mlp_layer = 50                       # Number of hidden units per layer
outer_times = 6                      # Number of identification cycles
bias = False
activation = None

FOLDER_HEAD = './MLP/'
folder = FOLDER_HEAD + 'GT_results_layers_' + str(layers) + '_units_' + \
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
y_val = torch.tensor(regu_data['realVal'])
mlp_pred_val = torch.tensor(regu_data['predVal'])
mlp_sim_val = torch.tensor(regu_data['simVal'])

noregu_data = scio.loadmat(noregu_results)
noregu_sim_val = torch.tensor(noregu_data['simVal'])

convregu_data = scio.loadmat(convregu_results)
convregu_sim_val = torch.tensor(convregu_data['simVal'])

train_data = scio.loadmat(training_progress)
pred_losses = np.ndarray.flatten(train_data['predLosses'])
train_losses = np.ndarray.flatten(train_data['trainLosses'])
pred_vals = np.ndarray.flatten(train_data['predTotalValLosses'])
sim_vals = np.ndarray.flatten(train_data['simTotalValLosses'])

# Compute RMSE
mlp_loss_pred = torch.sqrt((torch.nn.functional.mse_loss(mlp_pred_val, y_val)))
mlp_loss_sim = torch.sqrt((torch.nn.functional.mse_loss(mlp_sim_val, y_val)))
convregu_mlp_loss_sim = torch.sqrt((torch.nn.functional.mse_loss(convregu_sim_val, y_val)))
noregu_mlp_loss_sim = torch.sqrt((torch.nn.functional.mse_loss(noregu_sim_val, y_val)))

############################ Load best model #################################

# Load Model and Bayesian parameters
layers = [u_est_data_pro.shape[1], mlp_layer, y_est_data_pro.shape[1]]
best_mlp_param = torch.load(model_filename)
mlp_model = MLPNetwork(layers, activation=activation, bias=bias,
                       save_device=TORCH_MACHINE).to(TORCH_MACHINE)

mlp_model.load_state_dict(best_mlp_param["model_state_dict"])
hessian_auto = best_mlp_param["hessianAuto"]
masks = best_mlp_param["masks"]
sum_gamma = best_mlp_param["sum_gamma"]

############################# Plot Input Data ################################

fig_est_input = plt.figure(figsize=[7, 4])
ax = fig_est_input.add_subplot(1, 1, 1)
plot_min = np.floor(np.min(u_est.cpu().numpy()))
plot_max = np.ceil(np.max(u_est.cpu().numpy()))
ax.plot(time, u_est.cpu().numpy(), color='tab:blue',
                      label=r'Estimation Input')
ax.axis([time[0], time[-1], plot_min, plot_max])
ax.grid(True, 'major', 'x')
# ax.set_title('Input Benchmark Dataset for GT Model Estimation')
ax.set_ylabel(r'drawing speed')
ax.set_xlabel(r'time [s]')
ax.legend(loc='upper right')
fig_est_input.subplots_adjust(left=0.15, bottom=0.12, right=0.9,
                          top=0.9, wspace=0.3, hspace=0.2)
plt.savefig("./figures/gt_input_estimation_data.pdf")

fig_val_input = plt.figure(figsize=[7, 4])
ax = fig_val_input.add_subplot(1, 1, 1)
plot_min = np.floor(np.min(u_val.cpu().numpy()))
plot_max = np.ceil(np.max(u_val.cpu().numpy()))
ax.plot(val_time, u_val.cpu().numpy(), color='tab:orange',
                      label=r'Validation Input')
ax.axis([val_time[0], val_time[-1], plot_min, plot_max])
ax.grid(True, 'major', 'x')
# ax.set_title(' Input Benchmark Dataset for GT Model Validation')
ax.set_ylabel(r'drawing speed')
ax.set_xlabel(r'time [s]')
ax.legend(loc='upper right')
fig_val_input.subplots_adjust(left=0.15, bottom=0.12, right=0.9,
                          top=0.9, wspace=0.3, hspace=0.2)
plt.savefig("./figures/gt_input_validation_data.pdf")

##################### Plot Prediction Validation  ###########################
pred_val_mean = y_val[:, 0].clone()
pred_val_std = torch.zeros(pred_val_mean.size())
pred_val_mean[lag:], pred_val_std[lag:] = lib.predict_distribution(
    mlp_model, u_val_data_pro, y_val_data_pro, masks, sum_gamma, hessian_auto,
    std_y, mean_y, repeat=10000)
minimalLossPred1 = torch.sqrt((torch.nn.functional.mse_loss(pred_val_mean, y_val[:, 0])))

fig_mlp0 = plt.figure(figsize=[7, 4])
ax = fig_mlp0.add_subplot(1, 1, 1)
bayes_str = 'Bayesian Model (RMSE = {:2.3f}'.format(minimalLossPred1)
true_pred = ax.plot(val_time, y_val.cpu().numpy(), color='tab:blue', label=r'True Output')
pred_regu = ax.plot(val_time, pred_val_mean.cpu().numpy(), '-',
                     color='tab:orange', label=bayes_str + ')')
ax.fill_between(val_time, pred_val_mean.cpu().numpy() - 2 * pred_val_std.cpu().numpy(),
                pred_val_mean.cpu().numpy() + 2 * pred_val_std.cpu().numpy(),
                color='tab:orange', alpha=0.2)
plot_min = -4
plot_max = 4
ax.axis([val_time[0], val_time[-1], plot_min, plot_max])
ax.grid(True, 'major', 'x')
ax.grid(True, 'both', 'y')
# ax.set_title('One Step Ahead prediction results for GT Benchmark.')
ax.set_ylabel(r'output glass thickness')
ax.set_xlabel(r'time [s]')
ax.legend(loc='upper right')
fig_mlp0.subplots_adjust(left=0.15, bottom=0.12, right=0.9,
                    top=0.9, wspace=0.3, hspace=0.2)
name = 'lag_' + str(lag) + '_lambda_' + str(index_lambda)
plt.savefig("./figures/gt_mlp_pred_results_" + name + ".pdf")

##################### Plot Simulation Validation ###########################
sim_val_mean, sim_val_std= lib.simulate_distribution(
    mlp_model, u_val_data, y_val_data, masks, sum_gamma, hessian_auto,
    std_y, mean_y, lag_input, lag_output, repeat=10000)
mlp_loss_sim1 = torch.sqrt((torch.nn.functional.mse_loss(sim_val_mean, y_val[:, 0])))

fig_mlp1 = plt.figure(figsize=[7, 4])
ax = fig_mlp1.add_subplot(1, 1, 1)
bayes_str = 'Bayesian Model (RMSE = {:2.4f}'.format(mlp_loss_sim1)
true_sim1 = ax.plot(val_time, y_val.cpu().numpy(), color='tab:blue', label=r'True Output')
sim_regu1 = ax.plot(val_time, sim_val_mean.cpu().numpy(), '-',
                    color='tab:orange', label=bayes_str + ')')
ax.fill_between(val_time, sim_val_mean.cpu().numpy() - sim_val_std.cpu().numpy(),
                sim_val_mean.cpu().numpy() + sim_val_std.cpu().numpy(),
                color='tab:orange', alpha=0.2)
plot_min = np.floor(np.nanmin(sim_val_mean.cpu().numpy() - sim_val_std.cpu().numpy()))
plot_max = np.ceil(np.nanmax(sim_val_mean.cpu().numpy() + sim_val_std.cpu().numpy()))
ax.axis([val_time[0], val_time[-1], plot_min, plot_max])
ax.grid(True, 'major', 'x')
ax.grid(True, 'both', 'y')
ax.set_title('Free-Run Simulation results of validation dataset.')
ax.set_ylabel(r'output glass thickness')
ax.set_xlabel(r'time [s]')
ax.legend()
fig_mlp1.subplots_adjust(left=0.15, bottom=0.12, right=0.9,
                         top=0.9, wspace=0.3, hspace=0.2)
name = 'lag_' + str(lag) + '_lambda_' + str(index_lambda)
plt.savefig("./figures/gt_mlp_sim_results_" + name + ".pdf")

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

fig_mlp2 = plt.figure(figsize=[7, 4])
ax = fig_mlp2.add_subplot(1, 1, 1)
ax.axis([0, len(pred_losses), 0.3, 0.9])
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
fig_mlp2.tight_layout()
fig_mlp2.subplots_adjust(left=0.12, bottom=0.12, right=0.88,
                         top=0.9, wspace=0.3, hspace=0.2)
name = 'lag_' + str(lag) + '_lambda_' + str(index_lambda)
plt.savefig("./figures/gt_mlp_training_progress_" + name + ".pdf")

######################### Plot Validation Progress ##########################
fig_mlp3 = plt.figure(figsize=[7, 4])
ax = fig_mlp3.add_subplot(1, 1, 1)
ax.axis([0, len(pred_vals), 0.63, 0.69])
ax.set_xticks(np.arange(0, outer_times+1, 1))
pred_prog = ax.plot(np.arange(outer_times) + 1, pred_vals, 'x',
                    color = 'tab:blue', linewidth=2,
                    label=r'One Step Ahead Prediction RMSE')
sim_prog = ax.plot(np.arange(outer_times) + 1, sim_vals, 'x',
                   color = 'tab:orange', linewidth=2,
                   label=r'Free Run Simulation RMSE')
chosen_model = ax.vlines(chosen_outer, 0, 1.5, color='tab:red',
                          linestyle='--', linewidth=3, label = r'Chosen Model')
ax.grid(True, 'major', 'x')
ax.grid(True, 'both', 'y')
# ax.set_title(r'Prediction and Simulation validation errors per ID cycle')
lns = pred_prog + sim_prog
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='upper right')
ax.set_xlabel(r'identification cycle [.]')
ax.set_ylabel(r'validation RMSE')
fig_mlp3.tight_layout()
fig_mlp3.subplots_adjust(left=0.12, bottom=0.12, right=0.88,
                         top=0.9, wspace=0.3, hspace=0.2)
name = 'lag_' + str(lag) + '_lambda_' + str(index_lambda)
plt.savefig("./figures/gt_mlp_validation_progress_" + name + ".pdf")

######################### Plot Sparsity MLP #################################
fig_mlp4 = plt.figure(figsize=[7, 4])
gs = fig_mlp4.add_gridspec(4, 4)
ax = fig_mlp4.add_subplot(gs[0:2, :])
ax.spy(np.abs(mlp_model.weights['linear_0'].detach().cpu().numpy().transpose()),
       markersize=4)
ax.tick_params(axis='both', which='both', left=False, right=False,
               bottom=False, top=False, labelbottom=False, labelsize=8)
ax.set_title('First MLP Layer', y=0, pad=-15, verticalalignment="top")
ax.set_ylabel('Inputs z')
ax = fig_mlp4.add_subplot(gs[3, :])
ax.spy(np.abs(mlp_model.weights['linear_1'].detach().cpu().numpy()),
       markersize=4)
ax.tick_params(axis='both', which='both', left=False, right=False,
               bottom=False, top=False, labelbottom=False, labelleft=False,
               labelsize=8)
ax.set_ylabel(' ')

ax.set_title('Output Layer', y=0, pad=-15, verticalalignment="top")
# fig_mlp4.suptitle('Sparsity plots of the estimated LSTM Network')
fig_mlp4.subplots_adjust(top=0.78, bottom=0.2, wspace=1)
name = 'lag_' + str(lag) + '_lambda_' + str(index_lambda)
plt.savefig("./figures/gt_mlp_sparse_model_" + name + ".pdf")

##############################################################################
##############################################################################
############################### IMPORT LSTM ##################################

index_lambda = float(2)              # Regularization parameter
rep_index = 5                        # Number of time to repeat experiment
layers = 1                           # Number of hidden layers
lstm_layer = 10                      # Number of hidden units per layer
outer_times = 6                      # Number of identification cycles
bias = False

FOLDER_HEAD = './LSTM/'
folder = FOLDER_HEAD + 'GT_results_layers_' + str(layers) + '_units_' + \
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
y_val = torch.tensor(regu_data['realVal'])
lstm_pred_val = torch.tensor(regu_data['predVal'])
lstm_sim_val = torch.tensor(regu_data['simVal'])

noregu_data = scio.loadmat(noregu_results)
noregu_sim_val = torch.tensor(noregu_data['simVal'])

convregu_data = scio.loadmat(convregu_results)
convregu_sim_val = torch.tensor(convregu_data['simVal'])

train_data = scio.loadmat(training_progress)
pred_losses = np.ndarray.flatten(train_data['predLosses'])
train_losses = np.ndarray.flatten(train_data['trainLosses'])
pred_vals = np.ndarray.flatten(train_data['predTotalValLosses'])
sim_vals = np.ndarray.flatten(train_data['simTotalValLosses'])

# Compute RMSE
lstm_loss_pred = torch.sqrt((torch.nn.functional.mse_loss(lstm_pred_val, y_val)))
lstm_loss_sim = torch.sqrt((torch.nn.functional.mse_loss(lstm_sim_val, y_val)))
convregu_lstm_loss_sim = torch.sqrt((torch.nn.functional.mse_loss(convregu_sim_val, y_val)))
noregu_lstm_loss_sim = torch.sqrt((torch.nn.functional.mse_loss(noregu_sim_val, y_val)))

###################### Load data and best model #############################

# Load Model and Bayesian parameters
layers = [u_est_data_pro.shape[1], lstm_layer, y_est_data_pro.shape[1]]
best_lstm_param = torch.load(model_filename)
lstm_model = RNNNetwork(layers, bias=bias, save_device=TORCH_MACHINE).to(TORCH_MACHINE)

lstm_model.load_state_dict(best_lstm_param["model_state_dict"])
hessian_auto = best_lstm_param["hessianAuto"]
masks = best_lstm_param["masks"]
sum_gamma = best_lstm_param["sum_gamma"]

###################### Plot Prediction Validation ###########################
pred_val_mean = y_val[:, 0].clone()
pred_val_std = torch.zeros(pred_val_mean.size())

pred_val_mean[lag:], pred_val_std[lag:] = lib.predict_distribution(
    lstm_model, u_val_data_pro, y_val_data_pro, masks, sum_gamma, hessian_auto,
    std_y, mean_y, repeat=10000)
lstm_loss_pred1 = torch.sqrt((torch.nn.functional.mse_loss(pred_val_mean, y_val[:, 0])))

fig_lstm0 = plt.figure(figsize=[7, 4])
ax = fig_lstm0.add_subplot(1, 1, 1)
bayes_str = 'Bayesian Model (RMSE = {:2.4f}'.format(lstm_loss_pred1)
true_pred1 = ax.plot(val_time, y_val.cpu().numpy(), color='tab:blue', label=r'True Output')
pred_regu1 = ax.plot(val_time, pred_val_mean.cpu().numpy(), '-',
                     color='tab:orange', label=bayes_str + ')')
ax.fill_between(val_time, pred_val_mean.cpu().numpy() - 2 * pred_val_std.cpu().numpy(),
                pred_val_mean.cpu().numpy() + 2 * pred_val_std.cpu().numpy(),
                color='tab:orange', alpha=0.2)
plot_min = -4
plot_max = 4
ax.axis([val_time[0], val_time[-1], plot_min, plot_max])
ax.grid(True, 'major', 'x')
ax.grid(True, 'both', 'y')
# ax.set_title('One Step Ahead prediction results of GT dataset.')
ax.set_ylabel(r'output glass thickness')
ax.set_xlabel(r'time [s]')
ax.legend(loc='upper right')
fig_lstm0.subplots_adjust(left=0.15, bottom=0.12, right=0.9,
                    top=0.9, wspace=0.3, hspace=0.2)
name = 'lag_' + str(lag) + '_lambda_' + str(index_lambda)
plt.savefig("./figures/gt_lstm_pred_results_" + name + ".pdf")

###################### Plot Simulation Validation ###########################
sim_val_mean, sim_val_std = lib.simulate_distribution(
    lstm_model, u_val_data, y_val_data, masks, sum_gamma, hessian_auto,
    std_y, mean_y, lag_input, lag_output, repeat=10000)
lstm_loss_sim1 = torch.sqrt((torch.nn.functional.mse_loss(sim_val_mean, y_val[:, 0])))

fig_lstm1 = plt.figure(figsize=[7, 4])
ax = fig_lstm1.add_subplot(1, 1, 1)
bayes_str = 'Bayesian Model (RMSE = {:2.4f}'.format(lstm_loss_sim1)
true_sim1 = ax.plot(val_time, y_val.cpu().numpy(), color='tab:blue', label=r'True Output')
sim_regu1 = ax.plot(val_time, sim_val_mean.cpu().numpy(), '-',
                    color='tab:orange', label=bayes_str + ')')
ax.fill_between(val_time, sim_val_mean.cpu().numpy() - sim_val_std.cpu().numpy(),
                sim_val_mean.cpu().numpy() + sim_val_std.cpu().numpy(),
                color='tab:orange', alpha=0.2)
plot_min = np.floor(np.nanmin(sim_val_mean.cpu().numpy() - sim_val_std.cpu().numpy()))
plot_max = np.ceil(np.nanmax(sim_val_mean.cpu().numpy() + sim_val_std.cpu().numpy()))
ax.axis([val_time[0], val_time[-1], plot_min, plot_max])
ax.grid(True, 'major', 'x')
ax.grid(True, 'both', 'y')
ax.set_title('Free-Run Simulation results of first dataset.')
ax.set_ylabel(r'output glass thickness')
ax.set_xlabel(r'time [s]')
ax.legend()
fig_lstm1.subplots_adjust(left=0.15, bottom=0.12, right=0.9,
                          top=0.9, wspace=0.3, hspace=0.2)
name = 'lag_' + str(lag) + '_lambda_' + str(index_lambda)
plt.savefig("./figures/gt_lstm_sim_results_" + name + ".pdf")

###################### Plot Training Progress #############################
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

fig_lstm2 = plt.figure(figsize=[7, 4])
ax = fig_lstm2.add_subplot(1, 1, 1)
ax.axis([0, len(pred_losses), 0.4, 4])
ax.set_xticks(np.arange(0, len(train_losses), epochs))
ax2 = ax.twinx()
train_prog = ax.plot(its, train_losses, color ='tab:green',
                     label=r'Prediction Loss + regu')
sparsity_prog = ax2.plot(outers*epochs, models_sparsity,'*', color='tab:blue',
                         linewidth=2, markersize=8, label=r'Model Sparsity')
chosen_model = ax.vlines(chosen_outer*epochs, 0, 4, color='tab:red',
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
fig_lstm2.tight_layout()
fig_lstm2.subplots_adjust(left=0.12, bottom=0.12, right=0.88,
                          top=0.9, wspace=0.3, hspace=0.2)
name = 'lag_' + str(lag) + '_lambda_' + str(index_lambda)
plt.savefig("./figures/gt_lstm_training_progress_" + name + ".pdf")

######################### Plot Validation Progress ##########################
fig_lstm3 = plt.figure(figsize=[7, 4])
ax = fig_lstm3.add_subplot(1, 1, 1)
ax.axis([0, len(pred_vals), 0.64, 0.74])
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
ax.legend(lns, labs, loc='upper right')
ax.set_xlabel(r'identification cycle [.]')
ax.set_ylabel(r'output glass thickness')
fig_lstm3.tight_layout()
fig_lstm3.subplots_adjust(left=0.12, bottom=0.12, right=0.88,
                          top=0.9, wspace=0.3, hspace=0.2)
name = 'lag_' + str(lag) + '_lambda_' + str(index_lambda)
plt.savefig("./figures/gt_lstm_validation_progress_" + name + ".pdf")

######################### Plot Sparsity LSTM #################################
fig_lstm4 = plt.figure(figsize=[7, 4])
gs = fig_lstm4.add_gridspec(4, 5)
ax = fig_lstm4.add_subplot(gs[0:2, 0])
ax.spy(np.abs(lstm_model.weights['lstm_0.linear_ii'].detach().numpy().transpose()),
       markersize=4)
ax.tick_params(axis='both', which='both', left=False, right=False,
               bottom=False, top=False, labelbottom=False, labelsize=8)
ax.set_title(r'$W_{ii}$', y=0, pad=-15, verticalalignment="top")
ax.set_ylabel('Inputs z')
ax = fig_lstm4.add_subplot(gs[0:2, 1])
ax.spy(np.abs(lstm_model.weights['lstm_0.linear_ij'].detach().numpy().transpose()),
markersize=4)
ax.tick_params(axis='both', which='both', left=False, right=False,
               bottom=False, top=False, labelbottom=False, labelsize=8)
ax.set_title(r'$W_{ij}$', y=0, pad=-15, verticalalignment="top")
ax = fig_lstm4.add_subplot(gs[0:2, 2])
ax.spy(np.abs(lstm_model.weights['lstm_0.linear_if'].detach().numpy().transpose()),
       markersize=4)
ax.tick_params(axis='both', which='both', left=False, right=False,
               bottom=False, top=False, labelbottom=False, labelsize=8)
ax.set_title(r'$W_{if}$', y=0, pad=-15, verticalalignment="top")
ax = fig_lstm4.add_subplot(gs[0:2, 3])
ax.spy(np.abs(lstm_model.weights['lstm_0.linear_io'].detach().numpy().transpose()),
       markersize=4)
ax.tick_params(axis='both', which='both', left=False, right=False,
               bottom=False, top=False, labelbottom=False, labelsize=8)
ax.set_title(r'$W_{io}$', y=0, pad=-15, verticalalignment="top")
ax = fig_lstm4.add_subplot(gs[2:4, 0])
ax.spy(np.abs(lstm_model.weights['lstm_0.linear_hi'].detach().numpy().transpose()),
       markersize=4)
ax.tick_params(axis='both', which='both', left=False, right=False,
               bottom=False, top=False, labelbottom=False, labelsize=8)
ax.set_title(r'$W_{hi}$', y=0, pad=-15, verticalalignment="top")
ax = fig_lstm4.add_subplot(gs[2:4, 1])
ax.spy(np.abs(lstm_model.weights['lstm_0.linear_hj'].detach().numpy().transpose()),
       markersize=4)
ax.tick_params(axis='both', which='both', left=False, right=False,
               bottom=False, top=False, labelbottom=False, labelsize=8)
ax.set_title(r'$W_{hj}$', y=0, pad=-15, verticalalignment="top")
ax = fig_lstm4.add_subplot(gs[2:4, 2])
ax.spy(np.abs(lstm_model.weights['lstm_0.linear_hf'].detach().numpy().transpose()),
       markersize=4)
ax.tick_params(axis='both', which='both', left=False, right=False,
               bottom=False, top=False, labelbottom=False, labelsize=8)
ax.set_title(r'$W_{hf}$', y=0, pad=-15, verticalalignment="top")
ax = fig_lstm4.add_subplot(gs[2:4, 3])
ax.spy(np.abs(lstm_model.weights['lstm_0.linear_ho'].detach().numpy().transpose()),
       markersize=4)
ax.tick_params(axis='both', which='both', left=False, right=False,
               bottom=False, top=False, labelbottom=False, labelsize=8)
ax.set_title(r'$W_{ho}$', y=0, pad=-15, verticalalignment="top")
ax = fig_lstm4.add_subplot(gs[1:3, 4])
ax.spy(np.abs(lstm_model.weights['linear_f'].detach().numpy().transpose()),
       markersize=4)
ax.tick_params(axis='both', which='both', left=False, right=False,
               bottom=False, top=False, labelbottom=False, labeltop=False, labelsize=8)
ax.set_title(r'$W_{f}$', y=0, pad=-15, verticalalignment="top")
fig_lstm4.subplots_adjust(left=0.08, bottom=0.05, right=0.98, top=0.85,
                          wspace=1, hspace=0.8)
# fig_lstm4.suptitle('Sparsity plots of the estimated LSTM Network')
name = 'lag_' + str(lag) + '_lambda_' + str(index_lambda)
plt.savefig("./figures/gt_lstm_sparse_model_" + name + ".pdf")

######################### Plot Simulation Comparison #########################
fig_sim = plt.figure(figsize=[7, 4])
ax = fig_sim.add_subplot(1, 1, 1)
bayes_mlp_str = 'Bayesian MLP Model (RMSE = {:2.4f}'.format(mlp_loss_sim)
bayes_lstm_str = 'Bayesian LSTM Model (RMSE = {:2.4f}'.format(lstm_loss_sim)

ax.plot(val_time, y_val.cpu().numpy(), color='tab:red', label=r'True Output')
ax.plot(val_time, lstm_sim_val.cpu().numpy(),
        color='tab:green', label=bayes_lstm_str + ')')
ax.plot(val_time, mlp_sim_val.cpu().numpy(),
        color='tab:blue', label=bayes_mlp_str + ')')
plot_min = -4
plot_max = 4
ax.axis([val_time[0], val_time[-1], plot_min, plot_max])
ax.grid(True, 'major', 'x')
ax.grid(True, 'both', 'y')
# ax.set_title('Simulation Comparison of GT validation dataset.')
ax.set_ylabel(r'output glass thickness')
ax.set_xlabel(r'time [s]')
ax.legend(loc='upper center')
fig_sim.subplots_adjust(left=0.15, bottom=0.12, right=0.9,
                        top=0.9, wspace=0.3, hspace=0.2)
name = 'lag_' + str(lag) + '_lambda_' + str(index_lambda)
plt.savefig("./figures/gt_sim_comparison_" + name + ".pdf")