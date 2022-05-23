import torch
import numpy as np
import model as md
import torch.autograd as autograd

def clean_data(inputs, outputs, detrend = False, normalize = False):
    """this part is used to normalize data"""
    std_y_train, mean_y_train = 1, 0
    std_x_train, mean_x_train = 1, 0
    x_train = inputs
    y_train = outputs
    if detrend or normalize:
        mean_x_train = torch.mean(inputs)
        mean_y_train = torch.mean(outputs)
        x_train = (inputs.clone() - mean_x_train)
        y_train = (outputs.clone() - mean_y_train)
    if normalize:
        std_x_train = torch.std(inputs, unbiased=True)
        std_y_train = torch.std(outputs, unbiased=True)
        x_train = x_train / std_x_train
        y_train = y_train / std_y_train

    return x_train, y_train, std_y_train, mean_y_train, std_x_train, \
           mean_x_train

def data_normalize(inputs, outputs, mean_x_train, std_x_train,
                        mean_y_train, std_y_train):
    x_test = (inputs.clone() - mean_x_train) / std_x_train
    y_test = (outputs.clone() - mean_y_train) / std_y_train
    return x_test, y_test

def divide_data(inputs, outputs, percentage):
    """this part is used to normalize data"""
    train_number = int(np.round(inputs.shape[0] * percentage))
    x_train = inputs[:train_number]
    y_train = outputs[:train_number]
    x_test= inputs[train_number:]
    y_test= outputs[train_number:]
    return x_train,y_train,x_test,y_test

def select_train_data(inputs, outputs, percentage):
    """this part is used to normalize data"""
    train_number = int(np.round(inputs.shape[0] * percentage))
    x_train = inputs[:train_number]
    y_train = outputs[:train_number]
    return x_train,y_train

def select_train_test_data(inputs, outputs, train_number):
    """this part is used to normalize data"""
    x_train = inputs[:train_number]
    x_test = inputs[train_number:]
    y_train = outputs[:train_number]
    y_test = outputs[train_number:]
    return x_train, y_train, x_test, y_test

def generate_prediction_data(inputs, outputs, lag_inputs, lag_outputs):
    """decide new dimension for data"""
    lag = max(lag_inputs,lag_outputs)
    new_val_sample_numbers = outputs.size(0) - lag
    new_val_feature_numbers = lag_outputs*outputs.size(1)+\
                              (lag_inputs+1)*inputs.size(1)
    target_inputs= torch.ones(new_val_sample_numbers, new_val_feature_numbers)
    target_outputs = torch.ones(new_val_sample_numbers, outputs.size(1))
    for i in range(target_inputs.size(0)):
        inputs_index = lag+i+1
        input_reg = inputs[inputs_index-lag_inputs-1:inputs_index]
        outputs_index = lag+i
        output_reg = outputs[outputs_index-lag_outputs:outputs_index]
        target_inputs[i] = torch.cat((input_reg, output_reg)
                                     ,0).reshape(target_inputs[i].size())
        target_outputs[i] = outputs[outputs_index]
    return target_inputs, target_outputs

def batchify(inputs, outputs, batch_size):
    batches = {}
    data_length = inputs.shape[0]
    batch_number = int(np.ceil(data_length / batch_size))
    for i in range(batch_number):
        start = batch_size*i
        end = min(batch_size*(i+1), data_length)
        batches['batch_' + str(i)] = {'x':inputs[start:end],
                                      'y':outputs[start:end]}
    return batches

def train(model, algorithm, optimizer, batches, onoff_regularization,
          onoff_hessian):
    hessian_auto = {}
    layer_n = len(model.layers) - 2
    pred_loss = 0
    n_total = 0
    model.train()
    for batchID in batches:
        x = batches[batchID]['x']
        y = batches[batchID]['y']
        n = x.shape[0]
        if isinstance(model,md.MLPNetwork):
            pred = model(x)
        elif isinstance(model,md.RNNNetwork):
            hidden = torch.zeros(layer_n, model.hiddenUnits,
                                 device=model.save_device, requires_grad=False)
            c = torch.zeros(layer_n, model.hiddenUnits,
                            device=model.save_device, requires_grad=False)
            pred, _, _ = model(x, hidden, c)
        loss_pred, loss = algorithm.loss_cal(pred, y, onoff_regularization)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if isinstance(model,md.MLPNetwork):
            pred = model(x)
        elif isinstance(model,md.RNNNetwork):
            hidden = torch.zeros(layer_n, model.hiddenUnits,
                                 device=model.save_device, requires_grad=False)
            c = torch.zeros(layer_n, model.hiddenUnits,
                            device=model.save_device, requires_grad=False)
            pred, _, _ = model(x, hidden, c)
        new_loss_pred, new_loss = algorithm.loss_cal(pred, y, onoff_regularization)
        pred_loss += torch.pow(new_loss_pred.detach(),2)*n
        n_total += n
    pred_loss = pred_loss/n_total
    train_loss = pred_loss + (new_loss.detach()-new_loss_pred.detach())
    if onoff_hessian:
        hessian_auto = second_order_derivative(model, new_loss_pred)
    return train_loss, pred_loss, hessian_auto

def second_order_derivative(model, loss):
    hessian_auto = {}
    for key,param in model.named_parameters():
        grad = torch.zeros_like(param)
        grad2 = torch.zeros_like(param)
        if 'weight' in key:
            grad = autograd.grad(loss, param, create_graph= True)[0]
            grad2=autograd.grad(grad.sum(), param, retain_graph=True)[0]
        for name in model.names:
            if 'weight' in key and name in key:
                hessian_auto[name] = grad2
        del grad, grad2
    torch.cuda.empty_cache()
    return hessian_auto

def validate(model, inputs, outputs, std_y_train, mean_y_train):
    model.eval()
    with torch.no_grad():
        layer_n = len(model.layers)-2
        x = inputs.to(model.save_device)
        if isinstance(model,md.MLPNetwork):
            val_pred = model(x)
        elif isinstance(model,md.RNNNetwork):
            hidden = torch.zeros(layer_n, model.hiddenUnits,
                                 device=model.save_device, requires_grad=False)
            c = torch.zeros(layer_n, model.hiddenUnits,
                            device=model.save_device, requires_grad=False)
            val_pred, _, _ = model(x, hidden, c)
        val_pred = restore_data(val_pred,std_y_train,mean_y_train)
        real_est = restore_data(outputs, std_y_train, mean_y_train)
        val_loss = torch.sqrt(torch.nn.functional.mse_loss(val_pred, real_est))
    return val_loss.item(), val_pred

def validate_sim(model, inputs, outputs, std_y_train, mean_y_train,
                 lag_in, lag_out):
    model.eval()
    with torch.no_grad():
        layer_n = len(model.layers)-2
        lag = max([lag_in,lag_out])
        simloop_num = outputs.size(0)
        sim_inputs = torch.zeros(1,lag_in+lag_out+1, device=model.save_device)
        sim_outputs = outputs[lag - lag_out:lag].clone()
        val_sim = torch.zeros(simloop_num, outputs.shape[1],
                              device=model.save_device)
        val_sim[0:lag] = restore_data(outputs[0:lag], std_y_train,
                                      mean_y_train)
        if isinstance(model, md.RNNNetwork):
            hidden = torch.zeros(layer_n,  1, model.hiddenUnits,
                                 device=model.save_device, requires_grad=False)
            c = torch.zeros(layer_n, 1, model.hiddenUnits,
                            device=model.save_device, requires_grad=False)
        for sim_iter in range(lag, simloop_num):
            sim_inputs[:,0:lag+1] = inputs[sim_iter - lag:sim_iter + 1,:].t()
            sim_inputs[:,-lag_out:] = sim_outputs.t()
            if isinstance(model, md.MLPNetwork):
                sim = model(sim_inputs)
            elif isinstance(model, md.RNNNetwork):
                sim, hidden[:,0,:], c[:,0,:] = model(sim_inputs, hidden, c)
            val_sim[sim_iter] = restore_data(sim, std_y_train, mean_y_train)
            sim_outputs[0:lag_out-1] = sim_outputs[1:lag_out].clone()
            sim_outputs[-1] = sim
        sim_real = restore_data(outputs, std_y_train, mean_y_train)
        sim_loss = torch.sqrt(torch.nn.functional.mse_loss(sim_real, val_sim))
    return sim_loss.item(), val_sim

def restore_data(inputs, std_y_train, mean_y_train):
    """this code is used for testing,only exeacuting the forward process"""
    with torch.no_grad():
        pred_outputs = inputs * std_y_train + mean_y_train
    return pred_outputs

def predict_distribution(model, inputs, outputs, masks, gamma, hessian,
                         std_y_train, mean_y_train, repeat=10000):
    """ Predictive distribution mode calculation by MC integration """
    model.eval()
    with torch.no_grad():
        weights = sample_posterior_weights(model, hessian, gamma, repeat)

        # Implement the validation process repeatly
        uncertainty_outputs = []
        uncertainty_loss = []
        for rep in range(repeat):
            # Set the sampled weights
            for name in model.names:
                param = model.weights[name]
                if 'linear' in name:
                    param.data = weights[name][:, :, rep].reshape(param.size())
                    param.data = param.masked_fill(~masks[name], 0)

            val_loss, val_pred = validate(model, inputs, outputs,
                                          std_y_train, mean_y_train)

            uncertainty_outputs.append(val_pred)
            uncertainty_loss.append(val_loss)

        # Calcule the mean/variance of the predicted outputs
        uncertainty_outputs_reshape = torch.cat(uncertainty_outputs, 1)
        val_pred_mean = torch.mean(uncertainty_outputs_reshape, 1)
        val_pred_std = torch.std(uncertainty_outputs_reshape, 1)

        # val_pred_mode, _ = torch.mode(uncertainty_outputs_reshape, 1)

    return val_pred_mean, val_pred_std

def simulate_distribution(model, inputs, outputs, masks, gamma, hessian,
                         std_y_train, mean_y_train, lag_in, lag_out,
                          repeat=10000):
    """ Predictive distribution mode calculation by MC integration """
    model.eval()
    with torch.no_grad():
        weights = sample_posterior_weights(model, hessian, gamma, repeat)

        # Implement the validation process repeatly
        uncertainty_outputs = []
        uncertainty_loss = []
        for rep in range(repeat):
            # Set the sampled weights
            for name in model.names:
                param = model.weights[name]
                if 'linear' in name:
                    param.data = weights[name][:, :, rep].reshape(param.size())
                    param.data = param.masked_fill(~masks[name], 0)

            val_loss, val_sim = validate_sim(model, inputs, outputs,
                                              std_y_train, mean_y_train,
                                              lag_in, lag_out)

            if np.isnan(val_loss) or np.isinf(val_loss) or val_loss>100:
                pass
            else:
                uncertainty_loss.append(val_loss)
                uncertainty_outputs.append(val_sim)

        # Calcule the mean/variance of the predicted outputs
        uncertainty_outputs_reshape = torch.cat(uncertainty_outputs, 1)
        val_pred_mean = torch.mean(uncertainty_outputs_reshape, 1)
        val_pred_std = torch.std(uncertainty_outputs_reshape, 1)
    return val_pred_mean, val_pred_std

# def simulate_distribution(model, inputs, outputs, masks, gamma, hessian,
#                          std_y_train, mean_y_train, lag_in, lag_out,
#                           repeat=10000):
#     """ Simulation distribution mode calculation by MC integration """
#     model.eval()
#     with torch.no_grad():
#         layer_n = len(model.layers)-2
#
#         # Sample Connection weights from posterior
#         weights = sample_posterior_weights(model, hessian, gamma, repeat)
#
#         # Initialize simulation variables
#         lag = max([lag_in, lag_out])
#         simloop_num = outputs.size(0)
#
#         # Initialize the final outputs
#         val_sim_mean = torch.zeros(simloop_num, outputs.shape[1],
#                                    device=model.save_device)
#         val_sim_mean[0:lag] = restore_data(outputs[0:lag], std_y_train,
#                                            mean_y_train)
#         val_sim_std = torch.zeros(simloop_num, outputs.shape[1],
#                                   device=model.save_device)
#         # Initialize the regressors
#         sim_inputs = torch.zeros(1, lag_in + lag_out + 1,
#                                  device=model.save_device)
#         sim_outputs_mean = outputs[lag - lag_out:lag].clone()
#         sim_outputs_std = torch.zeros(sim_outputs_mean.size(),
#                                       device=model.save_device)
#
#         # For LSTM initialize some derived parameters
#         if isinstance(model, md.RNNNetwork):
#             hidden_mean = torch.zeros(layer_n, 1, model.hiddenUnits,
#                                       device=model.save_device)
#             hidden_std = torch.ones(layer_n, 1, model.hiddenUnits,
#                                      device=model.save_device)*1e-10
#             c_mean = torch.zeros(layer_n, 1, model.hiddenUnits,
#                                  device=model.save_device)
#             c_std = torch.ones(layer_n, 1, model.hiddenUnits,
#                                 device=model.save_device)*1e-10
#
#         # Loop through time steps
#         for sim_iter in range(lag, simloop_num):
#             sim_inputs[:,0:lag+1] = inputs[sim_iter-lag:sim_iter+1,:].t()
#             sims = torch.zeros(repeat, 1, device=model.save_device)
#             if isinstance(model, md.RNNNetwork):
#                 sampled_hidden = torch.normal(hidden_mean, hidden_std)
#                 sampled_c = torch.normal(c_mean, c_std)
#                 hiddens = torch.zeros(repeat, layer_n, 1, model.hiddenUnits,
#                                       device=model.save_device)
#                 cs = torch.zeros(repeat, layer_n, 1, model.hiddenUnits,
#                                  device=model.save_device)
#
#             # Sample from posterior predictive of regressors
#             sampled_sim_outputs = sample_posterior_regressors(sim_outputs_mean,
#                                                               sim_outputs_std,
#                                                               repeat)
#
#             for rep in range(repeat):
#                 # Set the sampled weights
#                 for name in model.names:
#                     param = model.weights[name]
#                     if 'linear' in name:
#                         param.data = weights[name][:, :, rep].reshape(param.size())
#                         param.data = param.masked_fill(~masks[name], 0)
#                 # Set the sampled regressors
#                 sim_inputs[:, -lag_out:] = sampled_sim_outputs[:,:,rep].t()
#
#                 # Simulate the time step
#                 if isinstance(model, md.MLPNetwork):
#                     sims[rep,:] = model(sim_inputs)
#                 elif isinstance(model, md.RNNNetwork):
#                     hidden = sampled_hidden[rep]
#                     c = sampled_c[rep]
#                     res = model(sim_inputs, hidden, c)
#                     sims[rep,:], hiddens[rep,:,0,:], cs[rep,:,0,:] = res
#
#             # Update regressors for next sim_iter
#             sim_outputs_mean[0:lag_out-1] = sim_outputs_mean[1:lag_out].clone()
#             sim_outputs_std[0:lag_out-1] = sim_outputs_std[1:lag_out].clone()
#             sim_outputs_mean[-1] = torch.mean(sims,dim=0)
#             sim_outputs_std[-1] = torch.std(sims, dim=0)
#
#             # Update hidden and cell
#             if isinstance(model, md.RNNNetwork):
#                 hidden_mean = torch.mean(hiddens, dim=0)
#                 hidden_std = torch.std(hiddens, dim=0)
#                 c_mean = torch.mean(cs, dim=0)
#                 c_std = torch.std(cs, dim=0)
#
#             # Restore and store to return
#             real_out = restore_data(sims, std_y_train, mean_y_train)
#             val_sim_mean[sim_iter] = torch.mean(real_out,dim=0)
#             val_sim_std[sim_iter] = torch.std(real_out,dim=0)
#
#     return val_sim_mean[:,0], val_sim_std[:,0]

# def sample_posterior_regressors(sim_outputs_mean, sim_outputs_std, repeat):
#     sampled_sim_outputs = torch.zeros(sim_outputs_mean.size()+(repeat,))
#
#     for sim_out in range(len(sim_outputs_mean)):
#         mean = sim_outputs_mean[sim_out,:].item()
#         std = sim_outputs_std[sim_out,:].item()
#         if std==0: std = 1e-10
#         sampled_sim_outputs[sim_out] = torch.normal(mean, std, size=(1,repeat))
#     return sampled_sim_outputs

def sample_posterior_weights(model, hessian, gamma, repeat):
    means = {}
    variances = {}
    weights = {}
    with torch.no_grad():
        # Sample Connection weights from posterior
        for name in model.names:
            param = model.weights[name]
            if 'linear' in name:
                variances[name] = 1 / (torch.abs(hessian[name])
                                        + 1 / gamma[name])
                means[name] = param
                weights[name] = torch.zeros((param.size(0), param.size(1), repeat))

                # Sample weights from the posterior distribution
                nonzero_index_list = torch.nonzero(param)
                for index in nonzero_index_list:
                    mean = means[name][index[0], index[1]].item()
                    std = variances[name][index[0], index[1]].item()
                    if std==0 or mean <=1e-4  : std = 1e-10
                    weights[name][index[0], index[1]] = torch.normal(
                        mean, std, size=(1, repeat))
    return weights

def weight_overall_sparsity(weight):
    row, column = weight.size()
    amount_elements = row * column
    amount_zero = int((weight == 0).sum())
    return 1 - float(amount_zero / amount_elements)

def structure_sparsity(weight):
    weight_in = weight
    weight_out = weight.t()
    row, column = weight.size()
    amount_inputs = row
    amount_outputs = column
    weight_norm_in = torch.norm(weight_in, 2, 1)
    weight_norm_out = torch.norm(weight_out, 2, 1)
    amount_zero_in = torch.nonzero(weight_norm_in.data).size(0)
    amount_zero_out = torch.nonzero(weight_norm_out.data).size(0)
    strucutre_sparsity_in = float(amount_zero_in / amount_inputs)
    strucutre_sparsity_out = float(amount_zero_out / amount_outputs)
    return strucutre_sparsity_in, strucutre_sparsity_out

def model_overall_sparsity(model):
    amount_zero = 0
    amount_total = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            row, column = param.size()
            tmp_amount_elements = row * column
            tmp_amount_zero = (param == 0).sum()
            amount_total += tmp_amount_elements
            amount_zero += int(tmp_amount_zero)
            print(name, int(tmp_amount_zero)/tmp_amount_elements)
    return 1 - float(amount_zero / amount_total)
