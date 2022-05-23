import torch

class BayesianAlgorithm:
    def __init__(self, model, lambda_base):
        self.device = model.save_device
        self.model = model.to(self.device)
        self.lambda_base = lambda_base
        self.names = model.names

        #omage, gamma, alpha for weight
        self.gamma = {}
        self.omega = {}
        self.alpha = {}
        self.sum_gamma = {}
        for name in self.names:
            name_shape = self.model.weights[name].size()
            self.gamma[name + '_in'] = torch.ones(name_shape).to(self.device)
            self.omega[name + '_in'] = torch.ones(name_shape).to(self.device)
            self.alpha[name + '_in'] = torch.ones(name_shape).to(self.device)
            self.gamma[name + '_out'] = torch.ones(name_shape).to(self.device)
            self.omega[name + '_out'] = torch.ones(name_shape).to(self.device)
            self.alpha[name + '_out'] = torch.ones(name_shape).to(self.device)
            self.gamma[name + '_l1'] = torch.ones(name_shape).to(self.device)
            self.omega[name + '_l1'] = torch.ones(name_shape).to(self.device)
            self.alpha[name + '_l1'] = torch.ones(name_shape).to(self.device)
            self.gamma[name + '_l2'] = torch.ones(name_shape).to(self.device)
            self.omega[name + '_l2'] = torch.ones(name_shape).to(self.device)
            self.alpha[name + '_l2'] = torch.ones(name_shape).to(self.device)
            self.sum_gamma[name] = torch.ones(name_shape).to(self.device)

    def get_sum_gamma(self, name):
        return self.sum_gamma[name]

    def update(self, hessian_dic, onoff_regularization):
        with torch.no_grad():
            #for 2D-weight update        
            index = -1
            for name in self.names:
                sign_inner_in = False
                sign_inner_out = False
                sign_inner_l1 = False
                sign_inner_l2 = False
                gamma_inverse = 0
                gamma_inv = 0
                hessian = 0
                hessian_layer = hessian_dic[name]               
                weight_layer = self.model.weights[name]
                index += 1
                # update gamma for lambda in
                if onoff_regularization['group_in']:
                    in_gamma_layer = self.gamma[name + '_in']
                    in_omega_layer = self.omega[name + '_in']
                    in_temp_update_gamma = torch.zeros(in_gamma_layer.size()).to(self.device)
                    if float(self.lambda_base['lambda_in'][index])!=0:
                        sign_inner_in = True
                        for j in range(in_gamma_layer.size(0)):
                            in_gamma_column = in_gamma_layer[j]
                            in_omega_column = in_omega_layer[j]
                            in_length_column = len(in_gamma_column)
                            in_weight_column = weight_layer[j]
                            in_weight_norm2 = torch.norm(in_weight_column.clone().detach(), 2).to(self.device)
                            in_omega_column_value = torch.mean(in_omega_column)
                            in_gamma_column_value = in_weight_norm2 / in_omega_column_value
                            in_gamma_column_value = torch.clamp(in_gamma_column_value, min=1e-10, max=1e10)
                            in_temp_update_gamma[j] = torch.ones(in_length_column).to(self.device) * in_gamma_column_value
                        self.gamma[name + '_in'] = in_temp_update_gamma                      
                        in_gamma_inverse = 1 / in_temp_update_gamma
                        in_gamma_local_matrix = in_temp_update_gamma
                        gamma_inv += 1 / self.gamma[name + '_in']
                        gamma_inverse += in_gamma_inverse                           
                        hessian += 1 / float(self.lambda_base['lambda_in'][index]) * hessian_layer
                        del in_gamma_inverse, in_temp_update_gamma

                # update gamma for lambda out
                if onoff_regularization['group_out']:
                    out_gamma_layer = self.gamma[name + '_out']
                    out_omega_layer = self.omega[name + '_out']
                    out_temp_update_gamma = torch.zeros(out_gamma_layer.size()).to(self.device)
                    if float(self.lambda_base['lambda_out'][index])!=0:
                        sign_inner_out = True
                        for j in range(out_gamma_layer.size(1)):
                            out_gamma_row = out_gamma_layer.t()[j]
                            out_omega_row = out_omega_layer.t()[j]
                            out_length_row = len(out_gamma_row)
                            out_weight_row = weight_layer.t()[j]
                            out_weight_norm2 = torch.norm(out_weight_row.clone().detach(), 2).to(self.device)
                            out_omega_row_value = torch.sum(out_omega_row) / out_length_row
                            out_gamma_row_value = out_weight_norm2 / out_omega_row_value
                            out_gamma_row_value = torch.clamp(out_gamma_row_value, min=1e-10, max=1e10)
                            out_temp_update_gamma.t()[j] = torch.ones(out_length_row).to(self.device) * out_gamma_row_value
                        self.gamma[name + '_out'] = out_temp_update_gamma
                        out_gamma_inverse = 1 / out_temp_update_gamma
                        out_gamma_local_matrix = out_temp_update_gamma
                        gamma_inv += 1 / self.gamma[name + '_out']
                        gamma_inverse += out_gamma_inverse
                        hessian += 1 / float(float(self.lambda_base['lambda_out'][index])) * hessian_layer
                        del out_gamma_inverse, out_temp_update_gamma
      
                # update gamma for lambda l1
                if onoff_regularization['l1']:
                    if float(self.lambda_base['lambda_l1'][index])!=0:
                        sign_inner_l1= True
                        l1_omega_layer = self.omega[name + '_l1']            
                        l1_temp_update_gamma = torch.abs(weight_layer / l1_omega_layer).to(self.device)
                        l1_temp_update_gamma = torch.clamp(l1_temp_update_gamma, min=1e-10, max=1e10).to(self.device)
                        self.gamma[name + '_l1'] = l1_temp_update_gamma                    
                        l1_gamma_inverse = 1 / l1_temp_update_gamma
                        l1_gamma_local_matrix = l1_temp_update_gamma
                        gamma_inv += 1 / self.gamma[name + '_l1']
                        gamma_inverse += l1_gamma_inverse
                        hessian += 1 / float(float(self.lambda_base['lambda_l1'][index])) * hessian_layer

                # update gamma for lambda l2
                l2_update_order = 'after'
                if onoff_regularization['l2']:
                    if float(self.lambda_base['lambda_l2'][index]) != 0:
                        sign_inner_l2 = True
                        if l2_update_order == 'before':
                            l2_omega_layer = self.omega[name + '_l2']
                            l2_temp_update_gamma = torch.abs(weight_layer / l2_omega_layer)
                            l2_temp_update_gamma = torch.clamp(l2_temp_update_gamma, min=1e-10, max=1e10)
                            self.gamma[name + '_l2'] = l2_temp_update_gamma
                            l2_gamma_inverse = 1 / l2_temp_update_gamma
                            l2_gamma_local_matrix = l2_temp_update_gamma
                            gamma_inv += 1 / self.gamma[name + '_l2']
                            gamma_inverse += l2_gamma_inverse
                            hessian += float(float(self.lambda_base['lambda_l2'][index])) * hessian_layer
                        if l2_update_order == 'after':
                            l2_temp_update_gamma = self.gamma[name + '_l2']
                            l2_gamma_local_matrix = l2_temp_update_gamma
                            hessian += 1 / float(self.lambda_weight[name+'_l2']) * hessian_layer
 
               # update for C
                if sign_inner_in== True or sign_inner_out== True or sign_inner_l1== True or sign_inner_l2== True:
                    C = gamma_inverse + hessian
                    C = torch.reciprocal(C)
                    self.sum_gamma[name] = 1/gamma_inv
                    del gamma_inv, gamma_inverse, hessian, hessian_layer
                
                # group_in
                if onoff_regularization['group_in']:
                    if float(self.lambda_base['lambda_in'][index]) != 0:
                        in_alpha = -C / (in_gamma_local_matrix.pow(2)) + 1 / in_gamma_local_matrix

                        in_alpha_reshape = in_alpha
                        self.alpha[name + '_in'] = in_alpha_reshape
                        in_update_omega_temp = torch.ones(in_gamma_layer.size()).to(self.device)
                        for i in range(in_gamma_layer.size(0)):
                            in_alpha_i = in_alpha_reshape[i]
                            in_alpha_i_value = torch.sqrt(torch.abs(in_alpha_i).sum()/len(in_alpha_i))
                            in_alpha_i_value = torch.clamp(in_alpha_i_value, min=1e-10, max=1e10)
                            in_update_omega_temp[i] = torch.ones(in_gamma_layer.size(1)).to(self.device) * in_alpha_i_value
                        self.omega[name + '_in'] = in_update_omega_temp
                        del in_update_omega_temp, in_alpha, in_alpha_reshape
        
                # group_out
                if onoff_regularization['group_out']:
                    if float(self.lambda_base['lambda_out'][index]) != 0:
                        out_alpha = -C / (out_gamma_local_matrix.pow(2)) + 1 / out_gamma_local_matrix
                        out_alpha_reshape = out_alpha
                        self.alpha[name + '_out'] = out_alpha_reshape
                        out_update_omega_temp = torch.ones(out_gamma_layer.size()).to(self.device)
                        for i in range(out_gamma_layer.size(1)):
                            out_alpha_i = out_alpha_reshape.t()[i]
                            out_alpha_i_value = torch.sqrt(torch.abs(out_alpha_i).sum()/len(out_alpha_i))
                            out_alpha_i_value = torch.clamp(out_alpha_i_value, min=1e-10, max=1e10)
                            out_update_omega_temp.t()[i] = torch.ones(out_gamma_layer.size(0)).to(self.device) * out_alpha_i_value
                        self.omega[name + '_out'] = out_update_omega_temp
                        del out_update_omega_temp, out_alpha, out_alpha_reshape
        
                # l1
                if onoff_regularization['l1']:
                    if float(self.lambda_base['lambda_l1'][index]) != 0:
                        l1_alpha = -C / (l1_gamma_local_matrix.pow(2)) + 1 / l1_gamma_local_matrix
                        l1_alpha_reshape = l1_alpha
                        self.alpha[name + '_l1'] = l1_alpha_reshape
                        l1_update_omega_temp = torch.sqrt(torch.abs(l1_alpha_reshape))
                        self.omega[name + '_l1'] = l1_update_omega_temp
        
                # l2
                if onoff_regularization['l2']:
                    if float(self.lambda_base['lambda_l2'][index]) != 0:
                        if l2_update_order == 'before':
                            l2_alpha = -C / (l2_gamma_local_matrix.pow(2)) + 1 / l2_gamma_local_matrix
                            l2_alpha_reshape = l2_alpha
                            self.alpha[name + '_l2'] = l2_alpha_reshape
                            l2_update_omega_temp = torch.sqrt(torch.abs(l2_alpha_reshape))
                            self.omega[name + '_l2'] = l2_update_omega_temp
                        if l2_update_order == 'after':
                            l2_alpha = -C / (l2_gamma_local_matrix.pow(2)) + 1 / l2_gamma_local_matrix
                            l2_alpha_reshape = l2_alpha
                            self.alpha[name + '_l2'] = l2_alpha_reshape
                            l2_temp_updategamma = torch.clamp(torch.abs(weight_layer / torch.sqrt(torch.abs(l2_alpha_reshape))),
                                                              min=1e-10, max=1e10)
                            self.gamma[name + '_l2'] = l2_temp_updategamma
                            self.sum_gamma[name] += 1 / l2_temp_update_gamma
                            self.omega[name + '_l2'] = 1 / torch.sqrt(torch.abs(l2_temp_updategamma))
        return


    def loss_cal(self, prediction, target,  onoff_regularization):
        target = target.float()
        loss_prediction = torch.sqrt(torch.nn.functional.mse_loss(prediction, target))
        loss_reg = 0
        index = -1
        for name in self.names:
            index += 1
            if onoff_regularization['group_in']:
                if self.lambda_base["lambda_in"][index] != 0:
                    loss_reg += self.lambda_base["lambda_in"][index]*torch.abs(torch.norm(torch.mul(self.model.weights[name],self.omega[name + '_in']), 2, 1)).sum()
            if onoff_regularization['group_out']:
                if self.lambda_base["lambda_out"][index] != 0:
                    loss_reg += self.lambda_base["lambda_out"][index]*torch.abs(torch.norm(torch.mul(self.model.weights[name],self.omega[name + '_out']), 2, 0)).sum()
            if onoff_regularization['l1']:
                if self.lambda_base["lambda_l1"][index] != 0:
                    loss_reg += self.lambda_base["lambda_l1"][index]*torch.abs(torch.norm(torch.mul(self.model.weights[name],self.omega[name + '_l1']), 1)).sum()
            if onoff_regularization['l2']:
                if self.lambda_base["lambda_l2"][index] != 0:
                    loss_reg += self.lambda_base["lambda_l2"][index]*torch.abs(torch.norm(torch.mul(self.model.weights[name],self.omega[name + '_l2']), 2)).sum()

        loss = loss_prediction + loss_reg
        return loss_prediction, loss

