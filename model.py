import torch


class MLPNetwork(torch.nn.Module):
    def __init__(self, layers, activation=torch.nn.functional.sigmoid ,
                 bias=True, save_device=torch.device('cpu')):
        super(MLPNetwork, self).__init__()
        self.save_device = save_device
        self.activation = activation
        self.layers = layers
        self.weights = {}
        self.mask = {}

        if len(self.layers) == 2:
            self.linear_0 = torch.nn.Linear(in_features=self.layers[0],
                                            out_features=self.layers[1],
                                            bias=bias).to(self.save_device)
            self.weights['linear_0'] = self.linear_0.weight
            self.names = ['linear_0']
        
        if len(self.layers) == 3:
            self.linear_0 = torch.nn.Linear(in_features=self.layers[0],
                                            out_features=self.layers[1],
                                            bias=bias).to(self.save_device)
            self.linear_1 = torch.nn.Linear(in_features=self.layers[1],
                                            out_features=self.layers[2],
                                            bias=bias).to(self.save_device)
            self.weights['linear_0'] = self.linear_0.weight
            self.weights['linear_1'] = self.linear_1.weight 
            self.names = ['linear_0', 'linear_1']

        if len(self.layers) == 4:
            self.linear_0 = torch.nn.Linear(in_features=self.layers[0],
                                            out_features=self.layers[1],
                                            bias=bias).to(self.save_device)
            self.linear_1 = torch.nn.Linear(in_features=self.layers[1],
                                            out_features=self.layers[2],
                                            bias=bias).to(self.save_device)
            self.linear_2 = torch.nn.Linear(in_features=self.layers[2],
                                            out_features=self.layers[3],
                                            bias=bias).to(self.save_device)
            self.weights['linear_0'] = self.linear_0.weight
            self.weights['linear_1'] = self.linear_1.weight
            self.weights['linear_2'] = self.linear_2.weight
            self.names = ['linear_0', 'linear_1', 'linear_2']
                
        if len(self.layers) == 5:
            self.linear_0 = torch.nn.Linear(in_features=self.layers[0],
                                            out_features=self.layers[1],
                                            bias=bias).to(self.save_device)
            self.linear_1 = torch.nn.Linear(in_features=self.layers[1],
                                            out_features=self.layers[2],
                                            bias=bias).to(self.save_device)
            self.linear_2 = torch.nn.Linear(in_features=self.layers[2],
                                            out_features=self.layers[3],
                                            bias=bias).to(self.save_device)
            self.linear_3 = torch.nn.Linear(in_features=self.layers[3],
                                            out_features=self.layers[4],
                                            bias=bias).to(self.save_device)
            self.weights['linear_0'] = self.linear_0.weight
            self.weights['linear_1'] = self.linear_1.weight
            self.weights['linear_2'] = self.linear_2.weight
            self.weights['linear_3'] = self.linear_3.weight
            self.names = ['linear_0', 'linear_1', 'linear_2','linear_3']

    def forward(self, x):
        if len(self.layers) == 2:
            x = self.linear_0(x)

        elif len(self.layers) == 3:
            if self.activation is None:
                x = self.linear_0(x)
            else:
                x = self.activation(self.linear_0(x))
            x = self.linear_1(x)

        elif len(self.layers) == 4:
            if self.activation is None:
                x = self.linear_0(x)
                x = self.linear_1(x)
            else:
                x = self.activation(self.linear_0(x))
                x = self.activation(self.linear_1(x))
            x = self.linear_2(x)

        elif len(self.layers) == 5:
            if self.activation is None:
                x = self.linear_0(x)
                x = self.linear_1(x)
                x = self.linear_2(x)
            else:
                x = self.activation(self.linear_0(x))
                x = self.activation(self.linear_1(x))
                x = self.activation(self.linear_2(x))
            x = self.linear_3(x)
        return x

class LSTMLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, save_device, bias=True):
        super(LSTMLayer, self).__init__()
        self.save_device = save_device
        self.linear_ii = torch.nn.Linear(in_features=in_features,
                                         out_features=out_features,
                                         bias=bias).to(self.save_device)
        self.linear_ij = torch.nn.Linear(in_features=in_features,
                                         out_features=out_features,
                                         bias=bias).to(self.save_device)
        self.linear_if = torch.nn.Linear(in_features=in_features,
                                         out_features=out_features,
                                         bias=bias).to(self.save_device)
        self.linear_io = torch.nn.Linear(in_features=in_features,
                                         out_features=out_features,
                                         bias=bias).to(self.save_device)

        self.linear_hi = torch.nn.Linear(in_features=out_features,
                                         out_features=out_features,
                                         bias=bias).to(self.save_device)
        self.linear_hj = torch.nn.Linear(in_features=out_features,
                                         out_features=out_features,
                                         bias=bias).to(self.save_device)
        self.linear_hf = torch.nn.Linear(in_features=out_features,
                                         out_features=out_features,
                                         bias=bias).to(self.save_device)
        self.linear_ho = torch.nn.Linear(in_features=out_features,
                                         out_features=out_features,
                                         bias=bias).to(self.save_device)

    def forward(self, inputs, hidden, c):
        new_hidden = hidden
        new_c = c
        inputs = inputs.unbind(0)
        outputs = []
        for t in range(len(inputs)):
            i_output = torch.sigmoid(self.linear_ii(inputs[t])
                                     + self.linear_hi(new_hidden))
            j_output = torch.tanh(self.linear_ij(inputs[t])
                                  + self.linear_hj(new_hidden))
            f_output = torch.sigmoid(self.linear_if(inputs[t])
                                     + self.linear_hf(new_hidden))
            o_output = torch.sigmoid(self.linear_io(inputs[t])
                                     + self.linear_ho(new_hidden))
            new_c = f_output * new_c + i_output * j_output
            new_hidden = o_output * torch.tanh(new_c)

            outputs += [new_hidden]

        return torch.stack(outputs), new_hidden, new_c


class RNNNetwork(torch.nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
    def __init__(self, layers, save_device=torch.device('cpu'), bias=True):
        super(RNNNetwork, self).__init__()
        self.save_device = save_device
        self.layers = layers
        self.hiddenUnits = layers[1]
        self.weights = {}
        self.mask = {}

        if len(layers) == 2:
            self.linear_f = torch.nn.Linear(in_features=layers[0],
                                            out_features=layers[1],
                                            bias=bias).to(self.save_device)
            self.weights['linear_f'] = self.linear_f.weight
            self.names = ['linear_f']

        elif len(layers) == 3:
            self.lstm_0 = LSTMLayer(in_features=layers[0],
                                    out_features=layers[1],
                                    save_device=self.save_device,
                                    bias=bias)
            self.linear_f = torch.nn.Linear(in_features=layers[1],
                                            out_features=layers[2],
                                            bias=bias).to(self.save_device)
            self.weights['lstm_0.linear_ii'] = self.lstm_0.linear_ii.weight
            self.weights['lstm_0.linear_ij'] = self.lstm_0.linear_ij.weight
            self.weights['lstm_0.linear_if'] = self.lstm_0.linear_if.weight
            self.weights['lstm_0.linear_io'] = self.lstm_0.linear_io.weight
            self.weights['lstm_0.linear_hi'] = self.lstm_0.linear_hi.weight
            self.weights['lstm_0.linear_hj'] = self.lstm_0.linear_hj.weight
            self.weights['lstm_0.linear_hf'] = self.lstm_0.linear_hf.weight
            self.weights['lstm_0.linear_ho'] = self.lstm_0.linear_ho.weight
            self.weights['linear_f'] = self.linear_f.weight
            self.names = ['lstm_0.linear_ii', 'lstm_0.linear_ij',
                          'lstm_0.linear_if', 'lstm_0.linear_io',
                          'lstm_0.linear_hi', 'lstm_0.linear_hj',
                          'lstm_0.linear_hf', 'lstm_0.linear_ho', 'linear_f']

        elif len(layers) == 4:
            self.lstm_0 = LSTMLayer(in_features=layers[0],
                                    out_features=layers[1],
                                    save_device= self.save_device,
                                    bias=bias)
            self.lstm_1 = LSTMLayer(in_features=layers[1],
                                    out_features=layers[2],
                                    save_device=self.save_device,
                                    bias=bias)
            self.linear_f = torch.nn.Linear(in_features=layers[2],
                                            out_features=layers[3],
                                            bias=bias).to(self.save_device)
            self.weights['lstm_0.linear_ii'] = self.lstm_0.linear_ii.weight
            self.weights['lstm_0.linear_ij'] = self.lstm_0.linear_ij.weight
            self.weights['lstm_0.linear_if'] = self.lstm_0.linear_if.weight
            self.weights['lstm_0.linear_io'] = self.lstm_0.linear_io.weight
            self.weights['lstm_0.linear_hi'] = self.lstm_0.linear_hi.weight
            self.weights['lstm_0.linear_hj'] = self.lstm_0.linear_hj.weight
            self.weights['lstm_0.linear_hf'] = self.lstm_0.linear_hf.weight
            self.weights['lstm_0.linear_ho'] = self.lstm_0.linear_ho.weight
            self.weights['lstm_1.linear_ii'] = self.lstm_1.linear_ii.weight
            self.weights['lstm_1.linear_ij'] = self.lstm_1.linear_ij.weight
            self.weights['lstm_1.linear_if'] = self.lstm_1.linear_if.weight
            self.weights['lstm_1.linear_io'] = self.lstm_1.linear_io.weight
            self.weights['lstm_1.linear_hi'] = self.lstm_1.linear_hi.weight
            self.weights['lstm_1.linear_hj'] = self.lstm_1.linear_hj.weight
            self.weights['lstm_1.linear_hf'] = self.lstm_1.linear_hf.weight
            self.weights['lstm_1.linear_ho'] = self.lstm_1.linear_ho.weight
            self.weights['linear_f'] = self.linear_f.weight
            self.names = ['lstm_0.linear_ii', 'lstm_0.linear_ij',
                          'lstm_0.linear_if', 'lstm_0.linear_io',
                          'lstm_0.linear_hi', 'lstm_0.linear_hj',
                          'lstm_0.linear_hf', 'lstm_0.linear_ho',
                          'lstm_1.linear_ii', 'lstm_1.linear_ij',
                          'lstm_1.linear_if', 'lstm_1.linear_io',
                          'lstm_1.linear_hi', 'lstm_1.linear_hj',
                          'lstm_1.linear_hf', 'lstm_1.linear_ho','linear_f']

    def forward(self, inputs, hidden, c):
        output, new_h, new_c = [], hidden, c
        if len(self.layers) == 2:
            output = self.linear_f(inputs)
        elif len(self.layers) == 3:
            lstm_output, h, c = self.lstm_0(inputs, hidden[0], c[0])
            output = self.linear_f(lstm_output)
        elif len(self.layers) == 4:
            lstm_output0, h0, c0 = self.lstm_0(inputs, hidden[0], c[0])
            lstm_output1, h1, c1 = self.lstm_1(lstm_output0, hidden[1], c[1])
            h = torch.cat((h0,h1),dim=0)
            c = torch.cat((c0,c1),dim=0)
            output = self.linear_f(lstm_output1)
        return output, h, c