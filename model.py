from math import log
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.nn.init as init
import mlp
import siren

class INRModel(object):
    def __init__(self, hyperparameters, device):
        
        self.hyperparameters = hyperparameters
        self.device = device

        self.time = self.hyperparameters['time']
        
        if (self.time):
            self.dim_in = 4
            print("Dynamic reconstruction")
        else:
            self.dim_in = 3
            print("Static reconstruction")
                                           
        # Neural network model        
        if (self.hyperparameters['type'] == 'siren'):
            if 'w0_initial' in self.hyperparameters:
                w0_initial = self.hyperparameters['w0_initial']
            else:
                w0_initial = 30.0
            self.model = siren.SirenNet(dim_in=self.dim_in, 
                dim_hidden=self.hyperparameters['dim_hidden'], 
                dim_out=1, 
                num_layers=self.hyperparameters['n_hidden'], 
                w0_initial=w0_initial).to(self.device)
            print('SIREN model')

        if (self.hyperparameters['type'] == 'mlpFourier'):
            if (self.hyperparameters['activation'] == 'relu'):
                activation = nn.ReLU()
            if (self.hyperparameters['activation'] == 'leakyrelu'):
                activation = nn.LeakyReLU()
            if (self.hyperparameters['activation'] == 'elu'):
                activation = nn.ELU()

            sigma = self.hyperparameters['sigma']
            mapping_size = self.hyperparameters['mapping_size']
            
            self.model = mlp.MLPMultiFourier(n_input=self.dim_in, 
                                            n_output=1, 
                                            n_hidden=self.hyperparameters['dim_hidden'],                                             
                                            n_hidden_layers=self.hyperparameters['n_hidden'], 
                                            activation=activation, 
                                            mapping_size=mapping_size, 
                                            sigma=sigma).to(self.device)

            self.model.weights_init(type='kaiming')
            print('MLP Fourier model')
        
        # Count the number of trainable parameters
        print('N. total parameters : {0}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

    def load_weights(self, state_dict):
        self.model.load_state_dict(state_dict)

    def __call__(self, xyz):
        return self.model(xyz)

    def set_train(self):
        self.model.train()

    def set_eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()