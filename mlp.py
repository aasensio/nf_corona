import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from rotation import rotation

def xavier_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

class MLPConditioning(nn.Module):
    def __init__(self, n_input, n_output, n_hidden=1, n_hidden_layers=1, activation=nn.Tanh(), bias=True):
        """
        Simple fully connected network
        """
        super(MLPConditioning, self).__init__()

        self.layers = nn.ModuleList([])

        self.activation = activation

        self.layers.append(nn.Linear(n_input, n_hidden, bias=bias))
        self.layers.append(self.activation)

        for i in range(n_hidden_layers):
            self.layers.append(nn.Linear(n_hidden, n_hidden, bias=bias))
            self.layers.append(self.activation)

        self.gamma = nn.Linear(n_hidden, n_output)
        self.beta = nn.Linear(n_hidden, n_output)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        gamma = self.gamma(x)
        beta = self.beta(x)

        return gamma, beta

    def weights_init(self, type='xavier'):
        for module in self.modules():
            if (type == 'xavier'):
                xavier_init(module)
            if (type == 'kaiming'):
                kaiming_init(module)


class MLP(nn.Module):
    def __init__(self, n_input, n_output, n_hidden=1, n_hidden_layers=1, activation=nn.Tanh(), bias=True, final_activation=nn.Identity()):
        """
        Simple fully connected network
        """
        super(MLP, self).__init__()

        self.layers = nn.ModuleList([])

        self.activation = activation
        self.final_activation = final_activation
        
        self.layers.append(nn.Linear(n_input, n_hidden, bias=bias))
        
        for i in range(n_hidden_layers):
            self.layers.append(nn.Linear(n_hidden, n_hidden, bias=bias))
        
        self.layers.append(nn.Linear(n_hidden, n_output))
        
    def forward(self, x, gamma=None, beta=None):

        x = x.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input        

        for layer in self.layers[0:-1]:
            if (gamma is not None):
                x = layer(x) * gamma + beta
            else:
                x = layer(x)
            x = self.activation(x)

        x = self.layers[-1](x)
        x = self.final_activation(x)
        
        return x

    def weights_init(self, type='xavier'):
        for module in self.modules():
            if (type == 'xavier'):
                xavier_init(module)
            if (type == 'kaiming'):
                kaiming_init(module)

class MLPMultiFourier(nn.Module):
    def __init__(self, n_input, n_output, n_hidden=1, n_hidden_layers=1, mapping_size=128, sigma=[3.0], activation=nn.Tanh(), bias=True, final_activation=nn.Identity()):
        """
        Simple fully connected network with random Fourier embedding with several frequencies
        arxiv:2012.10047

        Parameters
        ----------
        n_input : int
            Dimensionality of the input
        n_output : int
            Dimensionality of the output
        n_hidden : int, optional
            Number of hidden neurons in each layer, by default 1
        n_hidden_layers : int, optional
            Number of hidden layers, by default 1
        mapping_size : int, optional
            Dimensionality of the initial Fourier embedding, by default 128
        sigma : numpy array or list, optional
            Array of size (n_scales,n_input) that define the maximum Fourier scale per input dimension.
            If it is an array of size (n_scales) or a list of the same size, all input dimensions are
            treated equally. This can be used to include larger/smaller frequencies in certain dimensions.
        activation : PyTorch activation function, optional
            Activation function for all layers, by default nn.Tanh()
        bias : bool, optional
            Whether to include biases in the MLP, by default True
        final_activation : PyTorch activation function, optional
            Final activation function, by default nn.Identity()
        """
        
        super(MLPMultiFourier, self).__init__()

        sigma = np.array(sigma)
        
        if (sigma.ndim == 2):
            self.n_scales = sigma.shape[0]
            assert n_input == sigma.shape[1], "Dimensions of sigma [n_scales, n_input] is not compatible with n_input"

        if (sigma.ndim == 1):
            self.n_scales = len(sigma)
            sigma = sigma[:, None]

        self.activation = activation
        self.final_activation = final_activation
                
        # Fourier matrix        
        self.sigma = torch.tensor(sigma.astype('float32'))
        B = torch.randn((self.n_scales, mapping_size, n_input))
        B *= self.sigma[:, None, :]
        
        self.register_buffer("B", B)

        # Layers
        self.layers = nn.ModuleList([])        
        
        self.layers.append(nn.Linear(2*mapping_size, n_hidden, bias=bias))
        
        for i in range(n_hidden_layers):
            self.layers.append(nn.Linear(n_hidden, n_hidden, bias=bias))
        
        self.layers.append(nn.Linear(self.n_scales * n_hidden, n_output))
        
    def forward(self, x, gamma=None, beta=None):
        # Fourier encoding        
        
        x = x.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input                

        n = x.shape[-1]
        tmp = (2. * np.pi * x.view(-1, n)) @ torch.transpose(self.B, 1, 2)
        tmp = torch.cat([torch.sin(tmp), torch.cos(tmp)], dim=-1) 
        
        for layer in self.layers[0:-1]:
            if (gamma is not None):
                n = gamma.size(-1)
                tmp = layer(tmp) * gamma.view(-1, n)[None, :, :] + beta.view(-1, n)[None, :, :]
            else:
                tmp = layer(tmp)

            tmp = self.activation(tmp)
                
        tmp = torch.transpose(tmp, 0, 1).contiguous().view(x.size(0), x.size(1), -1)
        
        tmp = self.layers[-1](tmp)
        tmp = self.final_activation(tmp)

        return tmp

    def weights_init(self, type='xavier'):
        for module in self.modules():
            if (type == 'xavier'):
                xavier_init(module)
            if (type == 'kaiming'):
                kaiming_init(module)

if (__name__ == '__main__'):
    import matplotlib.pyplot as pl
    dim_in = 3
    dim_hidden = 128
    dim_out = 1
    num_layers = 15
        
    tmp = MLPMultiFourier(n_input=dim_in, n_output=dim_out, n_hidden=dim_hidden, n_hidden_layers=num_layers, sigma=[0.05], activation=nn.ReLU()) #, 0.1, 1.0])
    tmp.weights_init(type='kaiming')

    print(f'N. parameters : {sum(x.numel() for x in tmp.parameters())}')

    with torch.no_grad():
        x = np.linspace(-1, 1, 32)
        y = np.linspace(-1, 1, 32)
        z = np.linspace(-1, 1, 32)
        X, Y, Z = np.meshgrid(x, y, z)

        xin = torch.tensor(np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T.astype('float32'))

        xin = xin.unsqueeze(0)
        
        out = tmp(xin).squeeze().reshape((32, 32, 32)).detach().numpy()

        fig, ax = pl.subplots(nrows=1, ncols=2)
        ax[0].imshow(out[:, :, 16])
        
        xin = torch.tensor(np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T.astype('float32'))
        
        xin = xin.unsqueeze(0)

        u = torch.tensor([0.0, 0.0, 1.0])
        angle = 180.0 * torch.ones(1)
        M = rotation(u, angle[0])

        xin = (M[None,None,:,:] @ xin[:,:,:,None]).squeeze()[None, :, :]
        
        out = tmp(xin).squeeze().reshape((32, 32, 32)).detach().numpy()
        ax[1].imshow(out[:, :, 16])

    dim_in = 2
    dim_hidden = 128
    dim_out = 1
    num_layers = 5

    sigma = np.ones((1, dim_in))
    sigma[0, 0] = 0.1    
    sigma[0, 1] = 0.01    
    tmp = MLPMultiFourierSeparate(n_input=dim_in, n_output=dim_out, n_hidden=dim_hidden, n_hidden_layers=num_layers, sigma=sigma, activation=nn.ReLU()) #, 0.1, 1.0])
    tmp.weights_init(type='kaiming')

    x = np.linspace(-1, 1, 32)
    y = np.linspace(-1, 1, 32)    
    X, Y = np.meshgrid(x, y)

    xin = torch.tensor(np.vstack([X.flatten(), Y.flatten()]).T.astype('float32'))

    xin = xin.unsqueeze(0)
        
    out = tmp(xin).squeeze().reshape((32, 32)).detach().numpy()

    pl.imshow(out)