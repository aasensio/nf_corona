import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import torch.nn.init as init

def kaiming_init(m):
    if isinstance(m, nn.Linear):
        # torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        # torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)

class MLPMultiFourier(nn.Module):
    """A general-purpose MLP with multi-scale Fourier encoding
        arxiv:2012.10047
    """

    def __init__(self, dim_in, dim_hidden, n_hidden, dim_out, mapping_size=256, sigma=[3.0], activation=nn.ReLU, final_activation=None, return_coordinates=False):
        super().__init__()

        self.activation = activation        
        self.n_hidden = n_hidden
        self.n_scales = len(sigma)
        self.dim_hidden = dim_hidden
        self.sigma = torch.tensor(np.array(sigma).astype('float32'))
        self.return_coordinates = return_coordinates
        
        B = torch.randn((self.n_scales, mapping_size, dim_in))
        B *= self.sigma[:, None, None]
        
        self.register_buffer("B", B)        
        
        self.net = nn.ModuleList()

        self.net.append(nn.Linear(2*mapping_size, dim_hidden))
        self.net.append(self.activation())
        
        for i in range(self.n_hidden):
            self.net.append(nn.Linear(dim_hidden, dim_hidden))
            self.net.append(self.activation())

        self.net.append(nn.Linear(self.n_scales * dim_hidden, dim_out))
        self.final_activation = False
        if (final_activation is not None):
            self.final_activation = True
            self.net.append(final_activation())

    def init(self):
        self.net.apply(kaiming_init)

    def forward(self, x):

        nb, nrays, ndim = x.shape

        x = x.view(-1, ndim)

        # x is in the range [0, 1]
        x = x.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input        

        
        # Fourier encoding
        tmp = (2. * np.pi * x) @ torch.transpose(self.B, 1, 2) #.t()
        tmp = torch.cat([torch.sin(tmp), torch.cos(tmp)], dim=-1)        

        # And now the neural network
        if (self.final_activation):
            index = -2
        else:
            index = -1

        for layer in self.net[0:index]:
            tmp = layer(tmp)
        
        tmp = torch.transpose(tmp, 0, 1).reshape(x.size(0), -1)
        tmp = self.net[index](tmp)

        if (self.final_activation):
            tmp = self.net[-1](tmp)

        x = x.view(nb, nrays, -1)
        tmp = tmp.view(nb, nrays, -1)

        if (self.return_coordinates):
            return tmp, x
        else:
            return tmp


class MLP(nn.Module):
    """A general-purpose MLP"""

    def __init__(self, dim_in, dim_hidden, n_hidden, dim_out, activation=nn.ReLU, final_activation=None, return_coordinates=False):
        super().__init__()

        self.activation = activation        
        self.n_hidden = n_hidden
        self.return_coordinates = return_coordinates

        self.net = nn.ModuleList()

        self.net.append(nn.Linear(dim_in, dim_hidden))
        self.net.append(self.activation())

        for i in range(self.n_hidden):
            self.net.append(nn.Linear(dim_hidden, dim_hidden))
            self.net.append(self.activation())

        self.net.append(nn.Linear(dim_hidden, dim_out))

        self.final_activation = False
        if (final_activation is not None):
            self.final_activation = True
            self.net.append(final_activation())

    def init(self):
        self.net.apply(kaiming_init)

    def forward(self, x):

        # x is in the range [0, 1]
        x = x.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input        
        
        tmp = x
        for layer in self.net:
            tmp = layer(tmp)

        if (self.return_coordinates):
            return tmp, x
        else:
            return tmp

if __name__ == '__main__':
    x = torch.rand((100, 100, 3))

    mlp = MLPMultiFourier(dim_in=3, dim_hidden=12, n_hidden=2, dim_out=1, mapping_size=32, sigma=[0.1], activation=nn.ReLU, final_activation=nn.Sigmoid, return_coordinates=True)
    tmp, x = mlp(x)