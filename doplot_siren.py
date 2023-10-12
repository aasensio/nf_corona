import numpy as np
import matplotlib.pyplot as pl
import siren
import torch

dim_in = 2
dim_hidden = 128
dim_out = 1
num_layers = 5

w0 = [1.0, 10.0, 30.0]
n = len(w0)
pl.close('all')

fig, ax = pl.subplots(nrows=n, ncols=n, figsize=(10,10), sharex=True, sharey=True)

for i in range(n):
    for j in range(n):
        tmp = siren.SirenNet(dim_in=dim_in, dim_hidden=dim_hidden, dim_out=1, num_layers=num_layers, w0_initial=[w0[i], w0[j]])

        x = np.linspace(-1, 1, 128)
        y = np.linspace(-1, 1, 128)    
        X, Y = np.meshgrid(x, y)

        xin = torch.tensor(np.vstack([X.flatten(), Y.flatten()]).T.astype('float32'))

        xin = xin.unsqueeze(0)
            
        out = tmp(xin).squeeze().reshape((128, 128)).detach().numpy()
        
        ax[i, j].imshow(out, extent=(-1, 1, -1, 1))
        ax[i, j].text(-0.8, 0.8, f'w$_x$={w0[i]}', color='white', weight='bold', size='large')
        ax[i, j].text(-0.8, 0.65, f'w$_y$={w0[j]}', color='white', weight='bold', size='large')

fig.supxlabel('x')
fig.supylabel('y')
pl.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)
pl.tight_layout()
pl.show()
pl.savefig('figs/siren_w0.pdf')