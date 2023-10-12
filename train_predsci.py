import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as pl
import h5py
from tqdm import tqdm
import model
try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False

class Dataset(torch.utils.data.Dataset):
    """
    Dataset class that will provide data during training. Modify it accordingly
    for your dataset. This one shows how to do augmenting during training for a 
    very simple training set    
    """
    def __init__(self, x, rho):
        """
        Very simple training set made of 200 Gaussians of width between 0.5 and 1.5
        We later augment this with a velocity and amplitude.
        
        Args:
            n_training (int): number of training examples including augmenting
        """
        super(Dataset, self).__init__()
        
        self.x = x
        self.rho = rho

        self.n_training = len(self.rho)        
        
    def __getitem__(self, index):

        xout = self.x[index, :]
        rho = self.rho[index]

        return xout.astype('float32'), rho.astype('float32')

    def __len__(self):
        return self.n_training

gpu = 0
device = torch.device(f"cuda:{gpu}")

if (NVIDIA_SMI):
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu)
    print("Computing in {0} : {1}".format(device, nvidia_smi.nvmlDeviceGetName(handle)))

# f = h5py.File('/net/drogon/scratch1/aasensio/pred_sci/cr2204-medium__hmi_mast_mas_std_0101.h5', 'r')
f = h5py.File('/net/drogon/scratch1/aasensio/pred_sci/cr2205-medium__hmi_mast_mas_std_0101.h5', 'r')
pm = f['pm'][:]
tm = f['tm'][:]
rm = f['rm'][:]

r_min = 2.0
r_max = 10.0
smooth = 0.15
train = False

ind = np.where((rm > r_min) & (rm < r_max))[0]
rm = rm[ind]

rho = np.log(f['rho'][:][:, :, ind])

npm, ntm, nrm = rho.shape

P, T, R = np.meshgrid(pm, tm, rm, indexing='ij')

X = R * np.sin(T) * np.cos(P)
Y = R * np.sin(T) * np.sin(P)
Z = R * np.cos(T)

hyperparameters = {
        'time': False,
        'type': 'siren',
        'dim_hidden': 256,
        'n_hidden' : 8,
        'activation': 'relu',        
        'sigma': [0.05, 1.0],
        'mapping_size': 64,        
        'n_epochs': 200,
        'lr': 5e-3,
        'r_max': r_max
    }

xin = np.concatenate([X.flatten()[:, None], Y.flatten()[:, None], Z.flatten()[:, None]], axis=-1)

model = model.INRModel(hyperparameters, device=device)

if (hyperparameters['type'] == 'siren'):
    xin /= r_max
if (hyperparameters['type'] == 'mlpFourier'):
    xin /= r_max
    xin = (xin + 1.0) / 2.0

if (train):
    model.set_train()

    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, hyperparameters['n_epochs'], eta_min=0.1*hyperparameters['lr'])
    loss_L2 = nn.MSELoss().to(device)

    dataset = Dataset(xin, rho.flatten())

    loader = torch.utils.data.DataLoader(dataset, batch_size=16384, shuffle=True, num_workers=4)

    best_loss = 1e10

    for epoch in range(hyperparameters['n_epochs']):
        
        loss_avg = 0.0

        t = tqdm(loader)

        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']

        for batch_idx, (x, r) in enumerate(t):

            x = x.to(device)
            r = r.to(device)

            optimizer.zero_grad()
            
            out = model(x[None, :, :]).squeeze()
            
            loss = loss_L2(out, r)

            loss.backward()

            optimizer.step()

            if (batch_idx == 0):
                loss_avg = loss.item()
            else:
                loss_avg = smooth * loss.item() + (1.0 - smooth) * loss_avg
            
            tmp = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            t.set_postfix(epoch=f"{epoch}/{hyperparameters['n_epochs']}", loss=loss_avg, gpu=tmp.gpu, mem=tmp.memory, lr=current_lr)

        scheduler.step()
        
        if (loss_avg < best_loss):

            checkpoint = {                  
                'hyperparameters'  : hyperparameters,
                'state_dict': model.state_dict(),                    
                'best_loss': best_loss,
            }
            
            print("Saving model...")
            torch.save(checkpoint, f'pred_sci_cr2205.pth')

            best_loss = loss_avg
    
if (not train):
    checkpoint = torch.load('pred_sci_cr2205.pth')    
    model.load_weights(checkpoint['state_dict'])

xin = torch.tensor(xin.astype('float32')).to(device)
rho_3d = torch.tensor(rho.astype('float32')).to(device)
with torch.no_grad():    
    out_3d = model(xin[None, :, :]).squeeze().reshape((npm, ntm, nrm))

fig, ax = pl.subplots(nrows=3, ncols=2, figsize=(10,10))
ax[0,0].imshow(rho_3d[:,:,5].cpu().numpy())
ax[0,1].imshow(out_3d[:,:,5].cpu().numpy())

ax[1,0].imshow(rho_3d[:,:,10].cpu().numpy())
ax[1,1].imshow(out_3d[:,:,10].cpu().numpy())

ax[2,0].imshow(rho_3d[:,:,20].cpu().numpy())
ax[2,1].imshow(out_3d[:,:,20].cpu().numpy())
pl.savefig('im1.png')

fig, ax = pl.subplots(nrows=3, ncols=2, figsize=(10,10))
ax[0,0].imshow(rho_3d[:,20,:].cpu().numpy())
ax[0,1].imshow(out_3d[:,20,:].cpu().numpy())

ax[1,0].imshow(rho_3d[:,50,:].cpu().numpy())
ax[1,1].imshow(out_3d[:,50,:].cpu().numpy())

ax[2,0].imshow(rho_3d[:,90,:].cpu().numpy())
ax[2,1].imshow(out_3d[:,90,:].cpu().numpy())
pl.savefig('im2.png')

fig, ax = pl.subplots(nrows=3, ncols=2, figsize=(10,10))
ax[0,0].imshow(rho_3d[20,:,:].cpu().numpy())
ax[0,1].imshow(out_3d[20,:,:].cpu().numpy())

ax[1,0].imshow(rho_3d[50,:,:].cpu().numpy())
ax[1,1].imshow(out_3d[50,:,:].cpu().numpy())

ax[2,0].imshow(rho_3d[90,:,:].cpu().numpy())
ax[2,1].imshow(out_3d[90,:,:].cpu().numpy())
pl.savefig('im3.png')