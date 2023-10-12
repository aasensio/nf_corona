from math import log
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import matplotlib.pyplot as pl
from tqdm import tqdm
from rotation import rotation
import glob
from astropy.io import fits
import datetime
from modules.Siren import SirenNet
try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False
from pytorch_memlab import MemReporter


class MLP(nn.Module):
    def __init__(self, *, dim_input, dim_hidden, dim_out, depth_hidden = 3, activation='relu'):
        super().__init__()

        if (activation == 'relu'):
            activation = nn.ReLU()
        if (activation == 'elu'):
            activation = nn.ELU()

        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(dim_input, dim_hidden))
        self.layers.append(activation)

        for i in range(depth_hidden):
            self.layers.append(nn.Linear(dim_hidden, dim_hidden))
            self.layers.append(activation)
        
        self.layers.append(nn.Linear(dim_hidden, dim_out))
        self.layers.append(nn.Sigmoid())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

class CoronaTomography(object):
    def __init__(self, gpu=0):

        self.cuda = torch.cuda.is_available()
        self.gpu = gpu

        if (self.gpu < 0):
            self.device = torch.device('cpu')
        else:
            if (not self.cuda):
                print("Computing in CPU")
                self.device = torch.device('cpu')
            else:
                self.device = torch.device(f"cuda:{self.gpu}")

                if (NVIDIA_SMI):
                    nvidia_smi.nvmlInit()
                    self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.gpu)
                    print(f"Computing in {self.device} : {nvidia_smi.nvmlDeviceGetName(self.handle)}")
                else:
                    print(f"Computing in {self.device}")
        
    def define_reference_system(self):
        """
        Define the reference systems and compute all auxiliary quantities
        """
        
        torch.set_default_tensor_type(torch.FloatTensor)

        self.sigma_thomson = 6.6524587321e-23            # Thomson scattering cross-section
        self.const = 3.0 * self.sigma_thomson / 8.0
        self.R_sun = 6.957e+10                   
        self.delta_LOS = self.R_sun * (2.0 * self.FOV) / self.n_pixels  # Integration step in cm
        self.L_sun = 1.0                                 # Solar luminosity

        # Coordinate system where X is the LOS
        x = np.linspace(-self.FOV, self.FOV, self.n_pixels, dtype='float32')
        y = np.linspace(-self.FOV, self.FOV, self.n_pixels, dtype='float32')
        z = np.linspace(-self.FOV, self.FOV, self.n_pixels, dtype='float32')
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        self.X = torch.tensor(X)
        self.Y = torch.tensor(Y)
        self.Z = torch.tensor(Z)
        self.X_flat = self.X.flatten()
        self.Y_flat = self.Y.flatten()
        self.Z_flat = self.Z.flatten()
        
        # Impact parameter
        self.p = torch.sqrt(self.Y_flat**2 + self.Z_flat**2)
        self.p_POS = torch.sqrt(self.Y[self.n_pixels//2, :, :]**2 + self.Z[self.n_pixels//2, :, :]**2).flatten()

        # XYZ coordinates of the LOS reference system in batch form
        self.XYZ_LOS = torch.cat([self.X_flat[:, None], self.Y_flat[:, None], self.Z_flat[:, None]], dim=1)
        self.r_LOS = torch.sqrt(torch.sum(self.XYZ_LOS**2, dim=-1))

        # Scattering angle
        self.theta_scatt = torch.asin(self.p / self.r_LOS)

        # Scattering efficiencies
        omegaB = self.const * self.L_sun / self.p**2 * (torch.sin(self.theta_scatt)**2 - 0.5 * torch.sin(self.theta_scatt)**4)
        omegapB = 0.5 * self.const * self.L_sun / self.p**2 * torch.sin(self.theta_scatt)**4

        self.omegaB = omegaB.reshape((self.n_pixels, self.n_pixels, self.n_pixels))
        self.omegapB = omegapB.reshape((self.n_pixels, self.n_pixels, self.n_pixels))

        # Central and outer masks
        self.mask_in = (self.p_POS < self.mask_size_min).reshape((self.n_pixels, self.n_pixels))
        self.mask_out = (self.p_POS > self.mask_size_max).reshape((self.n_pixels, self.n_pixels))
        self.mask = torch.logical_or(self.mask_in, self.mask_out)
        
        self.mask_3d_in = (self.r_LOS < self.mask_size_min).reshape((self.n_pixels, self.n_pixels, self.n_pixels))
        self.mask_3d_out = (self.r_LOS > self.mask_size_max).reshape((self.n_pixels, self.n_pixels, self.n_pixels))
        self.mask_3d = torch.logical_or(self.mask_3d_in, self.mask_3d_out)

        # Now mask all observations        
        for i in range(self.n_files):
            self.obspB[i, self.mask] = 1e-10
        
        self.weight = torch.exp(-self.p_POS**2).reshape((self.n_pixels, self.n_pixels))
        self.weight[self.mask] = 0.0

        # Unit vector along the Z axis which is used for the rotation
        self.u = torch.tensor([0.0, 0.0, 1.0], device=self.device)

        # Define rotation matrix that transforms the LOS reference system into the 
        # local reference system where we evaluate the electron density
        # This is equivalent to rotating the reference system by -alpha      
        self.M = torch.zeros((self.n_files, 3, 3))
        for i in range(self.n_files):
            self.M[i, :, :] = rotation(self.u, -self.angles[i])

        # Move to the GPU (if used) all necessary tensors
        self.XYZ_LOS = self.XYZ_LOS.to(self.device)
        self.omegaB = self.omegaB.to(self.device)
        self.omegapB = self.omegapB.to(self.device)
        self.mask = self.mask.to(self.device)
        self.M = self.M.to(self.device)

    def read_obs(self, directory, reduction=1):
        """
        Read observations

        Parameters
        ----------
        directory : str
            Directory where all FITS files are found
        reduction : int, optional
            Resolution reduction, by default 1 (keep the original resolution)
        """

        # Find all files in the directory and sort by name
        files = glob.glob(f'{directory}/*.fts')
        files.sort()

        self.n_files = len(files)

        self.obspB = [None] * self.n_files        
        self.angles = torch.zeros(self.n_files, device=self.device)
        self.center = np.zeros((2, self.n_files))
        self.rsun = np.zeros((self.n_files))

        # Read all files, extracting the observing time, the size of the Sun
        for i in range(self.n_files):
            ff = fits.open(files[i])
            strdate = f"{ff[0].header['DATE_OBS']} {ff[0].header['TIME_OBS']}"            
            date = datetime.datetime.strptime(strdate, '%Y/%m/%d %H:%M:%S.%f')
            if (i == 0):
                self.angles[i] = 0.0
                date0 = date                
            else:
                deltat = date - date0
                # Calculate angle assuming rigid rotation (14.7 deg/day)           
                self.angles[i] = 14.7 * deltat.total_seconds() / (24 * 60.0 * 60.0)

            tmp = ff[0].data.astype('<f8')
            if (reduction):
                self.obspB[i] = torch.tensor(tmp[::reduction, ::reduction].astype('float32'))[None, :, :]
                self.obspB[i] += 1e-10

            self.center[0, i] = ff[0].header['XSUN_MED'] / reduction
            self.center[1, i] = ff[0].header['YSUN_MED'] / reduction
            self.rsun[i] = ff[0].header['RSUN_PIX'] / reduction

            # Compute the FOV (in solar radii) by dividing the size in pixels by twice the radius of the Sun
            self.FOV = tmp.shape[0] / ( 2.0 * ff[0].header['RSUN_PIX'])

            print(f'{strdate} -> {self.angles[i]:5.3f} deg')

        # Number of pixels in X,Y,Z
        self.n_pixels = self.obspB[0].shape[1]

        # Size of the LASCO C2 mask
        self.mask_size_min = 2.1
        self.mask_size_max = 6.3

        # Concatenate all observations in the tensor [Nobs, X, Y]
        self.obspB = torch.cat(self.obspB, dim=0)

    def synth_parallel(self, M_rot, log_electron_neural, logNe_range=[0.0, 15.0], compute_electron_LOS=False):
        """
        Synthesize an image using a function for the logarithm of Ne, typically a fully connected neural network

        Parameters
        ----------
        angles : 1D torch tensor
            Angles in degrees of the observations            
        log_electron_neural : callable
            A function or a PyTorch module that can be called using [N_points, 3] coordinates and returns the
            log-electron density at these points
        range : list, optional
            Minimum and maximum values that are applied to the output of the log-Ne function, by default [0.0, 15.0]
        compute_electron_LOS : bool, optional
            Flag to compute the electron density in the LOS reference system for plotting, by default False
        
        """
        n_angles = M_rot.shape[0]

        XYZ_rotated = torch.matmul(M_rot[:, None, :, :], self.XYZ_LOS[:, :, None]).squeeze()

        # Evaluate the log-electron density using the neural network        
        if (self.log_model):
            Ne = torch.exp((logNe_range[1] - logNe_range[0]) * log_electron_neural(XYZ_rotated / self.FOV) + logNe_range[0])            
        else:
            Ne = (logNe_range[1] - logNe_range[0]) * log_electron_neural(XYZ_rotated / self.FOV) + logNe_range[0]
        Ne_3d = Ne.reshape((n_angles, self.n_pixels, self.n_pixels, self.n_pixels))

        # Compute the integrated light along the LOS
        # imB[i] = self.delta_LOS * torch.sum(Ne_3d * self.omegaB, dim=0) / 1e-10
        imB = None
        impB = self.delta_LOS * torch.sum(Ne_3d * self.omegapB[None, :], dim=1) / 1e-10
        impB[:, self.mask] = 1e-10
        
        # Optionally compute the 3D electron density at the LOS reference frame
        if (compute_electron_LOS):            
            if (self.log_model):
                logNe = (logNe_range[1] - logNe_range[0]) * log_electron_neural(self.XYZ_LOS / self.FOV) + logNe_range[0]
                Ne = torch.exp(logNe).reshape((self.n_pixels, self.n_pixels, self.n_pixels))
            else:
                Ne = (logNe_range[1] - logNe_range[0]) * log_electron_neural(self.XYZ_LOS / self.FOV) + logNe_range[0]
                Ne = Ne.reshape((self.n_pixels, self.n_pixels, self.n_pixels))

            Ne[self.mask_3d] = 0.0
        else:
            # Else return one of the computed 3D electron densities
            Ne = Ne_3d.detach()

        return imB, impB, Ne

    def optimize(self, batch_size, n_epochs, lr=6e-3, logNe_range=[0.0, 15.0], log_model=True):
        self.logNe_range = logNe_range
        self.log_model = log_model
        self.dataset = torch.utils.data.TensorDataset(self.obspB, self.M)
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=False)

        # Neural network for producing the electron density

        self.model = SirenNet(dim_in=3, dim_hidden=128, dim_out=1, num_layers=5, final_activation = nn.Sigmoid()).to(self.device)

        # self.model = MLP(dim_input=3, dim_hidden=128, dim_out=1, depth_hidden=5, activation='relu').to(self.device)

        # Count the number of trainable parameters
        print('N. total parameters : {0}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, n_epochs, eta_min=0.1*lr)
        
        # Loss function
        self.loss_fn = nn.MSELoss().to(self.device)

        reporter = MemReporter()

        for epoch in range(n_epochs):

            for param_group in self.optimizer.param_groups:
                current_lr = param_group['lr']

            t = tqdm(self.loader)

            for batch_idx, (obspB, M_rot) in enumerate(t):

                obspB, M_rot = obspB.to(self.device), M_rot.to(self.device)
                
                self.optimizer.zero_grad()                    
                
                imB, impB, Ne = self.synth_parallel(M_rot, self.model, logNe_range=self.logNe_range)
                                
                loss = torch.mean((torch.log(obspB) - torch.log(impB))**2)
                loss.backward()
                
                # reporter.report()

                self.optimizer.step()

                # reporter.report()
                
                min_Ne = torch.min(Ne).item()
                max_Ne = torch.max(Ne).item()
                current_loss = loss.item()
                
                if (NVIDIA_SMI):
                    usage = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle)
                    memory = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
                    t.set_postfix(epoch=f'{epoch}/{n_epochs}', batch=batch_idx, lr=current_lr, loss=f'{current_loss:4.2f}', maxne=f'{max_Ne:5.1e}', minne=f'{min_Ne:5.1e}', gpu=usage.gpu, memfree=f'{memory.free/1024**2:5.1f} MB', memused=f'{memory.used/1024**2:5.1f} MB')
                else:
                    t.set_postfix(epoch=epoch, batch=batch_idx, loss=self.loss[-1].item(), lr=current_lr, maxne=torch.max(self.Ne).item(), minne=torch.min(self.Ne).item())

            self.scheduler.step()
        

        # Do a final synthesis to get the final result
        # for batch_idx, (obspB, angles) in enumerate(t):
            # imB, impB, Ne = self.synth(self.angles, self.model, logNe_range=self.logNe_range, compute_electron_LOS=True)
        self.impB = []
        with torch.no_grad():
            for batch_idx, (obspB, M_rot) in enumerate(t):

                obspB, M_rot = obspB.to(self.device), M_rot.to(self.device)
                                
                if (batch_idx == 0):
                    imB, impB, Ne = self.synth_parallel(M_rot, self.model, logNe_range=self.logNe_range, compute_electron_LOS=True)
                else:
                    imB, impB, Ne = self.synth_parallel(M_rot, self.model, logNe_range=self.logNe_range)

                self.impB.append(impB)

        self.imB = None
        self.impB = torch.cat(self.impB, dim=0)
        
        return self.imB, self.impB, Ne

    def spherical_caps(self, r):
        
        n_caps = len(r)
        Ne_cap = [None] * n_caps

        # Spherical caps at different radial distances
        theta = np.linspace(-np.pi/2.0, np.pi/2.0, self.n_pixels, dtype='float32')
        phi = np.linspace(0.0, 2*np.pi, self.n_pixels, dtype='float32')
        Theta, Phi = np.meshgrid(theta, phi, indexing='ij')
        theta = torch.tensor(Theta).flatten()
        phi = torch.tensor(Phi).flatten()
        
        # Spherical caps
        for i in range(n_caps):        
            x = r[i] * torch.sin(theta) * torch.cos(phi)
            y = r[i] * torch.sin(theta) * torch.sin(phi)
            z = r[i] * torch.cos(theta)
            xyz = torch.cat([x[:, None], y[:, None], z[:, None]], dim=-1).to(self.device)
            if (self.log_model):
                logNe = (self.logNe_range[1] - self.logNe_range[0]) * self.model(xyz / self.FOV) + self.logNe_range[0]
                Ne_cap[i] = torch.exp(logNe).reshape((self.n_pixels, self.n_pixels)).detach().cpu().numpy()
            else:
                logNe = (self.logNe_range[1] - self.logNe_range[0]) * self.model(xyz / self.FOV) + self.logNe_range[0]
                Ne_cap[i] = logNe.reshape((self.n_pixels, self.n_pixels)).detach().cpu().numpy()
        return Ne_cap

    def longitudinal_cuts(self, longitude):

        # Cuts at different longitudinal angles
        r = np.linspace(self.mask_size_min, self.mask_size_max, self.n_pixels, dtype='float32')
        theta = np.linspace(-np.pi/2.0, np.pi/2.0, self.n_pixels, dtype='float32')        
        R, Theta = np.meshgrid(r, theta, indexing='ij')
        r = torch.tensor(R).flatten()
        theta = torch.tensor(Theta).flatten()      
        
        n_long = len(longitude)
        Ne_long = [None] * n_long

        # Longitudinal cuts
        for i in range(n_long):        
            x = r * torch.sin(theta) * torch.cos(longitude[i] * np.pi / 180.0)
            y = r * torch.sin(theta) * torch.sin(longitude[i] * np.pi / 180.0)
            z = r * torch.cos(theta)
            xyz = torch.cat([x[:, None], y[:, None], z[:, None]], dim=-1).to(self.device)
            if (self.log_model):
                logNe = (self.logNe_range[1] - self.logNe_range[0]) * self.model(xyz / self.FOV) + self.logNe_range[0]
                Ne_long[i] = torch.exp(logNe).reshape((self.n_pixels, self.n_pixels)).detach().cpu().numpy()
            else:
                logNe = (self.logNe_range[1] - self.logNe_range[0]) * self.model(xyz / self.FOV) + self.logNe_range[0]
                Ne_long[i] = logNe.reshape((self.n_pixels, self.n_pixels)).detach().cpu().numpy()                
        return Ne_long

    def do_plot(self, r, longitude):

        fig, ax = pl.subplots(nrows=8, ncols=2, figsize=(8,15))
        for i in range(8):
            im = ax[i, 0].imshow(torch.log10(self.obspB[4*i, :, :]).detach().cpu().numpy(), vmin=np.log10(0.1), vmax=np.log10(500.0))
            pl.colorbar(im, ax=ax[i ,0])
            im = ax[i, 1].imshow(torch.log10(self.impB[4*i, :, :]).detach().cpu().numpy(), vmin=np.log10(0.1), vmax=np.log10(500.0))
            pl.colorbar(im, ax=ax[i ,1])
        
        Ne_cap = self.spherical_caps(r)

        fig, ax = pl.subplots(nrows=1, ncols=3, figsize=(10,6))
        for i in range(len(r)):
            im = ax[i].imshow(Ne_cap[i], extent=[0.0, 360.0, -90, 90], aspect='equal')
            pl.colorbar(im, ax=ax[i])
            ax[i].set_title(f'{r[i]}') 
        pl.ticklabel_format(useOffset=False)

        Ne_long = self.longitudinal_cuts(longitude)

        fig, ax = pl.subplots(nrows=1, ncols=3, figsize=(10,6))
        for i in range(len(longitude)):
            im = ax[i].imshow(Ne_long[i], extent=[-90.0, 90.0, self.mask_size_max, self.mask_size_min], aspect='auto')
            pl.colorbar(im, ax=ax[i])
            ax[i].set_title(f'{longitude[i]}') 
        pl.ticklabel_format(useOffset=False)

if __name__ == '__main__':
    pl.close('all')

    corona = CoronaTomography(gpu=0)

    corona.read_obs('data', reduction=8)

    corona.define_reference_system()

    imB, impB, Ne = corona.optimize(batch_size=2, n_epochs=50, lr=1e-2, logNe_range=[0.0, 15.0], log_model=True)
    # imB, impB, Ne = corona.optimize(batch_size=2, n_epochs=100, lr=1e-2, logNe_range=[0.0, 1.e5], log_model=False)

    r = torch.tensor([2.5, 4.0, 5.5])
    long = torch.tensor([0.0, 90.0, 180.0])
    corona.do_plot(r, long)