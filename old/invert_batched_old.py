from math import log
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.nn.init as init
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

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

class MLP(nn.Module):
    def __init__(self, *, dim_input, dim_hidden, dim_out, depth_hidden = 3, activation=None):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(dim_input, dim_hidden))
        self.layers.append(activation)

        for i in range(depth_hidden):
            self.layers.append(nn.Linear(dim_hidden, dim_hidden))
            self.layers.append(activation)
        
        self.layers.append(nn.Linear(dim_hidden, dim_out))
        # self.layers.append(nn.Sigmoid())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def weights_init(self):
        for module in self.modules():
            kaiming_init(module)


class Dataset(torch.utils.data.Dataset):
    """
    Dataset class that will provide data during training. Modify it accordingly
    for your dataset. This one shows how to do augmenting during training for a 
    very simple training set    
    """
    def __init__(self, directory, reduction):
        """
        Main dataset

        Parameters
        ----------
        directory : str
            Directory where all FITS files are found
        reduction : int
            Resolution reduction, by default 1 (keep the original resolution)
        """
        super(Dataset, self).__init__()

        # Find all files in the directory and sort by name
        files = glob.glob(f'{directory}/*.fts')
        files.sort()

        self.n_files = len(files)
        
        self.obspB = [None] * self.n_files        
        self.angles = torch.zeros(self.n_files)
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
                # self.obspB[i] += 1e-10

            self.center[0, i] = ff[0].header['XSUN_MED'] / reduction
            self.center[1, i] = ff[0].header['YSUN_MED'] / reduction
            self.rsun[i] = ff[0].header['RSUN_PIX'] / reduction

            # Compute the FOV (in solar radii) by dividing the size in pixels by twice the radius of the Sun
            self.FOV = tmp.shape[0] / ( 2.0 * ff[0].header['RSUN_PIX'])

            print(f'{strdate} -> {self.angles[i]:5.3f} deg')

        # Number of pixels in X,Y,Z
        self.n_pixels = self.obspB[0].shape[1]

        # Unit vector along the Z axis which is used for the rotation
        self.u = torch.tensor([0.0, 0.0, 1.0])

        # Define rotation matrix that transforms the LOS reference system into the 
        # local reference system where we evaluate the electron density
        # This is equivalent to rotating the reference system by -alpha
        self.M = torch.zeros((self.n_files, 3, 3))
        for i in range(self.n_files):
            self.M[i, :, :] = rotation(self.u, -self.angles[i])           

        # Number of training images
        self.n_training = self.n_files

        # Concatenate all observations in the tensor [Nobs, X, Y]
        self.obspB = torch.cat(self.obspB, dim=0)
        
    def __getitem__(self, index):
        """
        Return each item in the training set

        Parameters
        ----------
        """
        
        return self.obspB[index, :], self.M[index, :]
    
    def __len__(self):
        return self.n_training

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

        torch.set_default_tensor_type(torch.FloatTensor)
        
    def observations_and_reference_system(self, directory, reduction=1, n_pixels_integration=64):
        """
        Read the observations, define the reference systems and compute all auxiliary quantities

        Parameters
        ----------
        directory : str
            Directory where all FITS files are found
        reduction : int, optional
            Resolution reduction, by default 1 (keep the original resolution)
        """
        
        # Number of pixels to use for LOS integration
        self.n_pixels_integration = n_pixels_integration

        # Instantiate dataset
        self.dataset = Dataset(directory, reduction)

        # Define constants and reference system

        # Thomson scattering cross-section
        # https://hesperia.gsfc.nasa.gov/ssw/packages/nrl/idl/nrlgen/analysis/eltheory.pro
        self.sigma_thomson = 6.6524587321e-25
        self.mean_solar_brightness = 2.009e10

        # Solar radius
        self.R_sun = 6.957e+10

        # Step for the LOS integration
        self.delta_LOS = self.R_sun * (2.0 * self.dataset.FOV) / self.n_pixels_integration
        self.L_sun = 3.845e33

        # Size of the LASCO C2 mask
        self.mask_size_min = 2.1
        self.mask_size_max = 6.3

        # Coordinate system where X is the LOS
        x = np.linspace(-self.dataset.FOV, self.dataset.FOV, self.n_pixels_integration, dtype='float32')
        y = np.linspace(-self.dataset.FOV, self.dataset.FOV, self.dataset.n_pixels, dtype='float32')
        z = np.linspace(-self.dataset.FOV, self.dataset.FOV, self.dataset.n_pixels, dtype='float32')
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        self.X = torch.tensor(X)
        self.Y = torch.tensor(Y)
        self.Z = torch.tensor(Z)        
        
        # Impact parameter
        self.p = torch.sqrt(self.Y**2 + self.Z**2)
        self.p_POS = torch.sqrt(self.Y[self.dataset.n_pixels//2, :, :]**2 + self.Z[self.dataset.n_pixels//2, :, :]**2)

        # XYZ coordinates of the LOS reference system in 3D form
        self.XYZ_LOS = torch.cat([self.X[..., None], self.Y[..., None], self.Z[..., None]], dim=-1)
        self.r_LOS = torch.sqrt(torch.sum(self.XYZ_LOS**2, dim=-1))

        # Sine of the scattering angle for each point in the 3D volume
        self.sin_theta_scatt = self.p / self.r_LOS        
        
        # Scattering efficiencies
        self.omegaB = 3.0 * self.sigma_thomson / 8.0 * self.L_sun / (self.p * self.R_sun)**2 * (self.sin_theta_scatt**2 - 0.5 * self.sin_theta_scatt**4)
        self.omegapB = 3.0 * self.sigma_thomson / 16.0 * self.L_sun / (self.p * self.R_sun)**2 * self.sin_theta_scatt**4
    
        # Central and outer masks in the 2D plane-of-the-sky
        self.mask_in = (self.p_POS < self.mask_size_min)
        self.mask_out = (self.p_POS > self.mask_size_max)
        self.mask = torch.logical_or(self.mask_in, self.mask_out)
        
        # 3D mask produced by the occulters
        self.mask_3d_in = (self.r_LOS < self.mask_size_min)
        self.mask_3d_out = (self.r_LOS > self.mask_size_max)
        self.mask_3d = torch.logical_or(self.mask_3d_in, self.mask_3d_out)

        tmp = self.dataset.obspB[:, ~self.mask]
        print(f'Observations inside mask : min={torch.min(tmp).item()} - max={torch.max(tmp).item()}')
                
        self.weight = torch.exp((5-self.p_POS) / 1.25).reshape((self.dataset.n_pixels, self.dataset.n_pixels))
        self.weight = torch.ones((self.dataset.n_pixels, self.dataset.n_pixels))
        self.weights = (1.0 / self.p_POS).reshape((self.dataset.n_pixels, self.dataset.n_pixels))
        # self.weight[self.mask] = 0.0

    def neural_logNe(self, xyz):
        return (self.logNe_range[1] - self.logNe_range[0]) * self.model(xyz) + self.logNe_range[0]        

    def synth(self, XYZ_LOS, M_rot, omegapB, compute_electron_LOS=False):
        """
        Synthesize an image using a function for the logarithm of Ne, typically a fully connected neural network

        Parameters
        ----------
        XYZ_LOS : Torch tensor
            Coordinates of the LOS reference system where to compute the image
        M_rot : Torch tensor
            Rotation matrices for each observing angle. It is of size [n_angles,3,3]
        log_electron_neural : callable
            A function or a PyTorch module that can be called using [N_points, 3] coordinates and returns the
            log-electron density at these points
        omegapB : Torch tensor
            Efficiency of the scattering for the pB images
        range : list, optional
            Minimum and maximum values that are applied to the output of the log-Ne function, by default [0.0, 15.0]
        compute_electron_LOS : bool, optional
            Flag to compute the electron density in the LOS reference system for plotting, by default False
        
        """
        n_angles = M_rot.shape[0]
        n_pixels_X, n_pixels_Y, n_pixels_Z, _ = XYZ_LOS.shape

        # Flatten the XYZ_LOS reference system for the current patch
        XYZ_LOS_flat = XYZ_LOS.reshape(-1, 3)

        # Rotate it at the desired angle
        XYZ_rotated = torch.matmul(M_rot[:, None, :, :], XYZ_LOS_flat[:, :, None]).squeeze()

        # Evaluate the log-electron density using the neural network        
        logNe = self.neural_logNe(XYZ_rotated / self.dataset.FOV)
        Ne_3d = torch.exp(logNe).reshape((n_angles, n_pixels_X, n_pixels_Y, n_pixels_Z))
                                
        # Compute the integrated light along the LOS        
        imB = None        
        impB = self.delta_LOS * torch.sum(Ne_3d * omegapB[None, :], dim=1) / (1e-10 * self.mean_solar_brightness)        

        Ne = Ne_3d.detach()

        return imB, impB, Ne

    def optimize(self, batch_size, n_epochs, patch_size=None, lr=6e-3, logNe_range=[0.0, 15.0]):

        # If no patch size is provided, then use the full image size
        if (patch_size is None):
            patch_size = self.dataset.n_pixels

        # Min and max of the log(N_e)
        self.logNe_range = logNe_range        

        # Dataloader that will produce images at different rotation angles
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        # Neural network model
        # self.model = SirenNet(dim_in=3, dim_hidden=128, dim_out=1, num_layers=5, final_activation = nn.Sigmoid()).to(self.device)
        self.model = MLP(dim_input=3, dim_hidden=128, dim_out=1, depth_hidden=5, activation=nn.LeakyReLU()).to(self.device)
        self.model.weights_init()

        # Count the number of trainable parameters
        print('N. total parameters : {0}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, n_epochs, eta_min=0.5*lr)
                
        # Patchify the tensors if needed by unfolding the tensor with the provided patch size            
        XYZ_LOS_patch = self.XYZ_LOS.unfold(2, patch_size, patch_size).unfold(1, patch_size, patch_size)
        XYZ_LOS_patch = XYZ_LOS_patch.permute(0,4,5,3,1,2)
        ni = XYZ_LOS_patch.shape[4]
        nj = XYZ_LOS_patch.shape[5]

        omegapB_patch = self.omegapB.unfold(2, patch_size, patch_size).unfold(1, patch_size, patch_size)
        mask_patch = (~self.mask).unfold(1, patch_size, patch_size).unfold(0, patch_size, patch_size).float()
        weight_patch = self.weight.unfold(1, patch_size, patch_size).unfold(0, patch_size, patch_size)        
        
        for epoch in range(n_epochs):

            self.impB = []

            for param_group in self.optimizer.param_groups:
                current_lr = param_group['lr']

            t = tqdm(self.loader)

            for batch_idx, (obspB, M_rot) in enumerate(t):

                # Patchify the observations using unfold with the provided patch size                
                obspB_patch = obspB.unfold(2, patch_size, patch_size).unfold(1, patch_size, patch_size)

                # Move the rotation matrix to GPU because it is common for all patches
                M_rot = M_rot.to(self.device)
                
                # Zero the gradient computation
                self.optimizer.zero_grad()

                impB = [None] * ni

                # Accumulate gradients for all patches, passing to GPU only those that we care
                for i in range(ni):
                    impBj = [None] * nj
                    for j in range(nj):
                        
                        obspB = obspB_patch[:, i, j, :, :].to(self.device)
                        XYZ_LOS = XYZ_LOS_patch[:, :, :, :, i, j].to(self.device)
                        omegapB = omegapB_patch[:, i, j, :, :].to(self.device)                        
                        mask = mask_patch[i, j, :, :].to(self.device)
                        weight = weight_patch[i, j, :, :].to(self.device)

                        imB, impB_tmp, Ne = self.synth(XYZ_LOS, M_rot, omegapB)
                                                                        
                        loss = torch.mean( mask[None, :, :] * (torch.log(obspB + 1e-10) - torch.log(impB_tmp + 1e-10))**2)
                        # loss = torch.mean( mask[None, :, :] * torch.abs(torch.log(obspB + 1e-10) - torch.log(impB_tmp + 1e-10)))
                        
                        loss.backward()

                        impBj[j] = impB_tmp[..., None]

                    impB[i] = torch.cat(impBj, dim=-1)[..., None]
                                                
                impB = torch.cat(impB, dim=-1).transpose(1, 2).transpose(3, 4).reshape((-1, patch_size*patch_size, ni*nj))
                            
                impB = F.fold(impB, (self.dataset.n_pixels, self.dataset.n_pixels), patch_size, stride=patch_size).squeeze(1)

                self.impB.append(impB)
                
                self.optimizer.step()                
                
                min_Ne = torch.min(Ne).item()
                max_Ne = torch.max(Ne).item()
                current_loss = loss.item()
                
                if (NVIDIA_SMI):
                    usage = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle)
                    memory = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
                    t.set_postfix(epoch=f'{epoch}/{n_epochs}', lr=f'{current_lr:6.4f}', loss=f'{current_loss:5.3f}', maxne=f'{max_Ne:5.1e}', minne=f'{min_Ne:5.1e}', gpu=usage.gpu, memfree=f'{memory.free/1024**2:5.1f} MB', memused=f'{memory.used/1024**2:5.1f} MB')
                else:
                    t.set_postfix(epoch=epoch, loss=f'{current_loss:5.3f}', lr=f'{current_lr:6.4f}', maxne=torch.max(self.Ne).item(), minne=torch.min(self.Ne).item())

            self.scheduler.step()


        self.imB = None
        self.impB = torch.cat(self.impB, dim=0).detach().cpu().numpy()
        
        return self.imB, self.impB

    def spherical_caps(self, r, n_pixels):
        
        n_caps = len(r)
        Ne_cap = [None] * n_caps

        # Spherical caps at different radial distances
        theta = np.linspace(-np.pi/2.0, np.pi/2.0, n_pixels, dtype='float32')
        phi = np.linspace(0.0, 2*np.pi, n_pixels, dtype='float32')
        Theta, Phi = np.meshgrid(theta, phi, indexing='ij')
        theta = torch.tensor(Theta).flatten()
        phi = torch.tensor(Phi).flatten()
        
        # Spherical caps
        for i in range(n_caps):        
            x = r[i] * torch.sin(theta) * torch.cos(phi)
            y = r[i] * torch.sin(theta) * torch.sin(phi)
            z = r[i] * torch.cos(theta)
            xyz = torch.cat([x[:, None], y[:, None], z[:, None]], dim=-1).to(self.device)
            
            logNe = self.neural_logNe(xyz / self.dataset.FOV)
            Ne_cap[i] = torch.exp(logNe).reshape((n_pixels, n_pixels)).detach().cpu().numpy()
            
        return Ne_cap

    def longitudinal_cuts(self, longitude, n_pixels):

        # Cuts at different longitudinal angles
        r = np.linspace(self.mask_size_min, self.mask_size_max, n_pixels, dtype='float32')
        theta = np.linspace(-np.pi/2.0, np.pi/2.0, n_pixels, dtype='float32')        
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
            
            logNe = self.neural_logNe(xyz / self.dataset.FOV)
            Ne_long[i] = torch.exp(logNe).reshape((n_pixels, n_pixels)).detach().cpu().numpy()
            
        return Ne_long

    def do_plot(self, r, longitude, n_pixels):
        
        mask = (~self.mask).cpu().numpy()
        obspB = self.dataset.obspB.cpu().numpy()

        fig, ax = pl.subplots(nrows=8, ncols=3, figsize=(8,15))
        for i in range(8):
            im = ax[i, 0].imshow(np.log10(mask * obspB[4*i, :, :]), vmin=np.log10(0.01), vmax=np.log10(60.0))
            pl.colorbar(im, ax=ax[i ,0])
            im = ax[i, 1].imshow(np.log10(mask * self.impB[4*i, :, :]), vmin=np.log10(0.01), vmax=np.log10(60.0))
            pl.colorbar(im, ax=ax[i ,1])
            im = ax[i, 2].imshow(mask * obspB[4*i, :, :] / self.impB[4*i, :, :], vmin=0.2, vmax=5.0)
            pl.colorbar(im, ax=ax[i ,2])

        
        Ne_cap = self.spherical_caps(r, n_pixels)

        fig, ax = pl.subplots(nrows=1, ncols=3, figsize=(10,6))
        for i in range(len(r)):
            im = ax[i].imshow(Ne_cap[i], extent=[0.0, 360.0, -90, 90], aspect='equal')
            pl.colorbar(im, ax=ax[i])
            ax[i].set_title(f'{r[i]}') 
        pl.ticklabel_format(useOffset=False)

        Ne_long = self.longitudinal_cuts(longitude, n_pixels)

        fig, ax = pl.subplots(nrows=1, ncols=3, figsize=(10,6))
        for i in range(len(longitude)):
            im = ax[i].imshow(Ne_long[i], extent=[-90.0, 90.0, self.mask_size_max, self.mask_size_min], aspect='auto')
            pl.colorbar(im, ax=ax[i])
            ax[i].set_title(f'{longitude[i]}') 
        pl.ticklabel_format(useOffset=False)

if __name__ == '__main__':
    pl.close('all')

    corona = CoronaTomography(gpu=0)
    
    corona.observations_and_reference_system('data', reduction=8)

    imB, impB = corona.optimize(batch_size=4, n_epochs=50, patch_size=32, lr=3e-4, logNe_range=[-5.0, 18.0])
    # imB, impB = corona.optimize(batch_size=4, n_epochs=10, patch_size=32, lr=3e-4, logNe_range=[0.0, 18.0])

    r = torch.tensor([2.5, 4.0, 5.5])
    long = torch.tensor([0.0, 90.0, 180.0])
    corona.do_plot(r, long, n_pixels=64)