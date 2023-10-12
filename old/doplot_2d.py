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
import mlp_new as mlp
# import napari
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
        
        return self.obspB[index, :], self.M[index, :], self.angles[index]
    
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
        
    def observations_and_reference_system(self, directory, checkpoint, reduction=1, n_pixels_integration=64):
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
        self.sigma_thomson = 6.65e-25
        self.constant = self.sigma_thomson * np.pi / 2.0
        self.u = 0.63
        self.limb_darkening = 1.0 - self.u/3.0
        self.constant = self.constant / self.limb_darkening

        # Solar radius
        self.R_sun = 6.957e+10

        # Step for the LOS integration
        self.delta_LOS = self.R_sun * (2.0 * self.dataset.FOV) / self.n_pixels_integration
        
        # Size of the LASCO C2 mask
        self.mask_size_min = 2.1
        self.mask_size_max = 6.3

        # Coordinate system where X is the LOS
        x = np.linspace(-self.dataset.FOV, self.dataset.FOV, self.dataset.n_pixels, dtype='float32')
        y = np.linspace(-self.dataset.FOV, self.dataset.FOV, self.dataset.n_pixels, dtype='float32')
        z = 0.7 * np.ones(1, dtype='float32')
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        self.X = torch.tensor(X)
        self.Y = torch.tensor(Y)
        self.Z = torch.tensor(Z)        
        
        # XYZ coordinates of the LOS reference system in 3D form
        self.XYZ_LOS = torch.cat([self.X[..., None], self.Y[..., None], self.Z[..., None]], dim=-1)
        
# Load checkpoint and rebuild the model
        self.checkpoint = '{0}'.format(checkpoint)
        print("=> loading checkpoint '{}'".format(self.checkpoint))
        chk = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)
        print("=> loaded checkpoint '{}'".format(self.checkpoint))

        self.hyperparameters = chk['hyperparameters']

        self.time = self.hyperparameters['time']
        if (self.time):
            self.dim_in = 5
            print("Dynamic reconstruction")
        else:
            self.dim_in = 3
            print("Static reconstruction")
                       
        # Neural network model        
        if (self.hyperparameters['type'] == 'siren'):
            self.model = SirenNet(dim_in=self.dim_in, dim_hidden=self.hyperparameters['dim_hidden'], dim_out=1, num_layers=self.hyperparameters['n_hidden'], final_activation = nn.Sigmoid()).to(self.device)

        if (self.hyperparameters['type'] == 'mlp'):
            if (self.hyperparameters['activation'] == 'relu'):
                activation = nn.ReLU()
            if (self.hyperparameters['activation'] == 'leakyrelu'):
                activation = nn.LeakyReLU()
            if (self.hyperparameters['activation'] == 'elu'):
                activation = nn.ELU()
            
            self.model = MLP(dim_input=self.dim_in, dim_hidden=self.hyperparameters['dim_hidden'], dim_out=1, depth_hidden=self.hyperparameters['n_hidden'], activation=activation).to(self.device)
            self.model.init()

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

        # Now we load the weights and the network should be on the same state as before
        self.model.load_state_dict(chk['state_dict'])

        # Count the number of trainable parameters
        print('N. total parameters : {0}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
    
        # Evaluation mode
        self.model.eval()                        
                                
        # Dataloader that will produce images at different rotation angles
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=False)
                                                
        t = tqdm(self.loader)

        with torch.no_grad():

            tmp = []
            angle = []
            coord = []

            for batch_idx, (obspB, M_rot, angles) in enumerate(t):
                
                # Move the rotation matrix to GPU because it is common for all patches
                M_rot = M_rot.to(self.device)
                angles = angles.to(self.device)

                n_angles = M_rot.shape[0]
                n_pixels_X, n_pixels_Y, n_pixels_Z, _ = self.XYZ_LOS.shape
                n_points = n_pixels_X * n_pixels_Y * n_pixels_Z

                # Flatten the XYZ_LOS reference system for the current patch
                XYZ_LOS_flat = self.XYZ_LOS.reshape(-1, 3).to(self.device)

                # Rotate it at the desired angle
                XYZ_rotated = (M_rot[:, None, :, :] @ XYZ_LOS_flat[None, :, :, None]).squeeze()

                # Evaluate the log-electron density using the neural network                
                if (XYZ_rotated.ndim == 2):
                    XYZ_rotated = XYZ_rotated[None, :, :]   

                if (self.time):
                    cosa = torch.cos(angles * np.pi / 180.0)[:, None, None].repeat(1, n_points, 1)
                    sina = torch.sin(angles * np.pi / 180.0)[:, None, None].repeat(1, n_points, 1)
                    inp = torch.cat([cosa, sina, XYZ_rotated / (self.dataset.FOV * np.sqrt(3))], dim=-1)
                else:
                    inp = XYZ_rotated / (self.dataset.FOV * np.sqrt(3))

                logNe = self.model(inp)

                tmp.append(torch.exp(logNe.squeeze()).reshape((n_angles, n_pixels_X, n_pixels_Y, n_pixels_Z)))
                angle.append(angles)
                coord.append(inp)
                
        out = torch.cat(tmp, dim=0).cpu().numpy()
        coord = torch.cat(coord, dim=0).cpu().numpy()
        angle = torch.cat(angle).cpu().numpy()

        loop = 0
        fig, ax = pl.subplots(nrows=7, ncols=7, figsize=(10,10))
        for i in range(7):
            for j in range(7):
                ax[i, j].imshow(out[loop, :, :, 0])
                ax[i, j].set_title(f'{angle[loop]:4.1f}')
                loop += 1

        loop = 0
        fig, ax = pl.subplots(nrows=7, ncols=7, figsize=(10,10))
        for i in range(7):
            for j in range(7):
                ax[i, j].imshow(coord[loop, :, 0].reshape((64, 64)))
                ax[i, j].set_title(f'{angle[loop]:4.1f}')
                loop += 1
                

if __name__ == '__main__':
    pl.close('all')

    corona = CoronaTomography(gpu=0)
    
    corona.observations_and_reference_system('data', 'model_time.pth', reduction=8)

    # viewer = napari.view_image(Ne)
    # viewer = napari.view_image(Ne * (~corona.mask_3d[None,:,:,:]).cpu().numpy())

    # breakpoint()

    # r = torch.tensor([2.5, 4.0, 5.5])
    # long = torch.tensor([0.0, 90.0, 180.0])
    # corona.do_plot(r, long, n_pixels=64)