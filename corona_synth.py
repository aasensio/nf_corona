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
import model
import napari
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
    def __init__(self, angles, M_rot=None, images=None):
        """
        Main dataset

        Parameters
        ----------
        angles : numpy array
            Array with all rotation angles
        images : numpy array
            Numpy array with all images of the observation. If None, they won't be used
        """
        super(Dataset, self).__init__()
        
        self.n_images = len(angles)
        
        self.obspB = [None] * self.n_images
        self.angles = angles
        
        # Read all files, extracting the observing time, the size of the Sun
        for i in range(self.n_images):
            
                # Data is given in 1e-10 mean solar brightness units
            if (images is not None):
                self.obspB[i] = torch.tensor(images[i, :, :].astype('float32'))[None, :, :]
            else:
                self.obspB[i] = torch.zeros((1,1,1))
        
        # Define rotation matrix that transforms the LOS reference system into the 
        # local reference system where we evaluate the electron density
        # This is equivalent to rotating the reference system by -alpha
        self.M = M_rot

        # Number of training images
        self.n_training = self.n_images

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

class CoronaSynthesis(object):
    def __init__(self, n_pixels=128, 
            n_pixels_integration=64, 
            angles=None, 
            FOV=1.0, 
            device=None, 
            rot_vector=[0.0, 0.0, 1.0], 
            images=None, 
            batch_size=4,
            r_max=1.0):

        self.device = device
                            
        # Number of pixels to use for LOS integration
        self.n_pixels_integration = n_pixels_integration
        self.n_pixels = n_pixels
        self.images = images
        self.r_max = r_max
        self.batch_size = batch_size
        
        # Field-of-view
        self.FOV = FOV

        self.n_angles = len(angles)
        self.angles = torch.tensor(angles.astype('float32'))

        # Unit vector along the Z axis which is used for the rotation
        self.u = torch.tensor(rot_vector)

        # Define rotation matrix that transforms the LOS reference system into the 
        # local reference system where we evaluate the electron density
        # This is equivalent to rotating the reference system by -alpha
        self.M = torch.zeros((self.n_angles, 3, 3))
        for i in range(self.n_angles):
            self.M[i, :, :] = rotation(self.u, -self.angles[i])

        # Define the dataset to loop over angles
        self.dataset = Dataset(self.angles, self.M, images=self.images)

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
        self.delta_LOS = self.R_sun * (2.0 * self.FOV) / self.n_pixels_integration
        
        # Size of the LASCO C2 mask
        self.mask_size_min = 2.3
        self.mask_size_max = 6.3

        # Coordinate system where X is the LOS
        x = np.linspace(-self.FOV, self.FOV, self.n_pixels_integration, dtype='float32')
        y = np.linspace(-self.FOV, self.FOV, self.n_pixels, dtype='float32')
        z = np.linspace(-self.FOV, self.FOV, self.n_pixels, dtype='float32')
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        self.X = torch.tensor(X)
        self.Y = torch.tensor(Y)
        self.Z = torch.tensor(Z)        
        
        # Impact parameter
        self.p = torch.sqrt(self.Y**2 + self.Z**2)
        self.p_POS = torch.sqrt(self.Y[self.n_pixels//2, :, :]**2 + self.Z[self.n_pixels//2, :, :]**2)

        # XYZ coordinates of the LOS reference system in 3D form
        self.XYZ_LOS = torch.cat([self.X[..., None], self.Y[..., None], self.Z[..., None]], dim=-1)
        self.r_LOS = torch.sqrt(torch.sum(self.XYZ_LOS**2, dim=-1))

        # Sine of the scattering angle for each point in the 3D volume
        self.sin_theta_scatt = self.p / self.r_LOS

        # https://hesperia.gsfc.nasa.gov/ssw/packages/nrl/idl/nrlgen/analysis/eltheory.pro
        # See also A&A, 393, 295
        sinchi2 = self.sin_theta_scatt**2
        s = torch.clamp(1.0/self.r_LOS, max=0.99999999)
        s2 = s*s
        c2 = torch.clamp((1.0 - s2), min=1e-8)
        c = torch.sqrt(c2)
        g = c2 * torch.log((1.0 + s) / c) / s

        # Compute Van de Hulst coefficients (Billings 1968 after Minnaert 1930)
        a_el = c*s2
        c_el = (4.0 - c*(3 + c2)) / 3.0
        b_el = -(1.0 - 3.0*s2 - g*(1.0 + 3.0*s2)) / 8.0
        d_el = (5.0 + s2 - g*(5.0-s2)) / 8.0

        Bt = self.constant * (c_el + self.u*(d_el - c_el))
        pB = self.constant * sinchi2 * ( (a_el + self.u*(b_el - a_el)) )
        
        Br = Bt - pB
                
        # Scattering efficiencies
        self.omegaB = Bt + Br
        self.omegapB = Bt - Br
        
        # Central and outer masks in the 2D plane-of-the-sky
        self.mask_in = (self.p_POS < self.mask_size_min)
        self.mask_out = (self.p_POS > self.mask_size_max)
        self.mask = torch.logical_or(self.mask_in, self.mask_out)
        
        # 3D mask produced by the occulters
        self.mask_3d_in = (self.r_LOS < self.mask_size_min)
        self.mask_3d_out = (self.r_LOS > self.mask_size_max)
        self.mask_3d = torch.logical_or(self.mask_3d_in, self.mask_3d_out)

        # Dataloader that will produce images at different rotation angles
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
                                
    def neural_logNe(self, xyz):
        return self.model(xyz) #(self.logNe_range[1] - self.logNe_range[0]) * self.model(xyz) + self.logNe_range[0]        

    def synth(self, model_ne, patch_size):
        # Patchify the tensors if needed by unfolding the tensor with the provided patch size
        XYZ_LOS_patch = self.XYZ_LOS.unfold(2, patch_size, patch_size).unfold(1, patch_size, patch_size)
        XYZ_LOS_patch = XYZ_LOS_patch.permute(0, 5, 4, 3, 1, 2)

        ni = XYZ_LOS_patch.shape[4]
        nj = XYZ_LOS_patch.shape[5]        

        omegapB_patch = self.omegapB.unfold(2, patch_size, patch_size).unfold(1, patch_size, patch_size).permute(0, 1, 2, 4, 3)
        mask_patch = (~self.mask).unfold(1, patch_size, patch_size).unfold(0, patch_size, patch_size).permute(0, 1, 3, 2).float()

        t = tqdm(self.loader)

        impB = torch.zeros((self.batch_size, ni, nj, patch_size, patch_size))

        with torch.no_grad():
            
            self.impB = []
            self.Ne_cube = []

            for batch_idx, (obspB, M_rot, angles) in enumerate(t):
                                
                # Move the rotation matrix to GPU because it is common for all patches
                M_rot = M_rot.to(self.device)
                angles = angles.to(self.device)

                self.impB_out = torch.zeros((len(angles), patch_size, patch_size)).to(self.device)

                # Compute all patches
                for i in range(ni):
                    
                    for j in range(nj):
                                        
                        XYZ_LOS = XYZ_LOS_patch[:, :, :, :, i, j].to(self.device)
                        omegapB = omegapB_patch[:, i, j, :, :].to(self.device)
                        mask = mask_patch[i, j, :, :].to(self.device)
                                    
                        imB, impB_patch, Ne = self.synth_patch(model_ne, XYZ_LOS, M_rot, angles, omegapB, mask)
                        
                        impB[:, i, j, :, :] = (impB_patch * mask[None, :, :]).detach().cpu()
                        

                impB_full = impB.permute(0, 3, 4, 1, 2).reshape((-1, patch_size*patch_size, ni*nj))
                impB_full = F.fold(impB_full, (self.n_pixels, self.n_pixels), patch_size, stride=patch_size).squeeze(1)    
                                
                self.impB.append(impB_full)
                
        self.imB = None
        self.impB = torch.cat(self.impB, dim=0).numpy()
        
        return self.imB, self.impB

    def synth_patch(self, model_ne, XYZ_LOS, M_rot, angles, omegapB, mask, time_variation=False):
        """
        Synthesize an image using a function for the logarithm of Ne, typically a fully connected neural network

        Parameters
        ----------
        XYZ_LOS : Torch model
            Model that takes XYZ(t) as input and returns the electron density
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
        n_points = n_pixels_X * n_pixels_Y * n_pixels_Z

        mask_extended = (mask[None, :, :] == 1).expand(n_angles, n_pixels_Y, n_pixels_Z)
        
        XYZ_LOS_active = torch.masked_select(XYZ_LOS, mask[None, :, :, None] == 1).view(n_pixels_X, -1, 3)
        omegapB_active = torch.masked_select(omegapB, mask[None, :, :] == 1).view(n_pixels_X, -1)

        # Flatten the XYZ_LOS reference system for the current patch
        XYZ_LOS_flat = XYZ_LOS_active.reshape(-1, 3)

        # Rotate it at the desired angle
        XYZ_rotated = torch.matmul(M_rot[:, None, :, :], XYZ_LOS_flat[None, :, :, None]).squeeze()
        
        # Evaluate the log-electron density using the neural network        
        if (XYZ_rotated.ndim == 2):
            XYZ_rotated = XYZ_rotated[None, :, :]

        if (time_variation):
            cosa = torch.cos(angles * np.pi / 180.0)[:, None, None].repeat(1, n_points, 1)
            sina = torch.sin(angles * np.pi / 180.0)[:, None, None].repeat(1, n_points, 1)
            inp = torch.cat([cosa, sina, XYZ_rotated / self.r_max], dim=-1)
        else:
            inp = XYZ_rotated / self.r_max
        
        if (isinstance(model_ne, list)):
            logNe_1 = model_ne[0](inp)
            logNe_2 = model_ne[1](inp)
            t = angles / 360.0
            logNe = (1.0 - t[:, None, None]) * logNe_1 + t[:, None, None] * logNe_2
        else:
            logNe = model_ne(inp)

        Ne_3d = torch.exp(logNe).reshape((n_angles, n_pixels_X, -1))
        impB = self.delta_LOS * torch.sum(Ne_3d * omegapB_active[None, :], dim=1)
                
        imB = None
        self.impB_out[mask_extended] = impB.view(-1)

        Ne = Ne_3d.detach()

        return imB, self.impB_out, Ne


    def synth_patch_old(self, model_ne, XYZ_LOS, M_rot, angles, omegapB, mask, time_variation=False):
        """
        Synthesize an image using a function for the logarithm of Ne, typically a fully connected neural network

        Parameters
        ----------
        XYZ_LOS : Torch model
            Model that takes XYZ(t) as input and returns the electron density
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
        n_points = n_pixels_X * n_pixels_Y * n_pixels_Z

        mask_extended = (mask[None, :, :] == 1).expand(n_angles, n_pixels_Y, n_pixels_Z)
        
        XYZ_LOS_active = torch.masked_select(XYZ_LOS, mask[None, :, :, None] == 1).view(n_pixels_X, -1, 3)
        omegapB_active = torch.masked_select(omegapB, mask[None, :, :] == 1).view(n_pixels_X, -1)

        # Flatten the XYZ_LOS reference system for the current patch
        XYZ_LOS_flat = XYZ_LOS_active.reshape(-1, 3)

        # Rotate it at the desired angle
        XYZ_rotated = torch.matmul(M_rot[:, None, :, :], XYZ_LOS_flat[None, :, :, None]).squeeze()
        
        # Evaluate the log-electron density using the neural network        
        if (XYZ_rotated.ndim == 2):
            XYZ_rotated = XYZ_rotated[None, :, :]

        if (time_variation):
            cosa = torch.cos(angles * np.pi / 180.0)[:, None, None].repeat(1, n_points, 1)
            sina = torch.sin(angles * np.pi / 180.0)[:, None, None].repeat(1, n_points, 1)
            inp = torch.cat([cosa, sina, XYZ_rotated / self.r_max], dim=-1)
        else:
            inp = XYZ_rotated / self.r_max
        
        logNe = model_ne(inp)

        Ne_3d = torch.exp(logNe).reshape((n_angles, n_pixels_X, -1))
        impB = self.delta_LOS * torch.sum(Ne_3d * omegapB_active[None, :], dim=1)

        impB_out = torch.zeros((n_angles, n_pixels_Y, n_pixels_Z))
        breakpoint()
        impB_out[mask_extended] = impB

        breakpoint()
        

        

        
        
        Ne_3d = torch.exp(logNe).reshape((n_angles, n_pixels_X, n_pixels_Y, n_pixels_Z))
                                
        # Compute the integrated light along the LOS        
        imB = None
                
        impB = self.delta_LOS * torch.sum(Ne_3d * omegapB[None, :], dim=1)

        Ne = Ne_3d.detach()

        return imB, impB, Ne

if __name__ == '__main__':
    pl.close('all')

    device = torch.device('cuda:1')

    checkpoint = 'pred_sci.pth'
    print("=> loading checkpoint '{}'".format(checkpoint))
    chk = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    print("=> loaded checkpoint '{}'".format(checkpoint))
    hyperparameters = chk['hyperparameters']

    model_ne = model.INRModel(hyperparameters, device=device)
    model_ne.load_weights(chk['state_dict'])

    angles = np.linspace(0.0, 180, 20)
    corona = CoronaSynthesis(n_pixels=64, n_pixels_integration=64, angles=angles, device=device, FOV=7.0, r_max=hyperparameters['r_max'])
    
    imB, impB = corona.synth(model_ne, patch_size=32)

    fig, ax = pl.subplots(nrows=5, ncols=4, figsize=(10,10))
    for i in range(20):
        ax.flat[i].imshow(impB[i, :, :] / 1e-10)
    pl.show()

    # fig, ax = pl.subplots()

    # ax.imshow(np.log(Ne))
    # pl.show()

    # viewer = napari.view_image(impB)