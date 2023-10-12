import numpy as np
import torch
import matplotlib.pyplot as pl
from tqdm import tqdm
from rotation import rotation


def electron(xyz):
    r = torch.sqrt(torch.sum(xyz**2, dim=-1))
    th = torch.atan2(xyz[:, 1], xyz[:, 0])
    return 2e5 * torch.exp(-r**2 / 12.0) * torch.cos(th)**2

class CoronaTomography(object):
    def __init__(self):
        self.sigma_thomson = 6.6524587321e-23            # Thomson scattering cross-section
        self.const = 3.0 * self.sigma_thomson / 8.0
        self.FOV = 5.0                                   # in solar radii
        self.n_pixels = 100                              # Number of pixels in X,Y,Z
        self.R_sun = 6.957e+10                   
        self.delta_LOS = self.R_sun * (2.0 * self.FOV) / self.n_pixels  # Integration step
        self.mask_size = 2.0                             # Coronographic mask size
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
        self.p_POS = torch.sqrt(self.Y[50, :, :]**2 + self.Z[50, :, :]**2).flatten()

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

        # Central mask
        self.mask = (self.p_POS < self.mask_size).reshape((self.n_pixels, self.n_pixels))

    def synth(self, angles, electron_fn):
        n_angles = len(angles)
    
        imB = [None] * n_angles
        impB = [None] * n_angles        

        # Now compute the images for all observing angles
        for i in range(n_angles):

            alpha = angles[i]
            
            # Define rotation matrix that transforms the LOS reference system into the 
            # local reference system where we evaluate the electron density
            # This is equivalent to rotating the reference system by -alpha
            u = torch.tensor([0.0, 0.0, 1.0])
            M = rotation(u, -alpha)

            # Apply the rotation matrix for all points
            XYZ_rotated = torch.matmul(M, self.XYZ_LOS[:, :, None]).squeeze()
            
            # Evaluate the electron density
            Ne = electron_fn(XYZ_rotated)
            
            Ne_3d = Ne.reshape((self.n_pixels, self.n_pixels, self.n_pixels))
            
            imB[i] = self.delta_LOS * torch.sum(Ne_3d * self.omegaB, dim=0)[:, :, None] / 1e-10
            impB[i] = self.delta_LOS * torch.sum(Ne_3d * self.omegapB, dim=0)[:, :, None] / 1e-10

            imB[i][self.mask] = 0.0
            impB[i][self.mask] = 0.0

        imB = torch.cat(imB, dim=-1)
        impB = torch.cat(impB, dim=-1)

        return imB, impB


if __name__ == '__main__':

    corona = CoronaTomography()

    # Observing angles
    angles = np.linspace(0.0, 180, 8)
    imB, impB = corona.synth(angles, electron)

    fig, ax = pl.subplots(nrows=8, ncols=2, figsize=(8,15))
    for i in range(8):
        ax[i, 0].imshow(imB[:, :, i].detach().numpy())
        ax[i, 1].imshow(impB[:, :, i].detach().numpy())

    