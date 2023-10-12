import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as pl
from tqdm import tqdm
from rotation import rotation
from modules.Siren import SirenNet

class MLP(nn.Module):
    def __init__(self, *, dim_input, dim_hidden, dim_out, depth_hidden = 3):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(dim_input, dim_hidden))
        self.layers.append(nn.ELU())

        for i in range(depth_hidden):
            self.layers.append(nn.Linear(dim_hidden, dim_hidden))
            self.layers.append(nn.ELU())
        
        self.layers.append(nn.Linear(dim_hidden, dim_out))
        # self.layers.append(nn.Sigmoid())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

def log_electron(xyz_rescaled):
    xyz = xyz_rescaled * 5.0
    r = torch.sqrt(torch.sum(xyz**2, dim=-1))
    th = torch.atan2(xyz[:, 2], xyz[:, 1])
    return torch.log(1e0 * torch.exp(-r**2 / 12.0) * (0.2 + 0.5*torch.cos(th)**2))

class CoronaTomography(object):
    def __init__(self, gpu=0):

        self.cuda = torch.cuda.is_available()
        self.gpu = gpu

        if (self.gpu < 0):
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")
        
        print(f'Computing in {self.device}')
        
        self.sigma_thomson = 6.6524587321e-23            # Thomson scattering cross-section
        self.const = 3.0 * self.sigma_thomson / 8.0
        self.FOV = 5.0                                  # in solar radii
        self.n_pixels = 50                              # Number of pixels in X,Y,Z
        self.R_sun = 6.957e+10                   
        self.delta_LOS = self.R_sun * (2.0 * self.FOV) / self.n_pixels  # Integration step
        self.mask_size = 2.0                             # Coronographic mask size
        self.L_sun = 1.0                                 # Solar luminosity

        # Coordinate system where X is the LOS
        x = np.linspace(-self.FOV, self.FOV, self.n_pixels, dtype='float32')
        y = np.linspace(-self.FOV, self.FOV, self.n_pixels, dtype='float32')
        z = np.linspace(-self.FOV, self.FOV, self.n_pixels, dtype='float32')
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        self.X = torch.tensor(X, device=self.device)
        self.Y = torch.tensor(Y, device=self.device)
        self.Z = torch.tensor(Z, device=self.device)
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

        # Central mask
        self.mask = (self.p_POS < self.mask_size).reshape((self.n_pixels, self.n_pixels))
        self.mask_3d = (self.r_LOS < self.mask_size).reshape((self.n_pixels, self.n_pixels, self.n_pixels))

        self.weight = torch.exp(-self.p_POS**2).reshape((self.n_pixels, self.n_pixels))
        self.weight[self.mask] = 0.0

        # Unit vector along the Z axis which is used for the rotation
        self.u = torch.tensor([0.0, 0.0, 1.0], device=self.device)

        # Spherical caps at different radial distances
        theta = np.linspace(0.0, np.pi, self.n_pixels, dtype='float32')
        phi = np.linspace(0.0, 2*np.pi, self.n_pixels, dtype='float32')
        Theta, Phi = np.meshgrid(theta, phi, indexing='ij')
        self.theta = torch.tensor(Theta, device=self.device)
        self.phi = torch.tensor(Phi, device=self.device)
        self.theta_flat = self.theta.flatten()
        self.phi_flat = self.phi.flatten()

    def synth(self, angles, log_electron_fn, factor=1.0):
        n_angles = len(angles)
    
        imB = [None] * n_angles
        impB = [None] * n_angles        

        # Now compute the images for all observing angles
        for i in range(n_angles):

            alpha = angles[i]
            
            # Define rotation matrix that transforms the LOS reference system into the 
            # local reference system where we evaluate the electron density
            # This is equivalent to rotating the reference system by -alpha            
            M = rotation(self.u, -alpha).to(self.device)

            # Apply the rotation matrix for all points
            XYZ_rotated = torch.matmul(M, self.XYZ_LOS[:, :, None]).squeeze()
            
            # Evaluate the electron density
            Ne = torch.exp(factor * log_electron_fn(XYZ_rotated / self.FOV))
            
            Ne_3d = Ne.reshape((self.n_pixels, self.n_pixels, self.n_pixels))
            
            imB[i] = self.delta_LOS * torch.sum(Ne_3d * self.omegaB, dim=0) / 1e-10
            impB[i] = self.delta_LOS * torch.sum(Ne_3d * self.omegapB, dim=0) / 1e-10

            imB[i][self.mask] = 1e-10
            impB[i][self.mask] = 1e-10

            imB[i] = imB[i][None, :, :]
            impB[i] = impB[i][None, :, :]

        imB = torch.cat(imB, dim=0)
        impB = torch.cat(impB, dim=0)

        Ne = torch.exp(factor * log_electron_fn(self.XYZ_LOS / self.FOV).reshape((self.n_pixels, self.n_pixels, self.n_pixels)))

        Ne[self.mask_3d] = 0.0

        return imB, impB, Ne

    def optimize(self, angles, obsB, obspB):

        # self.model = SirenNet(dim_in=3, dim_hidden=8, dim_out=1, num_layers=3, final_activation=nn.Sigmoid())

        self.model = MLP(dim_input=3, dim_hidden=8, dim_out=1, depth_hidden=3).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-2)
        # self.optimizer = torch.optim.LBFGS(self.model.parameters(), lr=1.0)
        
        self.loss_fn = nn.MSELoss().to(self.device)

        self.loss = []

        for i in range(200):

            def closure():
                self.optimizer.zero_grad()
                imB, impB, _ = self.synth(angles, self.model, factor=6.0)
                
                loss = torch.mean((torch.log(obsB) - torch.log(imB))**2) #+ self.loss_fn(torch.log(obspB), torch.log(impB))
                # loss = torch.mean((obsB - imB)**2) #+ self.loss_fn(torch.log(obspB), torch.log(impB))
                # loss = self.loss_fn(obsB, imB) #+ self.loss_fn(obspB, impB)
                self.loss.append(loss)
                loss.backward()
                return loss

            self.optimizer.step(closure)

            print(f'{i} - loss={self.loss[-1].item()}')
                    
        imB, impB, Ne = self.synth(angles, self.model, factor=6.0)

        Ne[self.mask_3d] = 0.0

        return imB, impB, Ne

    def spherical_caps(self, r, log_electron_fn, factor=1.0):
        
        n_caps = len(r)
        Ne_cap = [None] * n_caps
        obs_cap = [None] * n_caps

        # Spherical caps
        for i in range(n_caps):        
            x = r[i] * torch.sin(self.theta_flat) * torch.cos(self.phi_flat)
            y = r[i] * torch.sin(self.theta_flat) * torch.sin(self.phi_flat)
            z = r[i] * torch.cos(self.theta_flat)
            xyz = torch.cat([x[:, None], y[:, None], z[:, None]], dim=-1)
            Ne_cap[i] = torch.exp(factor * self.model(xyz / self.FOV)).reshape((self.n_pixels, self.n_pixels)).detach().cpu().numpy()
            
            obs_cap[i] = torch.exp(log_electron_fn(xyz / self.FOV)).reshape((self.n_pixels, self.n_pixels)).detach().cpu().numpy()

        return obs_cap, Ne_cap


if __name__ == '__main__':
    pl.close('all')

    corona = CoronaTomography(gpu=0)

    # Observing angles
    angles = torch.linspace(0.0, 180, 8, device=corona.device)

    obsB, obspB, obsNe = corona.synth(angles, log_electron, factor=1.0)

    # noise = torch.distributions.Normal(torch.zeros_like(obsB), 1e-10*torch.ones_like(obsB))
    # obsB += noise.sample()

    # fig, ax = pl.subplots(nrows=8, ncols=2, figsize=(8,15))
    # for i in range(8):
    #     ax[i, 0].imshow(imB[:, :, i].detach().numpy())
    #     ax[i, 1].imshow(impB[:, :, i].detach().numpy())

    imB, impB, Ne = corona.optimize(angles, obsB, obspB)

    r = torch.tensor([2.5, 3.5])
    obs_cap, Ne_cap = corona.spherical_caps(r, log_electron, factor=6.0)


    obsB = obsB.detach().cpu().numpy()
    obspB = obspB.detach().cpu().numpy()
    obsNe = obsNe.detach().cpu().numpy()

    imB = imB.detach().cpu().numpy()
    impB = impB.detach().cpu().numpy()
    Ne = Ne.detach().cpu().numpy()

    fig, ax = pl.subplots(nrows=2, ncols=3, figsize=(10,8))
    maxval = np.max(obsNe)
    minval = np.min(obsNe)
    im = ax[0, 0].imshow(obsNe[25, :, :], vmin=minval, vmax=maxval)
    pl.colorbar(im, ax=ax[0, 0])
    im = ax[0, 1].imshow(obsNe[:, 25, :], vmin=minval, vmax=maxval)
    pl.colorbar(im, ax=ax[0, 1])
    im = ax[0, 2].imshow(obsNe[:, :, 25], vmin=minval, vmax=maxval)
    pl.colorbar(im, ax=ax[0, 2])
    
    im = ax[1, 0].imshow(Ne[25, :, :], vmin=minval, vmax=maxval)
    pl.colorbar(im, ax=ax[1, 0])
    im = ax[1, 1].imshow(Ne[:, 25, :], vmin=minval, vmax=maxval)
    pl.colorbar(im, ax=ax[1, 1])
    im = ax[1, 2].imshow(Ne[:, :, 25], vmin=minval, vmax=maxval)
    pl.colorbar(im, ax=ax[1, 2])
    pl.ticklabel_format(useOffset=False)


    fig, ax = pl.subplots(nrows=2, ncols=8, figsize=(18,6))
    for i in range(8):
        maxval = np.max(np.log(obsB[i, :, :]))
        minval = np.min(np.log(obsB[i, :, :]))
        im = ax[0, i].imshow(np.log(obsB[i, :, :]), vmin=minval, vmax=maxval)
        pl.colorbar(im, ax=ax[0, i])
        im = ax[1, i].imshow(np.log(imB[i, :, :]), vmin=minval, vmax=maxval)
        pl.colorbar(im, ax=ax[1, i])
    pl.ticklabel_format(useOffset=False)

    fig, ax = pl.subplots(nrows=2, ncols=2)
    for i in range(2):
        im = ax[0, i].imshow(obs_cap[i])
        pl.colorbar(im, ax=ax[0, i])
        im = ax[1, i].imshow(Ne_cap[i])
        pl.colorbar(im, ax=ax[1, i])
    pl.ticklabel_format(useOffset=False)