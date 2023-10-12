from math import log
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.nn.init as init
import matplotlib.pyplot as pl
from tqdm import tqdm
import datetime
import model
import dataset
import azimuthal_average
from torch.profiler import profile, record_function, ProfilerActivity
try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False


class CoronalTomography(object):
    def __init__(self, gpu=0, hyperparameters=None, checkpoint=None):

        self.cuda = torch.cuda.is_available()
        self.gpu = gpu        

        # Hyperparameters
        if (hyperparameters is not None):
            self.hyperparameters = hyperparameters
        elif (checkpoint is not None):
            self.load_checkpoint(checkpoint)

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
        
    def observations_and_reference_system(self, directory=None, reduction=1, n_pixels_integration=64):
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
        if (self.hyperparameters['obs'] == 'lasco'):
            self.dataset = dataset.DatasetFITS(directory, reduction)

        if (self.hyperparameters['obs'] == 'predsci'):
            self.dataset = dataset.DatasetPREDSCI(n_angles=20, device=self.device, FOV=6.3)        

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

        # Size of the LASCO C2 mask
        self.mask_size_min = 2.3
        self.mask_size_max = 6.3

        # Maximum radial position of all XYZ points
        self.r_max = 10.0

        # Step for the LOS integration
        self.delta_LOS = self.R_sun * (2.0 * self.dataset.FOV) / self.n_pixels_integration
        
        # Coordinate system where X is the LOS. We use a cube of length equal to the full LASCO FOV
        x = np.linspace(-self.dataset.FOV, self.dataset.FOV, self.n_pixels_integration, dtype='float32')
        y = np.linspace(-self.dataset.FOV, self.dataset.FOV, self.dataset.n_pixels, dtype='float32')
        z = np.linspace(-self.dataset.FOV, self.dataset.FOV, self.dataset.n_pixels, dtype='float32')
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        self.X = torch.tensor(X)
        self.Y = torch.tensor(Y)
        self.Z = torch.tensor(Z)
        
        # Impact parameter
        self.p = torch.sqrt(self.Y**2 + self.Z**2)
        self.p_POS = torch.sqrt(self.Y[self.n_pixels_integration//2, :, :]**2 + self.Z[self.n_pixels_integration//2, :, :]**2)

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

        tmp = self.dataset.obspB[:, ~self.mask]
        print(f'Observations inside mask : min={torch.min(tmp).item()} - max={torch.max(tmp).item()}')

        # avg = torch.mean(self.dataset.obspB, dim=0).cpu().numpy()
        # rr, az_avg = azimuthal_average.azimuthal_average(avg, returnradii=True)
        # az_avg = np.nan_to_num(az_avg)
        
        self.weight = torch.exp(((self.p_POS - 2.0))).reshape((self.dataset.n_pixels, self.dataset.n_pixels))

        # self.weight = torch.exp((5-self.p_POS) / 1.25).reshape((self.dataset.n_pixels, self.dataset.n_pixels))
        # self.weight = torch.ones((self.dataset.n_pixels, self.dataset.n_pixels))
        # self.weight = (1.0 / self.p_POS**2).reshape((self.dataset.n_pixels, self.dataset.n_pixels))        
        
    def init_optimize(self, optimizer=True):
        
        batch_size = self.hyperparameters['batch_size']
        self.n_epochs = self.hyperparameters['n_epochs']
        self.patch_size = self.hyperparameters['patch_size']
        lr = self.hyperparameters['lr']

        # If no patch size is provided, then use the full image size
        if (self.patch_size is None):
            self.patch_size = self.dataset.n_pixels

        # Include the effect of time
        self.time = self.hyperparameters['time']
        
        # Dataloader that will produce images at different rotation angles
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        # Neural network model
        self.model = model.INRModel(self.hyperparameters, device=self.device)
        
        # Optimizer
        if (optimizer):
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

            # Scheduler
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.n_epochs, eta_min=0.1*lr)        

    def neural_logNe(self, xyz):
        return self.model(xyz)

    def load_checkpoint(self, checkpoint):
        
        # Load checkpoint and rebuild the model
        self.checkpoint = '{0}'.format(checkpoint)
        print("=> loading checkpoint '{}'".format(self.checkpoint))
        self.chk = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)
        print("=> loaded checkpoint '{}'".format(self.checkpoint))

        self.hyperparameters = self.chk['hyperparameters']

    def load_weights(self):

        # Now we load the weights and the network should be on the same state as before
        self.model.load_weights(self.chk['state_dict'])

    def synth(self, XYZ_LOS, M_rot, angles, omegapB, mask, compute_electron_LOS=False):
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
        n_points = n_pixels_X * n_pixels_Y * n_pixels_Z

        impB_out = torch.zeros((n_angles, n_pixels_Y, n_pixels_Z)).to(self.device)

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

        if (self.time):
            cosa = torch.cos(angles * np.pi / 180.0)[:, None, None].repeat(1, n_points, 1)
            sina = torch.sin(angles * np.pi / 180.0)[:, None, None].repeat(1, n_points, 1)
            inp = torch.cat([cosa, sina, XYZ_rotated / self.r_max], dim=-1)
        else:
            inp = XYZ_rotated / self.r_max

        logNe = self.neural_logNe(inp)
        
        Ne_3d = torch.exp(logNe).reshape((n_angles, n_pixels_X, -1))
        impB = self.delta_LOS * torch.sum(Ne_3d * omegapB_active[None, :], dim=1)
        
        imB = None
        impB_out[mask_extended] = impB.view(-1)

        Ne = Ne_3d.detach()

        return imB, impB_out, Ne

    def synth_old(self, XYZ_LOS, M_rot, angles, omegapB, mask, compute_electron_LOS=False):
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
        n_points = n_pixels_X * n_pixels_Y * n_pixels_Z

        # Flatten the XYZ_LOS reference system for the current patch
        XYZ_LOS_flat = XYZ_LOS.reshape(-1, 3)        

        # Rotate it at the desired angle
        XYZ_rotated = torch.matmul(M_rot[:, None, :, :], XYZ_LOS_flat[None, :, :, None]).squeeze()        
        
        # Evaluate the log-electron density using the neural network        
        if (XYZ_rotated.ndim == 2):
            XYZ_rotated = XYZ_rotated[None, :, :]

        if (self.time):
            cosa = torch.cos(angles * np.pi / 180.0)[:, None, None].repeat(1, n_points, 1)
            sina = torch.sin(angles * np.pi / 180.0)[:, None, None].repeat(1, n_points, 1)
            inp = torch.cat([cosa, sina, XYZ_rotated / self.r_max], dim=-1)
        else:
            inp = XYZ_rotated / self.r_max

        logNe = self.neural_logNe(inp)
        
        Ne_3d = torch.exp(logNe).reshape((n_angles, n_pixels_X, n_pixels_Y, n_pixels_Z))
                                
        # Compute the integrated light along the LOS        
        imB = None
                
        impB = self.delta_LOS * torch.sum(Ne_3d * omegapB[None, :], dim=1)        

        Ne = Ne_3d.detach().cpu()

        del Ne_3d
        del XYZ_LOS_flat
        del XYZ_rotated
        del inp

        return imB, impB, Ne

    def train_epoch_full(self, epoch, optimizer=True):
        self.impB = []
        self.Ne_cube = []

        ni = self.XYZ_LOS_patch.shape[4]
        nj = self.XYZ_LOS_patch.shape[5]

        if (optimizer):
            for param_group in self.optimizer.param_groups:
                current_lr = param_group['lr']

        t = tqdm(self.loader)
        
        total_loss = 0

        # Batch covers all rotations
        for batch_idx, (obspB, M_rot, angles) in enumerate(t):
            
            # Patchify the observations using unfold with the provided patch size                
            obspB_patch = obspB.unfold(2, self.patch_size, self.patch_size).unfold(1, self.patch_size, self.patch_size)

            # Move the rotation matrix to GPU because it is common for all patches
            M_rot = M_rot.to(self.device)
            angles = angles.to(self.device)
            
            # Zero the gradient computation
            if (optimizer):
                self.optimizer.zero_grad()

            impB = [None] * ni
            Ne_cube = [None] * ni

            loss = 0.0

            self.impB_out = torch.zeros((len(angles), self.patch_size, self.patch_size)).to(self.device)

            # Accumulate gradients for all patches, passing to GPU only those that we care
            for i in range(ni):
                impBj = [None] * nj
                Ne_cubej = [None] * nj
                for j in range(nj):
                    
                    obspB = obspB_patch[:, i, j, :, :].to(self.device)
                    XYZ_LOS = self.XYZ_LOS_patch[:, :, :, :, i, j].to(self.device)
                    omegapB = self.omegapB_patch[:, i, j, :, :].to(self.device)                        
                    mask = self.mask_patch[i, j, :, :].to(self.device)
                    weight = self.weight_patch[i, j, :, :].to(self.device)                        

                    imB, impB_patch, Ne = self.synth(XYZ_LOS, M_rot, angles, omegapB, mask)
                                                                                            
                    # The units of obspB are in 1e-10 times the mean solar brightness                    
                    if (optimizer):
                        loss += torch.mean( weight[None, :, :] * mask[None, :, :] * (obspB - impB_patch * 1e10)**2)
                                                                
                    impBj[j] = impB_patch[..., None].detach().cpu()
                    # Ne_cubej[j] = Ne[..., None]                     

                impB[i] = torch.cat(impBj, dim=-1)[..., None]
                # Ne_cube[i] = torch.cat(Ne_cubej, dim=-1)[..., None]

            if (optimizer):
                loss.backward()

                if (batch_idx == 0):
                    loss_avg = loss.item() / (ni * nj)
                else:
                    loss_avg = 0.05 * loss.item() / (ni * nj) + (1.0 - 0.05) * loss_avg
                
                self.optimizer.step()

            del self.impB_out
            
            impB = torch.cat(impB, dim=-1).transpose(1, 2).transpose(3, 4).reshape((-1, self.patch_size*self.patch_size, ni*nj))
            # Ne_cube = torch.cat(Ne_cube, dim=-1).reshape((-1, self.n_pixels_integration, self.patch_size, self.patch_size, ni, nj))
            # Ne_cube = Ne_cube.transpose(2, 3).transpose(4, 5).reshape((-1, self.patch_size*self.patch_size, ni*nj))
                        
            impB = F.fold(impB, (self.dataset.n_pixels, self.dataset.n_pixels), self.patch_size, stride=self.patch_size).squeeze(1)
            # Ne_cube = F.fold(Ne_cube, (self.dataset.n_pixels, self.dataset.n_pixels), self.patch_size, stride=self.patch_size).squeeze(1)
                
            # Ne_cube = Ne_cube.reshape((-1, self.n_pixels_integration, self.patch_size*ni, self.patch_size*nj))

            self.impB.append(impB)
            # self.Ne_cube.append(Ne_cube)
                                            
            min_Ne = torch.min(Ne).item()
            max_Ne = torch.max(Ne).item()
            if (optimizer):
                current_loss = loss.item() / (ni * nj * len(t))
            
                if (NVIDIA_SMI):
                    usage = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle)
                    memory = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
                    t.set_postfix(epoch=f'{epoch}/{self.n_epochs}', lr=f'{current_lr:8.5f}', loss_avg=f'{loss_avg:5.3f}', maxne=f'{max_Ne:5.1e}', minne=f'{min_Ne:5.1e}', gpu=usage.gpu, memfree=f'{memory.free/1024**2:5.1f} MB', memused=f'{memory.used/1024**2:5.1f} MB')
                else:
                    t.set_postfix(epoch=epoch, loss=f'{current_loss:5.3f}', lr=f'{current_lr:6.4f}', maxne=torch.max(self.Ne).item(), minne=torch.min(self.Ne).item())

        if (optimizer):
            return loss_avg
        else:
            return


    def train_epoch(self, epoch, optimizer=True):

        self.impB = []
        self.Ne_cube = []

        # torch.autograd.set_detect_anomaly(True)

        ni = self.XYZ_LOS_patch.shape[4]
        nj = self.XYZ_LOS_patch.shape[5]

        if (optimizer):
            for param_group in self.optimizer.param_groups:
                current_lr = param_group['lr']

        t = tqdm(self.loader)
                
        # Batch covers all rotations
        for batch_idx, (obspB, M_rot, angles) in enumerate(t):
            
            # Patchify the observations using unfold with the provided patch size                
            obspB_patch = obspB.unfold(2, self.patch_size, self.patch_size).unfold(1, self.patch_size, self.patch_size).permute(0, 1, 2, 4, 3)
            
            # Move the rotation matrix to GPU because it is common for all patches
            M_rot = M_rot.to(self.device)
            angles = angles.to(self.device)
                        
            impB = [None] * ni
            # pB = [None] * ni
            Ne_cube = [None] * ni

            loss = 0.0
            
            # Accumulate gradients for all patches, passing to GPU only those that we care
            for i in range(ni):
                impBj = [None] * nj
                # pBj = [None] * nj
                Ne_cubej = [None] * nj
                for j in range(nj):

                    print(i, j)

                    # Zero the gradient computation
                    if (optimizer):
                        self.optimizer.zero_grad()
                    
                    obs_pB = obspB_patch[:, i, j, :, :].to(self.device)
                    XYZ_LOS = self.XYZ_LOS_patch[:, :, :, :, i, j].to(self.device)
                    omegapB = self.omegapB_patch[:, i, j, :, :].to(self.device)                        
                    mask = self.mask_patch[i, j, :, :].to(self.device)
                    weight = self.weight_patch[i, j, :, :].to(self.device)                        

                    # imB, impB_patch, Ne = self.synth(XYZ_LOS, M_rot, angles, omegapB, mask)
                    imB, impB_patch, Ne = self.synth(XYZ_LOS, M_rot, angles, omegapB, mask)
                                                                                            
                    # The units of obspB are in 1e-10 times the mean solar brightness                    
                    if (optimizer):
                        loss = torch.mean( weight[None, :, :] * mask[None, :, :] * (obs_pB - impB_patch * 1e10)**2)
                                                                
                    impBj[j] = impB_patch[..., None].detach().cpu()
                    # pBj[j] = obs_pB[..., None].cpu()
                    # Ne_cubej[j] = Ne[..., None]                     

                    if (optimizer):
                        loss.backward(retain_graph=True)

                        if (batch_idx == 0):
                            loss_avg = loss.item() / (ni * nj)
                        else:
                            loss_avg = 0.05 * loss.item() / (ni * nj) + (1.0 - 0.05) * loss_avg
                        
                        self.optimizer.step()

                impB[i] = torch.cat(impBj, dim=-1)[..., None]
                # pB[i] = torch.cat(pBj, dim=-1)[..., None]
                # Ne_cube[i] = torch.cat(Ne_cubej, dim=-1)[..., None]
            
            impB = torch.cat(impB, dim=-1).transpose(1, 2).transpose(3, 4).reshape((-1, self.patch_size*self.patch_size, ni*nj))
            # pB = torch.cat(pB, dim=-1).transpose(1, 2).transpose(3, 4).reshape((-1, self.patch_size*self.patch_size, ni*nj))
            # Ne_cube = torch.cat(Ne_cube, dim=-1).reshape((-1, self.n_pixels_integration, self.patch_size, self.patch_size, ni, nj))
            # Ne_cube = Ne_cube.transpose(2, 3).transpose(4, 5).reshape((-1, self.patch_size*self.patch_size, ni*nj))
                        
            impB = F.fold(impB, (self.dataset.n_pixels, self.dataset.n_pixels), self.patch_size, stride=self.patch_size).squeeze(1)
            # pB = F.fold(pB, (self.dataset.n_pixels, self.dataset.n_pixels), self.patch_size, stride=self.patch_size).squeeze(1)
            # Ne_cube = F.fold(Ne_cube, (self.dataset.n_pixels, self.dataset.n_pixels), self.patch_size, stride=self.patch_size).squeeze(1)
                
            # Ne_cube = Ne_cube.reshape((-1, self.n_pixels_integration, self.patch_size*ni, self.patch_size*nj))

            breakpoint()
            self.impB.append(impB)
            # self.Ne_cube.append(Ne_cube)
                                            
            min_Ne = torch.min(Ne).item()
            max_Ne = torch.max(Ne).item()
            if (optimizer):
                current_loss = loss.item() / (ni * nj * len(t))
            
                if (NVIDIA_SMI):
                    usage = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle)
                    memory = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
                    t.set_postfix(epoch=f'{epoch}/{self.n_epochs}', lr=f'{current_lr:8.5f}', loss_avg=f'{loss_avg:5.3f}', maxne=f'{max_Ne:5.1e}', minne=f'{min_Ne:5.1e}', gpu=usage.gpu, memfree=f'{memory.free/1024**2:5.1f} MB', memused=f'{memory.used/1024**2:5.1f} MB')
                else:
                    t.set_postfix(epoch=epoch, loss=f'{current_loss:5.3f}', lr=f'{current_lr:6.4f}', maxne=torch.max(self.Ne).item(), minne=torch.min(self.Ne).item())

        if (optimizer):
            return loss_avg
        else:
            return

    def optimize(self):
                
        # Patchify the tensors if needed by unfolding the tensor with the provided patch size
        # Take into account the special way unfolding works, which always appends the new dimension at the end
        self.XYZ_LOS_patch = self.XYZ_LOS.unfold(2, self.patch_size, self.patch_size).unfold(1, self.patch_size, self.patch_size)   # (X, Ny, Nz, 3, Pz, Py)
        self.XYZ_LOS_patch = self.XYZ_LOS_patch.permute(0, 5, 4, 3, 1, 2)                                                           # (X, Ny, Nz, 3, Py, Pz)

        self.omegapB_patch = self.omegapB.unfold(2, self.patch_size, self.patch_size).unfold(1, self.patch_size, self.patch_size).permute(0, 1, 2, 4, 3)
        self.mask_patch = (~self.mask).unfold(1, self.patch_size, self.patch_size).unfold(0, self.patch_size, self.patch_size).permute(0, 1, 3, 2).float()
        self.weight_patch = self.weight.unfold(1, self.patch_size, self.patch_size).unfold(0, self.patch_size, self.patch_size).permute(0, 1, 3, 2)

        best_loss = 1e10

        self.model.set_train()

        loss = np.zeros(self.n_epochs)

        filename = f"models/{self.hyperparameters['obs']}_{self.hyperparameters['type']}_h{self.hyperparameters['dim_hidden']}_nh{self.hyperparameters['n_hidden']}_w{self.hyperparameters['w0_initial']}.pth"
        
        for epoch in range(self.n_epochs):

            loss_avg = self.train_epoch(epoch, optimizer=True)

            loss[epoch] = loss_avg

            self.scheduler.step()

            if (loss_avg < best_loss):

                checkpoint = {                  
                    'hyperparameters'  : self.hyperparameters,
                    'state_dict': self.model.state_dict(),
                    'best_loss': best_loss,
                    'loss': loss
                }                                                

                print(f"Saving model {filename}...")
                
                torch.save(checkpoint, filename)

                best_loss = loss_avg

        self.imB = None
        self.impB = torch.cat(self.impB, dim=0).detach().cpu().numpy()

        # Do a final save to update the loss function for all epochs
        checkpoint = torch.load(filename)
        checkpoint['loss'] = loss
        torch.save(checkpoint, filename)
        
        return self.imB, self.impB

    def synthesize(self):
                
        # Patchify the tensors if needed by unfolding the tensor with the provided patch size
        self.XYZ_LOS_patch = self.XYZ_LOS.unfold(2, self.patch_size, self.patch_size).unfold(1, self.patch_size, self.patch_size)
        self.XYZ_LOS_patch = self.XYZ_LOS_patch.permute(0,4,5,3,1,2)        

        self.omegapB_patch = self.omegapB.unfold(2, self.patch_size, self.patch_size).unfold(1, self.patch_size, self.patch_size)
        self.mask_patch = (~self.mask).unfold(1, self.patch_size, self.patch_size).unfold(0, self.patch_size, self.patch_size).float()
        self.weight_patch = self.weight.unfold(1, self.patch_size, self.patch_size).unfold(0, self.patch_size, self.patch_size)

        self.model.set_eval()
        
        with torch.no_grad():
            loss_avg = self.train_epoch(1, optimizer=False)
            
        self.imB = None
        self.impB = torch.cat(self.impB, dim=0).detach().cpu().numpy()
        # self.Ne_cube = torch.cat(self.Ne_cube, dim=0).detach().cpu().numpy()
        
        return self.imB, self.impB#, self.Ne_cube

if (__name__ == '__main__'):
    
    corona = CoronalTomography(gpu=0, checkpoint='model_time.pth')    

    corona.observations_and_reference_system('data', reduction=8, n_pixels_integration=64)

    corona.init_optimize(optimizer=False)
    corona.load_weights()

    imB, impB = corona.synthesize()