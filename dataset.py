import numpy as np
import torch
import torch.nn as nn
import torch.utils
from rotation import rotation
import glob
from astropy.io import fits
import datetime
import model
import corona_synth


class DatasetFITS(torch.utils.data.Dataset):
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
        super(DatasetFITS, self).__init__()


        # https://lasco-www.nrl.navy.mil/content/retrieve/polarize/2009_03/vig/c2/

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
            strdate = f"{ff[0].header['DATE-OBS']} {ff[0].header['TIME-OBS']}"            
            date = datetime.datetime.strptime(strdate, '%Y/%m/%d %H:%M:%S.%f')
            
            if (i == 0):
                self.angles[i] = 0.0
                date0 = date                
            else:
                deltat = date - date0
                # Calculate angle assuming rigid rotation (14.7 deg/day)           
                self.angles[i] = 14.7 * deltat.total_seconds() / (24 * 60.0 * 60.0)

            tmp = ff[0].data.astype('<f8')

            # This flip and transpose is needed to put the images in accordance with the
            # XYZ reference system!!!
            tmp = np.flipud(tmp.T)

            if (reduction):

                # Data is transformed to 1e-10 mean solar brightness units
                self.obspB[i] = torch.tensor(tmp[::reduction, ::reduction].astype('float32'))[None, :, :] / 1e-10

            # self.rsun[i] = ff[0].header['RSUN_PIX'] / reduction

            # Compute the FOV (in solar radii) by dividing the size in pixels by twice the radius of the Sun            
            self.FOV = tmp.shape[0] / 2.0 * ff[0].header['CDELT1']    # The FOV goes from [-x,x] in arcsec
            self.FOV /= 959.90                                        # FOV in solar radii            

            print(f'{i:03d} - {strdate} -> {self.angles[i]:7.3f} deg - {tmp.shape[0]} x {tmp.shape[1]} px -> {self.obspB[i].shape[1]} x {self.obspB[i].shape[2]} px - FOV={self.FOV}')

        # Number of pixels in X,Y,Z
        self.n_pixels = self.obspB[0].shape[1]
        
        # Unit vector along the Z axis which is used for the rotation
        self.u = torch.tensor([0.0, 0.0, 1.0])

        # Define rotation matrix that transforms the LOS reference system into the 
        # local reference system where we evaluate the electron density
        # This is equivalent to rotating the reference system by -alpha
        self.M = torch.zeros((self.n_files, 3, 3))
        for i in range(self.n_files):
            self.M[i, :, :] = rotation(self.u, self.angles[i])

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
        # ALSO RETURN COORDINATES TO AND FLATTEN ALL ANGLES
        return self.obspB[index, :], self.M[index, :], self.angles[index]
    
    def __len__(self):
        return self.n_training


class DatasetPREDSCI(torch.utils.data.Dataset):
    """
    Dataset class that will provide data during training. Modify it accordingly
    for your dataset. This one shows how to do augmenting during training for a 
    very simple training set    
    """
    def __init__(self, n_angles, device, FOV):
        """
        Main dataset

        Parameters
        ----------
        n_angles : int
            Number of angles to be used        
        """
        super(DatasetPREDSCI, self).__init__()

        self.FOV = FOV

        checkpoint = 'pred_sci_cr2204.pth'
        print("=> loading checkpoint '{}'".format(checkpoint))
        chk = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        print("=> loaded checkpoint '{}'".format(checkpoint))
        hyperparameters = chk['hyperparameters']

        self.model = model.INRModel(hyperparameters, device=device)
        self.model.load_weights(chk['state_dict'])

        angles = np.linspace(0.0, 180, n_angles)
        corona = corona_synth.CoronaSynthesis(n_pixels=64, n_pixels_integration=64, angles=angles, device=device, FOV=self.FOV, r_max=hyperparameters['r_max'])
        
        with torch.no_grad():
            imB, impB = corona.synth(self.model, patch_size=32)

        # Add Poisson noise
                
        impB /= 7.3e-12     # Transform to DN s-1 pix-1 using the calibrated PCF https://link.springer.com/article/10.1007/s11207-014-0635-2        
        
        impB_noisy = np.random.poisson(impB) * 7.3e-12

        self.obspB = torch.tensor(impB_noisy.astype('float32') / 1e-10)
                
        # Number of pixels in X,Y,Z
        self.n_pixels = self.obspB[0, :, :].shape[1]
        
        # Unit vector along the Z axis which is used for the rotation
        self.u = torch.tensor([0.0, 0.0, 1.0])

        # Rotation matrices
        self.M = corona.M
        self.angles = torch.tensor(angles.astype('float32'))
        
        # Number of training images
        self.n_training = n_angles
        
    def __getitem__(self, index):
        """
        Return each item in the training set

        Parameters
        ----------
        """
        
        return self.obspB[index, :], self.M[index, :], self.angles[index]
    
    def __len__(self):
        return self.n_training
    

class DatasetPREDSCI_Time(torch.utils.data.Dataset):
    """
    Dataset class that will provide data during training. Modify it accordingly
    for your dataset. This one shows how to do augmenting during training for a 
    very simple training set    
    """
    def __init__(self, n_angles, device, FOV):
        """
        Main dataset

        Parameters
        ----------
        n_angles : int
            Number of angles to be used        
        """
        super(DatasetPREDSCI_Time, self).__init__()

        self.FOV = FOV

        checkpoint = 'pred_sci_cr2204.pth'
        print("=> loading checkpoint '{}'".format(checkpoint))
        chk = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        print("=> loaded checkpoint '{}'".format(checkpoint))
        hyperparameters = chk['hyperparameters']

        self.model = model.INRModel(hyperparameters, device=device)
        self.model.load_weights(chk['state_dict'])

        checkpoint = 'pred_sci_cr2205.pth'
        print("=> loading checkpoint '{}'".format(checkpoint))
        chk = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        print("=> loaded checkpoint '{}'".format(checkpoint))
        hyperparameters = chk['hyperparameters']

        self.model2 = model.INRModel(hyperparameters, device=device)
        self.model2.load_weights(chk['state_dict'])

        angles = np.linspace(0.0, 180, n_angles)
        corona = corona_synth.CoronaSynthesis(n_pixels=64, n_pixels_integration=64, angles=angles, device=device, FOV=self.FOV, r_max=hyperparameters['r_max'])
        
        with torch.no_grad():
            imB, impB = corona.synth([self.model, self.model2], patch_size=32)

        # Add Poisson noise
                
        impB /= 7.3e-12     # Transform to DN s-1 pix-1 using the calibrated PCF https://link.springer.com/article/10.1007/s11207-014-0635-2        
        
        impB_noisy = np.random.poisson(impB) * 7.3e-12

        self.obspB = torch.tensor(impB_noisy.astype('float32') / 1e-10)
                
        # Number of pixels in X,Y,Z
        self.n_pixels = self.obspB[0, :, :].shape[1]
        
        # Unit vector along the Z axis which is used for the rotation
        self.u = torch.tensor([0.0, 0.0, 1.0])

        # Rotation matrices
        self.M = corona.M
        self.angles = torch.tensor(angles.astype('float32'))
        
        # Number of training images
        self.n_training = n_angles
        
    def __getitem__(self, index):
        """
        Return each item in the training set

        Parameters
        ----------
        """
        
        return self.obspB[index, :], self.M[index, :], self.angles[index]
    
    def __len__(self):
        return self.n_training