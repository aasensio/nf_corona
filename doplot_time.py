from math import log
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.nn.init as init
import matplotlib.pyplot as pl
from tqdm import tqdm
import model
import dataset
import napari
import coronal_tomography
try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False

def spherical_caps(corona, model, r, n_pixels):
    
    n_caps = len(r)
    Ne_cap = [None] * n_caps

    # Spherical caps at different radial distances
    theta = np.linspace(0.0, np.pi, n_pixels, dtype='float32')
    phi = np.linspace(0.0, 2*np.pi, n_pixels, dtype='float32')
    Theta, Phi = np.meshgrid(theta, phi, indexing='ij')
    theta = torch.tensor(Theta).flatten()
    phi = torch.tensor(Phi).flatten()
    
    # Spherical caps
    for i in range(n_caps):        
        x = r[i] * torch.sin(theta) * torch.cos(phi)
        y = r[i] * torch.sin(theta) * torch.sin(phi)
        z = r[i] * torch.cos(theta)
        xyz = torch.cat([x[:, None], y[:, None], z[:, None]], dim=-1).to(corona.device)
    
        logNe = model(xyz[None, :, :] / 10.0)
        Ne_cap[i] = torch.exp(logNe).reshape((n_pixels, n_pixels)).detach().cpu().numpy()
        
    return Ne_cap

def longitudinal_cuts(corona, model, longitude, n_pixels):

    # Cuts at different longitudinal angles
    r = np.linspace(corona.mask_size_min, corona.mask_size_max, n_pixels, dtype='float32')
    theta = np.linspace(0, np.pi, n_pixels, dtype='float32')        
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
        xyz = torch.cat([x[:, None], y[:, None], z[:, None]], dim=-1).to(corona.device)
        
        logNe = model(xyz[None, :, :] / 10.0)
        Ne_long[i] = torch.exp(logNe).reshape((n_pixels, n_pixels)).detach().cpu().numpy()
        
    return Ne_long


def latitudinal_cuts(corona, model, latitude, n_pixels):

    # Cuts at different latitudinal angles
    r = np.linspace(corona.mask_size_min, corona.mask_size_max, n_pixels, dtype='float32')        
    phi = np.linspace(0.0, 2*np.pi, n_pixels, dtype='float32')
    R, Phi = np.meshgrid(r, phi, indexing='ij')
    r = torch.tensor(R).flatten()
    phi = torch.tensor(Phi).flatten()      
    
    n_lat = len(latitude)
    Ne_lat = [None] * n_lat

    # Longitudinal cuts
    for i in range(n_lat):        
        x = r * torch.sin(latitude[i] * np.pi / 180.0) * torch.cos(phi)
        y = r * torch.sin(latitude[i] * np.pi / 180.0) * torch.sin(phi)
        z = r * torch.cos(latitude[i] * np.pi / 180.0)
        xyz = torch.cat([x[:, None], y[:, None], z[:, None]], dim=-1).to(corona.device)
        
        logNe = model(xyz[None, :, :] / 10.0)
        Ne_lat[i] = torch.exp(logNe).reshape((n_pixels, n_pixels)).detach().cpu().numpy()
        
    return Ne_lat

def do_plot_emission(corona):        
    
    mask = (~self.mask).cpu().numpy()
    obspB = self.dataset.obspB.cpu().numpy()

    fig, ax = pl.subplots(nrows=8, ncols=6, figsize=(12,15), sharex='col')
    loop = 0
    for j in range(2):
        for i in range(8):
            im = ax[i, 3*j+0].imshow(np.log10(mask * obspB[loop, :, :]), vmin=np.log10(0.01), vmax=np.log10(60.0))
            pl.colorbar(im, ax=ax[i ,3*j+0])
            ax[i, 3*j+0].set_title(f'Obs - {self.dataset.angles[loop]:5.2f}')
            im = ax[i, 3*j+1].imshow(np.log10(mask * self.impB[loop, :, :]), vmin=np.log10(0.01), vmax=np.log10(60.0))
            pl.colorbar(im, ax=ax[i ,3*j+1])
            ax[i, 3*j+1].set_title(f'Syn - {self.dataset.angles[loop]:5.2f}')
            im = ax[i, 3*j+2].imshow(mask * obspB[loop, :, :] / self.impB[loop, :, :], vmin=0.2, vmax=5.0)
            pl.colorbar(im, ax=ax[i ,3*j+2])
            ax[i, 3*j+2].set_title(f'Ratio')
            loop += 3

    # pl.savefig('pB.png')

    fig, ax = pl.subplots(nrows=8, ncols=6, figsize=(12,15), sharex='col')
    loop = 0
    for j in range(2):
        for i in range(8):
            im = ax[i, 3*j+0].imshow(mask * obspB[loop, :, :], vmin=0.01, vmax=10.0)
            pl.colorbar(im, ax=ax[i ,3*j+0])
            ax[i, 3*j+0].set_title(f'Obs - {self.dataset.angles[loop]:5.2f}')
            im = ax[i, 3*j+1].imshow(mask * self.impB[loop, :, :], vmin=0.01, vmax=10.0)
            pl.colorbar(im, ax=ax[i ,3*j+1])
            ax[i, 3*j+1].set_title(f'Syn - {self.dataset.angles[loop]:5.2f}')
            im = ax[i, 3*j+2].imshow(mask * obspB[loop, :, :] / self.impB[loop, :, :], vmin=0.5, vmax=2.0)
            pl.colorbar(im, ax=ax[i ,3*j+2])
            ax[i, 3*j+2].set_title(f'Ratio')
            loop += 3

    # pl.savefig('pB.png')

def do_plot_cuts(corona, r, longitude, latitude, n_pixels):
    
    Ne_cap = spherical_caps(corona, corona.model, r, n_pixels)    
    Ne_cap_orig = spherical_caps(corona, corona.dataset.model, r, n_pixels)    
    
    Ne_long = longitudinal_cuts(corona, corona.model, longitude, n_pixels)
    Ne_long_orig = longitudinal_cuts(corona, corona.dataset.model, longitude, n_pixels)

    Ne_lat = latitudinal_cuts(corona, corona.model, latitude, n_pixels)
    Ne_lat_orig = latitudinal_cuts(corona, corona.dataset.model, latitude, n_pixels)

    fig, ax = pl.subplots(nrows=6, ncols=3, figsize=(12,10))

    for i in range(len(r)):
        im = ax[0, i].imshow(np.log10(Ne_cap[i]), extent=[0.0, 360.0, -90, 90], aspect='equal')
        pl.colorbar(im, ax=ax[0, i])
        ax[0, i].set_title(f'r={r[i]:4.1f}') 

    for i in range(len(r)):
        im = ax[1, i].imshow(np.log10(Ne_cap_orig[i]), extent=[0.0, 360.0, -90, 90], aspect='equal')
        pl.colorbar(im, ax=ax[1, i])
        ax[1, i].set_title(f'r={r[i]:4.1f}') 
    
    pl.ticklabel_format(useOffset=False)

    
    for i in range(len(longitude)):
        im = ax[2, i].imshow(np.log10(Ne_long[i]), extent=[-90.0, 90.0, corona.mask_size_max, corona.mask_size_min], aspect='auto')
        pl.colorbar(im, ax=ax[2, i])
        ax[2, i].set_title(f'lon={longitude[i]}') 

    for i in range(len(longitude)):
        im = ax[3, i].imshow(np.log10(Ne_long_orig[i]), extent=[-90.0, 90.0, corona.mask_size_max, corona.mask_size_min], aspect='auto')
        pl.colorbar(im, ax=ax[3, i])
        ax[3, i].set_title(f'lon={longitude[i]}') 

    pl.ticklabel_format(useOffset=False)
    
    
    for i in range(len(latitude)):
        im = ax[4, i].imshow(np.log10(Ne_lat[i]), extent=[0.0, 360.0, corona.mask_size_max, corona.mask_size_min], aspect='auto')
        pl.colorbar(im, ax=ax[4, i])
        ax[4, i].set_title(f'lon={latitude[i]}') 

    for i in range(len(latitude)):
        im = ax[5, i].imshow(np.log10(Ne_lat_orig[i]), extent=[0.0, 360.0, corona.mask_size_max, corona.mask_size_min], aspect='auto')
        pl.colorbar(im, ax=ax[5, i])
        ax[5, i].set_title(f'lon={latitude[i]}') 

    pl.ticklabel_format(useOffset=False)

if __name__ == '__main__':
    pl.close('all')

    corona = coronal_tomography.CoronalTomography(gpu=1, checkpoint='models/predsci_siren_h128_nh5_w30.0.pth')

    corona.observations_and_reference_system(n_pixels_integration=64)

    corona.init_optimize(optimizer=False)
    corona.load_weights()

    imB, impB = corona.synthesize()

    r = torch.tensor([3.15, 4.05, 5.95])
    long = torch.tensor([0.0, 90.0, 180.0])
    lat = torch.tensor([0.0, 30.0, 60.0])
    
    do_plot_cuts(corona, r, long, lat, n_pixels=64)

    # viewer = napari.view_image(Ne)
    # viewer = napari.view_image(Ne * (~corona.mask_3d[None,:,:,:]).cpu().numpy())

    
    # fig, ax = pl.subplots(nrows=7, ncols=7, figsize=(15, 15))

    # loop = 0

    # for i in range(7):
    #     for j in range(7):
    #         ax[i, j].imshow(np.log10(Ne[loop, :, :, 32]))
    #         loop += 1


    # breakpoint()

    
    # corona.do_plot_emission()
    # corona.do_plot_cuts(r, long, lat, n_pixels=64)