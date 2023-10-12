import matplotlib
matplotlib.use('TKAgg')
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
import matplotlib as mpl
from matplotlib.patches import Circle
try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False

##################################################
# Evaluate N_e in differente geometries
##################################################
def spherical_caps(corona, model, angles, r, n_pixels):
    """
    Evaluate the electron density at surfaces of constant radial distance
    """
    
    n_caps = len(angles)
    Ne_cap = [None] * n_caps

    # Spherical caps at different radial distances
    theta = np.linspace(0.0, np.pi, n_pixels, dtype='float32')
    phi = np.linspace(0.0, 2*np.pi, n_pixels, dtype='float32')
    Phi, Theta = np.meshgrid(phi, theta, indexing='ij')
    theta = torch.tensor(Theta).flatten()
    phi = torch.tensor(Phi).flatten()
    
    # Spherical caps
    for i in range(n_caps):        
        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * torch.cos(theta)                
        ang = torch.ones_like(z) * angles[i]
        xyz = torch.cat([x[:, None] / 10.0, y[:, None] / 10.0, z[:, None] / 10.0], dim=-1).to(corona.device)
        xyza = torch.cat([1.8 / 200.0 * ang[:, None] - 0.9, x[:, None] / 10.0, y[:, None] / 10.0, z[:, None] / 10.0], dim=-1).to(corona.device)
    
        if (isinstance(model, list)):
            logNe_1 = model[0](xyz[None, :, :])
            logNe_2 = model[1](xyz[None, :, :])
            t = angles[i] / 180.0            
            logNe = (1.0 - t) * logNe_1 + t * logNe_2
        else:
            logNe = model(xyza[None, :, :])
                
        Ne_cap[i] = torch.exp(logNe).reshape((n_pixels, n_pixels)).detach().cpu().numpy()
        
    return Phi, Theta, Ne_cap


def longitudinal_caps(corona, model, angles, longitude, n_pixels):
    """
    Evaluate the electron density at constant longitudes
    """
    # Cuts at different longitudinal angles
    
    n_long = len(angles)
    Ne_long = [None] * n_long

    r = np.linspace(2.3, 6.8, n_pixels, dtype='float32')
    theta = np.linspace(0.0, np.pi, n_pixels, dtype='float32')
    R, Theta = np.meshgrid(r, theta, indexing='ij')            
    r = torch.tensor(R).flatten()
    theta = torch.tensor(Theta).flatten()             

    # Longitudinal cuts
    for i in range(n_long):        
        x = r * torch.sin(theta) * torch.cos(longitude * np.pi / 180.0)
        y = r * torch.sin(theta) * torch.sin(longitude * np.pi / 180.0)        
        z = r * torch.cos(theta)
        ang = torch.ones_like(z) * angles[i]
        xyz = torch.cat([1.8 / 200.0 * ang[:, None] - 0.9, x[:, None] / 10.0, y[:, None] / 10.0, z[:, None] / 10.0], dim=-1).to(corona.device)
        
        logNe = model(xyz[None, :, :])
        Ne_long[i] = torch.exp(logNe).reshape((n_pixels, n_pixels)).detach().cpu().numpy()
        
    return Theta, R, Ne_long

def latitudinal_caps(corona, model, angles, latitude, n_pixels):
    """
    Evaluate the electron density at constant latitudes
    """
    
    n_lat = len(angles)
    Ne_lat = [None] * n_lat


    # Cuts at different latitudinal angles
    r = np.linspace(2.3, 6.8, n_pixels, dtype='float32')        
    phi = np.linspace(0.0, 2*np.pi, n_pixels, dtype='float32')
    R, Phi = np.meshgrid(r, phi, indexing='ij')
    r = torch.tensor(R).flatten()
    phi = torch.tensor(Phi).flatten()          
    
    # Longitudinal cuts
    for i in range(n_lat):        
        x = r * torch.sin(latitude * np.pi / 180.0) * torch.cos(phi)
        y = r * torch.sin(latitude * np.pi / 180.0) * torch.sin(phi)
        z = r * torch.cos(latitude * np.pi / 180.0)
        ang = torch.ones_like(z) * angles[i]
        xyz = torch.cat([x[:, None] / 10.0, y[:, None] / 10.0, z[:, None] / 10.0], dim=-1).to(corona.device)
        xyza = torch.cat([1.8 / 200.0 * ang[:, None] - 0.9, x[:, None] / 10.0, y[:, None] / 10.0, z[:, None] / 10.0], dim=-1).to(corona.device)

        if (isinstance(model, list)):
            logNe_1 = model[0](xyz[None, :, :])
            logNe_2 = model[1](xyz[None, :, :])
            t = angles[i] / 180.0            
            logNe = (1.0 - t) * logNe_1 + t * logNe_2
        else:        
            logNe = model(xyza[None, :, :])
        Ne_lat[i] = torch.exp(logNe).reshape((n_pixels, n_pixels)).detach().cpu().numpy()
        
    return Phi, R, Ne_lat

######################################################
# Do the actual plots
######################################################
def do_plot_radial(corona, model, r, n_pixels, ax, cmap, norm, label=''):
    
    n = len(ax)
    angles = torch.linspace(0.0, 180.0, n)
    t = angles / 14.7
    Phi, Theta, Ne_cap = spherical_caps(corona, model, angles, r, n_pixels)
    
    for i in range(n):
        
        # im = ax[i].imshow(np.log10(Ne_cap[i]), extent=[0.0, 360.0, -90, 90], aspect='auto', cmap=cmap[i], norm=norm[i])
        im = ax[i].pcolormesh(Phi * 180.0 / np.pi, Theta * 180.0 / np.pi, np.log10(Ne_cap[i]), cmap=cmap, norm=norm, edgecolors='face', rasterized=True)
        if (label != ''):
            ax[i].text(10, 150, f'{label}={t[i]:3.1f} days', color='white', weight='bold', size='large')

def do_plot_longitude_cart(corona, model, longitude, n_pixels, ax, cmap, norm, label=''):
    
    n = len(ax)
    angles = torch.linspace(0.0, 180.0, n)
    t = angles / 14.7
    theta, r, Ne_cap = longitudinal_caps(corona, model, angles, longitude, n_pixels)
    
    for i in range(n):
        ax[i].pcolormesh(theta * 180.0 / np.pi, r, np.log10(Ne_cap[i]), cmap=cmap, norm=norm, edgecolors='face', rasterized=True)
        if (label != ''):
            ax[i].text(10, 6, f'{label}={t[i]:3.1f} days', color='white', weight='bold', size='large')

def do_plot_latitude(corona, model, latitude, n_pixels, ax, cmap, norm, label=''):
    
    n = len(ax)
    angles = torch.linspace(0.0, 180.0, n)
    t = angles / 14.7
    theta, r, Ne_cap = latitudinal_caps(corona, model, angles, latitude, n_pixels)
    
    for i in range(n):
        ax[i].pcolormesh(theta * 180.0 / np.pi, r, np.log10(Ne_cap[i]), cmap=cmap, norm=norm, edgecolors='face', rasterized=True)
        if (label != ''):
            ax[i].text(20, 6.3, f'{label}={t[i]:3.1f} days', color='white', weight='bold', size='large')

##################################################
# LASCO
##################################################
def doplot_lasco(checkpoint, reduction=8, save=False, range=[1.5,5.0]):
    
    if (reduction == 8):
        n_pixels_integration = 64
        n_pixels = 64

    if (reduction == 4):
        n_pixels_integration = 128
        n_pixels = 128

    # Initialize the class
    corona = coronal_tomography.CoronalTomography(gpu=0, checkpoint=checkpoint)

    corona.observations_and_reference_system(directory='datapB', reduction=reduction, n_pixels_integration=n_pixels_integration)

    corona.init_optimize(optimizer=False)
    corona.load_weights()
        
    cmap = mpl.colormaps.get_cmap('viridis')
    norm = mpl.colors.Normalize(range[0], range[1])
            
    # Radial shells
    fig, ax = pl.subplots(nrows=3, ncols=3, figsize=(14.6,9), sharex=True, sharey=True, constrained_layout=True)    
    do_plot_radial(corona, corona.model, torch.tensor([3.15]), n_pixels=n_pixels, ax=ax.flat, cmap=cmap, norm=norm, label='t')
    im = mpl.cm.ScalarMappable(norm=norm)
    cbar = fig.colorbar(im, ax=ax, location='top')
    cbar.set_label("log N$_e$")

    fig.supxlabel('Longitude [deg]')
    fig.supylabel('Latitude [deg]')

    if (save):
        pl.savefig(f'figs/lasco_time_radialshells.pdf')


    fig, ax = pl.subplots(nrows=3, ncols=3, figsize=(14.6,9), sharex=True, sharey=True, constrained_layout=True)    
    do_plot_latitude(corona, corona.model, torch.tensor([90.0]), n_pixels=n_pixels, ax=ax.flat, cmap=cmap, norm=norm, label='t')
    im = mpl.cm.ScalarMappable(norm=norm)
    cbar = fig.colorbar(im, ax=ax, location='top')
    cbar.set_label("log N$_e$")

    fig.supxlabel('Longitude [deg]')
    fig.supylabel('r [R$_\odot$]')

    if (save):
        pl.savefig(f'figs/lasco_time_angularshells.pdf')

    # Vertical cuts
    # fig, ax = pl.subplots(nrows=3, ncols=2, figsize=(6,10), sharex=False, sharey=True, constrained_layout=True)

    # x = torch.tensor([-3, 0.0, 3.0])
    # do_plot_vertical_x(corona, corona.model, x, n_pixels=64, ax=ax[:, 0], cmap=cmap, norm=norm, label='x')

    # y = torch.tensor([-3, 0.0, 3.0])
    # do_plot_vertical_y(corona, corona.model, y, n_pixels=64, ax=ax[:, 1], cmap=cmap, norm=norm, label='y')

    # im = mpl.cm.ScalarMappable(norm=norm[0])
    # cbar = fig.colorbar(im, ax=ax, location='top')
    # cbar.set_label("log N$_e$")

    # ax[-1, 0].set_xlabel('y [R$_\odot$]')
    # ax[-1, 1].set_xlabel('x [R$_\odot$]')

    # fig.supylabel('z [R$_\odot$]')

    # if (save):
    #     pl.savefig(f'figs/lasco_verticalshells.pdf')
    
    # fig, ax = pl.subplots(nrows=1, ncols=3, constrained_layout=True, subplot_kw={'polar': True})
    # long = torch.tensor([0.0, 90, 180.0])
    # do_plot_longitude_polar(corona, corona.model, long, n_pixels=64, ax=ax, cmap=cmap, norm=norm, label='x')

##################################################
# LASCO
##################################################
def doplot_predsci(checkpoint, reduction=8, save=False, range=[1.5,5.0]):
    
    if (reduction == 8):
        n_pixels_integration = 64
        n_pixels = 64

    if (reduction == 4):
        n_pixels_integration = 128
        n_pixels = 128

    # Initialize the class
    corona = coronal_tomography.CoronalTomography(gpu=0, checkpoint=checkpoint)

    corona.observations_and_reference_system(reduction=reduction, n_pixels_integration=n_pixels_integration)

    corona.init_optimize(optimizer=False)
    corona.load_weights()
    
    cmap = mpl.colormaps.get_cmap('viridis')
    norm = mpl.colors.Normalize(range[0], range[1])
            
    # Radial shells
    fig, ax = pl.subplots(nrows=3, ncols=3, figsize=(14.6,9), sharex=True, sharey=True, constrained_layout=True)    
    do_plot_radial(corona, corona.model, torch.tensor([3.15]), n_pixels=n_pixels, ax=ax.flat, cmap=cmap, norm=norm, label='t')
    im = mpl.cm.ScalarMappable(norm=norm)
    cbar = fig.colorbar(im, ax=ax, location='top')
    cbar.set_label("log N$_e$")

    fig.supxlabel('Longitude [deg]')
    fig.supylabel('Latitude [deg]')

    if (save):
        pl.savefig(f'figs/predsci_time_radialshells_neural.pdf')

    fig, ax = pl.subplots(nrows=3, ncols=3, figsize=(14.6,9), sharex=True, sharey=True, constrained_layout=True)    
    do_plot_radial(corona, [corona.dataset.model, corona.dataset.model2], torch.tensor([3.15]), n_pixels=n_pixels, ax=ax.flat, cmap=cmap, norm=norm, label='t')
    im = mpl.cm.ScalarMappable(norm=norm)
    cbar = fig.colorbar(im, ax=ax, location='top')
    cbar.set_label("log N$_e$")

    fig.supxlabel('Longitude [deg]')
    fig.supylabel('Latitude [deg]')

    if (save):
        pl.savefig(f'figs/predsci_time_radialshells_original.pdf')


    fig, ax = pl.subplots(nrows=3, ncols=3, figsize=(14.6,9), sharex=True, sharey=True, constrained_layout=True)    
    do_plot_latitude(corona, corona.model, torch.tensor([90.0]), n_pixels=n_pixels, ax=ax.flat, cmap=cmap, norm=norm, label='t')
    im = mpl.cm.ScalarMappable(norm=norm)
    cbar = fig.colorbar(im, ax=ax, location='top')
    cbar.set_label("log N$_e$")

    fig.supxlabel('Longitude [deg]')
    fig.supylabel('r [R$_\odot$]')

    if (save):
        pl.savefig(f'figs/predsci_time_angularshells_neural.pdf')

    fig, ax = pl.subplots(nrows=3, ncols=3, figsize=(14.6,9), sharex=True, sharey=True, constrained_layout=True)    
    do_plot_latitude(corona, [corona.dataset.model, corona.dataset.model2], torch.tensor([90.0]), n_pixels=n_pixels, ax=ax.flat, cmap=cmap, norm=norm, label='t')
    im = mpl.cm.ScalarMappable(norm=norm)
    cbar = fig.colorbar(im, ax=ax, location='top')
    cbar.set_label("log N$_e$")

    fig.supxlabel('Longitude [deg]')
    fig.supylabel('r [R$_\odot$]')

    if (save):
        pl.savefig(f'figs/predsci_time_angularshells_original.pdf')
    

def doplot_lasco_synthesis(checkpoint, save=False):
    
    corona = coronal_tomography.CoronalTomography(gpu=0, checkpoint=checkpoint)

    corona.observations_and_reference_system(directory='datapB', reduction=4, n_pixels_integration=128)

    corona.init_optimize(optimizer=False)
    corona.load_weights()

    imB, impB_all = corona.synthesize()

    y = np.linspace(-6.8, 6.8, corona.dataset.n_pixels, dtype='float32')    
    z = np.linspace(-6.8, 6.8, corona.dataset.n_pixels, dtype='float32')
    Y, Z = np.meshgrid(y, z, indexing='ij')
    
    mask = (~corona.mask)[None, :, :]
    obspB = np.log10(mask * corona.dataset.obspB[::3, :, :] + 1e-13)
    impB = np.log10(mask * impB_all[::3, :, :] * 1e10 + 1e-13)
    angles = corona.dataset.angles[::3]
    loop = 0
    fov = corona.dataset.FOV

    fig, ax = pl.subplots(figsize=(16,16), nrows=6, ncols=8, sharex=False, sharey=False, constrained_layout=True)
    
    cmap = mpl.colormaps.get_cmap('viridis')
    norm = mpl.colors.Normalize(-1.0, 1.0)
    
    for i in range(6):
        ax[i, 0].pcolormesh(Y, Z, obspB[loop, :, :], cmap=cmap, norm=norm, edgecolors='face', rasterized=True)
        ax[i, 1].pcolormesh(Y, Z, impB[loop, :, :], cmap=cmap, norm=norm, edgecolors='face', rasterized=True)
        ax[i, 2].pcolormesh(Y, Z, obspB[loop, :, :] - impB[loop, :, :], cmap=cmap, norm=norm, edgecolors='face', rasterized=True)

        t1 = obspB[loop, :, :].cpu().numpy().flatten()
        t2 = impB[loop, :, :].cpu().numpy().flatten()
        ind = np.where(t1 > -13.0)
        ratio = 10**t1[ind] / 10**t2[ind]

        ax[i, 3].hist(ratio, bins=20, range=(0, 2), density=True, histtype='step', color='black')
        ax[i, 3].set_xlim(0, 2)
        ax[i, 3].text(0.6, 0.85, f'm={np.mean(ratio):4.1f}', size='small', transform=ax[i, 3].transAxes)
        ax[i, 3].text(0.6, 0.75, f'sd={np.std(ratio):4.1f}', size='small', transform=ax[i, 3].transAxes)

        ax[i, 0].text(-5.5, 4.5, f'{angles[loop]:4.1f} deg', color='white', weight='bold', size='large')

        if (i < 5):
            for j in range(8):
                ax[i, j].set_xticks([])
        for j in range(1, 8):
            ax[i, j].set_yticks([])

        loop += 1
    
    for i in range(6):
        ax[i, 4].pcolormesh(Y, Z, obspB[loop, :, :], cmap=cmap, norm=norm, edgecolors='face', rasterized=True)
        ax[i, 5].pcolormesh(Y, Z, impB[loop, :, :], cmap=cmap, norm=norm, edgecolors='face', rasterized=True)
        ax[i, 6].pcolormesh(Y, Z, obspB[loop, :, :] - impB[loop, :, :], cmap=cmap, norm=norm, edgecolors='face', rasterized=True)

        t1 = obspB[loop, :, :].cpu().numpy().flatten()
        t2 = impB[loop, :, :].cpu().numpy().flatten()
        ind = np.where(t1 > -13.0)
        ratio = 10**t1[ind] / 10**t2[ind]

        ax[i, 7].hist(ratio, bins=20, range=(0, 2), density=True, histtype='step', color='black')
        ax[i, 7].set_xlim(0, 2)
        ax[i, 7].text(0.6, 0.85, f'm={np.mean(ratio):4.1f}', size='small', transform = ax[i, 7].transAxes)
        ax[i, 7].text(0.6, 0.75, f'sd={np.std(ratio):4.1f}', size='small', transform = ax[i, 7].transAxes)

        ax[i, 4].text(-5.5, 4.5, f'{angles[loop]:4.1f} deg', color='white', weight='bold', size='large')

        if (i < 5):
            for j in range(8):
                ax[i, j].set_xticks([])
        for j in range(1, 8):
            ax[i, j].set_yticks([])

        loop += 1

    cax, kw = mpl.colorbar.make_axes(ax, location='top', fraction=0.05, shrink=0.5)
    im = mpl.cm.ScalarMappable(norm=norm)
    cbar = fig.colorbar(im, cax=cax, **kw)
    cbar.set_label(r"$\log_{10}$ I [$\times$10$^{-10}$ MSB]")

    fig.supylabel('y [R$_\odot$]')
    ax[-1, 1].set_xlabel('x [R$_\odot$]')
    ax[-1, 5].set_xlabel('x [R$_\odot$]')

    ax[0, 0].set_title('LASCO')
    ax[0, 1].set_title('SYN')
    ax[0, 2].set_title('LASCO - SYN')
    ax[0, 3].set_title('LASCO/SYN')

    ax[0, 4].set_title('LASCO')    
    ax[0, 5].set_title('SYN')    
    ax[0, 6].set_title('LASCO - SYN')
    ax[0, 7].set_title('LASCO/SYN')

    if (save):
        pl.savefig('figs/lasco_time_emission.pdf')

    return corona, impB_all

def doplot_predsci_synthesis(checkpoint, save=False):
    
    corona = coronal_tomography.CoronalTomography(gpu=0, checkpoint=checkpoint)

    corona.observations_and_reference_system(reduction=4, n_pixels_integration=128)

    corona.init_optimize(optimizer=False)
    corona.load_weights()

    imB, impB_all = corona.synthesize()

    y = np.linspace(-6.8, 6.8, corona.dataset.n_pixels, dtype='float32')    
    z = np.linspace(-6.8, 6.8, corona.dataset.n_pixels, dtype='float32')
    Y, Z = np.meshgrid(y, z, indexing='ij')
    
    mask = (~corona.mask)[None, :, :]
    obspB = np.log10(mask * corona.dataset.obspB[::3, :, :] + 1e-13)
    impB = np.log10(mask * impB_all[::3, :, :] * 1e10 + 1e-13)
    angles = corona.dataset.angles[::3]
    loop = 0
    fov = corona.dataset.FOV

    fig, ax = pl.subplots(figsize=(16,16), nrows=6, ncols=8, sharex=False, sharey=False, constrained_layout=True)
    
    cmap = mpl.colormaps.get_cmap('viridis')
    norm = mpl.colors.Normalize(-1.0, 1.0)
    
    for i in range(6):
        ax[i, 0].pcolormesh(Y, Z, obspB[loop, :, :], cmap=cmap, norm=norm, edgecolors='face', rasterized=True)
        ax[i, 1].pcolormesh(Y, Z, impB[loop, :, :], cmap=cmap, norm=norm, edgecolors='face', rasterized=True)
        ax[i, 2].pcolormesh(Y, Z, obspB[loop, :, :] - impB[loop, :, :], cmap=cmap, norm=norm, edgecolors='face', rasterized=True)

        t1 = obspB[loop, :, :].cpu().numpy().flatten()
        t2 = impB[loop, :, :].cpu().numpy().flatten()
        ind = np.where(t1 > -13.0)
        ratio = 10**t1[ind] / 10**t2[ind]

        ax[i, 3].hist(ratio, bins=20, range=(0, 2), density=True, histtype='step', color='black')
        ax[i, 3].set_xlim(0, 2)
        ax[i, 3].text(0.6, 0.85, f'm={np.mean(ratio):4.1f}', size='small', transform=ax[i, 3].transAxes)
        ax[i, 3].text(0.6, 0.75, f'sd={np.std(ratio):4.1f}', size='small', transform=ax[i, 3].transAxes)

        ax[i, 0].text(-5.5, 4.5, f'{angles[loop]:4.1f} deg', color='white', weight='bold', size='large')

        if (i < 5):
            for j in range(8):
                ax[i, j].set_xticks([])
        for j in range(1, 8):
            ax[i, j].set_yticks([])

        loop += 1
    
    for i in range(6):
        ax[i, 4].pcolormesh(Y, Z, obspB[loop, :, :], cmap=cmap, norm=norm, edgecolors='face', rasterized=True)
        ax[i, 5].pcolormesh(Y, Z, impB[loop, :, :], cmap=cmap, norm=norm, edgecolors='face', rasterized=True)
        ax[i, 6].pcolormesh(Y, Z, obspB[loop, :, :] - impB[loop, :, :], cmap=cmap, norm=norm, edgecolors='face', rasterized=True)

        t1 = obspB[loop, :, :].cpu().numpy().flatten()
        t2 = impB[loop, :, :].cpu().numpy().flatten()
        ind = np.where(t1 > -13.0)
        ratio = 10**t1[ind] / 10**t2[ind]

        ax[i, 7].hist(ratio, bins=20, range=(0, 2), density=True, histtype='step', color='black')
        ax[i, 7].set_xlim(0, 2)
        ax[i, 7].text(0.6, 0.85, f'm={np.mean(ratio):4.1f}', size='small', transform = ax[i, 7].transAxes)
        ax[i, 7].text(0.6, 0.75, f'sd={np.std(ratio):4.1f}', size='small', transform = ax[i, 7].transAxes)

        ax[i, 4].text(-5.5, 4.5, f'{angles[loop]:4.1f} deg', color='white', weight='bold', size='large')

        if (i < 5):
            for j in range(8):
                ax[i, j].set_xticks([])
        for j in range(1, 8):
            ax[i, j].set_yticks([])

        loop += 1

    cax, kw = mpl.colorbar.make_axes(ax, location='top', fraction=0.05, shrink=0.5)
    im = mpl.cm.ScalarMappable(norm=norm)
    cbar = fig.colorbar(im, cax=cax, **kw)
    cbar.set_label(r"$\log_{10}$ I [$\times$10$^{-10}$ MSB]")

    fig.supylabel('y [R$_\odot$]')
    ax[-1, 1].set_xlabel('x [R$_\odot$]')
    ax[-1, 5].set_xlabel('x [R$_\odot$]')

    ax[0, 0].set_title('LASCO')
    ax[0, 1].set_title('SYN')
    ax[0, 2].set_title('LASCO - SYN')
    ax[0, 3].set_title('LASCO/SYN')

    ax[0, 4].set_title('LASCO')    
    ax[0, 5].set_title('SYN')    
    ax[0, 6].set_title('LASCO - SYN')
    ax[0, 7].set_title('LASCO/SYN')

    if (save):
        pl.savefig('figs/lasco_time_emission.pdf')

    return corona, impB_all

if __name__ == '__main__':
    pl.close('all')

    # PREDSCI    
    # filename = 'models/predsci_time_siren_h256_nh8_w[2.0, 15.0, 15.0, 15.0].pth'
    # doplot_predsci(filename, reduction=4, save=True, range=[4.0, 5.8])
    # corona, impB = doplot_predsci_synthesis(filename, save=False)

    # LASCO    
    filename = "models/lasco_siren_h256_nh8_w[5.0, 10.0, 10.0, 10.0].pth"
    #doplot_lasco(filename, reduction=4, save=False, range=[1.5,5.0])
    corona, impB = doplot_lasco_synthesis(filename, save=True)

    # viewer = napari.view_image(impB)