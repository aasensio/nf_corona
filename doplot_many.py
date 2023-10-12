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
def spherical_caps(corona, model, r, n_pixels):
    """
    Evaluate the electron density at surfaces of constant radial distance
    """
    
    n_caps = len(r)
    Ne_cap = [None] * n_caps

    # Spherical caps at different radial distances
    theta = np.linspace(0.0, np.pi, n_pixels, dtype='float32')    
    phi = np.linspace(0.0, 2*np.pi, n_pixels, dtype='float32')
    Phi, Theta = np.meshgrid(phi, theta, indexing='ij')
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
        
    return Phi, Theta, Ne_cap

def polar_caps(corona, model, r, phi, n_pixels):
    """
    Evaluate the electron density along a certain meridian at distance r and latitude phi
    """
    n_caps = len(r)
    Ne_cap = [None] * n_caps

    # Spherical caps at different radial distances
    theta = np.linspace(0.0, np.pi, n_pixels, dtype='float32')    
    theta = torch.tensor(theta)
    
    # Spherical caps
    for i in range(n_caps):        
        x = r[i] * torch.sin(theta) * torch.cos(phi)
        y = r[i] * torch.sin(theta) * torch.sin(phi)
        z = r[i] * torch.cos(theta)
        xyz = torch.cat([x[:, None], y[:, None], z[:, None]], dim=-1).to(corona.device)
    
        logNe = model(xyz[None, :, :] / 10.0)
        Ne_cap[i] = torch.exp(logNe).reshape((n_pixels)).detach().cpu().numpy()
        
    return Ne_cap

def vertical_x_caps(corona, model, x_cap, n_pixels):
    """
    Evaluate the electron density at vertical cuts along constant X planes
    """
    n_caps = len(x_cap)
    Ne_cap = [None] * n_caps

    # Spherical caps at different radial distances
    y = np.linspace(-6.8, 6.8, n_pixels, dtype='float32')    
    z = np.linspace(-6.8, 6.8, n_pixels, dtype='float32')
    Y, Z = np.meshgrid(y, z, indexing='ij')
    Y = torch.tensor(Y).flatten()
    Z = torch.tensor(Z).flatten()
    
    # Spherical caps
    for i in range(n_caps):        
        x = torch.ones_like(Y) * x_cap[i]        
        xyz = torch.cat([x[:, None], Y[:, None], Z[:, None]], dim=-1).to(corona.device)
    
        logNe = model(xyz[None, :, :] / 10.0)
        Ne_cap[i] = torch.exp(logNe).reshape((n_pixels, n_pixels)).detach().cpu().numpy()
        
    return Y.reshape((n_pixels, n_pixels)), Z.reshape((n_pixels, n_pixels)), Ne_cap

def vertical_y_caps(corona, model, y_cap, n_pixels):
    """
    Evaluate the electron density at vertical cuts along constant Y planes
    """
    n_caps = len(y_cap)
    Ne_cap = [None] * n_caps

    # Spherical caps at different radial distances
    x = np.linspace(-6.8, 6.8, n_pixels, dtype='float32')    
    z = np.linspace(-6.8, 6.8, n_pixels, dtype='float32')
    X, Z = np.meshgrid(x, z, indexing='ij')
    X = torch.tensor(X).flatten()
    Z = torch.tensor(Z).flatten()
    
    # Spherical caps
    for i in range(n_caps):        
        y = torch.ones_like(X) * y_cap[i]
        xyz = torch.cat([X[:, None], y[:, None], Z[:, None]], dim=-1).to(corona.device)
    
        logNe = model(xyz[None, :, :] / 10.0)
        Ne_cap[i] = torch.exp(logNe).reshape((n_pixels, n_pixels)).detach().cpu().numpy()
        
    return X.reshape((n_pixels, n_pixels)), Z.reshape((n_pixels, n_pixels)), Ne_cap

def longitudinal_caps(corona, model, longitude, n_pixels):
    """
    Evaluate the electron density at constant longitudes
    """
    # Cuts at different longitudinal angles
    r = np.linspace(2.3, 6.8, n_pixels, dtype='float32')
    theta = np.linspace(0.0, np.pi, n_pixels, dtype='float32')
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
        
    return Theta, R, Ne_long


def latitudinal_caps(corona, model, latitude, n_pixels):
    """
    Evaluate the electron density at constant latitudes
    """
    # Cuts at different latitudinal angles
    r = np.linspace(2.3, 6.8, n_pixels, dtype='float32')        
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
        
    return Phi, R, Ne_lat

######################################################
# Do the actual plots
######################################################
def do_plot_radial(corona, model, r, n_pixels, ax, cmap, norm, label=''):
    
    Phi, Theta, Ne_cap = spherical_caps(corona, model, r, n_pixels)
    
    for i in range(len(r)):
        # im = ax[i].imshow(np.log10(Ne_cap[i]), extent=[0.0, 360.0, -90, 90], aspect='auto', cmap=cmap[i], norm=norm[i])
        im = ax[i].pcolormesh(Phi * 180.0 / np.pi, Theta * 180.0 / np.pi, np.log10(Ne_cap[i]), cmap=cmap[i], norm=norm[i], edgecolors='face', rasterized=True)
        if (label != ''):
            ax[i].text(10, 165, f'{label}={r[i]:3.1f} R$_\odot$', color='white', weight='bold', size='large')

def do_plot_polar(corona, model, r, phi, n_pixels, ax, cmap, norm, label=''):
    
    theta = np.linspace(0.0, np.pi, n_pixels)
    Ne_cap = polar_caps(corona, model, r, phi, n_pixels)
    
    for i in range(len(r)):
        ax.plot(theta * 180.0 / np.pi, np.log10(Ne_cap[i]))
        
        # if (label != ''):
            # ax[i].text(-6.5, 5.5, f'{label}={x_cap[i]:3.1f}', color='white', weight='bold', size='large')

def do_plot_vertical_x(corona, model, x_cap, n_pixels, ax, cmap, norm, label=''):
    
    Y, Z, Ne_cap = vertical_x_caps(corona, model, x_cap, n_pixels)
    
    for i in range(len(x_cap)):
        # im = ax[i].imshow((~corona.mask) * np.log10(Ne_cap[i]), extent=[-7, 7, -7, 7], aspect='equal', cmap=cmap[i], norm=norm[i])        
        im = ax[i].pcolormesh(Y, Z, (~corona.mask) * np.log10(Ne_cap[i]), cmap=cmap[i], norm=norm[i], edgecolors='face', rasterized=True)
        circ = Circle((0, 0), 1.0, color='yellow')
        ax[i].add_patch(circ)
        if (label != ''):
            ax[i].text(-6.5, 5.5, f'{label}={x_cap[i]:3.1f} R$_\odot$', color='white', weight='bold', size='large')

def do_plot_vertical_y(corona, model, y_cap, n_pixels, ax, cmap, norm, label=''):
    
    X, Z, Ne_cap = vertical_y_caps(corona, model, y_cap, n_pixels)
    
    for i in range(len(y_cap)):
        # im = ax[i].imshow((~corona.mask) * np.log10(Ne_cap[i]), extent=[-7, 7, -7, 7], aspect='equal', cmap=cmap[i], norm=norm[i])
        im = ax[i].pcolormesh(X, Z, (~corona.mask) * np.log10(Ne_cap[i]), cmap=cmap[i], norm=norm[i], edgecolors='face', rasterized=True)
        circ = Circle((0, 0), 1.0, color='yellow')
        ax[i].add_patch(circ)
        if (label != ''):
            ax[i].text(-6.5, 5.5, f'{label}={y_cap[i]:3.1f} R$_\odot$', color='white', weight='bold', size='large')


def do_plot_longitude_polar(corona, model, longitude, n_pixels, ax, cmap, norm, label=''):
    
    theta, r, Ne_cap = longitudinal_caps(corona, model, longitude, n_pixels)
    
    for i in range(len(longitude)):
        ax[i].pcolormesh(theta, r, np.log10(Ne_cap[i]), edgecolors='face', rasterized=True)
        # im = ax[i].imshow(np.log10(Ne_cap[i]), extent=[2.3, 6.8, 180.0, 0.0], aspect='auto', cmap=cmap[i], norm=norm[i])        
        # ax[i].set_theta_zero_location('N')
        # ax[i].set_theta_direction(-1)
        if (label != ''):
            ax[i].text(25, 60, f'{label}', color='white', weight='bold', size='large')

def do_plot_longitude_cart(corona, model, longitude, n_pixels, ax, cmap, norm, label=''):
    
    theta, r, Ne_cap = longitudinal_caps(corona, model, longitude, n_pixels)
    
    for i in range(len(longitude)):        
        im = ax[i].pcolormesh(theta * 180.0 / np.pi, r, np.log10(Ne_cap[i]), cmap=cmap[i], norm=norm[i], edgecolors='face', rasterized=True)
        if (label != ''):
            ax[i].text(10, 6.3, f'{label}={longitude[i]:4.1f} deg', color='white', weight='bold', size='large')

def do_plot_latitude(corona, model, latitude, n_pixels, ax, cmap, norm, label=''):
    
    phi, r, Ne_cap = latitudinal_caps(corona, model, latitude, n_pixels)
    
    for i in range(len(latitude)):
        im = ax[i].pcolormesh(phi * 180.0 / np.pi, r, np.log10(Ne_cap[i]), cmap=cmap[i], norm=norm[i], edgecolors='face', rasterized=True)
        if (label != ''):
            ax[i].text(20, 6.3, f'{label}={latitude[i]:4.1f} deg', color='white', weight='bold', size='large')

##################################################
# Put all plots together
##################################################
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

##################################################
# PREDSCI
##################################################
def doplot_radial_all():

    n_hidden_list = [5, 8]
    dim_hidden_list = [128, 256]

    for n_hidden in n_hidden_list:
        for dim_hidden in dim_hidden_list:

            corona = coronal_tomography.CoronalTomography(gpu=1, checkpoint=f'models/predsci_siren_h{dim_hidden}_nh{n_hidden}_w5.0.pth')

            corona.observations_and_reference_system(n_pixels_integration=64)

            corona.init_optimize(optimizer=False)
            corona.load_weights()
            
            r = torch.tensor([3.15, 4.05, 5.95])
            long = torch.tensor([0.0, 90.0, 180.0])
            lat = torch.tensor([0.0, 30.0, 60.0])

            cmap = [None] * 3
            norm = [None] * 3
            im = [None] * 3

            cmap[0] = mpl.cm.get_cmap('viridis')
            norm[0] = mpl.colors.Normalize(4.5, 5.7)
            
            cmap[1] = mpl.cm.get_cmap('viridis')
            norm[1] = mpl.colors.Normalize(4, 5.3)

            cmap[2] = mpl.cm.get_cmap('viridis')
            norm[2] = mpl.colors.Normalize(3, 4.7)
                
            fig, ax = pl.subplots(nrows=4, ncols=3, figsize=(12,10), sharex=True, sharey=True, constrained_layout=True)

            do_plot_radial(corona, corona.dataset.model, r, n_pixels=64, ax=ax[0, :], cmap=cmap, norm=norm)
            do_plot_radial(corona, corona.model, r, n_pixels=64, ax=ax[1, :], cmap=cmap, norm=norm)
            for i in range(3):
                ax[1, i].text(25, 160, 'w$_0$=5', color='white', weight='bold', size='large')

            corona.load_checkpoint(checkpoint=f'models/predsci_siren_h{dim_hidden}_nh{n_hidden}_w15.0.pth')
            corona.init_optimize(optimizer=False)
            corona.load_weights()
            do_plot_radial(corona, corona.model, r, n_pixels=64, ax=ax[2, :], cmap=cmap, norm=norm)
            for i in range(3):
                ax[2, i].text(25, 160, 'w$_0$=15', color='white', weight='bold', size='large')

            corona.load_checkpoint(checkpoint=f'models/predsci_siren_h{dim_hidden}_nh{n_hidden}_w30.0.pth')
            corona.init_optimize(optimizer=False)
            corona.load_weights()
            do_plot_radial(corona, corona.model, r, n_pixels=64, ax=ax[3, :], cmap=cmap, norm=norm)
            for i in range(3):
                ax[3, i].text(25, 160, 'w$_0$=30', color='white', weight='bold', size='large')

            for i in range(3):
                ax[0, i].text(25, 160, f'r={r[i]:4.2f} R$_\odot$', color='white', weight='bold', size='large')
                if (i == 0):
                    ax[0, i].text(25, 140, f'(n,h)={n_hidden,dim_hidden}', color='white', weight='bold', size='large')

            for i in range(3):
                cax, kw = mpl.colorbar.make_axes([ax for ax in ax[:, i].flat], location='top')    
                im = mpl.cm.ScalarMappable(norm=norm[i])
                cbar = pl.colorbar(im, cax=cax, **kw)
                cbar.set_label("log N$_e$")

            fig.supxlabel('Longitude [deg]')
            fig.supylabel('Latitude [deg]')

            pl.ticklabel_format(useOffset=False)

            pl.savefig(f'figs/predsci_radial_h{dim_hidden}_nh{n_hidden}_wchange.pdf')


def doplot_angular():
    dim_hidden = 256
    n_hidden = 8

    corona = coronal_tomography.CoronalTomography(gpu=1, checkpoint=f'models/predsci_siren_h{dim_hidden}_nh{n_hidden}_w15.0.pth')

    corona.observations_and_reference_system(n_pixels_integration=64)

    corona.init_optimize(optimizer=False)
    corona.load_weights()
        
    long = torch.tensor([0.0, 45.0, 90.0])
    lat = torch.tensor([0.0, 30.0, 60.0])

    cmap = [None] * 3
    norm = [None] * 3
    im = [None] * 3

    fig = pl.figure(figsize=(10,11), constrained_layout=True)
    fig1, fig2 = fig.subfigures(nrows=2, ncols=1)
    
    # Subfigure 1    
    cmap[0] = mpl.cm.get_cmap('viridis')
    norm[0] = mpl.colors.Normalize(3.7, 5.1)
    
    cmap[1] = mpl.cm.get_cmap('viridis')
    norm[1] = mpl.colors.Normalize(4, 5.3)

    cmap[2] = mpl.cm.get_cmap('viridis')
    norm[2] = mpl.colors.Normalize(3, 5.2)
            
    ax1 = fig1.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
    
    do_plot_latitude(corona, corona.dataset.model, lat, n_pixels=64, ax=ax1[0, :], cmap=cmap, norm=norm, label='lat')    
    do_plot_latitude(corona, corona.model, lat, n_pixels=64, ax=ax1[1, :], cmap=cmap, norm=norm)

    # for i in range(3):
        # ax1[0, i].text(20, , f'lat={long[i]:4.2f}', color='white', weight='bold', size='large')        

    for i in range(3):
        cax, kw = mpl.colorbar.make_axes([ax for ax in ax1[:, i].flat], location='top')    
        im = mpl.cm.ScalarMappable(norm=norm[i])
        cbar = fig1.colorbar(im, cax=cax, **kw)
        cbar.set_label("log N$_e$")

    fig1.supxlabel('Longitude [deg]')
    fig1.supylabel('r [R$_\odot$]')
    
    # Subfigure 2
    cmap[0] = mpl.cm.get_cmap('viridis')
    norm[0] = mpl.colors.Normalize(3.7, 5.8)
    
    cmap[1] = mpl.cm.get_cmap('viridis')
    norm[1] = mpl.colors.Normalize(4, 5.8)

    cmap[2] = mpl.cm.get_cmap('viridis')
    norm[2] = mpl.colors.Normalize(3.5, 5.5)
    ax2 = fig2.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
    
    do_plot_longitude_cart(corona, corona.dataset.model, long, n_pixels=64, ax=ax2[0, :], cmap=cmap, norm=norm, label='lon')    
    do_plot_longitude_cart(corona, corona.model, long, n_pixels=64, ax=ax2[1, :], cmap=cmap, norm=norm)
    
    # for i in range(3):        
        # ax2[0, i].text(-75, 5, f'lon={lat[i]:4.2f}', color='white', weight='bold', size='large')
        
    for i in range(3):
        cax, kw = mpl.colorbar.make_axes([ax for ax in ax2[:, i].flat], location='top')    
        im = mpl.cm.ScalarMappable(norm=norm[i])
        cbar = fig2.colorbar(im, cax=cax, **kw)
        cbar.set_label("log N$_e$")

    fig2.supxlabel('Latitude [deg]')
    fig2.supylabel('r [R$_\odot$]')

    pl.ticklabel_format(useOffset=False)


    fig.savefig(f'figs/predsci_angular_h{dim_hidden}_nh{n_hidden}.pdf')

##################################################
# LASCO
##################################################
def doplot_lasco(checkpoint, w0, reduction=8, save=False):

    if (reduction == 8):
        n_pixels_integration = 64
        n_pixels = 64

    if (reduction == 4):
        n_pixels_integration = 128
        n_pixels = 128

    n_models = len(checkpoint)

    fig, ax = pl.subplots(nrows=n_models, ncols=3, figsize=(14.6,3*n_models), sharex=True, sharey=True, constrained_layout=True, squeeze=False)    
    fig2, ax2 = pl.subplots(nrows=n_models, ncols=3, figsize=(14.6,3*n_models), sharex=True, sharey=True, constrained_layout=True, squeeze=False)    

    for i in range(n_models):
    
    # Initialize the class
        corona = coronal_tomography.CoronalTomography(gpu=0, checkpoint=checkpoint[i])

        corona.observations_and_reference_system(directory='datapB', reduction=reduction, n_pixels_integration=n_pixels_integration)

        corona.init_optimize(optimizer=False)
        corona.load_weights()
        
        r = torch.tensor([3.15, 4.05, 5.95])

        cmap = [None] * 3
        norm = [None] * 3
        im = [None] * 3

        cmap[0] = mpl.cm.get_cmap('viridis')
        norm[0] = mpl.colors.Normalize(1.5, 5.0)
        
        cmap[1] = cmap[0]
        norm[1] = norm[0]

        cmap[2] = cmap[0]
        norm[2] = norm[0]
        
    # Radial shells

        if (i == 0):
            label = 'r'
        else:
            label = ''
        do_plot_radial(corona, corona.model, r, n_pixels=n_pixels, ax=ax[i, :], cmap=cmap, norm=norm, label=label)
        
        if (i == 0):
            im = mpl.cm.ScalarMappable(norm=norm[0])
            cbar = fig.colorbar(im, ax=ax[i,:], location='top')
            cbar.set_label("log N$_e$")

        fig.supxlabel('Longitude [deg]')
        fig.supylabel('Latitude [deg]')

        ax[i, 0].text(10, 145, f'w$_0$={w0[i]}', color='white', weight='bold', size='large')
        

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
    
    
        lat = torch.tensor([30.0, 60.0, 90.0])

        if (i == 0):
            label = 'lat'
        else:
            label = ''

        do_plot_latitude(corona, corona.model, lat, n_pixels=n_pixels, ax=ax2[i, :], cmap=cmap, norm=norm, label=label)
        if (i == 0):
            im = mpl.cm.ScalarMappable(norm=norm[0])
            cbar = fig2.colorbar(im, ax=ax2[i, :], location='top')
            cbar.set_label("log N$_e$")

        fig2.supylabel('r [R$_\odot$]')
        fig2.supxlabel('Longitude [deg]')

        ax2[i, 0].text(20, 5.8, f'w$_0$={w0[i]}', color='white', weight='bold', size='large')
        
    if (save):
        fig.savefig(f'figs/lasco_radialshells.pdf')
        fig2.savefig(f'figs/lasco_latitudinalcut.pdf')
    

def doplot_lasco_synthesis(checkpoint, reduction=8, save=False):

    if (reduction == 8):
        n_pixels_integration = 64
        n_pixels = 64

    if (reduction == 4):
        n_pixels_integration = 128
        n_pixels = 128
    
    corona = coronal_tomography.CoronalTomography(gpu=0, checkpoint=checkpoint)

    corona.observations_and_reference_system(directory='datapB', reduction=reduction, n_pixels_integration=n_pixels_integration)

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
    
    cmap = mpl.cm.get_cmap('viridis')
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
        pl.savefig('figs/lasco_emission.pdf')

    return corona, impB_all

    # delta = 


if __name__ == '__main__':
    pl.close('all')

    # PREDSCI
    # doplot_radial_all()
    # doplot_angular()

    # LASCO    
    # filenames = ['models/lasco_siren_h256_nh8_w5.0.pth', 'models/lasco_siren_h256_nh8_w10.0.pth', 'models/lasco_siren_h256_nh8_w30.0.pth']
    # w0 = ['5', '10', '30']
    # doplot_lasco(filenames, w0, save=False)
    # corona, impB = doplot_lasco_synthesis(filenames[1], save=True)


    filenames = ['models/lasco_siren_h256_nh8_w10.0.pth']
    w0 = ['10']
    # doplot_lasco(filenames, w0, reduction=4, save=False)
    corona, impB = doplot_lasco_synthesis(filenames[0], reduction=4, save=True)

    # viewer = napari.view_image(impB)