import model
import dataset
import coronal_tomography
import matplotlib.pyplot as pl


if __name__ == '__main__':
    pl.close('all')

    time = True

    if (time):
        w0_initial = [5.0, 10.0, 10.0, 10.0]
    else:
        w0_initial = 10.0
    
    hyperparameters = {
        'time': time,
        'type': 'siren',
        'dim_hidden': 256,
        'n_hidden' : 8,
        'w0_initial': w0_initial,
        'batch_size': 4,
        'n_epochs': 25,
        'patch_size': 16,
        'lr': 8e-5,
        'obs': 'lasco'        
    }

    corona = coronal_tomography.CoronalTomography(gpu=0, hyperparameters=hyperparameters)
    
    corona.observations_and_reference_system(directory='datapB', reduction=4, n_pixels_integration=128)

    corona.init_optimize()

    if (time):
        corona.load_checkpoint(checkpoint='models/final/lasco_siren_h256_nh8_w[5.0, 10.0, 10.0, 10.0].pth')
    else:
        corona.load_checkpoint(checkpoint='models/final/lasco_siren_h256_nh8_w10.0.pth')

    corona.load_weights()

    imB, impB = corona.optimize()