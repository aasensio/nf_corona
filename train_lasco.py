import model
import dataset
import coronal_tomography
import matplotlib.pyplot as pl


if __name__ == '__main__':
    pl.close('all')

    time = False

    if (time):
        w0_initial = [1.0, 1.0, 1.0, 1.0]
    else:
        w0_initial = 10.0
    
    hyperparameters = {
        'time': time,
        'type': 'siren',
        'dim_hidden': 256,
        'n_hidden' : 8,
        'w0_initial': w0_initial,
        'batch_size': 4,
        'n_epochs': 150,
        'patch_size': 32,
        'lr': 3e-4,
        'obs': 'lasco'        
    }

    corona = coronal_tomography.CoronalTomography(gpu=0, hyperparameters=hyperparameters)
    
    corona.observations_and_reference_system(directory='datapB', reduction=8, n_pixels_integration=64)

    corona.init_optimize()
    
    imB, impB = corona.optimize()