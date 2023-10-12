import model
import dataset
import coronal_tomography
import matplotlib.pyplot as pl


if __name__ == '__main__':
    pl.close('all')

    mode = 'single'
    obs = 'predsci_time'
    time = True

    if (obs == 'predsci'):
        if (mode == 'many'):
            w0_list = [5.0, 15.0, 30.0]
            n_hidden_list = [5, 8]
            dim_hidden_list = [128, 256]

            for w0 in w0_list:
                for n_hidden in n_hidden_list:
                    for dim_hidden in dim_hidden_list:
                    
                        hyperparameters = {
                            'time': False,
                            'type': 'siren',
                            'dim_hidden': dim_hidden,
                            'n_hidden' : n_hidden,
                            'w0_initial': w0,
                            'batch_size': 4,
                            'n_epochs': 150,
                            'patch_size': 32,
                            'lr': 3e-4,
                            'obs': 'predsci'        
                        }

                        corona = coronal_tomography.CoronalTomography(gpu=0, hyperparameters=hyperparameters)

                        corona.observations_and_reference_system(n_pixels_integration=64)

                        corona.init_optimize()
                        imB, impB = corona.optimize()

                        del corona

        if (mode == 'single'):
            hyperparameters = {
                'time': False,
                'type': 'siren',
                'dim_hidden': 128,
                'n_hidden' : 5,
                'w0_initial': 30.0,
                'batch_size': 4,
                'n_epochs': 150,
                'patch_size': 32,
                'lr': 3e-4,
                'obs': 'predsci'        
            }

            corona = coronal_tomography.CoronalTomography(gpu=0, hyperparameters=hyperparameters)

            corona.observations_and_reference_system(n_pixels_integration=64)

            corona.init_optimize()
            imB, impB = corona.optimize()

    if (obs == 'predsci_time'):

        if (mode == 'single'):
            hyperparameters = {
                'time': time,
                'type': 'siren',
                'dim_hidden': 256,
                'n_hidden' : 8,
                'w0_initial': [2.0, 15.0, 15.0, 15.0],
                'batch_size': 4,
                'n_epochs': 150,
                'patch_size': 32,
                'lr': 3e-4,
                'obs': 'predsci_time'        
            }

            corona = coronal_tomography.CoronalTomography(gpu=0, hyperparameters=hyperparameters)

            corona.observations_and_reference_system(n_pixels_integration=64)

            corona.init_optimize()
            imB, impB = corona.optimize()

    if (obs == 'lasco'):
        if (time):
            w0_initial = [1.0, 1.0, 1.0, 1.0]
        else:
            w0_initial = 10.0
        if (mode == 'single'):
            hyperparameters = {
                'time': time,
                'type': 'siren',
                'dim_hidden': 256,
                'n_hidden' : 8,
                'w0_initial': w0_initial,
                'batch_size': 4,
                'n_epochs': 150,
                'patch_size': 16,
                'lr': 3e-5,
                'obs': 'lasco'        
            }

            corona = coronal_tomography.CoronalTomography(gpu=0, hyperparameters=hyperparameters)
            
            corona.observations_and_reference_system(directory='datapB', reduction=4, n_pixels_integration=128)

            corona.init_optimize()

            corona.load_checkpoint(checkpoint='models/final/lasco_siren_h256_nh8_w10.0.pth')
            corona.load_weights()

            imB, impB = corona.optimize()