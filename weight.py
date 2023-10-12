import numpy as np
import matplotlib.pyplot as pl
import dataset

mask_size_min = 2.3
mask_size_max = 6.3

dset = dataset.DatasetFITS('datapB', reduction=4)
x = np.linspace(-dset.FOV, dset.FOV, dset.n_pixels, dtype='float32')
y = np.linspace(-dset.FOV, dset.FOV, dset.n_pixels, dtype='float32')
X, Y = np.meshgrid(x, y, indexing='ij')
p = np.sqrt(X**2 + Y**2)
mask_in = (p < mask_size_min)
mask_out = (p > mask_size_max)
mask = ~np.logical_or(mask_in, mask_out)

obspB = dset.obspB.numpy()

avg = np.mean(obspB, axis=0) * mask
std = np.std(obspB, axis=0) * mask

b = 5.5
weight = np.exp(p - b)
weight[p >= b] = 1.0
fig, ax = pl.subplots(nrows=3, ncols=3)
ax[0, 0].imshow(avg)
ax[0, 1].imshow(weight)
ax[1, 1].imshow(avg * weight)
ax[2, 1].imshow(avg * weight / p)
