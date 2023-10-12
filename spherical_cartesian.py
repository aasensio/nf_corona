import numpy as np
import matplotlib.pyplot as pl

y = np.linspace(0, 1, 100)
z = np.linspace(-1, 1, 100)

Y, Z = np.meshgrid(y, z, indexing='ij')

r = np.linspace(0, 1.0, 100)
theta = np.linspace(0.0, np.pi, 100)

Theta, R = np.meshgrid(theta, r, indexing='ij')

Theta = 2*np.pi - Theta

Y2 = R * np.sin(Theta)
Z2 = R * np.cos(Theta)

pl.close('all')
fig, ax = pl.subplots(nrows=2, ncols=1)
imY = ax[0].imshow(Y)
fig.colorbar(imY, ax=ax[0])
ax[0].set_title('Y')
imZ = ax[1].imshow(Z)
ax[1].set_title('Z')
fig.colorbar(imZ, ax=ax[1])


fig, ax = pl.subplots(nrows=2, ncols=1, subplot_kw={'polar': True})
ax[0].pcolormesh(Theta, R, Y2)
ax[0].set_title('Y2')

ax[1].pcolormesh(Theta, R, Z2)
ax[1].set_title('Z2')
# pl.tight_layout()
