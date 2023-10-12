import numpy as np
import torch
import matplotlib.pyplot as pl
from modules.Siren import SirenNet
from tqdm import tqdm
from rotation import rotation


siren = SirenNet(dim_in=3, dim_hidden=8, dim_out=1, num_layers=3)


# def electron(x, y, z):
#     r = np.sqrt(x**2 + y**2 + z**2)
#     th = np.arctan2(y, x)
#     return np.exp(-r**2 / 12.0) * np.cos(th)**2

const = 1.0

x = np.linspace(-3, 3, 100, dtype='float32')
y = np.linspace(-3, 3, 100, dtype='float32')
X, Y = np.meshgrid(x, y)
X_flat = torch.tensor(X.flatten())
Y_flat = torch.tensor(Y.flatten())
n_pix = 100 * 100

nz = 1000
Zs = np.linspace(-3, 3, nz, dtype='float32')

imB = torch.zeros((100, 100))
impB = torch.zeros((100, 100))

p = torch.tensor(X**2 + Y**2)
mask = p < 2.0

Z = 0.0

u = torch.tensor([0.0, 0.0, 1.0])
M = rotation(u, 0.0)

XYZ_LOS = torch.cat([X_flat[:, None], Y_flat[:, None], torch.zeros_like(X_flat)[:, None]], dim=1)

XYZ_rotated = torch.matmul(M, XYZ_LOS[:, :, None]).squeeze()

density = siren(XYZ_rotated)

# for i in tqdm(range(nz)):
#     Z = Zs[i]

#     theta = np.arctan2(p, Z)
    
#     omegaB = const / p**2 * (np.sin(theta)**2 - 0.5 * np.sin(theta)**4)
#     omegapB = 0.5 * const / p**2 * np.sin(theta)**4

#     Ne = electron(X, Y, Z)

#     imB += omegaB * Ne
#     impB += omegapB * Ne

# imB[mask] = 0.0
# impB[mask] = 0.0

# fig, ax = pl.subplots(nrows=1, ncols=2)
# ax[0].imshow(np.log10(imB))
# ax[1].imshow(np.log10(impB))
# pl.show()