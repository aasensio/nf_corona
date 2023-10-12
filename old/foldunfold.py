import torch
import torch.nn.functional as F
import matplotlib.pyplot as pl

def unfold_fold(X):
    X_unfold = X.unfold(2, 25, 25).unfold(1, 25, 25)

    X2 = [None] * 2
    for i in range(2):
        X2j = [None] * 2
        for j in range(2):
            X2j[j] = X_unfold[0, i, j, :, :][..., None]
        X2[i] = torch.cat(X2j, dim=-1)[..., None]

    X_final = torch.cat(X2, dim=-1).transpose(0,1).transpose(2,3).reshape((-1,25*25,2*2))
    Xf = F.fold(X_final, (50, 50), 25, stride=25).squeeze()

    return Xf

x = torch.linspace(-2, 2, 50)
y = torch.linspace(-2, 2, 50)
X, Y = torch.meshgrid(x, y)

X = X[None, ...]
Y = Y[None, ...]

Xf = unfold_fold(X)
Yf = unfold_fold(Y)

fig, ax = pl.subplots(nrows=2, ncols=2)
ax[0,0].imshow(X[0, :, :])
ax[0,1].imshow(Xf)
ax[1,0].imshow(Y[0, :, :])
ax[1,1].imshow(Yf)

