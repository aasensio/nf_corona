import torch
import numpy as np

def rotation(u, angle):
    """Rotation matrix in cartesian coordinates around an axis u for an angle alpha

    Parameters
    ----------
    u : vector
        Axis around which rotate
    angle : float
        Angle in deg

    Returns
    -------
    array
        3x3 rotation matrix
    """
    M = torch.zeros((3,3), device=u.device)
    u /= torch.dot(u, u)
    
    cosa = torch.cos(angle * np.pi / 180.0)
    sina = torch.sin(angle * np.pi / 180.0)

    M[0, 0] = cosa + u[0]**2 * (1.0 - cosa)
    M[0, 1] = u[0] * u[1] * (1.0 - cosa) - u[2] * sina
    M[0, 2] = u[0] * u[2] * (1.0 - cosa) + u[1] * sina

    M[1, 0] = u[1] * u[0] * (1.0 - cosa) + u[2] * sina
    M[1, 1] = cosa + u[1]**2 * (1.0 - cosa)
    M[1, 2] = u[1] * u[2] * (1.0 - cosa) - u[1] * sina

    M[2, 0] = u[2] * u[0] * (1.0 - cosa) - u[1] * sina
    M[2, 1] = u[2] * u[1] * (1.0 - cosa) + u[0] * sina
    M[2, 2] = cosa + u[2]**2 * (1.0 - cosa)

    return M

if (__name__ == '__main__'):
    u = torch.tensor([0.0, 0.0, 1.0])
    angles = torch.linspace(0, 90.0, 2)

    M = rotation(u, angles[0])