import numpy as np

def grid2cloud(grid):
    n, m, *_ = grid.shape

    x1 = np.arange(n)
    x2 = np.arange(m)
    x = np.transpose(np.meshgrid(x1, x2)).reshape((-1,2))

    cloud = np.ndarray(n * m, 3)

    cloud[:,:2] = x
    cloud[:,2] = grid.ravel()
    
    return cloud

def cloud2grid(cloud):
    n, m  = np.max(cloud[:,0]), np.max(cloud[:,1])

    grid = cloud[:,2].reshape((n,m))
    return grid

