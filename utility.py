import numpy as np

def grid2cloud(grid):
    pass

def cloud2grid(cloud):
    x_min = (np.min(cloud[:,0]), np.min(cloud[:,1]))
    x_max = (np.max(cloud[:,0]), np.max(cloud[:,1]))

    n, m = cloud.shape[0], cloud.shape[1]

    dx = tuple(np.subtract(x_max, x_min))

    y_min = np.min(cloud[:,2])
    y_max = np.max(cloud[:,2])

    x1 = np.arange(n)
    x2 = np.arange(m)

    x = np.transpose(np.meshgrid(x1, x2)).reshape((-1,2))
    pass

