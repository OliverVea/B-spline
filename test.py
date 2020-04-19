import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from tqdm import tqdm

from central_model import CentralModel

grid_size = (200, 200)
img_size = (200, 200)
scale = 1

order = 3
shape = (6,6,3)
np.random.seed(0)
a = np.random.normal(0, np.sqrt(np.average(grid_size)), np.prod(shape))
a = np.reshape(a, shape)

pts_size = (int((grid_size[0]) / scale), int((grid_size[1]) / scale))

cm = CentralModel(
    image_dimensions=img_size, 
    grid_dimensions=grid_size, 
    control_points=a, 
    order=order,
    knot_method='open_uniform',
    end_divergence=0.01,
    min_basis_value=0
)

d = np.divide(np.subtract(grid_size, 1), np.subtract(shape[:-1], 1))
c = np.divide(np.subtract(grid_size, img_size), 2)

ctrl = np.round(np.array([[[i * d[0], j * d[1], a[i,j,0]] for j in range(shape[1])] for i in range(shape[0])]))
ctrl = np.reshape(ctrl, (-1, 3))
ctrl[:,:2] -= c

pts = np.ndarray(pts_size + (3,))

for u in tqdm(range(0, img_size[0], scale)):
    for v in range(0, img_size[1], scale):
        pts[int(u / scale), int(v / scale), :] = cm.sample(u, v)

y = pts[:,:,0]

x = np.array([[[i * scale, j * scale, y[i,j]] for j in range(int((img_size[1]) / scale))] for i in range(int((img_size[0]) / scale))])
x = np.reshape(x, (-1,3))

np.savez('points.npz', x, ctrl)