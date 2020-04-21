import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from tqdm import tqdm

from central_model import CentralModel, fit_central_model
import utility as util

grid_size = (200, 200)
img_size = (200, 200)

order = 3
shape = (6,6,3)

knot_method = 'open_uniform'

end_divergence = 0.01
min_basis_value = 0

np.random.seed(0)
target_values = np.random.normal(0, np.sqrt(np.average(grid_size)), np.prod(shape))
target_values = np.reshape(target_values, shape)

fit = False

if fit:
    cm, results = fit_central_model(
        target_values,
        img_shape=img_size, 
        grid_shape=grid_size, 
        order=order,
        knot_method=knot_method,
        end_divergence=end_divergence,
        min_basis_value=min_basis_value,
        verbose=2
    )
else:
    cm = CentralModel(
        img_size,
        grid_size,
        target_values,
        order,
        knot_method,
        end_divergence,
        min_basis_value
    )

d = np.divide(np.subtract(grid_size, 1), np.subtract(shape[:-1], 1))
c = np.divide(np.subtract(grid_size, img_size), 2)

ctrl = np.round(np.array([[[i * d[0], j * d[1], cm.a[i,j,0]] for j in range(shape[1])] for i in range(shape[0])]))
ctrl = np.reshape(ctrl, (-1, 3))
ctrl[:,:2] -= c

tv = np.round(np.array([[[i * d[0], j * d[1], target_values[i,j,0]] for j in range(shape[1])] for i in range(shape[0])]))
tv = np.reshape(tv, (-1, 3))
tv[:,:2] -= c

pts = np.ndarray(grid_size + (3,))

for u in tqdm(range(img_size[0])):
    for v in range(img_size[1]):
        pts[u, v, :] = cm.sample(u, v)

y = pts[:,:,0]

x = np.array([[[i, j, y[i,j]] for j in range(img_size[1])] for i in range(img_size[0])])
x = np.reshape(x, (-1,3))

np.savez('points.npz', x, ctrl, tv)