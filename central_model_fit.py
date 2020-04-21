from central_model import CentralModel

from tqdm import tqdm

import numpy as np
from scipy.optimize import least_squares

def fun(params, target_values, grid_shape, image_dimensions, grid_dimensions, order, knot_method='open_uniform', end_divergence = 0):
    """Compute residuals.
    
    Keyword arguments: \n
    params: contains control points for the b-spline.\n
    target_values: the target values for the b-spline in the sampled points. (u, v, x, y, z)\n
    """
    grid = np.reshape(params, grid_shape)

    cm = CentralModel(
        image_dimensions=image_dimensions, 
        grid_dimensions=grid_dimensions, 
        control_points=grid, 
        order=order,
        knot_method=knot_method,
        end_divergence=end_divergence)

    grid_samples = cm.sample_grid().ravel()
    return target_values - grid_samples

if __name__ == '__main__':
    np.random.seed(0)

    # B-spline parameters
    grid_size = (1000,1000)
    img_size = grid_size
    scale = 1
    order = 3
    shape = (24,24,3)
    knot_method = 'open_uniform'
    end_divergence = 0


    #target_values = np.random.normal(0, np.sqrt(np.average(grid_size)), np.prod(shape))
    #target_values = np.reshape(target_values, shape)

    target_values = np.ndarray(shape)

    omega = 4
    amplitude = 100

    for u in range(shape[0]):
        for v in range(shape[0]):
            target_values[u,v] = np.array([np.cos(np.sqrt(u^2 + v^2) * omega), 0, 0]) * amplitude

    x0 = target_values.reshape((-1,))

    res = fun(x0.ravel(), target_values.ravel(), shape, img_size, grid_size, order, knot_method, end_divergence)

    print('Fitting started. This might take a while.')
    a = least_squares(fun, x0, verbose=2, args=(target_values.ravel(), shape, img_size, grid_size, order, knot_method, end_divergence))
    target_values = np.reshape(target_values, (-1,3))

    ctrl = a['x'].reshape((-1,3))

    ctrl_x = np.ndarray(ctrl.shape)
    ctrl_y = np.ndarray(ctrl.shape)
    ctrl_z = np.ndarray(ctrl.shape)
    tv_x = np.ndarray(ctrl.shape)
    tv_y = np.ndarray(ctrl.shape)
    tv_z = np.ndarray(ctrl.shape)
    tx = np.arange(0, shape[0]) * (grid_size[0] - 1) / (shape[0] - 1)
    ty = np.arange(0, shape[1]) * (grid_size[1] - 1) / (shape[1] - 1)
    ctrl_x[:,:2] = ctrl_y[:,:2] = ctrl_z[:,:2] = tv_x[:,:2] = tv_y[:,:2] = tv_z[:,:2] = np.transpose(np.meshgrid(tx, ty)).reshape((-1,2))
    ctrl_x[:,2] = ctrl[:,0]
    ctrl_y[:,2] = ctrl[:,1]
    ctrl_z[:,2] = ctrl[:,2]
    tv_x[:,2] = target_values[:,0]
    tv_y[:,2] = target_values[:,1]
    tv_z[:,2] = target_values[:,2]

    cm = CentralModel(img_size, grid_size, ctrl.reshape(shape), order, knot_method, end_divergence=end_divergence)

    pts_x = np.ndarray((np.product(img_size),3))
    pts_y = np.ndarray((np.product(img_size),3))
    pts_z = np.ndarray((np.product(img_size),3))

    print('Sampling from camera model.')
    pbar = tqdm(total=img_size[0] * img_size[1])
    for u in range(img_size[0]):
        for v in range(img_size[1]):
            s = cm.sample(u, v)
            pts_x[u + v * img_size[0]] = np.array([u, v, s[0]])
            pts_y[u + v * img_size[0]] = np.array([u, v, s[1]])
            pts_z[u + v * img_size[0]] = np.array([u, v, s[2]])
            pbar.update(1)

    np.savez('fitted_data.npz', ctrl_x, ctrl_y, ctrl_z, tv_x, tv_y, tv_z, pts_x, pts_y, pts_z)
    print('Points saved. Feel free to stop the program.')