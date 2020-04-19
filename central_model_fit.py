from central_model import CentralModel

import numpy as np
from scipy.optimize import least_squares

def fun(params, target_values, grid_shape, image_dimensions, grid_dimensions, order, min_basis_value = 0.01):
    """Compute residuals.
    
    Keyword arguments: \n
    params: contains control points for the b-spline.\n
    target_values: the target values for the b-spline in the sampled points. (u, v, x, y, z)\n
    """
    grid = np.reshape(params, grid_shape)

    cm = CentralModel(image_dimensions, grid_dimensions, grid, order, min_basis_value)
    grid_samples = cm.sample_grid().ravel()
    return target_values - grid_samples

if __name__ == '__main__':
    np.random.seed(0)

    # B-spline parameters
    grid_size = (100,100)
    img_size = grid_size
    scale = 1
    order = 3
    shape = (5,5,3)

    target_values = np.random.normal(0, np.sqrt(np.average(grid_size)), np.prod(shape))
    target_values = np.reshape(target_values, shape)

    x0 = target_values

    res = fun(x0.ravel(), target_values.ravel(), shape, img_size, grid_size, order)

    res = np.reshape(res, (5,5,3))

    pass