from central_model import CentralModel

import numpy as np
from scipy.optimize import least_squares


import sys
from PyQt5.QtWidgets import QApplication

## Set up windows for plotting.
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui

def fun(params, target_values, grid_shape, image_dimensions, grid_dimensions, order, knot_method='open_uniform', min_basis_value = 1e-4):
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
        min_basis_value=min_basis_value,
        end_divergence=0)

    grid_samples = cm.sample_grid().ravel()
    return target_values - grid_samples

if __name__ == '__main__':
    np.random.seed(0)

    # B-spline parameters
    grid_size = (100,100)
    img_size = grid_size
    scale = 1
    order = 3
    shape = (6,6,3)

    target_values = np.random.normal(0, np.sqrt(np.average(grid_size)), np.prod(shape))
    target_values = np.reshape(target_values, shape)

    x0 = target_values.reshape((-1,))

    res = fun(x0.ravel(), target_values.ravel(), shape, img_size, grid_size, order)

    a = least_squares(fun, x0, verbose=2, args=(target_values.ravel(), shape, img_size, grid_size, order))
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

    cm = CentralModel(img_size, grid_size, ctrl.reshape(shape), order)

    pts_x = np.ndarray((np.product(img_size),3))
    pts_y = np.ndarray((np.product(img_size),3))
    pts_z = np.ndarray((np.product(img_size),3))
    
    for u in range(img_size[0]):
        for v in range(img_size[1]):
            s = cm.sample(u, v)
            pts_x[u + v * img_size[0]] = np.array([u, v, s[0]])
            pts_y[u + v * img_size[0]] = np.array([u, v, s[1]])
            pts_z[u + v * img_size[0]] = np.array([u, v, s[2]])

    app = QApplication(sys.argv)
    w = gl.GLViewWidget()
    w.show()
    g = gl.GLGridItem()
    w.addItem(g)

    ## Adds point to scatter plot.
    ptcolor = np.ndarray((pts_x.shape[0], 4))
    c = np.divide(np.subtract(pts_x[:,2], np.min(pts_x[:,2])), np.max(pts_x[:,2]) - np.min(pts_x[:,2]))
    ptcolor[:,0] = c
    ptcolor[:,1] = 0
    ptcolor[:,2] = np.subtract(1, c)
    ptcolor[:,3] = 1

    ctrlcolor = np.full((ctrl.shape[0], 4), np.array([0, 1, 0, 1]))
    tvcolor = np.full((ctrl.shape[0], 4), np.array([1, 1, 1, 1]))

    scatterPlotItems = {}
    scatterPlotItems['pts'] = gl.GLScatterPlotItem(pos=pts_x, color=ptcolor, size = 2)
    scatterPlotItems['ctrl'] = gl.GLScatterPlotItem(pos=ctrl_x, color=ctrlcolor)
    scatterPlotItems['tvs'] = gl.GLScatterPlotItem(pos=tv_x, color=tvcolor)
    w.addItem(scatterPlotItems['pts'])
    w.addItem(scatterPlotItems['ctrl'])
    w.addItem(scatterPlotItems['tvs'])

    dx = ctrl_x - tv_x

    ## Start Qt event loop unless running in interactive mode.
    if __name__ == '__main__':
        import sys
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    pass