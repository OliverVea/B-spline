import numpy as np

import sys
from PyQt5.QtWidgets import QApplication

import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui

ctrl_x, ctrl_y, ctrl_z, tv_x, tv_y, tv_z, pts_x, pts_y, pts_z = np.load('fitted_data.npz').values()

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

ctrlcolor = np.full((ctrl_x.shape[0], 4), np.array([0, 1, 0, 1]))
tvcolor = np.full((tv_x.shape[0], 4), np.array([1, 1, 1, 1]))

scatterPlotItems = {}
#scatterPlotItems['pts'] = gl.GLScatterPlotItem(pos=pts_x, color=ptcolor, size = 2)
scatterPlotItems['ctrl'] = gl.GLScatterPlotItem(pos=ctrl_x, color=ctrlcolor)
scatterPlotItems['tvs'] = gl.GLScatterPlotItem(pos=tv_x, color=tvcolor)
#w.addItem(scatterPlotItems['pts'])
w.addItem(scatterPlotItems['ctrl'])
w.addItem(scatterPlotItems['tvs'])

x = np.arange(0, 1000)
y = np.arange(0, 1000)
z = pts_x[:,2].reshape(1000,1000)
c = ptcolor.reshape(1000,1000,4)

surface = gl.GLSurfacePlotItem(
    x, 
    y, 
    z, 
    c)

w.addItem(surface)

## Start Qt event loop unless running in interactive mode.
if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    QtGui.QApplication.instance().exec_()