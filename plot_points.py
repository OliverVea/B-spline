import numpy as np

import sys
from PyQt5.QtWidgets import QApplication

## Set up windows for plotting.
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui
app = QApplication(sys.argv)
w = gl.GLViewWidget()
w.show()
g = gl.GLGridItem()
w.addItem(g)

## Load and format data.
pts, ctrl = np.load('points.npz').values()

## Adds point to scatter plot.
ptcolor = np.ndarray((pts.shape[0], 4))
c = np.divide(np.subtract(pts[:,2], np.min(pts[:,2])), np.max(pts[:,2]) - np.min(pts[:,2]))
ptcolor[:,0] = c
ptcolor[:,1] = 0
ptcolor[:,2] = np.subtract(1, c)
ptcolor[:,3] = 1

ctrlcolor = np.full((ctrl.shape[0], 4), np.array([0, 1, 0, 1]))

scatterPlotItems = {}
scatterPlotItems['pts'] = gl.GLScatterPlotItem(pos=pts, color=ptcolor, size = 1)
scatterPlotItems['ctrl'] = gl.GLScatterPlotItem(pos=ctrl, color=ctrlcolor)
w.addItem(scatterPlotItems['pts'])
w.addItem(scatterPlotItems['ctrl'])

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()