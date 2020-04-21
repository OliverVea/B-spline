import numpy as np

import sys
from PyQt5.QtWidgets import QApplication

## Set up windows for plotting.
#import pyqtgraph.opengl as gl
#from pyqtgraph.Qt import QtCore, QtGui
#app = QApplication(sys.argv)
#w = gl.GLViewWidget()
#w.show()
#g = gl.GLGridItem()
#w.addItem(g)

save_images = False

import pyqtgraph as pg
app = pg.mkQApp()

import pyqtgraph.opengl as gl
w = gl.GLViewWidget()
w.show()

## Load and format data.
pts, ctrl, tv = np.load('points.npz').values()

## Adds point to scatter plot.
ptcolor = np.ndarray((pts.shape[0], 4))
c = np.divide(np.subtract(pts[:,2], np.min(pts[:,2])), np.max(pts[:,2]) - np.min(pts[:,2]))
ptcolor[:,0] = c
ptcolor[:,1] = 0
ptcolor[:,2] = np.subtract(1, c)
ptcolor[:,3] = 1

ctrlcolor = np.full((ctrl.shape[0], 4), np.array([0, 1, 0, 1]))
tvcolor = np.full((ctrl.shape[0], 4), np.array([1, 1, 1, 1]))

scatterPlotItems = {}
scatterPlotItems['pts'] = gl.GLScatterPlotItem(pos=pts, color=ptcolor, size=1)
#scatterPlotItems['ctrl'] = gl.GLScatterPlotItem(pos=ctrl, color=ctrlcolor)
scatterPlotItems['tv'] = gl.GLScatterPlotItem(pos=tv, color=tvcolor)
w.addItem(scatterPlotItems['pts'])
#w.addItem(scatterPlotItems['ctrl'])
w.addItem(scatterPlotItems['tv'])

centerpoint = np.average(pts, axis=0)

w.showFullScreen()
w.pan(centerpoint[0], centerpoint[1], centerpoint[2])
w.setCameraPosition(distance=400)

da = 0.2
current_angle = 0
i = 0
def update():
    global current_angle, da, i
    current_angle += da

    if current_angle < 360:
        w.grabFrameBuffer().save('images/image_{}.png'.format(i))
        i += 1

    w.orbit(da, 0)

timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
timer.start(1000/60)

app.exec_()