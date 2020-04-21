import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from OpenGL.GL import glReadBuffer, GL_FRONT

# can be 'rotation', 'single' or 'none'. 
export_option = 'single'
image_name = 'still'

draw_as_surface = True

draw_pts = True
draw_ctrl = True
draw_tv = True

orbit = False

frame_time = 1000/60 # in ms

# Make pyqtgraph window.
app = pg.mkQApp()
w = gl.GLViewWidget()
w.show()

## Load and format data.
pts, ctrl, tv = np.load('points.npz').values()
centerpoint = np.average(pts, axis=0)

## Adds point to scatter plot.
if draw_pts:
    ptcolor = np.ndarray((pts.shape[0], 4))
    c = np.divide(np.subtract(pts[:,2], np.min(pts[:,2])), np.max(pts[:,2]) - np.min(pts[:,2]))
    ptcolor[:,0] = c
    ptcolor[:,1] = 0
    ptcolor[:,2] = np.subtract(1, c)
    ptcolor[:,3] = 1

    # Creates surface/pointcloud for the b-spline samples.
    if draw_as_surface:
        x = np.arange(np.max(pts[:,0]) + 1)
        y = np.arange(np.max(pts[:,1]) + 1)
        pts = gl.GLSurfacePlotItem(x, y, pts[:,2].reshape(len(x), len(y)), colors=ptcolor)

    else:
        pts = gl.GLScatterPlotItem(pos=pts, color=ptcolor, size=1)

    w.addItem(pts)

scatterPlotItems = {}

# Creates points for the control points of the b-spline.
if draw_ctrl:
    ctrlcolor = np.full((ctrl.shape[0], 4), np.array([0, 1, 0, 1]))
    scatterPlotItems['ctrl'] = gl.GLScatterPlotItem(pos=ctrl, color=ctrlcolor)
    w.addItem(scatterPlotItems['ctrl'])

# Creates points for the target points of the b-spline.
if draw_tv:
    tvcolor = np.full((ctrl.shape[0], 4), np.array([1, 1, 1, 1]))
    scatterPlotItems['tv'] = gl.GLScatterPlotItem(pos=tv, color=tvcolor)
    w.addItem(scatterPlotItems['tv'])

# Orients camera in 3d.
w.showFullScreen()
w.pan(centerpoint[0], centerpoint[1], centerpoint[2])
w.setCameraPosition(distance=400)


# Orbiting and image exporting.
da = 0.2
current_angle = 0
i = 0
def update():

    global current_angle, da, i, export_option

    if export_option == 'single':
        w.grabFrameBuffer().save('images/{}_{}.png'.format(image_name, i))
        export_option = 'none'

    if current_angle < 720 and export_option == 'rotation':
        w.grabFrameBuffer().save('images/{}_{}.png'.format(image_name, i))
        current_angle += da

    if orbit:
        w.orbit(da, 0)
        
    i += 1

timera = pg.QtCore.QTimer()
timera.timeout.connect(update)

if export_option == 'rotation':
    timera.start(1)
else:
    timera.start(frame_time)

glReadBuffer(GL_FRONT) # A little bit of c-code in my lines

# Executes application
app.exec_()