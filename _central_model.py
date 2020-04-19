import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def CubicBezier(points, t):
    points = np.array(points)

    pt = np.ndarray(points.shape[-1])
    for i in range(points.shape[-1]):
        P = points[:,i]
        M = np.array([
            [-1, 3,-3, 1], 
            [ 3,-6, 3, 0], 
            [-3, 3, 0, 0], 
            [ 1, 0, 0, 0]])
        T = np.array([t**3, t**2, t, 1])

        pt[i] = np.dot(P, np.dot(M, T))
    return pt

def CubicBezierSurface(points, u, v):
    points = np.array(points)

    pt = np.ndarray(points.shape[-1])
    for i in range(points.shape[-1]):
        P = points[:,:,i]
        M = np.array([
            [-1, 3,-3, 1], 
            [ 3,-6, 3, 0], 
            [-3, 3, 0, 0], 
            [ 1, 0, 0, 0]])
        U = np.array([u**3, u**2, u, 1])
        V = np.array([v**3, v**2, v, 1])

        pt[i] = np.dot(U, np.dot(M, np.dot(P, np.dot(M, V))))
    return pt

uvalcounter = {}
udiff = []
oldu = 0

def BSplineSurface(points, u, v):  
    # TODO
    global uvalcounter, udiff, oldu
    n, m, _ = points.shape
    I = np.floor(np.multiply(np.array([u, v]), np.array([n - 1, m - 1])) - 1)
    I = np.maximum(np.array([0, 0]), np.minimum(I, np.array([n - 4, m - 4])))

    Ti = np.divide(I, np.array([n - 1, m - 1]))
    Te = np.divide(I + 3, np.array([n - 1, m - 1]))

    u, v = np.divide(np.subtract(np.array([u, v]), Ti), np.subtract(Te, Ti))

    ctrl = points[int(I[0]):int(I[0] + 4), int(I[1]):int(I[1] + 4),:]
    return CubicBezierSurface(ctrl, u, v)

def BezierCurveTest(num=15):
    plt.figure()

    T = np.linspace(0, 1, num=num)

    ctrl = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]])

    plt.plot(ctrl[:,0], ctrl[:,1], 'ro')

    pts = []
    for t in T:
        pts.append(CubicBezier(ctrl, t))
    pts = np.array(pts)

    plt.plot(pts[:,0], pts[:,1])

    plt.show()

def SurfaceFromZ(heights):
    s = heights.shape + (3,)
    pts = np.ndarray(s)

    for x in range(s[0]):
        for y in range(s[1]):
            pts[x,y,:] = np.array([x, y, heights[x,y]])

    return pts

def BezierSurfaceTest(num=15):
    heights = np.array([
        [0.0, -0.5, 1.0, 0.0],
        [1.0, 0.0, 1.0, -2.0],
        [0.0, 1.0, -1.0, 1.0],
        [0.0, -0.2, 0.0, 0.0]])
    ctrl = SurfaceFromZ(heights)

    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    for ctrln, c in zip(ctrl, ['ro', 'go', 'bo', 'yo']):
        ax.plot3D(ctrln[:,0], ctrln[:,1], ctrln[:,2], c)

    pts = []
    t = np.linspace(0, 1, num=num)
    tm = np.reshape(np.transpose(np.meshgrid(t,t)), (-1,2))
    for u, v in tm:
        pts.append(CubicBezierSurface(ctrl, u, v))
    pts = np.array(pts)

    ax.plot_trisurf(pts[:,0], pts[:,1], pts[:,2])
    plt.show()

def BSplineTest(num=15):
    # TODO
    heights = np.array([
        [0.0, 1.0,  2.0,  3.0,  4.0],
        [1.0, 2.0,  3.0,  4.0,  5.0],
        [2.0, 3.0,  4.0,  5.0,  6.0],
        [3.0, 4.0,  5.0,  6.0,  7.0],
        [4.0, 5.0,  6.0,  7.0,  8.0],
        [5.0, 6.0,  7.0,  8.0,  9.0]])

    heights1 = np.sin(np.arange(6)*2)
    heights2 = np.arange(5)

    for i in range(6):
        for j in range(5):
            heights[i,j] = heights1[i] + heights2[j]

    ctrl = SurfaceFromZ(heights)

    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    for ctrln in ctrl:
        ax.plot3D(ctrln[:,0], ctrln[:,1], ctrln[:,2], 'ro')
    
    pts = []
    t = np.linspace(0, 1, num=num)
    tm = np.reshape(np.transpose(np.meshgrid(t,t)), (-1,2))
    for u, v in tm:
        pts.append(BSplineSurface(ctrl, u, v))
    pts = np.array(pts)
    
    ax.plot_trisurf(pts[:,0], pts[:,1], pts[:,2])
    plt.show()
    pass

class CentralModel():
    def __init__(self, gridshape, imshape):
        assert isinstance(gridshape, tuple) and len(gridshape) == 2, 'gridshape must be a tuple with 2 positive integers.'
        assert isinstance(imshape, tuple) and len(imshape) == 2, 'imshape must be a tuple with 2 positive integers.'

        self.gridshape = gridshape
        self.imshape = imshape
        self.grid = np.full(gridshape + (3,), 0)
        self.dp = np.divide(imshape, np.subtract(gridshape, 3))
    
    def gridpositions(self):
        maxval = np.multiply(np.subtract(self.gridshape, 2), self.dp)

        x = np.linspace(-self.dp[0], maxval[0], self.gridshape[0])
        y = np.linspace(-self.dp[1], maxval[1], self.gridshape[1])

        result = np.transpose(np.meshgrid(x, y))
 
        return result

    def sample(self, point):
        neighbor = np.floor(np.divide(point, self.dp)) + 1 # Remember that the corner of the image starts at grid point (1,1), not (0,0).
        print(neighbor[0]-1)
        neighborvals = self.grid[int(neighbor[0]-1):int(neighbor[0]+3), int(neighbor[1]-1):int(neighbor[1]+3), :]
        [tx, ty] = 1/3 * (1 + np.divide(point - np.multiply(neighbor - 1, self.dp), self.dp))

        return CubicBezierSurface(neighborvals, tx, ty)

    def draw(self):
        # TODO
        pass

#BezierCurveTest(4)
#BezierSurfaceTest()
BSplineTest(80)

cm = CentralModel((11, 8), (800,500))
cm.gridpositions()
sample = cm.sample(np.array([270, 99]))
print(sample)
pass