# ----------------------------------------------------------------------------------------------------------------------
params = {
    # ------------------------------------------------------------------------------------------------------------------
    'S': [
        [5, 2, 1],
        [2, 7, 2],
        [1, 2, 3]
    ],
    'm': [2, 1, 5],
    # ------------------------------------------------------------------------------------------------------------------
    'd': 1.1,
    'e': None,    # if None will be calculated with ect
    # ------------------------------------------------------------------------------------------------------------------
    'n': 100000,   # desired amount of points in each set
    'ect': 0.1,   # e coefficient for e^2 ~= ect * d^2, ect in [0.01 .. 0.1]
    'seed': 2022  # numpy rng seed and points generation seed
    # ------------------------------------------------------------------------------------------------------------------
}
# ----------------------------------------------------------------------------------------------------------------------
if params['e'] is None:
    params['e'] = ((params['d'] ** 2) * params['ect']) ** 0.5

print('params: ' + str(params))

import numpy as np, scipy.stats as ss, pyqtgraph as pg
import pyqtgraph.opengl as gl
from OpenGL.GL import *

S = np.array(params['S'])
m = np.array(params['m'])
d = params['d']
e = params['e']
n = params['n']

d2, e2 = d ** 2, e ** 2

app = pg.mkQApp("Lab4")
w = gl.GLViewWidget()

class MyAxisItem(gl.GLAxisItem):  # custom axis: x red, y green, z blue
    def paint(self):
        self.setupGLState()
        if self.antialias:
            glEnable(GL_LINE_SMOOTH)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glBegin(GL_LINES)
        x, y, z = self.size()
        glColor4f(1, 0.5, 0.5, 1), glVertex3f(0, 0, 0), glVertex3f(x, 0, 0)
        glColor4f(0.5, 1, 0.5, 1), glVertex3f(0, 0, 0), glVertex3f(0, y, 0)
        glColor4f(0.5, 0.5, 1, 1), glVertex3f(0, 0, 0), glVertex3f(0, 0, z)
        glEnd()

w.addItem(MyAxisItem(pg.Vector(1,1,1), glOptions='opaque'))
w.addItem(gl.GLGridItem())

axm = MyAxisItem(pg.Vector(d/2,d/2,d/2), glOptions='opaque')
axm.translate(m[0], m[1], m[2])
w.addItem(axm)

axm2 = MyAxisItem(pg.Vector(-d/2,-d/2,-d/2), glOptions='opaque')
axm2.translate(m[0], m[1], m[2])
w.addItem(axm2)

rng = np.random.default_rng(params['seed'])

dist = ss.multivariate_normal(m, S)
pts1, pts2, gp, up, dp = [], [], 0, 0, 0

Si = np.linalg.inv(S)

print('Generating points in datasets...')
while True:
    pts = dist.rvs(n, params['seed'] + gp)
    gp += n
    for p in pts:
        di = (p - m).T.dot(Si).dot((p - m).T)
        if abs(di - d2) >= e2:
            if di < d2 and len(pts1) < n:
                pts1.append(p)
            elif di > d2 and len(pts2) < n:
                pts2.append(p)
            else:
                dp += 1
        else:
            up += 1
    if len(pts1) == n and len(pts2) == n:
        break

print('Obtained {} points for each set, generated {} points in total'.format(n, gp))
print('Unknown points: {}, discarded points: {}'.format(up, dp))

arr1 = np.array(pts1)
arr2 = np.array(pts2)

sp1 = gl.GLScatterPlotItem(pos=arr1, size=0.1, color=(1.0, 0.5, 0.5, 0.1), pxMode=False)
w.addItem(sp1)

sp2 = gl.GLScatterPlotItem(pos=arr2, size=0.1, color=(0.5, 0.5, 1.0, 0.1), pxMode=False)
w.addItem(sp2)

w.show()
w.setWindowTitle('Lab4 - Dense test')
w.setCameraPosition(pos=pg.Vector(m), distance=d*16)
pg.exec()  # MUST be last (event loop)
