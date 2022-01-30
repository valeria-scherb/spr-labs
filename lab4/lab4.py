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
    'd': 1.3,
    'e': None,     # if None will be calculated with ect
    # ------------------------------------------------------------------------------------------------------------------
    'n': 1000,     # desired amount of points in each set
    'ect': 0.1,    # e coefficient for e^2 ~= ect * d^2, ect in [0.01 .. 0.1]
    'seed': 2022,  # numpy rng seed and points generation seed
    # ------------------------------------------------------------------------------------------------------------------
    'cam_auto_rotate': True
}
# ----------------------------------------------------------------------------------------------------------------------
if params['e'] is None:
    params['e'] = ((params['d'] ** 2) * params['ect']) ** 0.5

print('params: ' + str(params))

import numpy as np, scipy.stats as ss, pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
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

batch = 0

print('Generating points in datasets...')
while True:
    pts = dist.rvs(n, params['seed'] + batch)
    if n == 1: pts = np.array([pts])
    gp += n
    batch += 1
    rel = pts - m
    for p in rel:
        di = p.T @ Si @ p
        if abs(di - d2) >= e2:
            if di < d2 and len(pts1) < n:
                pts1.append(p + m)
            elif di > d2 and len(pts2) < n:
                pts2.append(p + m)
            else:
                dp += 1
        else:
            up += 1
    if len(pts1) == n and len(pts2) == n:
        break
    print(len(pts1), len(pts2), up, dp)

print('Obtained {} points for each set, generated {} points in total'.format(n, gp))
print('Unknown points: {}, discarded points: {}'.format(up, dp))

arr1 = np.array(pts1)
arr2 = np.array(pts2)

w.addItem(gl.GLScatterPlotItem(pos=arr1, size=0.4, color=(1.0, 0.5, 0.5, 0.1), pxMode=False))
w.addItem(gl.GLScatterPlotItem(pos=arr1, size=0.3, color=(1.0, 0.5, 0.5, 0.1), pxMode=False))
w.addItem(gl.GLScatterPlotItem(pos=arr1, size=0.2, color=(1.0, 0.5, 0.5, 0.2), pxMode=False))
w.addItem(gl.GLScatterPlotItem(pos=arr1, size=0.1, color=(1.0, 0.5, 0.5, 0.2), pxMode=False))

w.addItem(gl.GLScatterPlotItem(pos=arr2, size=0.4, color=(0.5, 0.5, 1.0, 0.2), pxMode=False))
w.addItem(gl.GLScatterPlotItem(pos=arr2, size=0.3, color=(0.5, 0.5, 1.0, 0.2), pxMode=False))
w.addItem(gl.GLScatterPlotItem(pos=arr2, size=0.2, color=(0.5, 0.5, 1.0, 0.2), pxMode=False))
w.addItem(gl.GLScatterPlotItem(pos=arr2, size=0.1, color=(0.5, 0.5, 1.0, 0.2), pxMode=False))

print('Dataset (X Y Z L):')
data = np.concatenate((
    np.concatenate((pts1, np.full((n, 1), -1)), axis=1),
    np.concatenate((pts2, np.full((n, 1), +1)), axis=1)
))
print(data)
rng.shuffle(data)

def kernel(x):
    x1, x2, x3 = x[:3]
    return np.array([x1 * x1, x1 * x2, x1 * x3,
                     x2 * x1, x2 * x2, x2 * x3,
                     x3 * x1, x3 * x2, x3 * x3,
                     x1, x2, x3, 1])

def eigen_kernel(x):
    x1, x2, x3 = x
    return np.array([x1 * x1, x1 * x2, x1 * x3,
                     x2 * x1, x2 * x2, x2 * x3,
                     x3 * x1, x3 * x2, x3 * x3,
                     0, 0, 0, 0])

def iso_kernel(x):
    x1, x2, x3 = x
    return np.array([x1 * x1, x1 * x2, x1 * x3,
                     x2 * x1, x2 * x2, x2 * x3,
                     x3 * x1, x3 * x2, x3 * x3,
                     x1, x2, x3, 1], dtype=object)

def adjust(al, y):  # kozinec
    p = np.dot(np.negative(y), np.subtract(al, y)) / \
        np.linalg.norm(np.subtract(al, y)) ** 2
    p = max(min(p, 1), 0)
    return np.add(np.multiply(p, al), np.multiply(1-p, y))

alpha = kernel(pts2[0])  # can't be zero for kozinec
iso_alpha = alpha

itlog = []

iters = 0
miss, emiss = 1, 0
while miss + emiss > 0:
    iters += 1
    miss, emiss = 0, 0
    for x_y in data:
        ksi = kernel(x_y)
        perception = ksi @ alpha
        if np.sign(perception) != x_y[3]:
            miss += 1
            alpha = adjust(alpha, ksi * x_y[3])
    # always check and apply eigen kernels to keep alpha "in tone"
    vals, vects = np.linalg.eigh(np.reshape(alpha[:9], (3,3)))
    for i in range(3):
        if vals[i] <= 0:
            alpha = adjust(alpha, eigen_kernel(vects[:, i]))
            emiss += 1
    itlog.append([iters, alpha, miss, emiss, vals])
print('Iterations counts: ' + str(iters))

def restore_params(a):
    si = np.reshape(a[:9], (3,3))
    return np.linalg.solve(si, a[9:12] / (-2)), np.linalg.inv(si)

print('Alpha: ' + str(alpha))

em, ec = restore_params(alpha)
print('Separating ellipsoid params:')
print('Mean: ' + str(em))
print('Real mean: ' + str(m))

print('Cov:\n' + str(ec))
print('Real cov:\n' + str(S))

print('Normalized cov:\n' + str(ec / np.linalg.norm(ec)))
print('Normalized real cov:\n' + str(S / np.linalg.norm(S)))

iso_offs = 10   # range
iso_subdiv = 3  # precision
iso_range = iso_offs * 2 * iso_subdiv
def iso_surface(i, j, k):
    global iso_alpha
    return iso_kernel(np.array([i / iso_subdiv - iso_offs + m[0],
                                j / iso_subdiv - iso_offs + m[1],
                                k / iso_subdiv - iso_offs + m[2]])) @ iso_alpha

iso_data = np.fromfunction(iso_surface, (iso_range, iso_range, iso_range))
verts, faces = pg.isosurface(iso_data, 0)
md = gl.MeshData(vertexes=verts, faces=faces)
colors = np.ones((md.faceCount(), 4), dtype=float)
colors[:, 3] = 0.1
colors[:, 0] = 0
colors[:, 2] = 0
md.setFaceColors(colors)

mesh = gl.GLMeshItem(meshdata=md, smooth=True, shader='balloon')
mesh.setGLOptions('additive')
mesh.translate(-iso_offs+m[0], -iso_offs+m[1], -iso_offs+m[2])
mesh.scale(1./iso_subdiv, 1./iso_subdiv, 1./iso_subdiv)
w.addItem(mesh)

class MyPainterItem(pg.opengl.GLGraphicsItem.GLGraphicsItem):
    def __init__(self, **kwds):
        super().__init__()
        glopts = kwds.pop('glOptions', 'additive')
        self.setGLOptions(glopts)
        self.info = ''

    def paint(self):
        self.setupGLState()
        painter = QtGui.QPainter(self.view())
        self.draw(painter)
        painter.end()

    def draw(self, painter):
        painter.setPen(QtCore.Qt.GlobalColor.white)
        painter.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.TextAntialiasing)
        rect = self.view().rect()
        af = QtCore.Qt.AlignmentFlag
        painter.drawText(rect, af.AlignTop | af.AlignLeft, self.info)


text = MyPainterItem()
w.addItem(text)

rend_iter = -1
rend_limit = iters + 100  # 5 seconds

def update():
    global iso_alpha, rend_iter, rend_limit, iters, itlog
    rend_iter += 1
    if rend_iter == rend_limit:
        rend_iter = 0
    info = itlog[iters-1] if rend_iter >= iters else itlog[rend_iter]
    iso_alpha = info[1]
    iso_data = np.fromfunction(iso_surface, (iso_range, iso_range, iso_range))
    verts, faces = pg.isosurface(iso_data, 0)
    md = gl.MeshData(vertexes=verts, faces=faces)
    colors = np.ones((md.faceCount(), 4), dtype=float)
    colors[:, 3] = 0.1
    colors[:, 0] = 0
    colors[:, 2] = 0
    md.setFaceColors(colors)
    mesh.setMeshData(meshdata=md)
    em, ec = restore_params(iso_alpha)
    text.info = 'Iteration: {}/{}\nAlpha norm: {}\nAccuracy: {} ({} misses)\nNegative eigens: {}\nEigens: {}\n' \
                'Separating ellipsoid parameters\nmu: {}\ncov:\n{}'\
        .format(info[0], iters, np.linalg.norm(info[1]), (2*n-info[2])/(2*n), info[2], info[3], info[4], em, ec)
    if params['cam_auto_rotate']:
        w.setCameraPosition(azimuth=-rend_iter*2)

t = QtCore.QTimer()
t.timeout.connect(update)
t.start(20)

w.show()
w.setWindowTitle('Lab4')
w.setCameraPosition(pos=pg.Vector(m), distance=np.linalg.norm(S) * 1.5)
pg.exec()  # MUST be last (event loop)
