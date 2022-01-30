#!/usr/bin/env python
# coding: utf-8
import math, imageio as imio, numpy as np
import scipy.stats as ss, os, glob
from datetime import datetime
from numba import njit, prange

#########################################################################################################
#########################################################################################################
#########################################################################################################

configuration = {
    'image_file': 'image.jpg',         # image filename
    'cls_0_rect': (20,  20,  50, 50),  # class 0 sample rect
    'cls_1_rect': (20, 220,  50, 50),  # class 1 sample rect
    'prob_ndiff': 0.1,      # eps from lab 1, prob diff neigh
    'diffuse_it': 200,      # amount of diffusion iterations
    'reduction': {
        'iter_limit': 200,  # stop when this many iterations
        'eps_limit': 0,     # stop when eps reaches this
        'indec_want': 0,    # okay that many indecided cells
        'eps_dec': 0.9,     # if has markup: eps *= eps_dec
        'eps_inc': 1.05     # if no markup:  eps *= eps_inc
    },
    'images': {               # draw an image each nth iteration
        'diffusion_lnk': 10,  # links image - SLOW (~ 7 sec + compilation)
        'reduction_pix': 1,   # class image - FAST
        'reduction_lnk': 10   # links image - SLOW
    }
}

#########################################################################################################
#########################################################################################################
#########################################################################################################

R, D, L, U = 0, 1, 2, 3
SRC_K, DST_K, DIR_D = 1, 2, 4

#########################################################################################################
# ................................................. #          RIGHT          #          RIGHT          #
# ................................................. #   #####         #####   #   #####         #####   #
# ................................................. #   # 0 #  --0->  # 0 #   #   #   #    1->  # 0 #   #
# ................................................. #   #   #   \     #   #   #   #   #   /     #   #   #
# ................................................. #   #   #    2->  # 1 #   #   # 1 #  --3->  # 1 #   #
# ................................................. #   #####         #####   #   #####         #####   #
# ................................................. #                         #                         #
#########################################################################################################
#           DOWN          #           DOWN          #
#   #####         #####   #   #####         #####   #
#   # 0 #  --4->  # 0 #   #   #   #    5->  # 0 #   #
#   #   #   \     #   #   #   #   #   /     #   #   #
#   #   #    6->  # 1 #   #   # 1 #  --7->  # 1 #   #
#   #####         #####   #   #####         #####   #
#                         #                         #
#####################################################

@njit()
def rk_nat(w, h, y, x, kf, kt, dr):
    if x == 0 and dr == L: return -1, -1, -1
    if y == 0 and dr == U: return -1, -1, -1
    if x == w - 1 and dr == R: return -1, -1, -1,
    if y == h - 1 and dr == D: return -1, -1, -1
    if dr == L: x, dr, kf, kt = x - 1, R, kt, kf
    if dr == U: y, dr, kf, kt = y - 1, D, kt, kf
    return y, x, SRC_K * kf + DST_K * kt + DIR_D * dr
    #               1    0,1     2    0,1     4    R:0, D:1

#########################################################################################################

@njit()
def broadcast_pixel_probs_nat(w, h, lp0, lp1, gr):
    for y in range(0, h):
        for x in range(0, w):
            for kt in range(2):
                for dr in range(4):
                    iy, ix, ic = rk_nat(w, h, y, x, 0, kt, dr)
                    if ic != -1:
                        gr[iy, ix, ic] += lp0[y, x]
                        iy, ix, ic = rk_nat(w, h, y, x, 1, kt, dr)
                        gr[iy, ix, ic] += lp1[y, x]

#########################################################################################################

@njit()
def broadcast_neigh_probs_nat(w, h, p_diff, gr):
    g_same, g_diff = np.log(1 - p_diff), np.log(p_diff)
    for y in range(0, h):
        for x in range(0, w):
            for kf in range(2):
                for kt in range(2):
                    for dr in range(4):
                        iy, ix, ic = rk_nat(w, h, y, x, kf, kt, dr)
                        if ic != -1:
                            gr[iy, ix, ic] += g_same if kf == kt else g_diff

#########################################################################################################

@njit()
def diffuse_iter_nat(w, h, gr, gd):
    for y in range(0, h):
        for x in range(0, w):
            for k in range(0, 2):
                for d in range(0, 4):
                    iy, ix, ic = rk_nat(w, h, y, x, k, 0, d)
                    jy, jx, jc = rk_nat(w, h, y, x, k, 1, d)
                    if ic != -1:
                        gmk = max(gr[iy, ix, ic], gr[jy, jx, jc])
                        for kk in range(0, 2):
                            # ky, kx, kc = rk_nat(w, h, y, x, k, kk, d)
                            kc = ic if kk == 0 else jc
                            val, n = 0, 0
                            for dd in range(0, 4):
                                ly, lx, lc = rk_nat(w, h, y, x, k, 0, dd)
                                if lc == -1: continue
                                my, mx, mc = rk_nat(w, h, y, x, k, 1, dd)
                                val += max(gr[ly,lx,lc], gr[my,mx,mc])
                                n += 1
                            gd[iy, ix, kc] = gd[iy, ix, kc] - gmk + val / n

#########################################################################################################

@njit()
def find_max_min_nat(w, h, gr):
    fmin = math.inf
    fmax = -math.inf
    for y in range(0, h):
        for x in range(0, w):
            for kf in range(2):
                for kt in range(2):
                    for dr in range(4):
                        iy, ix, ic = rk_nat(w, h, y, x, kf, kt, dr)
                        if ic != -1:
                            v = gr[iy, ix, ic]
                            if v > fmax: fmax = v
                            if v < fmin: fmin = v
    return fmin, fmax

#########################################################################################################

@njit(parallel=True)
def fill_points_nat_par(w, h, src, img):
    for y in prange(0, h):
        for x in prange(0, w):
            img[12 * y:12 * y + 8, 12 * x:12 * x + 8] = src[y, x] / 512

#########################################################################################################

@njit(parallel=True)
def fill_marks_nat_par(w, h, ps, img):
    red = np.array([1,0,0])
    grn = np.array([0,.7,0])
    for y in prange(0, h):
        for x in prange(0, w):
            if ps[y, x, 0]:
                img[12*y+1, 12*x+1:12*x+6] = red
                img[12*y+1:12*y+6, 12*x+1] = red
            if ps[y, x, 1]:
                img[12*y+6, 12*x+2:12*x+7] = grn
                img[12*y+2:12*y+7, 12*x+6] = grn

#########################################################################################################

@njit(parallel=True)
def fill_links_nat_par(w, h, gr, gs, img):
    fmin, fmax = find_max_min_nat(w, h, gr)
    mul_r = 1 / (fmax - fmin)
    mul_g = mul_r * 0.7
    for y in prange(0, h):
        for x in prange(0, w):
            # RIGHT
            iy, ix, ic = rk_nat(w, h, y, x, 0, 0, R)
            if gs[iy, ix, ic]:
                img[12*y+2, 12*x+8:12*x+12, 0] = (gr[iy, ix, ic] - fmin) * mul_r
            iy, ix, ic = rk_nat(w, h, y, x, 1, 1, R)
            if gs[iy, ix, ic]:
                img[12*y+5, 12*x+8:12*x+12, 1] = (gr[iy, ix, ic] - fmin) * mul_g
            iy, ix, ic = rk_nat(w, h, y, x, 0, 1, R)
            if gs[iy, ix, ic]:
                img[12*y+3, 12*x+8:12*x+10, 0] = (gr[iy, ix, ic] - fmin) * mul_r
                img[12*y+3, 12*x+10:12*x+12, 1] = (gr[iy, ix, ic] - fmin) * mul_g
            iy, ix, ic = rk_nat(w, h, y, x, 1, 0, R)
            if gs[iy, ix, ic]:
                img[12*y+4, 12*x+8:12*x+10, 1] = (gr[iy, ix, ic] - fmin) * mul_g
                img[12*y+4, 12*x+10:12*x+12, 0] = (gr[iy, ix, ic] - fmin) * mul_r
            # DOWN
            iy, ix, ic = rk_nat(w, h, y, x, 0, 0, D)
            if gs[iy, ix, ic]:
                img[12*y+8:12*y+12, 12*x+2, 0] = (gr[iy, ix, ic] - fmin) * mul_r
            iy, ix, ic = rk_nat(w, h, y, x, 1, 1, D)
            if gs[iy, ix, ic]:
                img[12*y+8:12*y+12, 12*x+5, 1] = (gr[iy, ix, ic] - fmin) * mul_g
            iy, ix, ic = rk_nat(w, h, y, x, 0, 1, D)
            if gs[iy, ix, ic]:
                img[12*y+8:12*y+10, 12*x+3, 0] = (gr[iy, ix, ic] - fmin) * mul_r
                img[12*y+10:12*y+12, 12*x+3, 1] = (gr[iy, ix, ic] - fmin) * mul_g
            iy, ix, ic = rk_nat(w, h, y, x, 1, 0, D)
            if gs[iy, ix, ic]:
                img[12*y+8:12*y+10, 12*x+4, 1] = (gr[iy, ix, ic] - fmin) * mul_g
                img[12*y+10:12*y+12, 12*x+4, 0] = (gr[iy, ix, ic] - fmin) * mul_r

# #########################################################################################################
#
# @njit()
# def radial_max_nat(w, h, y, x, gr, gs):
#     mx = -math.inf
#     for kf in range(2):
#         for kt in range(2):
#             for dr in range(4):
#                 iy, ix, ic = rk_nat(w, h, y, x, kf, kt, dr)
#                 if ic != -1 and gs[iy, ix, ic]:
#                     v = gr[iy, ix, ic]
#                     if v > mx: mx = v
#     return mx
#
# #########################################################################################################
#
# @njit(parallel=True)
# def cross_links_sieve_nat_par(w, h, gr, gs, eps):
#     amt = 0
#     for y in prange(0, h):
#         for x in prange(0, w):
#             rmax = radial_max_nat(w, h, y, x, gr, gs)
#             for kf in range(2):
#                 for kt in range(2):
#                     for dr in range(4):
#                         iy, ix, ic = rk_nat(w, h, y, x, kf, kt, dr)
#                         if ic != -1 and gs[iy, ix, ic] and rmax - gr[iy, ix, ic] > eps:
#                             gs[iy, ix, ic] = False
#                             amt += 1
#     return amt
#
#########################################################################################################

@njit(parallel=True)
def cross_links_sieve_nat_par(w, h, gr, gs, eps):
    amt = 0
    for y in prange(0, h):
        for x in prange(0, w):
            for dr in range(4):
                for kf in range(2):
                    iy, ix, ic = rk_nat(w, h, y, x, kf, 0, dr)
                    jy, jx, jc = rk_nat(w, h, y, x, kf, 1, dr)
                    if ic != -1:
                        rmax = max(gr[iy, ix, ic], gr[jy, jx, jc])
                        if gs[iy, ix, ic] and rmax - gr[iy, ix, ic] > eps:
                            gs[iy, ix, ic] = False
                            amt += 1
                        if gs[jy, jx, jc] and rmax - gr[jy, jx, jc] > eps:
                            gs[jy, jx, jc] = False
                            amt += 1
    return amt

#########################################################################################################

@njit(parallel=True)
def cross_nodes_nat_par(w, h, gs, ps):
    amt = 0
    for y in prange(0, h):
        for x in prange(0, w):
            for kf in range(0, 2):
                if not ps[y, x, kf]:
                    continue
                has = True
                for dr in range(0, 4):
                    iy, ix, ic = rk_nat(w, h, y, x, kf, 0, dr)
                    jy, jx, jc = rk_nat(w, h, y, x, kf, 1, dr)
                    if ic == -1: continue
                    if (not gs[iy, ix, ic]) and (not gs[jy, jx, jc]):
                        has = False
                        break
                if not has:
                    ps[y, x, kf] = False
                    amt += 1
    return amt

#########################################################################################################

@njit(parallel=True)
def cross_links_nat_par(w, h, gs, ps):
    amt = 0
    for y in prange(0, h):
        for x in prange(0, w):
            for kf in range(0, 2):
                if not ps[y, x, kf]:
                    for kt in range(2):
                        for dr in range(4):
                            iy, ix, ic = rk_nat(w, h, y, x, kf, 0, dr)
                            if gs[iy, ix, ic]:
                                gs[iy, ix, ic] = False
                                amt += 1
    return amt

#########################################################################################################

@njit()
def markup_check_nat(w, h, ps):
    amt = 0
    for y in range(0, h):
        for x in range(0, w):
            if not (ps[y, x, 0] or ps[y, x, 1]):
                return -1
            if ps[y, x, 0] and ps[y, x, 1]:
                amt += 1
    return amt

#########################################################################################################

def overlay_num(img, it, bx=5, by=5, col=None):
    if col is None:
        col = [255, 255, 255]
    numbers = ['###  #  ### ### # # ### ### ### ### ###    ',
               '# #  #    #   # # # #   #     # # # # #    ',
               '# #  #  ### ### ### ### ###   # ### ###    ',
               '# #  #  #     #   #   # # #   # # #   #    ',
               '###  #  ### ###   # ### ###   # ### ###  # ']
    sit = str(it)
    offs = 0
    for n in sit:
        d = 40 if n == '.' else int(n) * 4
        for y in range(0, 5):
            for x in range(0, 3):
                if numbers[y][d+x] == '#':
                    img[by+y,bx+x+offs] = col
        offs += 4
    return img

def prepare_paths():
    if not os.path.exists('proc'): os.makedirs('proc')
    if not os.path.exists('proc/diff'): os.makedirs('proc/diff')
    if not os.path.exists('proc/redc_pix'): os.makedirs('proc/redc_pix')
    if not os.path.exists('proc/redc_lnk'): os.makedirs('proc/redc_lnk')
    files = glob.glob('proc/diff/*.png') + glob.glob('proc/redc_pix/*.png') +\
            glob.glob('proc/redc_lnk/*.png') + glob.glob('proc/*.png')
    for f in files:
        os.remove(f)

#########################################################################################################
#########################################################################################################

class Logic:
    def __init__(self):
        self.img = np.array([])  # image
        self.im = {}  # class images
        self.cd = {}  # class distributions
        self.pd = {}  # probability density
        self.lp = {}  # logarithmic pdf
        self.gr = np.array([])  # grid
        self.gd = np.array([])  # doubler grid
        self.gs = np.array([])  # strike grid
        self.ps = np.array([])  # strike points
        self.a = (0, 1)  # axis
        self.w, self.h = 0, 0

    def load_image(self, filename):
        self.img = imio.imread(filename)[:, :, :3]
        gs = self.img.shape
        self.gr = np.zeros((gs[0], gs[1], 8))
        self.gd = np.zeros((gs[0], gs[1], 8))
        self.gs = np.ones((gs[0], gs[1], 8), dtype=bool)
        self.ps = np.ones((gs[0], gs[1], 3), dtype=bool)
        self.w, self.h = gs[1], gs[0]

    def splice_sample(self, class_id, rect):
        self.im[class_id] = im = self.img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        mu = np.mean(im, self.a)
        cim = np.subtract(im, mu)
        cov = np.zeros([3, 3])
        for x in range(0, 3):
            for y in range(0, 3):
                cov[x, y] = np.mean(cim[:, :, x] * cim[:, :, y])
        self.cd[class_id] = cd = ss.multivariate_normal(mu, cov)
        self.pd[class_id] = cd.pdf(self.img)
        self.lp[class_id] = np.log(self.pd[class_id] + 10**-100)

    def rk(self, y, x, kf, kt, dr):  # resolve_key
        return rk_nat(self.w, self.h, y, x, kf, kt, dr)

    def broadcast_pixel_probs(self):
        broadcast_pixel_probs_nat(self.w, self.h, self.lp[0], self.lp[1], self.gr)

    def broadcast_neigh_probs(self, p_diff):
        broadcast_neigh_probs_nat(self.w, self.h, p_diff, self.gr)

    def diffuse_iter(self):
        np.copyto(self.gd, self.gr)
        diffuse_iter_nat(self.w, self.h, self.gr, self.gd)
        np.copyto(self.gr, self.gd)

    def get_min_max(self):
        return find_max_min_nat(self.w, self.h, self.gr)

    def make_links_image(self, mark_pts=False):
        img = np.zeros([self.h * 12, self.w * 12, 3])
        fill_points_nat_par(self.w, self.h, self.img, img)
        fill_links_nat_par(self.w, self.h, self.gr, self.gs, img)
        if mark_pts:
            fill_marks_nat_par(self.w, self.h, self.ps, img)
        return img

    def cross_links_nodes_reset(self):
        self.gs[:,:,:] = True
        self.ps[:,:,:] = True
        self.ps[:,:,2] = False

    def cross_links_sieve(self, eps):
        return cross_links_sieve_nat_par(self.w, self.h, self.gr, self.gs, eps)

    def cross_nodes(self):
        return cross_nodes_nat_par(self.w, self.h, self.gs, self.ps)

    def cross_links(self):
        return cross_links_nat_par(self.w, self.h, self.gs, self.ps)

    def markup_check(self):
        return markup_check_nat(self.w, self.h, self.ps)

    def do_reduction(self, iter_limit, eps_limit, indec_want, eps_dec, eps_inc, imwrite_pix=0, imwrite_lnk=0):
        fmin, fmax = self.get_min_max()
        eps, it = fmax - fmin, 1
        while True:
            self.cross_links_nodes_reset()
            tcs = self.cross_links_sieve(eps)
            tcn, tcl, nn, nl, cit = \
                0, tcs, 1, 1, 0
            while nn + nl > 0:
                nn = self.cross_nodes()
                nl = self.cross_links()
                tcn, tcl, cit = \
                    tcn + nn, tcl + nl, cit + 1
            mc = self.markup_check()
            end = False
            if mc > -1:
                print(datetime.now(),
                      '| (' + str(it) + ') Markup exists, indecisive elements: ' + str(mc) +
                      ', crossed ' + str(tcn) + ' nodes and ' + str(tcl) + ' links' +
                      ' in ' + str(cit) + ' cross iters for eps = ' + str(eps))
                if eps <= eps_limit:
                    print('Reached eps limit of ' + str(eps) + ' <= ' + str(eps_limit))
                    end = True
                eps *= eps_dec
                if mc <= indec_want:
                    end = True
                if it >= iter_limit != 0:
                    print('Reached iteration limit of ' + str(it))
                    end = True
            else:
                print(datetime.now(),
                      '| (' + str(it) + ') Markup does not exist, crossed ' +
                      str(tcn) + ' nodes and ' + str(tcl) + ' links' +
                      ' in ' + str(cit) + ' cross iters for eps = ' + str(eps))
                eps *= eps_inc
            if imwrite_lnk > 0 and (it % imwrite_lnk == 0 or it == 1 or end):
                print(datetime.now(), '| Writing links image to proc/redc_lnk/' + str(it) + '.png ...')
                imio.imwrite('proc/redc_lnk/' + str(it) + '.png', overlay_num((self.make_links_image(True) * 255).astype(np.uint8), it))
            if imwrite_pix > 0 and (it % imwrite_pix == 0 or it == 1 or end):
                imio.imwrite('proc/redc_pix/' + str(it) + '.png', overlay_num(self.ps.astype(np.uint8) * 255, it))
            if end: break
            it += 1

#########################################################################################################

def main():
    prepare_paths()
    l = Logic()
    print(datetime.now(), '| Loading image')
    l.load_image(configuration['image_file'])
    print(datetime.now(), '| Preparing')
    l.splice_sample(0, configuration['cls_0_rect'])
    l.splice_sample(1, configuration['cls_1_rect'])
    l.broadcast_pixel_probs()
    l.broadcast_neigh_probs(configuration['prob_ndiff'])
    print(datetime.now(), '| Diffusion start')
    for it in range(1, configuration['diffuse_it']+1):
        l.diffuse_iter()
        print(datetime.now(), '| Diffusion iteration ' + str(it))
        imwrite_lnk = configuration['images']['diffusion_lnk']
        if imwrite_lnk > 0 and (it % imwrite_lnk == 0 or it == 1 or it == configuration['diffuse_it']):
            print(datetime.now(), '| Writing links image to proc/diff/' + str(it) + '.png ...')
            imio.imwrite('proc/diff/' + str(it) + '.png', overlay_num((l.make_links_image(False) * 255).astype(np.uint8), it))
    print(datetime.now(), '| Diffusion end')
    print(datetime.now(), '| Reduction start')
    rc = configuration['reduction']
    l.do_reduction(rc['iter_limit'], rc['eps_limit'], rc['indec_want'], rc['eps_dec'], rc['eps_inc'],
                   configuration['images']['reduction_pix'], configuration['images']['reduction_lnk'])
    print(datetime.now(), '| Reduction end')
    print(datetime.now(), '| Writing final images to proc...')
    imio.imwrite('proc/links.png', (l.make_links_image(True) * 255).astype(np.uint8))
    imio.imwrite('proc/class.png', l.ps.astype(np.uint8) * 255)
    print(datetime.now(), '| Completed!')
    # plt.imshow(l.make_links_image(True))  #[1500:1604,756:920,:])
    # plt.show()

if __name__ == '__main__':
    main()
