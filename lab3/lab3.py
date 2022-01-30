import imageio, numpy as np, scipy.stats as ss, scipy.ndimage as nd, os
from datetime import datetime
from numpy import array as arr

def process(fname, eps, iters, r0, r1, em_iters):
    if not os.path.exists('proc'): os.makedirs('proc')
    if not os.path.exists('proc/iter'): os.makedirs('proc/iter')
    if not os.path.exists('proc/probs'): os.makedirs('proc/probs')
    print(datetime.now(), '| Processing started')
    im = imageio.imread(fname)[:,:,:3]  # Image without alpha
    im0 = im[r0[1]:r0[1]+r0[3], r0[0]:r0[0]+r0[2]]  # y, x
    im1 = im[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    d0, d1 = calc_class(im0), calc_class(im1)
    print('rect sampled dist_0:', d0.mean)
    print(d0.cov)
    print('rect sampled dist_1:', d1.mean)
    print(d1.cov)
    initial_cov = [
        [10000, 10, 10],
        [10, 10000, 10],
        [10, 10, 10000]
    ]
    initial_mu = [100, 50]
    initial_d0 = ss.multivariate_normal([initial_mu[0], initial_mu[0], initial_mu[0]], initial_cov)
    initial_d1 = ss.multivariate_normal([initial_mu[1], initial_mu[1], initial_mu[1]], initial_cov)
    d0, d1 = EM_process(im, initial_d0, initial_d1, em_iters, d0, d1)
    print('EM calculated dist_0:', d0.mean)
    print(d0.cov)
    print('EM calculated dist_1:', d1.mean)
    print(d1.cov)
    p0, p1 = d0.pdf(im), d1.pdf(im)
    ps = p0 + p1
    p0, p1 = p0 / ps, p1 / ps  # normalize
    same, diff, _ = 1 - eps, eps, 0
    npc = np.where(p0 > p1, 0, 1)
    img0 = write_cls_im(npc, npc, 0, True)
    write_cls_im(npc, npc, 'start', top=True)
    nps = [npc]
    imageio.imwrite('proc/probs/0.png', (np.dstack((p0, p0*0, p1)) * 255).astype(np.uint8))
    kernel = [[_, 1, _],
              [1, _, 1],
              [_, 1, _]]
    # --------------------------------------------------------
    for it in range(1, iters+1):
        nb1 = nd.correlate(npc, kernel)  # calc number of neighbours for each pixel
        nb0 = 4 - nb1  # nd.correlate(1 - npc, kernel)
        m0 = (same ** nb0) * (diff ** nb1)
        m1 = (same ** nb1) * (diff ** nb0)
        p0adj, p1adj = p0 * m0, p1 * m1
        psum = p0adj + p1adj
        p0adj, p1adj = p0adj / psum, p1adj / psum  # normalize
        npf = np.where(p0adj > p1adj, 0, 1)
        npc, p0, p1 = npf, p0adj, p1adj
        imageio.imwrite('proc/probs/' + str(it) + '.png', (np.dstack((p0, p0*0, p1)) * 255).astype(np.uint8))
        nps.append(npc)
        print(datetime.now(), '| Iteration ' + str(it) + '/' + str(iters))
    print(datetime.now(), '| Processing completed!')
    print(datetime.now(), '| Writing images...')
    write_cls_im(npc, npc, 'final', top=True)
    write_cls_im(nps[0], npc, 'diff', top=True)
    with imageio.get_writer('proc/animation.gif', mode='I', duration=0.1) as writer:
        writer.append_data(img0)
        for it in range(1, iters+1):
            print(datetime.now(), '| Writing image ' + str(it) + '/' + str(iters))
            img = write_cls_im(nps[it-1], nps[it], it, True)
            writer.append_data(img)
    print(datetime.now(), '| Completed!')

def EM_process(im, init0, init1, iters, rd0, rd1):
    pk0 = pk1 = 0.5
    mu0, cov0 = arr(init0.mean), arr(init0.cov)
    mu1, cov1 = arr(init1.mean), arr(init1.cov)
    # al0, al1 = np.zeros(im.shape), np.zeros(im.shape)
    chn = [im[:,:,i] for i in range(im.shape[2])]
    cnt = im.shape[0] * im.shape[1]
    sampler_stripe, sampler_height = 51, 50
    sampler_im = np.zeros((2 * sampler_height + 4*7, sampler_stripe * (iters + 2), 3))
    sampler = sampler_im, sampler_height, sampler_stripe
    EM_sampler(sampler, -1, init0, init1, '    - 0 -   ', '    - 1 -   ')
    for it in range(iters):
        # Expectation (Step 1)
        # 2a
        pr0 = ss.multivariate_normal(mu0, cov0).pdf(im)
        pr1 = ss.multivariate_normal(mu1, cov1).pdf(im)
        al0, al1 = pk0 * pr0, pk1 * pr1
        als = np.maximum(al0 + al1, 10 ** -100)
        al0, al1 = np.maximum(al0 / als, 10 ** -9), np.maximum(al1 / als, 10 ** -9)
        as0, as1 = np.sum(al0), np.sum(al1)
        # 2b
        pk0, pk1 = as0 / cnt, as1 / cnt
        # Maximization (Step 2)
        # 2c
        for c1 in range(len(chn)):
            mu0[c1] = np.sum(al0 * chn[c1]) / as0
            mu1[c1] = np.sum(al1 * chn[c1]) / as1
            for c2 in range(len(chn)):
                cov0[c1, c2] = np.sum(al0 * (chn[c1] - mu0[c1]) * (chn[c2] - mu0[c2])) / as0
                cov1[c1, c2] = np.sum(al1 * (chn[c1] - mu1[c1]) * (chn[c2] - mu1[c2])) / as1
        # Evaluation
        # calculate and output sum under argmax
        d0 = ss.multivariate_normal(mu0, cov0)
        d1 = ss.multivariate_normal(mu1, cov1)
        sk0 = np.sum(al0 * np.log(d0.pdf(im)))
        sk1 = np.sum(al1 * np.log(d1.pdf(im)))
        print(datetime.now(), '| EM', it+1, ': S =', sk0 + sk1)
        EM_sampler(sampler, it, d0, d1, sk0, sk1)
    print(datetime.now(), '| Writing EM iterational samplers to proc/em_iters.png')
    EM_sampler(sampler, iters, rd0, rd1, '------------', '------------')
    imageio.imwrite('proc/em_iters.png', sampler_im.astype(np.uint8))
    return ss.multivariate_normal(mu0, cov0), \
           ss.multivariate_normal(mu1, cov1)

def EM_sampler(sampler, it, d0, d1, sk0, sk1):
    sampler_im, sampler_height, sampler_stripe = sampler
    sampler_im[7:7+sampler_height, (it + 1) * sampler_stripe:(it + 2) * sampler_stripe - 1] = \
        np.clip(np.reshape(d0.rvs(sampler_height * (sampler_stripe - 1)),
                           (sampler_height, sampler_stripe - 1, 3)), 0, 255)
    sampler_im[7+sampler_height:7+2 * sampler_height, (it + 1) * sampler_stripe:(it + 2) * sampler_stripe - 1] = \
        np.clip(np.reshape(d1.rvs(sampler_height * (sampler_stripe - 1)),
                           (sampler_height, sampler_stripe - 1, 3)), 0, 255)
    if sk0 != '------------':
        overlay_num(sampler_im, it + 1, (it + 1) * sampler_stripe + 1, 1)
    overlay_num(sampler_im, str(sk0)[:12], (it + 1) * sampler_stripe + 1, sampler_height * 2 + 1 + 1*7, [255, 127, 127])
    overlay_num(sampler_im, str(sk1)[:12], (it + 1) * sampler_stripe + 1, sampler_height * 2 + 1 + 2*7, [127, 127, 255])
    if sk0 != '    - 0 -   ':
        overlay_num(sampler_im, str(sk0+sk1)[:12], (it + 1) * sampler_stripe + 1, sampler_height * 2 + 1 + 3*7, [255, 127, 255])
    else:
        overlay_num(sampler_im, '   - 0+1 -  ', (it + 1) * sampler_stripe + 1, sampler_height * 2 + 1 + 3 * 7, [255, 127, 255])

def calc_class(im):
    a = (0, 1)
    mu = np.mean(im, a)
    cim = np.subtract(im, mu)
    cov = np.zeros([3, 3])
    for x in range(0, 3):
        for y in range(0, 3):
            cov[x, y] = np.mean(cim[:, :, x] * cim[:, :, y])
    return ss.multivariate_normal(mu, cov)

def gen_cls_im(old, new):
    img = np.zeros(old.shape + (3,), dtype=np.uint8)
    msk = old + 2 * new  # 0: 0->0  #  1: 1->0  #  2: 0->1  #  3: 1->1
    cols = {
        0: [100,   0,   0],  # 0->0: dark red
        1: [255,   0,   0],  # 1->0: bright red
        2: [  0,   0, 255],  # 0->1: bright blue
        3: [  0,   0, 100]   # 1->1: dark blue
    }
    new0, new1 = 0, 0
    for ix in range(0, old.shape[0]):
        for iy in range(0, old.shape[1]):
            img[ix, iy] = cols[msk[ix, iy]]
            if msk[ix, iy] == 1: new0 += 1
            if msk[ix, iy] == 2: new1 += 1
    return img, new0, new1

def write_cls_im(old, new, it, overlay=False, top=False):
    img, new0, new1 = gen_cls_im(old, new)
    if overlay:
        overlay_num(img, it, col=[127, 255, 127])
        overlay_num(img, new0, 5, 11, [255, 127, 127])
        overlay_num(img, new1, 5, 17, [127, 127, 255])
    print('Written image ' + str(it) + ', new 0: ' + str(new0) + ', new 1: ' + str(new1))
    dr = '' if top else 'iter/'
    imageio.imwrite('proc/' + dr + str(it) + '.png', img)
    return img

def overlay_num(img, it, bx=5, by=5, col=None):
    if col is None:
        col = [255, 255, 255]
    numbers = ['###  #  ### ### # # ### ### ### ### ###            ',
               '# #  #    #   # # # #   #     # # # # #          # ',
               '# #  #  ### ### ### ### ###   # ### ### ###     ###',
               '# #  #  #     #   #   # # #   # # #   #          # ',
               '###  #  ### ###   # ### ###   # ### ###      #     ']
    sit = str(it)
    offs = 0
    for n in sit:
        if n != ' ':
            if n == '-': d = 40
            elif n == '.': d = 44
            elif n == '+': d = 48
            else: d = int(n) * 4
            for y in range(0, 5):
                for x in range(0, 3):
                    if numbers[y][d+x] == '#':
                        img[by+y,bx+x+offs] = col
        offs += 4
    return img

# def mvn_pdf(x, mu, cov):  # dropped in favor of
#     x_m = x - mu          #     ss.multivariate_normal
#     return (1. / (np.sqrt((2 * np.pi) ** 3 * np.linalg.det(cov))) *
#             np.exp(-(np.linalg.solve(cov, x_m).T.dot(x_m)) / 2))
    # print(mvn_pdf(im1[0,0], mu1, cov1), mvn_pdf(im1[0,0], mu2, cov2))
    # print(d1.pdf(im1[0,0]), d2.pdf(im1[0,0]))
    # 2.7134964240017446e-05    3.5603913656072235e-276
    # 2.7134964240017528e-05    3.5603913656065934e-276

process('image.jpg', 0.1, 10,  # Gibbs iters
        (20,  20,  50, 50),  # class 0 sample rect
        (20, 220,  50, 50),  # class 1 sample rect
        20)  # EM iters
