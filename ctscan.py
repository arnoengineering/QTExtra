import numpy as np
# import pandas as pd
from numpy import sin, cos
m = 10
n = 10
dep = 1

J = 10
k = 1


def main_f():
    body = np.random.randint(0, 255, (m, n, dep))

    eq_group = int(np.ceil(n*m/J))
    # grp = int(np.ceil(dep / k))

    cen = np.array(body.shape) // 2
    theta = np.linspace(0, np.pi, eq_group)

    eq_full = []
    si = body.shape
    siz = np.prod(si)
    eq_sum = np.zeros(siz, 'int')
    eq = np.zeros((siz, siz), 'int')
    print(body[:,:,0])
    ri = np.arange(max(m,n)/2, dtype='int').reshape((-1,1))
   
    ji = np.arange(J)-J/2
    ni = 0
    jk = 0
    for th in theta:
        print('th: ', th)
        th_r = np.array((cos(th), sin(th))).reshape((1,2))
        th2 = np.array((-th_r[0,1],th_r[0,0]))
        r_pos = ri@th_r
        for j in ji:
            print('j: ', j)

            j_sum = np.zeros(k,'int')

            p = r_pos+th2*j
            print('rp')
            p = np.fliplr(p)+cen[:2]
            print(p)

            # tp = cen[0] * mn * np.array([-1,1])-p[1]-p[0]
            # xr = (max(-cen[1],min(tp)),min(cen[1]-1, max(tp)))
            # tis = np.arange(-cen[1], cen[1])
            for lp in p:
                lp = np.array((m-lp[0],lp[1]),'int')
                if m > lp[0] >= 0 and n > lp[1] >= 0:
                    print('p', lp)
                    j_sum += body[lp[0],lp[1],:]
                    mn = m*lp[1]+lp[0]
                    for ki in range(k):
                        eq[ni*k+ki, m*n*ki+mn] = 1

                # loc *= np.array((1, m, m * n))
            ni+=1
            eq_sum[jk:jk+k] = j_sum
            jk += 1

    print('eq_sum')
    print(eq_sum)
    print('mat')
    nii = 0
    for i in eq:
        if np.all(i == 0):
            print('error, ', nii)
        nii += 1

    save(eq, eq_sum)
    so(eq, eq_sum, body.shape)


def save(eq_full, eq_sum):
    # pixel, in real?
    with open('full.npy', 'wb') as f:
        np.save(f, eq_full)

    with open('su.npy', 'wb') as fi:
        np.save(fi, eq_sum)
    np.savetxt('full.csv', eq_full, delimiter=',')
    np.savetxt('su.csv', eq_sum, delimiter=',')


def load():
    with open('full.npy', 'rb') as f:
        eq_full = np.load(f)
    with open('su.npy', 'rb') as fi:
        eq_sum = np.load(fi)
    return eq_full, eq_sum


def so(eq_full, eq_sum, sh):
    print(f'full size: {eq_full.shape}, sum_size: {eq_sum.shape}')
    print(eq_full)
    solve_v = np.linalg.solve(eq_full, eq_sum)
    solve_m = solve_v.reshape(sh)
    print('Sol')
    print(solve_m)


def read_norm():
    so(*load(), (m, n, dep))


main_f()
# read_norm()

