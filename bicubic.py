#!/usr/bin/env python
import logging
from time import perf_counter

logging.basicConfig(level=logging.INFO)

from numba import njit
import numpy as np


@njit(fastmath=True, inline='always')
def clip(x, a, b):
    return np.minimum(np.maximum(x, a), b)


@njit('(f8[:, :, :, ::1], f8, f8, f8, f8, f8[:], f8[:])', fastmath=True)
def bicubic_interp_eval(a, fs, ft, s0, t0, s, t):
    s = clip(s * fs + s0, 0, a.shape[0] - 1)
    t = clip(t * ft + t0, 0, a.shape[0] - 1)
    si = np.floor(s)
    ti = np.floor(t)
    s -= si
    t -= ti
    si = si.astype(np.intp)
    ti = si.astype(np.intp)

    aa = a.reshape(-1, 4, 4)[si * a.shape[0] + ti]
    t = np.expand_dims(t, -1)
    bb = ((aa[..., 0, :] * t + aa[..., 1, :]) * t + aa[..., 2, :]) + aa[..., 3, :]
    return ((bb[..., 0] * s + bb[..., 1]) * s + bb[..., 2]) * s + bb[..., 3]


    # s = np.minimum(np.maximum(s * fs + s0, 0), a.shape[0] - 1)
    # t = np.minimum(np.maximum(t * ft + t0, 0), a.shape[0] - 1)
    # si = np.floor(s)
    # ti = np.floor(t)
    # s -= si
    # t -= ti
    # si = si.astype(np.intp)
    # ti = si.astype(np.intp)
    # aa = a.reshape(-1, 4, 4)[si * a.shape[0] + ti]
    # t = np.expand_dims(t, -1)
    # bb = ((aa[..., 0, :] * t + aa[..., 1, :]) * t + aa[..., 2, :]) + aa[..., 3, :]
    # return ((bb[..., 0] * s + bb[..., 1]) * s + bb[..., 2]) * s + bb[..., 3]


    # def clip(x, a, b):
    #     return np.minimum(np.maximum(x, a), b)
    #
    # def cubic_interp_index(x, fx, x0, length):
    #     t = clip(x * fx + x0, 0, length - 1)
    #     i = np.floor(t)
    #     return t - i, t.astype(np.intp)
    #
    # s, si = cubic_interp_index(s, fs, s0, a.shape[0])
    # t, ti = cubic_interp_index(t, ft, t0, a.shape[1])
    # aa = a.reshape(-1, 4, 4)[si * a.shape[0] + ti]
    # t = np.expand_dims(t, -1)
    # bb = ((aa[..., 0, :] * t + aa[..., 1, :]) * t + aa[..., 2, :]) + aa[..., 3, :]
    # return ((bb[..., 0] * s + bb[..., 1]) * s + bb[..., 2]) * s + bb[..., 3]

    # tx = x * fx + x0
    # tx = np.maximum(tx, 0)
    # tx = np.minimum(tx, (np.asarray(a.shape) - 1)[np.newaxis, ...])
    # return tx
    # ix = np.floor(tx)
    # tx -= ix
    # ix = ix.astype(np.intp)
    # is_ = ix[..., 0]
    # it_ = ix[..., 1]
    # aa = a.reshape(-1, 4, 4)[is_ * a.shape[0] + it_]
    # s = tx[..., 0]
    # t = tx[..., 1][..., np.newaxis]
    # bb = ((aa[..., 0, :] * t + aa[..., 1, :]) * t + aa[..., 2, :]) + aa[..., 3, :]
    # return ((bb[..., 0] * s + bb[..., 1]) * s + bb[..., 2]) * s + bb[..., 3]


if __name__ == '__main__':
    a = np.random.uniform(size=(406, 406, 4, 4))
    fx = np.ones(2)
    x0 = np.asarray([0.0, 0.0])
    s, t = np.random.uniform(0.0, 400.0, size=(2, 1000000))
    result = bicubic_interp_eval(a, 1.0, 1.0, 0.0, 0.0, s, t)

    s, t = np.random.uniform(0.0, 400.0, size=(2, 1000000))

    start = perf_counter()
    result = bicubic_interp_eval(a, 1.0, 1.0, 0.0, 0.0, s, t)
    stop = perf_counter()
    print(stop - start)
