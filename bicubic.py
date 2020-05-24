#!/usr/bin/env python
from time import perf_counter

from numba import njit
import numpy as np


@njit(fastmath=True)
def bicubic_interp_eval(a, fx, x0, x):
    xlength = np.asarray(a.shape[:2]).astype(np.float64)
    x = np.minimum(np.maximum(x * fx + x0, 0.0), xlength - 1.0)
    xi = np.floor(x)
    x -= xi
    i = (xi[:, 0] * xlength[0] + xi[:, 1]).astype(np.intp)

    aa = a.reshape(-1, 4, 4)[i]
    t = np.expand_dims(x[:, 1], -1)
    bb = ((aa[..., 0, :] * t + aa[..., 1, :]) * t + aa[..., 2, :]) + aa[..., 3, :]
    s = x[:, 0]
    return ((bb[..., 0] * s + bb[..., 1]) * s + bb[..., 2]) * s + bb[..., 3]


if __name__ == '__main__':
    a = np.random.uniform(size=(406, 406, 4, 4))
    fx = np.asarray([1.0, 1.0])
    x0 = np.asarray([3.0, 3.0])
    x = np.random.uniform(0.0, 400.0, size=(100000, 2))
    result = bicubic_interp_eval(a, fx, x0, x)

    x = np.random.uniform(0.0, 400.0, size=(100000, 2))

    start = perf_counter()
    result = bicubic_interp_eval(a, fx, x0, x)
    stop = perf_counter()
    print(stop - start)
