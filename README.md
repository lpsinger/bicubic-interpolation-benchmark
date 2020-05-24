# Bicubic interpolation benchmark

Comparison of C, xtensor, and numba implementation of bicubic interpolation
with pre-evaluated coefficients, based on [cubic_interp.c] from [ligo.skymap].

[cubic_interp.c]: https://git.ligo.org/lscsoft/ligo.skymap/-/blob/master/src/cubic_interp.c
[ligo.skymap]: https://git.ligo.org/lscsoft/ligo.skymap

## To run

```
$ make
$ cat bench.txt 
C   0.005284
C++ 0.120126
Py  0.09751461100000003
```
