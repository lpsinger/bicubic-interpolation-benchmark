#include <stdalign.h>
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_rng.h>
#include <time.h>


#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)


typedef double v2df __attribute__ ((vector_size (2 * sizeof(double))));
typedef double v4df __attribute__ ((vector_size (4 * sizeof(double))));


typedef struct {
    double fs, ft, s0, t0, slength, tlength;
    alignas(sizeof(double) * 16) double a[][4][4];
} bicubic_interp;


static int clip_int(int t, int min, int max)
{
    if (t < min)
        return min;
    else if (t > max)
        return max;
    else
        return t;
}


static double clip_double(double t, double min, double max)
{
    if (t < min)
        return min;
    else if (t > max)
        return max;
    else
        return t;
}


/*
 * Calculate coefficients of the interpolating polynomial in the form
 *      a[0] * t^3 + a[1] * t^2 + a[2] * t + a[3]
 */
static void cubic_interp_init_coefficients(
    double *a, const double *z, const double *z1)
{
    if (UNLIKELY(!isfinite(z1[1] + z1[2])))
    {
        /* If either of the inner grid points are NaN or infinite,
         * then fall back to nearest-neighbor interpolation. */
        a[0] = 0;
        a[1] = 0;
        a[2] = 0;
        a[3] = z[1];
    } else if (UNLIKELY(!isfinite(z1[0] + z1[3]))) {
        /* If either of the outer grid points are NaN or infinite,
         * then fall back to linear interpolation. */
        a[0] = 0;
        a[1] = 0;
        a[2] = z[2] - z[1];
        a[3] = z[1];
    } else {
        /* Otherwise, all of the grid points are finite.
         * Use cubic interpolation. */
        a[0] = 1.5 * (z[1] - z[2]) + 0.5 * (z[3] - z[0]);
        a[1] = z[0] - 2.5 * z[1] + 2 * z[2] - 0.5 * z[3];
        a[2] = 0.5 * (z[2] - z[0]);
        a[3] = z[1];
    }
}


bicubic_interp *bicubic_interp_init(
    const double *data, int ns, int nt,
    double smin, double tmin, double ds, double dt)
{
    bicubic_interp *interp;
    const int slength = ns + 6;
    const int tlength = nt + 6;
    interp = aligned_alloc(alignof(*interp), sizeof(*interp) + slength * tlength * sizeof(*interp->a));
    if (LIKELY(interp))
    {
        interp->fs = 1 / ds;
        interp->ft = 1 / dt;
        interp->s0 = 3 - interp->fs * smin;
        interp->t0 = 3 - interp->ft * tmin;
        interp->slength = slength;
        interp->tlength = tlength;

        for (int is = 0; is < slength; is ++)
        {
            for (int it = 0; it < tlength; it ++)
            {
                double a[4][4], a1[4][4];
                for (int js = 0; js < 4; js ++)
                {
                    double z[4];
                    int ks = clip_int(is + js - 4, 0, ns - 1);
                    for (int jt = 0; jt < 4; jt ++)
                    {
                        int kt = clip_int(it + jt - 4, 0, nt - 1);
                        z[jt] = data[ks * ns + kt];
                    }
                    cubic_interp_init_coefficients(a[js], z, z);
                }
                for (int js = 0; js < 4; js ++)
                {
                    for (int jt = 0; jt < 4; jt ++)
                    {
                        a1[js][jt] = a[jt][js];
                    }
                }
                for (int js = 0; js < 4; js ++)
                {
                    cubic_interp_init_coefficients(a[js], a1[js], a1[3]);
                }
                memcpy(interp->a[is * slength + it], a, sizeof(a));
            }
        }
    }
    return interp;
}


double bicubic_interp_eval(const bicubic_interp *interp, double s, double t)
{
    if (UNLIKELY(isnan(s) || isnan(t)))
        return s + t;

    const v4df *a;
    v4df b;
    v2df ix;
    v2df tx = {s, t};
    v2df fx = {interp->fs, interp->ft};
    v2df x0 = {interp->s0, interp->t0};
    v2df xlength = {interp->slength, interp->tlength};

    tx *= fx;
    tx += x0;
    tx = _mm_max_pd(tx, (v2df) {0.0, 0.0}); // SSE2
    tx = _mm_min_pd(tx, xlength - 1.0); // SSE2
    ix = _mm_floor_pd(tx); // SSE4.1
    tx -= ix;
    s = tx[0];
    t = tx[1];

    a = (const v4df *) &interp->a[(int) (ix[0] * interp->slength + ix[1])][0][0];

    // This should emit three 256-bit FMA instructions.
    b = ((a[0] * t + a[1]) * t + a[2]) * t + a[3];

    // This should emit three 64-bit FMA instructions.
    return ((b[0] * s + b[1]) * s + b[2]) * s + b[3];
}


int main(int argc, char **argv)
{
    static const int n = 1000000;
    double data[400][400];
    double (*x)[2] = malloc(sizeof(*x) * n);
    double *y = malloc(sizeof(*y) * n);
    double duration;
    struct timespec start, stop;
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);
    bicubic_interp *interp;

    for (int i = 0; i < 400; i ++)
        for (int j = 0; j < 400; j ++)
            data[i][j] = gsl_rng_uniform(rng);

    interp = bicubic_interp_init(&data[0][0], 400, 400, 0, 0, 1, 1);

    for (int i = 0; i < n; i ++)
        for (int j = 0; j < 2; j ++)
            x[i][j] = gsl_rng_uniform(rng);

    for (int i = 0; i < n; i ++)
        y[i] = bicubic_interp_eval(interp, x[i][0], x[i][1]);

    for (int i = 0; i < n; i ++)
        for (int j = 0; j < 2; j ++)
            x[i][j] = gsl_rng_uniform(rng);

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < n; i ++)
        y[i] = bicubic_interp_eval(interp, x[i][0], x[i][1]);
    clock_gettime(CLOCK_MONOTONIC, &stop);

    duration = (stop.tv_sec - start.tv_sec) + 1e-9 * (stop.tv_nsec - start.tv_nsec);

    printf("%g\n", duration);

    return 0;
}
