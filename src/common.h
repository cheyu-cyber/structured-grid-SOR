#ifndef SOR_COMMON_H
#define SOR_COMMON_H

#define _POSIX_C_SOURCE 199309L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* NOTE on omega:
 * This project uses the Lab-7-style ping-pong Jacobi update form
 *   dst[i,j] = src[i,j] - omega * (src[i,j] - avg(neighbors))
 * which is equivalent to the damped Jacobi iteration
 *   M = (1 - omega) * I + omega * J
 * with J the Jacobi matrix of the 5-/7-point Laplacian. This form is stable
 * iff |1 - omega| + omega * rho(J) < 1; for large grids rho(J) -> 1, so
 * stability requires omega in (0, 1]. Lab 7 used omega = 1.85 and "validated"
 * against a CPU reference that was also diverging — in that setting both
 * sides hit inf/NaN in lockstep and the max_abs_diff check silently passed
 * (NaN > 0 is false, so the max stays 0). We use omega = 0.9 so the
 * iteration actually converges to the harmonic solution of the boundary
 * value problem and the correctness checks are meaningful. */
#define OMEGA_DEFAULT 0.9f
#define MINVAL  0.0f
#define MAXVAL  10.0f

static inline double wall_seconds(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static inline void init_array(float *a, long len, unsigned seed)
{
    srand(seed);
    for (long i = 0; i < len; i++)
        a[i] = MINVAL + (float)rand() / (float)RAND_MAX * (MAXVAL - MINVAL);
}

static inline float max_abs_diff(const float *a, const float *b, long len)
{
    float mx = 0.0f;
    int bad = 0;
    for (long i = 0; i < len; i++) {
        float av = a[i], bv = b[i];
        if (!isfinite(av) || !isfinite(bv)) { bad++; continue; }
        float d = fabsf(av - bv);
        if (d > mx) mx = d;
    }
    if (bad) {
        fprintf(stderr, "WARNING: %d non-finite values in max_abs_diff (diverged?)\n", bad);
        return INFINITY;
    }
    return mx;
}

static inline float max_abs(const float *a, long len)
{
    float mx = 0.0f;
    for (long i = 0; i < len; i++) {
        float v = fabsf(a[i]);
        if (v > mx) mx = v;
    }
    return mx;
}

#endif
