/*****************************************************************************

  EC527 Project — 3D SOR, serial CPU
  Baseline ping-pong sweep vs. temporally blocked (time-skewed) sweep over
  the 7-point 3D Laplacian.

  Stencil:
    dst[i,j,k] = src[i,j,k] - OMEGA * (src[i,j,k]
                 - (1/6) * (src[i-1,j,k] + src[i+1,j,k]
                          + src[i,j-1,k] + src[i,j+1,k]
                          + src[i,j,k-1] + src[i,j,k+1]))

  Same temporal-blocking scheme as sor2d_cpu.c but with cubic (B x B x B)
  tiles.  The 3D halo-to-interior ratio grows faster than 2D, so usable
  (B, T) pairs are tighter.

    Build: gcc -O1 -std=gnu11 sor3d_cpu.c -lrt -lm -o sor3d_cpu
    Run:   ./sor3d_cpu <N> <iters> [B=32] [T=2]

 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define CPNS    2.0         /* Cycles per nanosecond -- adjust per machine */

#define MINVAL  0.0f
#define MAXVAL  10.0f
#define OMEGA   0.9f

typedef float data_t;

#define IDX3(i,j,k,ldj,ldk) (((i)*(ldj) + (j))*(ldk) + (k))

static const data_t INV6 = 1.0f / 6.0f;

/* -=-=-=-=- Time measurement by clock_gettime() -=-=-=-=- */
double interval(struct timespec start, struct timespec end)
{
    struct timespec temp;
    temp.tv_sec  = end.tv_sec  - start.tv_sec;
    temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    if (temp.tv_nsec < 0) {
        temp.tv_sec  -= 1;
        temp.tv_nsec += 1000000000;
    }
    return ((double)temp.tv_sec) + ((double)temp.tv_nsec) * 1.0e-9;
}

/* -=-=-=-=- Array helpers -=-=-=-=- */
double fRand(double fMin, double fMax)
{
    double f = (double)rand() / (double)RAND_MAX;
    return fMin + f * (fMax - fMin);
}

void init_array(data_t *a, long len, unsigned seed)
{
    srand(seed);
    for (long i = 0; i < len; i++)
        a[i] = (data_t) fRand((double)MINVAL, (double)MAXVAL);
}

data_t max_diff(const data_t *a, const data_t *b, long len)
{
    data_t mx = 0.0f;
    int bad = 0;
    for (long i = 0; i < len; i++) {
        data_t av = a[i], bv = b[i];
        if (!isfinite(av) || !isfinite(bv)) { bad++; continue; }
        data_t d = fabsf(av - bv);
        if (d > mx) mx = d;
    }
    if (bad) {
        fprintf(stderr, "WARNING: %d non-finite values (diverged?)\n", bad);
        return INFINITY;
    }
    return mx;
}

data_t max_val(const data_t *a, long len)
{
    data_t mx = 0.0f;
    for (long i = 0; i < len; i++) {
        data_t v = fabsf(a[i]);
        if (v > mx) mx = v;
    }
    return mx;
}

/* ======================= Baseline: one sweep per iter ====================== */
void sor3d_baseline(data_t *src, data_t *dst, int N, int iters, float omega)
{
    for (int it = 0; it < iters; it++) {
        for (int i = 1; i < N-1; i++) {
            for (int j = 1; j < N-1; j++) {
                for (int k = 1; k < N-1; k++) {
                    data_t s = src[IDX3(i,j,k,N,N)];
                    data_t nb = INV6 * (src[IDX3(i-1,j,k,N,N)] + src[IDX3(i+1,j,k,N,N)] +
                                        src[IDX3(i,j-1,k,N,N)] + src[IDX3(i,j+1,k,N,N)] +
                                        src[IDX3(i,j,k-1,N,N)] + src[IDX3(i,j,k+1,N,N)]);
                    dst[IDX3(i,j,k,N,N)] = s - omega * (s - nb);
                }
            }
        }
        for (int a = 0; a < N; a++) {
            for (int b = 0; b < N; b++) {
                dst[IDX3(0,a,b,N,N)]   = src[IDX3(0,a,b,N,N)];
                dst[IDX3(N-1,a,b,N,N)] = src[IDX3(N-1,a,b,N,N)];
                dst[IDX3(a,0,b,N,N)]   = src[IDX3(a,0,b,N,N)];
                dst[IDX3(a,N-1,b,N,N)] = src[IDX3(a,N-1,b,N,N)];
                dst[IDX3(a,b,0,N,N)]   = src[IDX3(a,b,0,N,N)];
                dst[IDX3(a,b,N-1,N,N)] = src[IDX3(a,b,N-1,N,N)];
            }
        }
        data_t *tmp = src; src = dst; dst = tmp;
    }
}

/* =================== Temporal super-step: advance by T sweeps ============== */
void sor3d_temporal_superstep(const data_t *src, data_t *dst,
                              int N, int B, int T, float omega,
                              data_t *sa, data_t *sb)
{
    int S = B + 2*T;

    /* Boundary pass-through (all six faces). */
    for (int a = 0; a < N; a++) {
        for (int b = 0; b < N; b++) {
            dst[IDX3(0,a,b,N,N)]   = src[IDX3(0,a,b,N,N)];
            dst[IDX3(N-1,a,b,N,N)] = src[IDX3(N-1,a,b,N,N)];
            dst[IDX3(a,0,b,N,N)]   = src[IDX3(a,0,b,N,N)];
            dst[IDX3(a,N-1,b,N,N)] = src[IDX3(a,N-1,b,N,N)];
            dst[IDX3(a,b,0,N,N)]   = src[IDX3(a,b,0,N,N)];
            dst[IDX3(a,b,N-1,N,N)] = src[IDX3(a,b,N-1,N,N)];
        }
    }

    for (int i0 = 1; i0 < N-1; i0 += B) {
        for (int j0 = 1; j0 < N-1; j0 += B) {
            for (int k0 = 1; k0 < N-1; k0 += B) {
                int bi = (i0 + B <= N-1) ? B : (N-1 - i0);
                int bj = (j0 + B <= N-1) ? B : (N-1 - j0);
                int bk = (k0 + B <= N-1) ? B : (N-1 - k0);

                int gi0 = i0 - T, gj0 = j0 - T, gk0 = k0 - T;
                int Si = bi + 2*T, Sj = bj + 2*T, Sk = bk + 2*T;

                /* Clamp-to-edge load into sa. */
                for (int si = 0; si < Si; si++) {
                    int gi = gi0 + si; if (gi < 0) gi = 0; else if (gi > N-1) gi = N-1;
                    for (int sj = 0; sj < Sj; sj++) {
                        int gj = gj0 + sj; if (gj < 0) gj = 0; else if (gj > N-1) gj = N-1;
                        const data_t *srow = src + ((size_t)gi * N + gj) * N;
                        data_t *arow = sa + ((size_t)si * S + sj) * S;
                        for (int sk = 0; sk < Sk; sk++) {
                            int gk = gk0 + sk; if (gk < 0) gk = 0; else if (gk > N-1) gk = N-1;
                            arow[sk] = srow[gk];
                        }
                    }
                }

                data_t *A = sa, *Bp = sb;
                for (int t = 0; t < T; t++) {
                    int lo_i = 1 + t, hi_i = Si - 1 - t;
                    int lo_j = 1 + t, hi_j = Sj - 1 - t;
                    int lo_k = 1 + t, hi_k = Sk - 1 - t;

                    int ulo_i = lo_i; if (gi0 + ulo_i < 1)   ulo_i = 1 - gi0;
                    int uhi_i = hi_i; if (gi0 + uhi_i > N-1) uhi_i = N-1 - gi0;
                    int ulo_j = lo_j; if (gj0 + ulo_j < 1)   ulo_j = 1 - gj0;
                    int uhi_j = hi_j; if (gj0 + uhi_j > N-1) uhi_j = N-1 - gj0;
                    int ulo_k = lo_k; if (gk0 + ulo_k < 1)   ulo_k = 1 - gk0;
                    int uhi_k = hi_k; if (gk0 + uhi_k > N-1) uhi_k = N-1 - gk0;

                    /* Pre-copy carries through everything outside the update box. */
                    memcpy(Bp, A, (size_t)Si * Sj * Sk * sizeof(data_t));

                    for (int si = ulo_i; si < uhi_i; si++) {
                        for (int sj = ulo_j; sj < uhi_j; sj++) {
                            const data_t *am_i = A + (((size_t)(si-1)*Sj + sj)*Sk);
                            const data_t *ac_i = A + (((size_t) si   *Sj + sj)*Sk);
                            const data_t *ap_i = A + (((size_t)(si+1)*Sj + sj)*Sk);
                            const data_t *am_j = A + (((size_t) si   *Sj + (sj-1))*Sk);
                            const data_t *ap_j = A + (((size_t) si   *Sj + (sj+1))*Sk);
                            data_t       *bc   = Bp + (((size_t) si  *Sj + sj)*Sk);
                            for (int sk = ulo_k; sk < uhi_k; sk++) {
                                data_t s = ac_i[sk];
                                data_t nb = INV6 * (am_i[sk] + ap_i[sk] +
                                                    am_j[sk] + ap_j[sk] +
                                                    ac_i[sk-1] + ac_i[sk+1]);
                                bc[sk] = s - omega * (s - nb);
                            }
                        }
                    }
                    data_t *tmp = A; A = Bp; Bp = tmp;
                }

                /* Write central BxBxB region back to dst. */
                for (int si = T; si < T + bi; si++) {
                    int gi = gi0 + si;
                    for (int sj = T; sj < T + bj; sj++) {
                        int gj = gj0 + sj;
                        memcpy(dst + ((size_t)gi * N + gj) * N + (gk0 + T),
                               A   + ((size_t)si * S + sj) * S + T,
                               (size_t)bk * sizeof(data_t));
                    }
                }
            }
        }
    }
}

/* =====================================================================
   MAIN
   ===================================================================== */
int main(int argc, char **argv)
{
    struct timespec time_start, time_stop;

    if (argc < 3) {
        fprintf(stderr, "usage: %s N iters [B=32] [T=2]\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    int iters = atoi(argv[2]);
    int B = (argc >= 4) ? atoi(argv[3]) : 32;
    int T = (argc >= 5) ? atoi(argv[4]) : 2;
    float omega = OMEGA;

    if (N < 4 || iters < 1 || B < 1 || T < 1) {
        fprintf(stderr, "bad args\n"); return 1;
    }
    if (iters % T != 0) {
        fprintf(stderr, "iters (%d) must be a multiple of T (%d)\n", iters, T);
        return 1;
    }

    size_t bytes = (size_t)N * N * N * sizeof(data_t);
    data_t *base_a = (data_t*) malloc(bytes);
    data_t *base_b = (data_t*) malloc(bytes);
    data_t *temp_a = (data_t*) malloc(bytes);
    data_t *temp_b = (data_t*) malloc(bytes);
    if (!base_a || !base_b || !temp_a || !temp_b) {
        fprintf(stderr, "oom at N=%d (%.1f MB each)\n", N, bytes/1048576.0);
        return 1;
    }

    init_array(base_a, (long)N*N*N, 527u);
    memcpy(base_b, base_a, bytes);
    memcpy(temp_a, base_a, bytes);
    memcpy(temp_b, base_a, bytes);

    /* ---- Baseline ---- */
    clock_gettime(CLOCK_REALTIME, &time_start);
    sor3d_baseline(base_a, base_b, N, iters, omega);
    clock_gettime(CLOCK_REALTIME, &time_stop);
    double t_base = interval(time_start, time_stop);
    data_t *base_result = (iters % 2 == 0) ? base_a : base_b;

    /* ---- Temporal ---- */
    int super = iters / T;
    int S = B + 2*T;
    data_t *sa = (data_t*) malloc((size_t)S * S * S * sizeof(data_t));
    data_t *sb = (data_t*) malloc((size_t)S * S * S * sizeof(data_t));

    clock_gettime(CLOCK_REALTIME, &time_start);
    {
        data_t *cur_src = temp_a, *cur_dst = temp_b;
        for (int s = 0; s < super; s++) {
            sor3d_temporal_superstep(cur_src, cur_dst, N, B, T, omega, sa, sb);
            data_t *tmp = cur_src; cur_src = cur_dst; cur_dst = tmp;
        }
    }
    clock_gettime(CLOCK_REALTIME, &time_stop);
    double t_temp = interval(time_start, time_stop);
    data_t *temp_result = (super % 2 == 0) ? temp_a : temp_b;

    /* ---- Validation and reporting ---- */
    data_t diff  = max_diff(base_result, temp_result, (long)N*N*N);
    data_t scale = max_val(base_result, (long)N*N*N);
    data_t rel   = (scale > 0.f) ? diff / scale : 0.f;

    double pts = (double)(N-2) * (double)(N-2) * (double)(N-2) * (double)iters;
    printf("N=%d iters=%d B=%d T=%d OMEGA=%.3f\n", N, iters, B, T, (double)omega);
    printf("  baseline : %9.4f s  (%9.2f Mupdates/s, %10.3g cycles)\n",
           t_base, pts / t_base / 1e6, (double)CPNS * 1.0e9 * t_base);
    printf("  temporal : %9.4f s  (%9.2f Mupdates/s, %10.3g cycles)  speedup %5.2fx\n",
           t_temp, pts / t_temp / 1e6, (double)CPNS * 1.0e9 * t_temp,
           t_base / t_temp);
    printf("  max|base-temp| = %.4e   rel = %.4e\n", (double)diff, (double)rel);

    free(sa); free(sb);
    free(base_a); free(base_b); free(temp_a); free(temp_b);
    return 0;
}
