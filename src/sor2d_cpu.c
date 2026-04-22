/*****************************************************************************

  EC527 Project — 2D SOR, serial CPU
  Baseline ping-pong sweep vs. temporally blocked (time-skewed) sweep.

  Stencil (Jacobi-like ping-pong form, same as Lab 7):
    dst[i,j] = src[i,j] - OMEGA * (src[i,j]
               - 0.25 * (src[i-1,j]+src[i+1,j]+src[i,j-1]+src[i,j+1]))

  Temporal blocking:
    Split the domain into BxB output tiles.  Per super-step of T sweeps, load
    a (B+2T)^2 scratch region from src with clamp-to-edge, run T in-scratch
    sweeps whose valid region shrinks by 1 on each side per sub-step, then
    write the central BxB back to dst.  DRAM traffic drops by ~T at the cost
    of redundant halo work between tiles.

    Build: gcc -O1 -std=gnu11 sor2d_cpu.c -lrt -lm -o sor2d_cpu
    Run:   ./sor2d_cpu <N> <iters> [B=64] [T=4] [--ppm path]

 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define CPNS    2.0         /* Cycles per nanosecond -- adjust per machine */

#define MINVAL  0.0f
#define MAXVAL  10.0f
#define OMEGA   0.9f        /* Damped Jacobi form (ping-pong) is stable only
                               for omega in (0, 1] on large grids; see
                               report.md for the stability analysis. */

typedef float data_t;

#define IDX2(i,j,ld) ((i)*(ld) + (j))

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

/* Binary grayscale PPM (P5) writer -- normalise a[] to [0, 255]. */
int write_ppm_gray(const char *path, const data_t *a, int N)
{
    data_t mn =  INFINITY, mx = -INFINITY;
    for (long i = 0; i < (long)N*N; i++) {
        data_t v = a[i];
        if (!isfinite(v)) continue;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    if (!isfinite(mn) || mn > mx) return -1;
    data_t scale = (mx > mn) ? 255.0f / (mx - mn) : 0.0f;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fprintf(f, "P5\n%d %d\n255\n", N, N);
    for (long i = 0; i < (long)N*N; i++) {
        int v = (int)((a[i] - mn) * scale);
        if (v < 0) v = 0; else if (v > 255) v = 255;
        unsigned char c = (unsigned char)v;
        fwrite(&c, 1, 1, f);
    }
    fclose(f);
    return 0;
}

/* ======================= Baseline: one sweep per iter ====================== */
void sor2d_baseline(data_t *src, data_t *dst, int N, int iters, float omega)
{
    for (int k = 0; k < iters; k++) {
        for (int i = 1; i < N-1; i++) {
            for (int j = 1; j < N-1; j++) {
                data_t s = src[IDX2(i,j,N)];
                data_t nb = 0.25f * (src[IDX2(i-1,j,N)] + src[IDX2(i+1,j,N)] +
                                     src[IDX2(i,j-1,N)] + src[IDX2(i,j+1,N)]);
                dst[IDX2(i,j,N)] = s - omega * (s - nb);
            }
        }
        for (int j = 0; j < N; j++) {
            dst[IDX2(0,j,N)]   = src[IDX2(0,j,N)];
            dst[IDX2(N-1,j,N)] = src[IDX2(N-1,j,N)];
        }
        for (int i = 0; i < N; i++) {
            dst[IDX2(i,0,N)]   = src[IDX2(i,0,N)];
            dst[IDX2(i,N-1,N)] = src[IDX2(i,N-1,N)];
        }
        data_t *tmp = src; src = dst; dst = tmp;
    }
}

/* =================== Temporal super-step: advance by T sweeps ============== */
/* One super-step: advance src -> dst by T Jacobi-SOR sweeps via per-tile
   scratch buffers. */
void sor2d_temporal_superstep(const data_t *src, data_t *dst,
                              int N, int B, int T, float omega,
                              data_t *scratch_a, data_t *scratch_b)
{
    int S = B + 2*T;

    /* Boundary pass-through */
    for (int j = 0; j < N; j++) {
        dst[IDX2(0,j,N)]   = src[IDX2(0,j,N)];
        dst[IDX2(N-1,j,N)] = src[IDX2(N-1,j,N)];
    }
    for (int i = 0; i < N; i++) {
        dst[IDX2(i,0,N)]   = src[IDX2(i,0,N)];
        dst[IDX2(i,N-1,N)] = src[IDX2(i,N-1,N)];
    }

    for (int i0 = 1; i0 < N-1; i0 += B) {
        for (int j0 = 1; j0 < N-1; j0 += B) {
            int bi = (i0 + B <= N-1) ? B : (N-1 - i0);
            int bj = (j0 + B <= N-1) ? B : (N-1 - j0);

            int gi0 = i0 - T;
            int gj0 = j0 - T;
            int Si = bi + 2*T;
            int Sj = bj + 2*T;

            /* Load scratch_a with clamp-to-edge; mirror into scratch_b so
               both ping-pong buffers start with correct boundary values. */
            for (int si = 0; si < Si; si++) {
                int gi = gi0 + si;
                if (gi < 0) gi = 0; else if (gi > N-1) gi = N-1;
                const data_t *srow = src + (size_t)gi * N;
                data_t *arow = scratch_a + (size_t)si * S;
                for (int sj = 0; sj < Sj; sj++) {
                    int gj = gj0 + sj;
                    if (gj < 0) gj = 0; else if (gj > N-1) gj = N-1;
                    arow[sj] = srow[gj];
                }
            }
            memcpy(scratch_b, scratch_a, (size_t)Si * S * sizeof(data_t));

            data_t *A = scratch_a, *Bp = scratch_b;
            for (int t = 0; t < T; t++) {
                /* Trapezoid bounds in scratch coords, intersected with the
                   global-interior slab [gi in (0, N-1)] so the inner loop is
                   branch-free.  Cells inside the trapezoid but outside the
                   valid slab are handled by the pre-copy below. */
                int lo_i = 1 + t, hi_i = Si - 1 - t;
                int lo_j = 1 + t, hi_j = Sj - 1 - t;

                int ulo_i = lo_i; if (gi0 + ulo_i < 1)   ulo_i = 1 - gi0;
                int uhi_i = hi_i; if (gi0 + uhi_i > N-1) uhi_i = N-1 - gi0;
                int ulo_j = lo_j; if (gj0 + ulo_j < 1)   ulo_j = 1 - gj0;
                int uhi_j = hi_j; if (gj0 + uhi_j > N-1) uhi_j = N-1 - gj0;

                /* Pre-copy A -> Bp so frozen frame + boundary-excluded strips
                   (and everything outside the trapezoid) are carried through. */
                memcpy(Bp, A, (size_t)Si * S * sizeof(data_t));

                for (int si = ulo_i; si < uhi_i; si++) {
                    const data_t *am = A + (size_t)(si-1) * S;
                    const data_t *ac = A + (size_t) si    * S;
                    const data_t *ap = A + (size_t)(si+1) * S;
                    data_t       *bc = Bp + (size_t) si   * S;
                    for (int sj = ulo_j; sj < uhi_j; sj++) {
                        data_t s = ac[sj];
                        data_t nb = 0.25f * (am[sj] + ap[sj] + ac[sj-1] + ac[sj+1]);
                        bc[sj] = s - omega * (s - nb);
                    }
                }
                data_t *tmp = A; A = Bp; Bp = tmp;
            }

            /* Write central BxB output back to dst.  After T swaps, A is the
               T-advanced state; interior output spans si in [T, T+bi). */
            for (int si = T; si < T + bi; si++) {
                int gi = gi0 + si;
                memcpy(dst + (size_t)gi * N + (gj0 + T),
                       A   + (size_t)si * S + T,
                       (size_t)bj * sizeof(data_t));
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
        fprintf(stderr, "usage: %s N iters [B=64] [T=4] [--ppm path]\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    int iters = atoi(argv[2]);
    int B = 64, T = 4;
    const char *ppm_path = NULL;
    int pos = 0;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--ppm") == 0 && i + 1 < argc) {
            ppm_path = argv[++i];
        } else if (pos == 0) { B = atoi(argv[i]); pos++; }
        else if (pos == 1)   { T = atoi(argv[i]); pos++; }
    }
    float omega = OMEGA;

    if (N < 4 || iters < 1 || B < 1 || T < 1) {
        fprintf(stderr, "bad args\n"); return 1;
    }
    if (iters % T != 0) {
        fprintf(stderr, "iters (%d) must be a multiple of T (%d)\n", iters, T);
        return 1;
    }

    size_t bytes = (size_t)N * N * sizeof(data_t);
    data_t *base_a = (data_t*) malloc(bytes);
    data_t *base_b = (data_t*) malloc(bytes);
    data_t *temp_a = (data_t*) malloc(bytes);
    data_t *temp_b = (data_t*) malloc(bytes);

    init_array(base_a, (long)N*N, 527u);
    memcpy(base_b, base_a, bytes);
    memcpy(temp_a, base_a, bytes);
    memcpy(temp_b, base_a, bytes);

    /* ---- Baseline: ping-pong iters times between base_a and base_b ---- */
    clock_gettime(CLOCK_REALTIME, &time_start);
    sor2d_baseline(base_a, base_b, N, iters, omega);
    clock_gettime(CLOCK_REALTIME, &time_stop);
    double t_base = interval(time_start, time_stop);
    data_t *base_result = (iters % 2 == 0) ? base_a : base_b;

    /* ---- Temporal: super = iters/T super-steps, each ping-pongs once --- */
    int super = iters / T;
    int S = B + 2*T;
    data_t *sa = (data_t*) malloc((size_t)S * S * sizeof(data_t));
    data_t *sb = (data_t*) malloc((size_t)S * S * sizeof(data_t));

    clock_gettime(CLOCK_REALTIME, &time_start);
    {
        data_t *cur_src = temp_a, *cur_dst = temp_b;
        for (int s = 0; s < super; s++) {
            sor2d_temporal_superstep(cur_src, cur_dst, N, B, T, omega, sa, sb);
            data_t *tmp = cur_src; cur_src = cur_dst; cur_dst = tmp;
        }
    }
    clock_gettime(CLOCK_REALTIME, &time_stop);
    double t_temp = interval(time_start, time_stop);
    free(sa); free(sb);
    data_t *temp_result = (super % 2 == 0) ? temp_a : temp_b;

    /* ---- Validation and reporting ---- */
    data_t diff  = max_diff(base_result, temp_result, (long)N*N);
    data_t scale = max_val(base_result, (long)N*N);
    data_t rel   = (scale > 0.f) ? diff / scale : 0.f;

    double pts = (double)(N-2) * (double)(N-2) * (double)iters;
    printf("N=%d iters=%d B=%d T=%d OMEGA=%.3f\n", N, iters, B, T, (double)omega);
    printf("  baseline : %9.4f s  (%9.2f Mupdates/s, %10.3g cycles)\n",
           t_base, pts / t_base / 1e6, (double)CPNS * 1.0e9 * t_base);
    printf("  temporal : %9.4f s  (%9.2f Mupdates/s, %10.3g cycles)  speedup %5.2fx\n",
           t_temp, pts / t_temp / 1e6, (double)CPNS * 1.0e9 * t_temp,
           t_base / t_temp);
    printf("  max|base-temp| = %.4e   rel = %.4e\n", (double)diff, (double)rel);

    if (ppm_path) {
        if (write_ppm_gray(ppm_path, base_result, N) == 0)
            printf("  wrote %s\n", ppm_path);
        else
            fprintf(stderr, "failed to write %s\n", ppm_path);
    }

    free(base_a); free(base_b); free(temp_a); free(temp_b);
    return 0;
}
