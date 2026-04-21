/*
 * 2D SOR — baseline ping-pong vs. temporally blocked (time-skewed).
 *
 * Stencil (Jacobi-like ping-pong form, same as Lab 7):
 *   dst[i,j] = src[i,j] - omega * (src[i,j]
 *              - 0.25 * (src[i-1,j]+src[i+1,j]+src[i,j-1]+src[i,j+1]))
 *
 * Temporal blocking:
 *   Split the domain into BxB output tiles. Per super-step (T sweeps), load a
 *   (B+2T)^2 scratch region from src with clamp-to-edge, run T in-scratch
 *   sweeps whose valid region shrinks by 1 on each side per sub-step, then
 *   write the central BxB back to dst. DRAM traffic drops by ~T; extra work
 *   comes from the halo overlap between tiles.
 *
 * Usage:  ./sor2d_cpu <N> <iters> [B=64] [T=4]
 * Build:  gcc -O3 -march=native -std=c11 sor2d_cpu.c -o sor2d_cpu -lm
 */

#include "common.h"

#define IDX2(i,j,ld) ((i)*(ld) + (j))

static void sor2d_baseline(float *src, float *dst, int N, int iters, float omega)
{
    for (int k = 0; k < iters; k++) {
        for (int i = 1; i < N-1; i++) {
            for (int j = 1; j < N-1; j++) {
                float s = src[IDX2(i,j,N)];
                float nb = 0.25f * (src[IDX2(i-1,j,N)] + src[IDX2(i+1,j,N)] +
                                    src[IDX2(i,j-1,N)] + src[IDX2(i,j+1,N)]);
                dst[IDX2(i,j,N)] = s - omega * (s - nb);
            }
        }
        for (int j = 0; j < N; j++) {
            dst[IDX2(0,j,N)]     = src[IDX2(0,j,N)];
            dst[IDX2(N-1,j,N)]   = src[IDX2(N-1,j,N)];
        }
        for (int i = 0; i < N; i++) {
            dst[IDX2(i,0,N)]     = src[IDX2(i,0,N)];
            dst[IDX2(i,N-1,N)]   = src[IDX2(i,N-1,N)];
        }
        float *tmp = src; src = dst; dst = tmp;
    }
}

/* One super-step: advance src -> dst by T Jacobi-SOR sweeps via per-tile scratch. */
static void sor2d_temporal_superstep(const float *src, float *dst,
                                     int N, int B, int T, float omega,
                                     float *scratch_a, float *scratch_b)
{
    int S = B + 2*T;

    /* Boundary pass-through */
    for (int j = 0; j < N; j++) {
        dst[IDX2(0,j,N)]     = src[IDX2(0,j,N)];
        dst[IDX2(N-1,j,N)]   = src[IDX2(N-1,j,N)];
    }
    for (int i = 0; i < N; i++) {
        dst[IDX2(i,0,N)]     = src[IDX2(i,0,N)];
        dst[IDX2(i,N-1,N)]   = src[IDX2(i,N-1,N)];
    }

    for (int i0 = 1; i0 < N-1; i0 += B) {
        for (int j0 = 1; j0 < N-1; j0 += B) {
            int bi = (i0 + B <= N-1) ? B : (N-1 - i0);
            int bj = (j0 + B <= N-1) ? B : (N-1 - j0);

            int gi0 = i0 - T;
            int gj0 = j0 - T;
            int Si = bi + 2*T;
            int Sj = bj + 2*T;

            /* Load scratch_a with clamp-to-edge. Then mirror into scratch_b so
             * both ping-pong buffers start with correct global-boundary values. */
            for (int si = 0; si < Si; si++) {
                int gi = gi0 + si;
                if (gi < 0) gi = 0; else if (gi > N-1) gi = N-1;
                const float *srow = src + (size_t)gi * N;
                float *arow = scratch_a + (size_t)si * S;
                for (int sj = 0; sj < Sj; sj++) {
                    int gj = gj0 + sj;
                    if (gj < 0) gj = 0; else if (gj > N-1) gj = N-1;
                    arow[sj] = srow[gj];
                }
            }
            memcpy(scratch_b, scratch_a, (size_t)Si * S * sizeof(float));

            float *A = scratch_a, *Bp = scratch_b;
            for (int t = 0; t < T; t++) {
                /* Trapezoid bounds in scratch coords, intersected with the
                 * global-interior slab [gi ∈ (0, N-1)] so the inner loop is
                 * branch-free. Cells inside the trapezoid but outside the
                 * valid slab are handled by the pre-copy below. */
                int lo_i = 1 + t, hi_i = Si - 1 - t;
                int lo_j = 1 + t, hi_j = Sj - 1 - t;

                int ulo_i = lo_i; if (gi0 + ulo_i < 1)   ulo_i = 1 - gi0;
                int uhi_i = hi_i; if (gi0 + uhi_i > N-1) uhi_i = N-1 - gi0;
                int ulo_j = lo_j; if (gj0 + ulo_j < 1)   ulo_j = 1 - gj0;
                int uhi_j = hi_j; if (gj0 + uhi_j > N-1) uhi_j = N-1 - gj0;

                /* Pre-copy A -> Bp so frozen frame + boundary-excluded strips
                 * (and everything outside the trapezoid) are carried through. */
                memcpy(Bp, A, (size_t)Si * S * sizeof(float));

                for (int si = ulo_i; si < uhi_i; si++) {
                    const float *am = A + (size_t)(si-1) * S;
                    const float *ac = A + (size_t) si    * S;
                    const float *ap = A + (size_t)(si+1) * S;
                    float       *bc = Bp + (size_t) si   * S;
                    for (int sj = ulo_j; sj < uhi_j; sj++) {
                        float s = ac[sj];
                        float nb = 0.25f * (am[sj] + ap[sj] + ac[sj-1] + ac[sj+1]);
                        bc[sj] = s - omega * (s - nb);
                    }
                }
                float *tmp = A; A = Bp; Bp = tmp;
            }

            /* Write central BxB output back to dst. After T swaps, A is the
             * T-advanced state; interior output spans si in [T, T+bi). */
            for (int si = T; si < T + bi; si++) {
                int gi = gi0 + si;
                memcpy(dst + (size_t)gi * N + (gj0 + T),
                       A   + (size_t)si * S + T,
                       (size_t)bj * sizeof(float));
            }
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s N iters [B=64] [T=4]\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    int iters = atoi(argv[2]);
    int B = (argc >= 4) ? atoi(argv[3]) : 64;
    int T = (argc >= 5) ? atoi(argv[4]) : 4;
    float omega = OMEGA_DEFAULT;

    if (N < 4 || iters < 1 || B < 1 || T < 1) {
        fprintf(stderr, "bad args\n"); return 1;
    }
    if (iters % T != 0) {
        fprintf(stderr, "iters (%d) must be a multiple of T (%d)\n", iters, T);
        return 1;
    }

    size_t bytes = (size_t)N * N * sizeof(float);
    float *base_a = (float*) malloc(bytes);
    float *base_b = (float*) malloc(bytes);
    float *temp_a = (float*) malloc(bytes);
    float *temp_b = (float*) malloc(bytes);

    init_array(base_a, (long)N*N, 527u);
    memcpy(base_b, base_a, bytes);
    memcpy(temp_a, base_a, bytes);
    memcpy(temp_b, base_a, bytes);

    /* Baseline: ping-pongs iters times between base_a and base_b. */
    double t0 = wall_seconds();
    sor2d_baseline(base_a, base_b, N, iters, omega);
    double t_base = wall_seconds() - t0;
    float *base_result = (iters % 2 == 0) ? base_a : base_b;

    /* Temporal: super = iters/T super-steps, each ping-pongs once. */
    int super = iters / T;
    t0 = wall_seconds();
    {
        int S = B + 2*T;
        float *sa = (float*) malloc((size_t)S * S * sizeof(float));
        float *sb = (float*) malloc((size_t)S * S * sizeof(float));
        float *cur_src = temp_a, *cur_dst = temp_b;
        for (int s = 0; s < super; s++) {
            sor2d_temporal_superstep(cur_src, cur_dst, N, B, T, omega, sa, sb);
            float *tmp = cur_src; cur_src = cur_dst; cur_dst = tmp;
        }
        free(sa); free(sb);
    }
    double t_temp = wall_seconds() - t0;
    float *temp_result = (super % 2 == 0) ? temp_a : temp_b;

    float diff = max_abs_diff(base_result, temp_result, (long)N*N);
    float scale = max_abs(base_result, (long)N*N);
    float rel = (scale > 0.f) ? diff / scale : 0.f;

    double pts = (double)(N-2) * (double)(N-2) * (double)iters;
    printf("N=%d iters=%d B=%d T=%d omega=%.3f\n", N, iters, B, T, omega);
    printf("  baseline : %9.4f s   (%7.2f Mupdates/s)\n",
           t_base, pts / t_base / 1e6);
    printf("  temporal : %9.4f s   (%7.2f Mupdates/s)  speedup %5.2fx\n",
           t_temp, pts / t_temp / 1e6, t_base / t_temp);
    printf("  max|base-temp| = %.4e   rel = %.4e\n", diff, rel);

    free(base_a); free(base_b); free(temp_a); free(temp_b);
    return 0;
}
