/*
 * 2D SOR — OpenMP baseline vs. OpenMP temporally blocked.
 *
 * Mirrors sor2d_cpu.c but parallelizes both kernels:
 *   - baseline: #pragma omp parallel for on the outer i loop of every sweep
 *   - temporal: #pragma omp parallel for collapse(2) over tiles in each
 *               super-step, with per-thread scratch buffers
 *
 * Usage:  ./sor2d_omp <N> <iters> [B=64] [T=4]
 *         OMP_NUM_THREADS=16 ./sor2d_omp ...
 * Build:  gcc -O3 -march=native -fopenmp sor2d_omp.c -o sor2d_omp -lm
 */

#include "common.h"
#include <omp.h>

#define IDX2(i,j,ld) ((i)*(ld) + (j))

static void sor2d_baseline_omp(float *src, float *dst, int N, int iters, float omega)
{
    for (int k = 0; k < iters; k++) {
        #pragma omp parallel for schedule(static)
        for (int i = 1; i < N-1; i++) {
            const float *sm = src + (size_t)(i-1) * N;
            const float *sc = src + (size_t) i    * N;
            const float *sp = src + (size_t)(i+1) * N;
            float       *dc = dst + (size_t) i    * N;
            for (int j = 1; j < N-1; j++) {
                float s = sc[j];
                float nb = 0.25f * (sm[j] + sp[j] + sc[j-1] + sc[j+1]);
                dc[j] = s - omega * (s - nb);
            }
        }
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < N; j++) {
            dst[IDX2(0,j,N)]     = src[IDX2(0,j,N)];
            dst[IDX2(N-1,j,N)]   = src[IDX2(N-1,j,N)];
        }
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            dst[IDX2(i,0,N)]     = src[IDX2(i,0,N)];
            dst[IDX2(i,N-1,N)]   = src[IDX2(i,N-1,N)];
        }
        float *tmp = src; src = dst; dst = tmp;
    }
}

static void sor2d_temporal_superstep_omp(const float *src, float *dst,
                                         int N, int B, int T, float omega)
{
    int S = B + 2*T;

    #pragma omp parallel for schedule(static)
    for (int j = 0; j < N; j++) {
        dst[IDX2(0,j,N)]     = src[IDX2(0,j,N)];
        dst[IDX2(N-1,j,N)]   = src[IDX2(N-1,j,N)];
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        dst[IDX2(i,0,N)]     = src[IDX2(i,0,N)];
        dst[IDX2(i,N-1,N)]   = src[IDX2(i,N-1,N)];
    }

    int ntiles_i = (N - 2 + B - 1) / B;
    int ntiles_j = (N - 2 + B - 1) / B;

    #pragma omp parallel
    {
        float *sa = (float*) malloc((size_t)S * S * sizeof(float));
        float *sb = (float*) malloc((size_t)S * S * sizeof(float));

        #pragma omp for collapse(2) schedule(static)
        for (int ti = 0; ti < ntiles_i; ti++) {
            for (int tj = 0; tj < ntiles_j; tj++) {
                int i0 = 1 + ti * B;
                int j0 = 1 + tj * B;
                int bi = (i0 + B <= N-1) ? B : (N-1 - i0);
                int bj = (j0 + B <= N-1) ? B : (N-1 - j0);

                int gi0 = i0 - T, gj0 = j0 - T;
                int Si = bi + 2*T, Sj = bj + 2*T;

                for (int si = 0; si < Si; si++) {
                    int gi = gi0 + si;
                    if (gi < 0) gi = 0; else if (gi > N-1) gi = N-1;
                    const float *srow = src + (size_t)gi * N;
                    float *arow = sa + (size_t)si * S;
                    for (int sj = 0; sj < Sj; sj++) {
                        int gj = gj0 + sj;
                        if (gj < 0) gj = 0; else if (gj > N-1) gj = N-1;
                        arow[sj] = srow[gj];
                    }
                }
                memcpy(sb, sa, (size_t)Si * S * sizeof(float));

                float *A = sa, *Bp = sb;
                for (int t = 0; t < T; t++) {
                    int lo_i = 1 + t, hi_i = Si - 1 - t;
                    int lo_j = 1 + t, hi_j = Sj - 1 - t;

                    int ulo_i = lo_i; if (gi0 + ulo_i < 1)   ulo_i = 1 - gi0;
                    int uhi_i = hi_i; if (gi0 + uhi_i > N-1) uhi_i = N-1 - gi0;
                    int ulo_j = lo_j; if (gj0 + ulo_j < 1)   ulo_j = 1 - gj0;
                    int uhi_j = hi_j; if (gj0 + uhi_j > N-1) uhi_j = N-1 - gj0;

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

                for (int si = T; si < T + bi; si++) {
                    int gi = gi0 + si;
                    memcpy(dst + (size_t)gi * N + (gj0 + T),
                           A   + (size_t)si * S + T,
                           (size_t)bj * sizeof(float));
                }
            }
        }

        free(sa); free(sb);
    }
}

int main(int argc, char **argv)
{
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
    float omega = OMEGA_DEFAULT;

    if (N < 4 || iters < 1 || B < 1 || T < 1) { fprintf(stderr, "bad args\n"); return 1; }
    if (iters % T != 0) {
        fprintf(stderr, "iters (%d) must be a multiple of T (%d)\n", iters, T);
        return 1;
    }

    int nthreads = omp_get_max_threads();
    printf("OMP_NUM_THREADS=%d\n", nthreads);

    size_t bytes = (size_t)N * N * sizeof(float);
    float *base_a = (float*) malloc(bytes);
    float *base_b = (float*) malloc(bytes);
    float *temp_a = (float*) malloc(bytes);
    float *temp_b = (float*) malloc(bytes);

    init_array(base_a, (long)N*N, 527u);
    memcpy(base_b, base_a, bytes);
    memcpy(temp_a, base_a, bytes);
    memcpy(temp_b, base_a, bytes);

    double t0 = wall_seconds();
    sor2d_baseline_omp(base_a, base_b, N, iters, omega);
    double t_base = wall_seconds() - t0;
    float *base_result = (iters % 2 == 0) ? base_a : base_b;

    int super = iters / T;
    t0 = wall_seconds();
    {
        float *cur_src = temp_a, *cur_dst = temp_b;
        for (int s = 0; s < super; s++) {
            sor2d_temporal_superstep_omp(cur_src, cur_dst, N, B, T, omega);
            float *tmp = cur_src; cur_src = cur_dst; cur_dst = tmp;
        }
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

    if (ppm_path) {
        if (write_ppm_gray(ppm_path, base_result, N) == 0)
            printf("  wrote %s\n", ppm_path);
        else
            fprintf(stderr, "failed to write %s\n", ppm_path);
    }

    free(base_a); free(base_b); free(temp_a); free(temp_b);
    return 0;
}
