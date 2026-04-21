/*
 * 3D SOR — OpenMP baseline vs. OpenMP temporally blocked (7-point stencil).
 *
 * Mirrors sor3d_cpu.c, parallelized the same way as sor2d_omp.c:
 *   - baseline: #pragma omp parallel for on the outer i loop of every sweep
 *   - temporal: #pragma omp parallel for collapse(3) over tiles, with
 *               per-thread scratch buffers
 *
 * Usage:  ./sor3d_omp <N> <iters> [B=32] [T=2]
 *         OMP_NUM_THREADS=16 ./sor3d_omp ...
 * Build:  gcc -O3 -march=native -fopenmp sor3d_omp.c -o sor3d_omp -lm
 */

#include "common.h"
#include <omp.h>

#define IDX3(i,j,k,ldj,ldk) (((i)*(ldj) + (j))*(ldk) + (k))

static const float INV6 = 1.0f / 6.0f;

static void sor3d_baseline_omp(float *src, float *dst, int N, int iters, float omega)
{
    for (int it = 0; it < iters; it++) {
        #pragma omp parallel for schedule(static)
        for (int i = 1; i < N-1; i++) {
            for (int j = 1; j < N-1; j++) {
                const float *sm_i = src + ((size_t)(i-1)*N + j) * N;
                const float *sc_i = src + ((size_t) i   *N + j) * N;
                const float *sp_i = src + ((size_t)(i+1)*N + j) * N;
                const float *sm_j = src + ((size_t) i   *N + (j-1)) * N;
                const float *sp_j = src + ((size_t) i   *N + (j+1)) * N;
                float       *dc   = dst + ((size_t) i   *N + j) * N;
                for (int k = 1; k < N-1; k++) {
                    float s = sc_i[k];
                    float nb = INV6 * (sm_i[k] + sp_i[k] +
                                       sm_j[k] + sp_j[k] +
                                       sc_i[k-1] + sc_i[k+1]);
                    dc[k] = s - omega * (s - nb);
                }
            }
        }
        #pragma omp parallel for schedule(static)
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
        float *tmp = src; src = dst; dst = tmp;
    }
}

static void sor3d_temporal_superstep_omp(const float *src, float *dst,
                                         int N, int B, int T, float omega)
{
    int S = B + 2*T;

    #pragma omp parallel for schedule(static)
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

    int nt_i = (N - 2 + B - 1) / B;
    int nt_j = (N - 2 + B - 1) / B;
    int nt_k = (N - 2 + B - 1) / B;

    #pragma omp parallel
    {
        size_t sbytes = (size_t)S * S * S * sizeof(float);
        float *sa = (float*) malloc(sbytes);
        float *sb = (float*) malloc(sbytes);

        #pragma omp for collapse(3) schedule(static)
        for (int ti = 0; ti < nt_i; ti++) {
            for (int tj = 0; tj < nt_j; tj++) {
                for (int tk = 0; tk < nt_k; tk++) {
                    int i0 = 1 + ti * B;
                    int j0 = 1 + tj * B;
                    int k0 = 1 + tk * B;
                    int bi = (i0 + B <= N-1) ? B : (N-1 - i0);
                    int bj = (j0 + B <= N-1) ? B : (N-1 - j0);
                    int bk = (k0 + B <= N-1) ? B : (N-1 - k0);

                    int gi0 = i0 - T, gj0 = j0 - T, gk0 = k0 - T;
                    int Si = bi + 2*T, Sj = bj + 2*T, Sk = bk + 2*T;

                    for (int si = 0; si < Si; si++) {
                        int gi = gi0 + si; if (gi < 0) gi = 0; else if (gi > N-1) gi = N-1;
                        for (int sj = 0; sj < Sj; sj++) {
                            int gj = gj0 + sj; if (gj < 0) gj = 0; else if (gj > N-1) gj = N-1;
                            const float *srow = src + ((size_t)gi*N + gj)*N;
                            float *arow = sa + ((size_t)si*S + sj)*S;
                            for (int sk = 0; sk < Sk; sk++) {
                                int gk = gk0 + sk; if (gk < 0) gk = 0; else if (gk > N-1) gk = N-1;
                                arow[sk] = srow[gk];
                            }
                        }
                    }

                    float *A = sa, *Bp = sb;
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

                        memcpy(Bp, A, sbytes);

                        for (int si = ulo_i; si < uhi_i; si++) {
                            for (int sj = ulo_j; sj < uhi_j; sj++) {
                                const float *am_i = A + (((size_t)(si-1)*Sj + sj)*Sk);
                                const float *ac_i = A + (((size_t) si   *Sj + sj)*Sk);
                                const float *ap_i = A + (((size_t)(si+1)*Sj + sj)*Sk);
                                const float *am_j = A + (((size_t) si   *Sj + (sj-1))*Sk);
                                const float *ap_j = A + (((size_t) si   *Sj + (sj+1))*Sk);
                                float       *bc   = Bp + (((size_t) si  *Sj + sj)*Sk);
                                for (int sk = ulo_k; sk < uhi_k; sk++) {
                                    float s = ac_i[sk];
                                    float nb = INV6 * (am_i[sk] + ap_i[sk] +
                                                       am_j[sk] + ap_j[sk] +
                                                       ac_i[sk-1] + ac_i[sk+1]);
                                    bc[sk] = s - omega * (s - nb);
                                }
                            }
                        }
                        float *tmp = A; A = Bp; Bp = tmp;
                    }

                    for (int si = T; si < T + bi; si++) {
                        int gi = gi0 + si;
                        for (int sj = T; sj < T + bj; sj++) {
                            int gj = gj0 + sj;
                            memcpy(dst + ((size_t)gi*N + gj)*N + (gk0 + T),
                                   A   + ((size_t)si*S + sj)*S + T,
                                   (size_t)bk * sizeof(float));
                        }
                    }
                }
            }
        }

        free(sa); free(sb);
    }
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s N iters [B=32] [T=2]\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    int iters = atoi(argv[2]);
    int B = (argc >= 4) ? atoi(argv[3]) : 32;
    int T = (argc >= 5) ? atoi(argv[4]) : 2;
    float omega = OMEGA_DEFAULT;

    if (N < 4 || iters < 1 || B < 1 || T < 1) { fprintf(stderr, "bad args\n"); return 1; }
    if (iters % T != 0) {
        fprintf(stderr, "iters (%d) must be a multiple of T (%d)\n", iters, T);
        return 1;
    }

    int nthreads = omp_get_max_threads();
    printf("OMP_NUM_THREADS=%d\n", nthreads);

    size_t bytes = (size_t)N * N * N * sizeof(float);
    float *base_a = (float*) malloc(bytes);
    float *base_b = (float*) malloc(bytes);
    float *temp_a = (float*) malloc(bytes);
    float *temp_b = (float*) malloc(bytes);
    if (!base_a || !base_b || !temp_a || !temp_b) {
        fprintf(stderr, "oom at N=%d (%.1f MB each)\n", N, bytes/1048576.0);
        return 1;
    }

    init_array(base_a, (long)N*N*N, 527u);
    memcpy(base_b, base_a, bytes);
    memcpy(temp_a, base_a, bytes);
    memcpy(temp_b, base_a, bytes);

    double t0 = wall_seconds();
    sor3d_baseline_omp(base_a, base_b, N, iters, omega);
    double t_base = wall_seconds() - t0;
    float *base_result = (iters % 2 == 0) ? base_a : base_b;

    int super = iters / T;
    t0 = wall_seconds();
    {
        float *cur_src = temp_a, *cur_dst = temp_b;
        for (int s = 0; s < super; s++) {
            sor3d_temporal_superstep_omp(cur_src, cur_dst, N, B, T, omega);
            float *tmp = cur_src; cur_src = cur_dst; cur_dst = tmp;
        }
    }
    double t_temp = wall_seconds() - t0;
    float *temp_result = (super % 2 == 0) ? temp_a : temp_b;

    float diff = max_abs_diff(base_result, temp_result, (long)N*N*N);
    float scale = max_abs(base_result, (long)N*N*N);
    float rel = (scale > 0.f) ? diff / scale : 0.f;

    double pts = (double)(N-2) * (double)(N-2) * (double)(N-2) * (double)iters;
    printf("N=%d iters=%d B=%d T=%d omega=%.3f\n", N, iters, B, T, omega);
    printf("  baseline : %9.4f s   (%7.2f Mupdates/s)\n",
           t_base, pts / t_base / 1e6);
    printf("  temporal : %9.4f s   (%7.2f Mupdates/s)  speedup %5.2fx\n",
           t_temp, pts / t_temp / 1e6, t_base / t_temp);
    printf("  max|base-temp| = %.4e   rel = %.4e\n", diff, rel);

    free(base_a); free(base_b); free(temp_a); free(temp_b);
    return 0;
}
