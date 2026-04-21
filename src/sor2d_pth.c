/*
 * 2D SOR — pthreads baseline.
 *
 * Same Jacobi-like ping-pong stencil as the rest of the project. Threads split
 * the interior rows [1, N-1) into contiguous strips; a pthread_barrier
 * synchronizes between sweeps since every cell read comes from `src` and
 * every write goes to `dst`, so no thread may start the next sweep before
 * all threads have finished writing the current one.
 *
 * Validated against a single-threaded run of the identical stencil.
 *
 * Usage:  ./sor2d_pth <N> <iters> [nthreads=4] [--ppm path]
 * Build:  gcc -O3 -march=native -std=c11 -pthread sor2d_pth.c -o sor2d_pth -lm
 */

#include "common.h"
#include <pthread.h>

typedef struct {
    int tid;
    int nthreads;
    int N;
    int iters;
    float omega;
    /* Double-pointers so thread 0's post-sweep buffer swap is visible to all. */
    float **src;
    float **dst;
    pthread_barrier_t *bar;
} thread_args_t;

static void *worker(void *p)
{
    thread_args_t *a = (thread_args_t*) p;
    const int N = a->N;
    const int tid = a->tid;
    const int nt = a->nthreads;
    const float omega = a->omega;

    /* Strip decomposition over interior rows i in [1, N-1). */
    const int start = 1 + ((N - 2) * tid)       / nt;
    const int end   = 1 + ((N - 2) * (tid + 1)) / nt;

    for (int k = 0; k < a->iters; k++) {
        const float *src = *a->src;
        float       *dst = *a->dst;

        for (int i = start; i < end; i++) {
            const float *sm = src + (size_t)(i-1) * N;
            const float *sc = src + (size_t) i    * N;
            const float *sp = src + (size_t)(i+1) * N;
            float       *dc = dst + (size_t) i    * N;
            for (int j = 1; j < N-1; j++) {
                float s = sc[j];
                float nb = 0.25f * (sm[j] + sp[j] + sc[j-1] + sc[j+1]);
                dc[j] = s - omega * (s - nb);
            }
            /* Carry through the two boundary columns for this row. */
            dc[0]     = sc[0];
            dc[N-1]   = sc[N-1];
        }
        /* Thread 0 carries through the top/bottom boundary rows. */
        if (tid == 0) {
            for (int j = 0; j < N; j++) {
                dst[j]             = src[j];
                dst[(N-1)*N + j]   = src[(N-1)*N + j];
            }
        }

        pthread_barrier_wait(a->bar);
        if (tid == 0) {
            float *tmp = *a->src; *a->src = *a->dst; *a->dst = tmp;
        }
        pthread_barrier_wait(a->bar);
    }
    return NULL;
}

/* Serial reference with the same stencil — used to validate the pthreads result
 * is bit-identical (or to within floating-point noise). */
static void sor2d_serial(float *src, float *dst, int N, int iters, float omega)
{
    for (int k = 0; k < iters; k++) {
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
        for (int j = 0; j < N; j++) {
            dst[j]           = src[j];
            dst[(N-1)*N + j] = src[(N-1)*N + j];
        }
        for (int i = 0; i < N; i++) {
            dst[i*N]           = src[i*N];
            dst[i*N + (N-1)]   = src[i*N + (N-1)];
        }
        float *tmp = src; src = dst; dst = tmp;
    }
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s N iters [nthreads=4] [--ppm path]\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    int iters = atoi(argv[2]);
    int nthreads = 4;
    const char *ppm_path = NULL;

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--ppm") == 0 && i + 1 < argc) {
            ppm_path = argv[++i];
        } else {
            nthreads = atoi(argv[i]);
        }
    }
    float omega = OMEGA_DEFAULT;
    if (N < 4 || iters < 1 || nthreads < 1) { fprintf(stderr, "bad args\n"); return 1; }

    size_t bytes = (size_t)N * N * sizeof(float);
    float *pth_a = (float*) malloc(bytes);
    float *pth_b = (float*) malloc(bytes);
    float *ref_a = (float*) malloc(bytes);
    float *ref_b = (float*) malloc(bytes);

    init_array(pth_a, (long)N*N, 527u);
    memcpy(pth_b, pth_a, bytes);
    memcpy(ref_a, pth_a, bytes);
    memcpy(ref_b, pth_a, bytes);

    /* Serial reference. */
    double t0 = wall_seconds();
    sor2d_serial(ref_a, ref_b, N, iters, omega);
    double t_ref = wall_seconds() - t0;
    float *ref_result = (iters % 2 == 0) ? ref_a : ref_b;

    /* Pthreads run. */
    pthread_barrier_t bar;
    pthread_barrier_init(&bar, NULL, nthreads);
    pthread_t *threads = (pthread_t*) malloc((size_t)nthreads * sizeof(pthread_t));
    thread_args_t *args = (thread_args_t*) malloc((size_t)nthreads * sizeof(thread_args_t));
    float *cur_src = pth_a, *cur_dst = pth_b;
    for (int t = 0; t < nthreads; t++) {
        args[t].tid = t; args[t].nthreads = nthreads;
        args[t].N = N;   args[t].iters = iters; args[t].omega = omega;
        args[t].src = &cur_src; args[t].dst = &cur_dst; args[t].bar = &bar;
    }

    t0 = wall_seconds();
    for (int t = 0; t < nthreads; t++) pthread_create(&threads[t], NULL, worker, &args[t]);
    for (int t = 0; t < nthreads; t++) pthread_join(threads[t], NULL);
    double t_pth = wall_seconds() - t0;
    pthread_barrier_destroy(&bar);

    /* After every sweep thread 0 swaps (*src, *dst); after `iters` sweeps the
     * last-written buffer is cur_src. */
    float *pth_result = cur_src;

    float diff = max_abs_diff(ref_result, pth_result, (long)N*N);
    float scale = max_abs(ref_result, (long)N*N);
    float rel = (scale > 0.f) ? diff / scale : 0.f;

    double pts = (double)(N-2) * (double)(N-2) * (double)iters;
    printf("N=%d iters=%d nthreads=%d omega=%.3f\n", N, iters, nthreads, omega);
    printf("  serial  : %9.4f s   (%7.2f Mupdates/s)\n",
           t_ref, pts / t_ref / 1e6);
    printf("  pthread : %9.4f s   (%7.2f Mupdates/s)  speedup %5.2fx\n",
           t_pth, pts / t_pth / 1e6, t_ref / t_pth);
    printf("  max|serial-pthread| = %.4e   rel = %.4e\n", diff, rel);

    if (ppm_path) {
        if (write_ppm_gray(ppm_path, pth_result, N) == 0)
            printf("  wrote %s\n", ppm_path);
        else
            fprintf(stderr, "failed to write %s\n", ppm_path);
    }

    free(pth_a); free(pth_b); free(ref_a); free(ref_b);
    free(threads); free(args);
    return 0;
}
