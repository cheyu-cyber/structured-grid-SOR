/*****************************************************************************

  EC527 Project — 2D SOR, pthreads decomposition study
  Lab-5-Part-4 redo: three decomposition strategies and two scheduling
  strategies, in one binary.  Validates each run against an in-binary
  single-threaded reference.

  Decomposition (--mode):
    strip        Each thread owns a contiguous block of interior rows.
                 Best spatial locality on row-major data; one shared
                 boundary row per neighbour pair.
    interleaved  Thread t handles rows 1+t, 1+t+nt, 1+t+2nt, ...
                 Perfectly load-balanced but stride-nt access pattern
                 evicts cache lines that another thread will read.
    block        2D block decomposition: nthreads is factored as pi*pj
                 (most-square factor near sqrt(nt)) and each thread owns
                 a rows×cols rectangle.  Halo-to-interior is worse than
                 strip for the same thread count, but no two threads
                 share a row (so first-touch / write-back conflicts on a
                 cache line halve).

  Scheduling (--sched):
    persistent   Single pthread_create per worker; workers loop iters
                 times with two pthread_barrier_wait per sweep.
    spawn        pthread_create + pthread_join per sweep.  Exposes the
                 full thread-spawn cost iters times — designed to make
                 the Lab-6-Part-1b OMP-vs-pthreads overhead comparison
                 quantitative.

  Build: gcc -O1 -std=gnu11 -pthread sor2d_pth_decomp.c -lpthread -lrt -lm \
             -o sor2d_pth_decomp
  Run:   ./sor2d_pth_decomp <N> <iters>
              [--threads NT] [--mode strip|interleaved|block]
              [--sched persistent|spawn] [--ppm path]

 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <pthread.h>

#define MINVAL 0.0
#define MAXVAL 10.0
#define OMEGA  0.9

typedef double data_t;

/* -=-=-=-=- Time measurement by clock_gettime() -=-=-=-=- */
static double interval(struct timespec start, struct timespec end)
{
    struct timespec t;
    t.tv_sec  = end.tv_sec  - start.tv_sec;
    t.tv_nsec = end.tv_nsec - start.tv_nsec;
    if (t.tv_nsec < 0) { t.tv_sec -= 1; t.tv_nsec += 1000000000; }
    return (double)t.tv_sec + (double)t.tv_nsec * 1.0e-9;
}

static double fRand(double a, double b)
{
    double f = (double)random() / (double)RAND_MAX;
    return a + f * (b - a);
}

static void init_array(data_t *a, long len, unsigned seed)
{
    srandom(seed);
    for (long i = 0; i < len; i++)
        a[i] = (data_t) fRand((double)MINVAL, (double)MAXVAL);
}

static data_t max_diff(const data_t *a, const data_t *b, long len)
{
    data_t mx = 0.0;
    int bad = 0;
    for (long i = 0; i < len; i++) {
        data_t av = a[i], bv = b[i];
        if (!isfinite(av) || !isfinite(bv)) { bad++; continue; }
        data_t d = fabs(av - bv);
        if (d > mx) mx = d;
    }
    if (bad) {
        fprintf(stderr, "WARNING: %d non-finite values (diverged?)\n", bad);
        return INFINITY;
    }
    return mx;
}

static data_t max_val(const data_t *a, long len)
{
    data_t mx = 0.0;
    for (long i = 0; i < len; i++) {
        data_t v = fabs(a[i]);
        if (v > mx) mx = v;
    }
    return mx;
}

static int write_ppm_gray(const char *path, const data_t *a, int N)
{
    data_t mn = INFINITY, mx = -INFINITY;
    for (long i = 0; i < (long)N*N; i++) {
        data_t v = a[i];
        if (!isfinite(v)) continue;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    if (!isfinite(mn) || mn > mx) return -1;
    data_t scale = (mx > mn) ? 255.0 / (mx - mn) : 0.0;
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

/* ======================== Decomposition / scheduling ====================== */
typedef enum { MODE_STRIP, MODE_INTERLEAVED, MODE_BLOCK } decomp_t;
typedef enum { SCHED_PERSIST, SCHED_SPAWN }              sched_t;

static const char *mode_name(decomp_t m)
{
    return m == MODE_STRIP       ? "strip"
         : m == MODE_INTERLEAVED ? "interleaved"
         :                         "block";
}
static const char *sched_name(sched_t s)
{
    return s == SCHED_PERSIST ? "persistent" : "spawn";
}

/* Most-square factorisation: returns pi <= sqrt(n) and pj = n/pi. */
static void factor_2d(int n, int *pi, int *pj)
{
    int best = 1;
    for (int p = 1; p * p <= n; p++)
        if (n % p == 0) best = p;
    *pi = best;
    *pj = n / best;
}

/* Compute this thread's interior region [i_start, i_end) x [j_start, j_end),
   plus a row stride (1 for contiguous, nthreads for interleaved). */
static void thread_region(int tid, int nthreads, int N, decomp_t mode,
                          int pi, int pj,
                          int *i_start, int *i_end, int *i_step,
                          int *j_start, int *j_end)
{
    *j_start = 1; *j_end = N - 1; *i_step = 1;
    if (mode == MODE_STRIP) {
        *i_start = 1 + ((long)(N-2) * tid)        / nthreads;
        *i_end   = 1 + ((long)(N-2) * (tid + 1))  / nthreads;
    } else if (mode == MODE_INTERLEAVED) {
        *i_start = 1 + tid;
        *i_end   = N - 1;
        *i_step  = nthreads;
    } else {
        int pi_idx = tid / pj;
        int pj_idx = tid % pj;
        *i_start = 1 + ((long)(N-2) * pi_idx)       / pi;
        *i_end   = 1 + ((long)(N-2) * (pi_idx + 1)) / pi;
        *j_start = 1 + ((long)(N-2) * pj_idx)       / pj;
        *j_end   = 1 + ((long)(N-2) * (pj_idx + 1)) / pj;
    }
}

/* One sweep on this thread's region (no boundary cells, no swap). */
static void do_sweep(int tid, int nthreads, int N, double omega,
                     decomp_t mode, int pi, int pj,
                     const data_t *src, data_t *dst)
{
    int i_start, i_end, i_step, j_start, j_end;
    thread_region(tid, nthreads, N, mode, pi, pj,
                  &i_start, &i_end, &i_step, &j_start, &j_end);

    for (int i = i_start; i < i_end; i += i_step) {
        const data_t *sm = src + (size_t)(i-1) * N;
        const data_t *sc = src + (size_t) i    * N;
        const data_t *sp = src + (size_t)(i+1) * N;
        data_t       *dc = dst + (size_t) i    * N;
        for (int j = j_start; j < j_end; j++) {
            data_t s = sc[j];
            data_t nb = 0.25 * (sm[j] + sp[j] + sc[j-1] + sc[j+1]);
            dc[j] = s - omega * (s - nb);
        }
    }
}

/* Carry global boundary cells from src to dst.  4N cells, called once per
   sweep by either thread 0 (persistent) or main (spawn). */
static void carry_boundaries(int N, const data_t *src, data_t *dst)
{
    for (int j = 0; j < N; j++) {
        dst[j]           = src[j];
        dst[(N-1)*N + j] = src[(N-1)*N + j];
    }
    for (int i = 0; i < N; i++) {
        dst[(size_t)i*N]            = src[(size_t)i*N];
        dst[(size_t)i*N + (N-1)]    = src[(size_t)i*N + (N-1)];
    }
}

/* ============================ Persistent worker ============================ */
typedef struct {
    int tid, nthreads;
    int N, iters;
    double omega;
    decomp_t mode;
    int pi, pj;
    /* Double-pointers so thread 0's swap is visible to all threads. */
    data_t **psrc, **pdst;
    pthread_barrier_t *bar;
} persist_args_t;

static void *persist_worker(void *p)
{
    persist_args_t *a = (persist_args_t*) p;
    for (int k = 0; k < a->iters; k++) {
        const data_t *src = *a->psrc;
        data_t       *dst = *a->pdst;
        do_sweep(a->tid, a->nthreads, a->N, a->omega,
                 a->mode, a->pi, a->pj, src, dst);
        if (a->tid == 0) carry_boundaries(a->N, src, dst);
        pthread_barrier_wait(a->bar);
        if (a->tid == 0) {
            data_t *t = *a->psrc; *a->psrc = *a->pdst; *a->pdst = t;
        }
        pthread_barrier_wait(a->bar);
    }
    return NULL;
}

/* Returns wall time, leaves final result buffer pointer in *out. */
static double run_persistent(int N, int iters, int nthreads, double omega,
                             decomp_t mode, int pi, int pj,
                             data_t *a, data_t *b, data_t **out)
{
    pthread_barrier_t bar;
    pthread_barrier_init(&bar, NULL, nthreads);
    pthread_t *tids = (pthread_t*) malloc((size_t)nthreads * sizeof(pthread_t));
    persist_args_t *args =
        (persist_args_t*) malloc((size_t)nthreads * sizeof(persist_args_t));
    data_t *cur_src = a, *cur_dst = b;

    for (int t = 0; t < nthreads; t++) {
        args[t] = (persist_args_t){
            .tid = t, .nthreads = nthreads,
            .N = N, .iters = iters, .omega = omega,
            .mode = mode, .pi = pi, .pj = pj,
            .psrc = &cur_src, .pdst = &cur_dst, .bar = &bar
        };
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_REALTIME, &t0);
    for (int t = 0; t < nthreads; t++)
        pthread_create(&tids[t], NULL, persist_worker, &args[t]);
    for (int t = 0; t < nthreads; t++)
        pthread_join(tids[t], NULL);
    clock_gettime(CLOCK_REALTIME, &t1);

    pthread_barrier_destroy(&bar);
    free(tids); free(args);
    *out = cur_src;
    return interval(t0, t1);
}

/* ============================== Spawn worker ============================== */
typedef struct {
    int tid, nthreads;
    int N;
    double omega;
    decomp_t mode;
    int pi, pj;
    const data_t *src;
    data_t *dst;
} spawn_args_t;

static void *spawn_worker(void *p)
{
    spawn_args_t *a = (spawn_args_t*) p;
    do_sweep(a->tid, a->nthreads, a->N, a->omega,
             a->mode, a->pi, a->pj, a->src, a->dst);
    return NULL;
}

static double run_spawn(int N, int iters, int nthreads, double omega,
                        decomp_t mode, int pi, int pj,
                        data_t *a, data_t *b, data_t **out)
{
    pthread_t *tids = (pthread_t*) malloc((size_t)nthreads * sizeof(pthread_t));
    spawn_args_t *args =
        (spawn_args_t*) malloc((size_t)nthreads * sizeof(spawn_args_t));
    data_t *src = a, *dst = b;

    struct timespec t0, t1;
    clock_gettime(CLOCK_REALTIME, &t0);
    for (int k = 0; k < iters; k++) {
        for (int t = 0; t < nthreads; t++) {
            args[t] = (spawn_args_t){
                .tid = t, .nthreads = nthreads,
                .N = N, .omega = omega,
                .mode = mode, .pi = pi, .pj = pj,
                .src = src, .dst = dst
            };
            pthread_create(&tids[t], NULL, spawn_worker, &args[t]);
        }
        for (int t = 0; t < nthreads; t++)
            pthread_join(tids[t], NULL);
        carry_boundaries(N, src, dst);
        data_t *tmp = src; src = dst; dst = tmp;
    }
    clock_gettime(CLOCK_REALTIME, &t1);

    free(tids); free(args);
    *out = src; /* after the final swap, last-written buffer is `src`. */
    return interval(t0, t1);
}

/* ============================ Serial reference ============================ */
static double run_serial(int N, int iters, double omega,
                         data_t *a, data_t *b, data_t **out)
{
    data_t *src = a, *dst = b;
    struct timespec t0, t1;
    clock_gettime(CLOCK_REALTIME, &t0);
    for (int k = 0; k < iters; k++) {
        for (int i = 1; i < N-1; i++) {
            const data_t *sm = src + (size_t)(i-1) * N;
            const data_t *sc = src + (size_t) i    * N;
            const data_t *sp = src + (size_t)(i+1) * N;
            data_t       *dc = dst + (size_t) i    * N;
            for (int j = 1; j < N-1; j++) {
                data_t s = sc[j];
                data_t nb = 0.25 * (sm[j] + sp[j] + sc[j-1] + sc[j+1]);
                dc[j] = s - omega * (s - nb);
            }
        }
        carry_boundaries(N, src, dst);
        data_t *tmp = src; src = dst; dst = tmp;
    }
    clock_gettime(CLOCK_REALTIME, &t1);
    *out = src;
    return interval(t0, t1);
}

/* ===================================================================
   MAIN
   =================================================================== */
int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,
            "usage: %s N iters [--threads NT] "
            "[--mode strip|interleaved|block] "
            "[--sched persistent|spawn] [--ppm path]\n", argv[0]);
        return 1;
    }
    int N        = atoi(argv[1]);
    int iters    = atoi(argv[2]);
    int nthreads = 4;
    decomp_t mode = MODE_STRIP;
    sched_t  sched = SCHED_PERSIST;
    const char *ppm_path = NULL;

    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            nthreads = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--mode") && i + 1 < argc) {
            const char *s = argv[++i];
            if      (!strcmp(s, "strip"))       mode = MODE_STRIP;
            else if (!strcmp(s, "interleaved")) mode = MODE_INTERLEAVED;
            else if (!strcmp(s, "block"))       mode = MODE_BLOCK;
            else { fprintf(stderr, "bad --mode: %s\n", s); return 1; }
        } else if (!strcmp(argv[i], "--sched") && i + 1 < argc) {
            const char *s = argv[++i];
            if      (!strcmp(s, "persistent")) sched = SCHED_PERSIST;
            else if (!strcmp(s, "spawn"))      sched = SCHED_SPAWN;
            else { fprintf(stderr, "bad --sched: %s\n", s); return 1; }
        } else if (!strcmp(argv[i], "--ppm") && i + 1 < argc) {
            ppm_path = argv[++i];
        } else {
            fprintf(stderr, "unknown arg: %s\n", argv[i]); return 1;
        }
    }
    if (N < 4 || iters < 1 || nthreads < 1) {
        fprintf(stderr, "bad args\n"); return 1;
    }

    int pi = 1, pj = nthreads;
    if (mode == MODE_BLOCK) factor_2d(nthreads, &pi, &pj);

    /* Sanity-check the per-thread region is non-empty.  A degenerate row
     * range (e.g. interleaved with nthreads > N-2) silently makes a
     * thread skip its work; flag instead of producing a wrong result. */
    if (mode == MODE_INTERLEAVED && nthreads > N - 2) {
        fprintf(stderr, "interleaved needs nthreads (%d) <= N-2 (%d)\n",
                nthreads, N - 2);
        return 1;
    }
    if (mode == MODE_BLOCK && (pi > N - 2 || pj > N - 2)) {
        fprintf(stderr, "block: factorisation %dx%d does not fit in %dx%d "
                        "interior\n", pi, pj, N - 2, N - 2);
        return 1;
    }

    size_t bytes = (size_t)N * N * sizeof(data_t);
    data_t *pth_a = (data_t*) malloc(bytes);
    data_t *pth_b = (data_t*) malloc(bytes);
    data_t *ref_a = (data_t*) malloc(bytes);
    data_t *ref_b = (data_t*) malloc(bytes);
    if (!pth_a || !pth_b || !ref_a || !ref_b) {
        fprintf(stderr, "alloc failed\n"); return 1;
    }

    init_array(pth_a, (long)N*N, 527u);
    memcpy(pth_b, pth_a, bytes);
    memcpy(ref_a, pth_a, bytes);
    memcpy(ref_b, pth_a, bytes);

    /* ---- Serial reference ---- */
    data_t *ref_out = NULL;
    double t_ref = run_serial(N, iters, OMEGA, ref_a, ref_b, &ref_out);

    /* ---- Threaded run ---- */
    data_t *pth_out = NULL;
    double t_pth =
        (sched == SCHED_PERSIST)
        ? run_persistent(N, iters, nthreads, OMEGA, mode, pi, pj,
                         pth_a, pth_b, &pth_out)
        : run_spawn     (N, iters, nthreads, OMEGA, mode, pi, pj,
                         pth_a, pth_b, &pth_out);

    /* ---- Validation ---- */
    data_t diff  = max_diff(ref_out, pth_out, (long)N*N);
    data_t scale = max_val(ref_out, (long)N*N);
    data_t rel   = (scale > 0.0) ? diff / scale : 0.0;

    double pts  = (double)(N-2) * (double)(N-2) * (double)iters;
    /* CPE (lab convention): wall-clock cycles per output cell at CPNS=2.0. */
    const double CPNS = 2.0;
    double cpe_ref = CPNS * 1.0e9 * t_ref / pts;
    double cpe_pth = CPNS * 1.0e9 * t_pth / pts;

    /* Human-readable. */
    printf("N=%d iters=%d threads=%d mode=%s sched=%s",
           N, iters, nthreads, mode_name(mode), sched_name(sched));
    if (mode == MODE_BLOCK) printf(" pi=%d pj=%d", pi, pj);
    printf(" OMEGA=%.3f\n", (double)OMEGA);
    printf("  serial   : %9.4f s  (%10.3g cycles, %7.3f CPE)\n",
           t_ref, CPNS * 1.0e9 * t_ref, cpe_ref);
    printf("  threaded : %9.4f s  (%10.3g cycles, %7.3f CPE)  speedup %5.2fx\n",
           t_pth, CPNS * 1.0e9 * t_pth, cpe_pth, t_ref / t_pth);
    printf("  max|serial-threaded| = %.4e   rel = %.4e\n",
           (double)diff, (double)rel);

    /* Machine-readable: one CSV line that sweep.sh / plot.py grep. */
    printf("CSV,sor2d_pth_decomp,2,%d,%d,%d,%s,%s,%.6e,%.6e,%.4e\n",
           N, iters, nthreads, mode_name(mode), sched_name(sched),
           t_pth, cpe_pth, (double)diff);

    if (ppm_path) {
        if (write_ppm_gray(ppm_path, pth_out, N) == 0)
            printf("  wrote %s\n", ppm_path);
        else
            fprintf(stderr, "failed to write %s\n", ppm_path);
    }

    free(pth_a); free(pth_b); free(ref_a); free(ref_b);
    return 0;
}
