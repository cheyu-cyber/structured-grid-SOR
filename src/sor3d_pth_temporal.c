/*****************************************************************************

  EC527 Project — 3D SOR, pthreads with temporal blocking
  Pthreads version of sor3d_omp.c.  Same shrinking-trapezoid scheme on a
  7-point Laplacian; threads share work via a static partition of the
  cube-tile grid (mirroring OMP's `collapse(3) schedule(static)`).
  Baseline pthread sweep (parallel over outer i, strip decomposition) is
  in the same binary so the per-thread baseline-vs-temporal comparison
  is taken on the exact same compilation and machine.

    Build: gcc -O1 -std=gnu11 -pthread sor3d_pth_temporal.c \
               -lpthread -lrt -lm -o sor3d_pth_temporal
    Run:   ./sor3d_pth_temporal <N> <iters> [B=32] [T=2]
                                [--threads NT=4]

  Notes vs. sor3d_omp.c:
    - Per-thread scratch is a flat S^3 array indexed with strides
      (S*S, S, 1).  We deliberately use `S` everywhere, including the
      compute strides, so non-full edge tiles index the same layout as
      the load.  (sor3d_omp.c uses Sj/Sk for compute strides — fine
      when Si=Sj=Sk=S, which is every tile when (N-2) is a multiple
      of B, but inconsistent otherwise.  Worth fixing on a separate
      pass.)
    - Cubic tiles only.  The "rectangular tile B x B x Nk" idea from
      report.md:230 (next-step #2 to amortise halo over the long axis)
      is left for future work.

 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <pthread.h>

#define MINVAL  0.0
#define MAXVAL 10.0
#define OMEGA   0.9

typedef double data_t;

#define IDX3(i,j,k,N) (((size_t)(i)*(N) + (j))*(N) + (k))

static const double INV6 = 1.0 / 6.0;

/* -=-=-=-=- Time, init, validation -=-=-=-=- */
static double interval(struct timespec a, struct timespec b)
{
    struct timespec t;
    t.tv_sec  = b.tv_sec  - a.tv_sec;
    t.tv_nsec = b.tv_nsec - a.tv_nsec;
    if (t.tv_nsec < 0) { t.tv_sec -= 1; t.tv_nsec += 1000000000; }
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static double fRand(double a, double b)
{
    double f = (double)random() / (double)RAND_MAX;
    return a + f * (b - a);
}

static void init_array(data_t *arr, long len, unsigned seed)
{
    srandom(seed);
    for (long i = 0; i < len; i++)
        arr[i] = (data_t) fRand((double)MINVAL, (double)MAXVAL);
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

/* ============================ Boundary copy =============================== */
static void copy_boundaries_3d(int N, const data_t *src, data_t *dst,
                               int br_lo, int br_hi)
{
    /* Each thread covers indices [br_lo, br_hi) of the perimeter loop.
       At thread-0 union all chunks cover all 6 faces. */
    for (int a = br_lo; a < br_hi; a++) {
        for (int b = 0; b < N; b++) {
            dst[IDX3(0, a, b, N)]     = src[IDX3(0, a, b, N)];
            dst[IDX3(N-1, a, b, N)]   = src[IDX3(N-1, a, b, N)];
            dst[IDX3(a, 0, b, N)]     = src[IDX3(a, 0, b, N)];
            dst[IDX3(a, N-1, b, N)]   = src[IDX3(a, N-1, b, N)];
            dst[IDX3(a, b, 0, N)]     = src[IDX3(a, b, 0, N)];
            dst[IDX3(a, b, N-1, N)]   = src[IDX3(a, b, N-1, N)];
        }
    }
}

/* ============================ Baseline worker ============================== */
typedef struct {
    int tid, nthreads;
    int N, iters;
    double omega;
    data_t **psrc, **pdst;
    pthread_barrier_t *bar;
} base_args_t;

static void *base_worker(void *p)
{
    base_args_t *a = (base_args_t*) p;
    int N = a->N, nt = a->nthreads, tid = a->tid;
    int row_lo = 1 + ((long)(N-2) * tid)       / nt;
    int row_hi = 1 + ((long)(N-2) * (tid + 1)) / nt;
    int br_lo  = ((long)N * tid)       / nt;
    int br_hi  = ((long)N * (tid + 1)) / nt;

    for (int it = 0; it < a->iters; it++) {
        const data_t *src = *a->psrc;
        data_t       *dst = *a->pdst;

        for (int i = row_lo; i < row_hi; i++) {
            for (int j = 1; j < N - 1; j++) {
                const data_t *sm_i = src + IDX3(i-1, j, 0, N);
                const data_t *sc_i = src + IDX3(i,   j, 0, N);
                const data_t *sp_i = src + IDX3(i+1, j, 0, N);
                const data_t *sm_j = src + IDX3(i, j-1, 0, N);
                const data_t *sp_j = src + IDX3(i, j+1, 0, N);
                data_t       *dc   = dst + IDX3(i,   j, 0, N);
                for (int k = 1; k < N - 1; k++) {
                    data_t s = sc_i[k];
                    data_t nb = INV6 * (sm_i[k] + sp_i[k] +
                                        sm_j[k] + sp_j[k] +
                                        sc_i[k-1] + sc_i[k+1]);
                    dc[k] = s - a->omega * (s - nb);
                }
            }
        }
        copy_boundaries_3d(N, src, dst, br_lo, br_hi);
        pthread_barrier_wait(a->bar);
        if (tid == 0) {
            data_t *tmp = *a->psrc; *a->psrc = *a->pdst; *a->pdst = tmp;
        }
        pthread_barrier_wait(a->bar);
    }
    return NULL;
}

/* ============================ Temporal worker ============================== */
typedef struct {
    int tid, nthreads;
    int N, B, T;
    int super;
    double omega;
    data_t **psrc, **pdst;
    pthread_barrier_t *bar;
} temp_args_t;

static void *temp_worker(void *p)
{
    temp_args_t *a = (temp_args_t*) p;
    const int N = a->N, B = a->B, T = a->T, S = B + 2 * T;
    const int tid = a->tid, nt = a->nthreads;
    const double omega = a->omega;

    /* Per-thread scratch: two ping-pong buffers, S^3 each. */
    size_t sbytes = (size_t)S * S * S * sizeof(data_t);
    data_t *sa = (data_t*) malloc(sbytes);
    data_t *sb = (data_t*) malloc(sbytes);
    if (!sa || !sb) {
        fprintf(stderr, "thread %d: scratch alloc failed (%zu)\n",
                tid, sbytes);
        return NULL;
    }

    int nti = (N - 2 + B - 1) / B;
    int ntj = (N - 2 + B - 1) / B;
    int ntk = (N - 2 + B - 1) / B;
    int total = nti * ntj * ntk;
    int t_start = ((long)total * tid)       / nt;
    int t_end   = ((long)total * (tid + 1)) / nt;
    int br_lo   = ((long)N * tid)       / nt;
    int br_hi   = ((long)N * (tid + 1)) / nt;

    for (int s = 0; s < a->super; s++) {
        const data_t *src = *a->psrc;
        data_t       *dst = *a->pdst;

        copy_boundaries_3d(N, src, dst, br_lo, br_hi);
        pthread_barrier_wait(a->bar);

        for (int idx = t_start; idx < t_end; idx++) {
            int ti = idx / (ntj * ntk);
            int rem = idx % (ntj * ntk);
            int tj = rem / ntk;
            int tk = rem % ntk;

            int i0 = 1 + ti * B;
            int j0 = 1 + tj * B;
            int k0 = 1 + tk * B;
            int bi = (i0 + B <= N - 1) ? B : (N - 1 - i0);
            int bj = (j0 + B <= N - 1) ? B : (N - 1 - j0);
            int bk = (k0 + B <= N - 1) ? B : (N - 1 - k0);

            int gi0 = i0 - T, gj0 = j0 - T, gk0 = k0 - T;
            int Si = bi + 2 * T, Sj = bj + 2 * T, Sk = bk + 2 * T;

            /* Load S^3 scratch with clamp-to-edge.  Cells beyond the
               (Si, Sj, Sk) active region stay uninitialised but are
               never read (the trapezoid is bounded by the active
               region). */
            for (int si = 0; si < Si; si++) {
                int gi = gi0 + si;
                if (gi < 0) gi = 0; else if (gi > N - 1) gi = N - 1;
                for (int sj = 0; sj < Sj; sj++) {
                    int gj = gj0 + sj;
                    if (gj < 0) gj = 0; else if (gj > N - 1) gj = N - 1;
                    const data_t *srow = src + IDX3(gi, gj, 0, N);
                    data_t *arow = sa + ((size_t)si * S + sj) * S;
                    for (int sk = 0; sk < Sk; sk++) {
                        int gk = gk0 + sk;
                        if (gk < 0) gk = 0; else if (gk > N - 1) gk = N - 1;
                        arow[sk] = srow[gk];
                    }
                }
            }
            memcpy(sb, sa, sbytes);

            /* T sub-steps; central B^3 ends up T-step-advanced. */
            data_t *A = sa, *Bp = sb;
            for (int t = 0; t < T; t++) {
                int lo_i = 1 + t, hi_i = Si - 1 - t;
                int lo_j = 1 + t, hi_j = Sj - 1 - t;
                int lo_k = 1 + t, hi_k = Sk - 1 - t;

                int ulo_i = lo_i; if (gi0 + ulo_i < 1)     ulo_i = 1 - gi0;
                int uhi_i = hi_i; if (gi0 + uhi_i > N - 1) uhi_i = N - 1 - gi0;
                int ulo_j = lo_j; if (gj0 + ulo_j < 1)     ulo_j = 1 - gj0;
                int uhi_j = hi_j; if (gj0 + uhi_j > N - 1) uhi_j = N - 1 - gj0;
                int ulo_k = lo_k; if (gk0 + ulo_k < 1)     ulo_k = 1 - gk0;
                int uhi_k = hi_k; if (gk0 + uhi_k > N - 1) uhi_k = N - 1 - gk0;

                memcpy(Bp, A, sbytes);

                for (int si = ulo_i; si < uhi_i; si++) {
                    for (int sj = ulo_j; sj < uhi_j; sj++) {
                        const data_t *am_i = A + ((size_t)(si-1)*S + sj) * S;
                        const data_t *ac_i = A + ((size_t) si   *S + sj) * S;
                        const data_t *ap_i = A + ((size_t)(si+1)*S + sj) * S;
                        const data_t *am_j = A + ((size_t) si   *S + (sj-1)) * S;
                        const data_t *ap_j = A + ((size_t) si   *S + (sj+1)) * S;
                        data_t       *bc   = Bp + ((size_t) si  *S + sj) * S;
                        for (int sk = ulo_k; sk < uhi_k; sk++) {
                            data_t v = ac_i[sk];
                            data_t nb = INV6 * (am_i[sk] + ap_i[sk] +
                                                am_j[sk] + ap_j[sk] +
                                                ac_i[sk-1] + ac_i[sk+1]);
                            bc[sk] = v - omega * (v - nb);
                        }
                    }
                }
                data_t *tmp = A; A = Bp; Bp = tmp;
            }

            /* Write central bi x bj x bk back. */
            for (int si = T; si < T + bi; si++) {
                int gi = gi0 + si;
                for (int sj = T; sj < T + bj; sj++) {
                    int gj = gj0 + sj;
                    memcpy(dst + IDX3(gi, gj, gk0 + T, N),
                           A   + ((size_t)si * S + sj) * S + T,
                           (size_t)bk * sizeof(data_t));
                }
            }
        }

        pthread_barrier_wait(a->bar);
        if (tid == 0) {
            data_t *tmp = *a->psrc; *a->psrc = *a->pdst; *a->pdst = tmp;
        }
        pthread_barrier_wait(a->bar);
    }

    free(sa); free(sb);
    return NULL;
}

/* ============================ Serial reference ============================ */
static double run_serial(int N, int iters, double omega,
                         data_t *a, data_t *b, data_t **out)
{
    data_t *src = a, *dst = b;
    struct timespec t0, t1;
    clock_gettime(CLOCK_REALTIME, &t0);
    for (int it = 0; it < iters; it++) {
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                const data_t *sm_i = src + IDX3(i-1, j, 0, N);
                const data_t *sc_i = src + IDX3(i,   j, 0, N);
                const data_t *sp_i = src + IDX3(i+1, j, 0, N);
                const data_t *sm_j = src + IDX3(i, j-1, 0, N);
                const data_t *sp_j = src + IDX3(i, j+1, 0, N);
                data_t       *dc   = dst + IDX3(i,   j, 0, N);
                for (int k = 1; k < N - 1; k++) {
                    data_t s = sc_i[k];
                    data_t nb = INV6 * (sm_i[k] + sp_i[k] +
                                        sm_j[k] + sp_j[k] +
                                        sc_i[k-1] + sc_i[k+1]);
                    dc[k] = s - omega * (s - nb);
                }
            }
        }
        copy_boundaries_3d(N, src, dst, 0, N);
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
            "usage: %s N iters [B=32] [T=2] [--threads NT]\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    int iters = atoi(argv[2]);
    int B = 32, T = 2, nthreads = 4;
    int pos = 0;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            nthreads = atoi(argv[++i]);
        } else if (pos == 0) { B = atoi(argv[i]); pos++; }
        else if (pos == 1)   { T = atoi(argv[i]); pos++; }
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 1; }
    }
    if (N < 4 || iters < 1 || B < 1 || T < 1 || nthreads < 1) {
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
    data_t *ref_a  = (data_t*) malloc(bytes);
    data_t *ref_b  = (data_t*) malloc(bytes);
    if (!base_a || !base_b || !temp_a || !temp_b || !ref_a || !ref_b) {
        fprintf(stderr, "alloc failed (%zu bytes each)\n", bytes); return 1;
    }
    init_array(base_a, (long)N*N*N, 527u);
    memcpy(base_b, base_a, bytes);
    memcpy(temp_a, base_a, bytes);
    memcpy(temp_b, base_a, bytes);
    memcpy(ref_a,  base_a, bytes);
    memcpy(ref_b,  base_a, bytes);

    /* ---- Serial reference ---- */
    data_t *ref_out = NULL;
    double t_ref = run_serial(N, iters, OMEGA, ref_a, ref_b, &ref_out);

    /* ---- Pthreads baseline ---- */
    pthread_barrier_t bar;
    pthread_barrier_init(&bar, NULL, nthreads);
    pthread_t *tids  = (pthread_t*) malloc((size_t)nthreads * sizeof(pthread_t));
    base_args_t *bargs = (base_args_t*) malloc((size_t)nthreads * sizeof(base_args_t));
    data_t *bsrc = base_a, *bdst = base_b;
    for (int t = 0; t < nthreads; t++) {
        bargs[t] = (base_args_t){
            .tid = t, .nthreads = nthreads, .N = N, .iters = iters,
            .omega = OMEGA, .psrc = &bsrc, .pdst = &bdst, .bar = &bar
        };
    }
    struct timespec t0, t1;
    clock_gettime(CLOCK_REALTIME, &t0);
    for (int t = 0; t < nthreads; t++)
        pthread_create(&tids[t], NULL, base_worker, &bargs[t]);
    for (int t = 0; t < nthreads; t++) pthread_join(tids[t], NULL);
    clock_gettime(CLOCK_REALTIME, &t1);
    double t_base = interval(t0, t1);
    data_t *base_out = bsrc;

    /* ---- Pthreads temporal ---- */
    int super = iters / T;
    temp_args_t *targs = (temp_args_t*) malloc((size_t)nthreads * sizeof(temp_args_t));
    data_t *tsrc = temp_a, *tdst = temp_b;
    for (int t = 0; t < nthreads; t++) {
        targs[t] = (temp_args_t){
            .tid = t, .nthreads = nthreads, .N = N, .B = B, .T = T,
            .super = super, .omega = OMEGA,
            .psrc = &tsrc, .pdst = &tdst, .bar = &bar
        };
    }
    clock_gettime(CLOCK_REALTIME, &t0);
    for (int t = 0; t < nthreads; t++)
        pthread_create(&tids[t], NULL, temp_worker, &targs[t]);
    for (int t = 0; t < nthreads; t++) pthread_join(tids[t], NULL);
    clock_gettime(CLOCK_REALTIME, &t1);
    double t_temp = interval(t0, t1);
    data_t *temp_out = tsrc;

    pthread_barrier_destroy(&bar);

    /* ---- Validation ---- */
    data_t diff_b = max_diff(ref_out, base_out, (long)N*N*N);
    data_t diff_t = max_diff(ref_out, temp_out, (long)N*N*N);
    data_t scale  = max_val(ref_out, (long)N*N*N);
    data_t rel_b  = (scale > 0.0) ? diff_b / scale : 0.0;
    data_t rel_t  = (scale > 0.0) ? diff_t / scale : 0.0;

    double pts = (double)(N-2) * (double)(N-2) * (double)(N-2) * (double)iters;
    const double CPNS = 2.0;
    double cpe_ref  = CPNS * 1.0e9 * t_ref  / pts;
    double cpe_base = CPNS * 1.0e9 * t_base / pts;
    double cpe_temp = CPNS * 1.0e9 * t_temp / pts;

    printf("N=%d iters=%d B=%d T=%d threads=%d OMEGA=%.3f\n",
           N, iters, B, T, nthreads, (double)OMEGA);
    printf("  serial   : %9.4f s  (%10.3g cycles, %7.3f CPE)\n",
           t_ref, CPNS * 1.0e9 * t_ref, cpe_ref);
    printf("  baseline : %9.4f s  (%10.3g cycles, %7.3f CPE)  "
           "speedup vs serial %5.2fx\n",
           t_base, CPNS * 1.0e9 * t_base, cpe_base, t_ref / t_base);
    printf("  temporal : %9.4f s  (%10.3g cycles, %7.3f CPE)  "
           "speedup vs serial %5.2fx, vs baseline %5.2fx\n",
           t_temp, CPNS * 1.0e9 * t_temp, cpe_temp,
           t_ref / t_temp, t_base / t_temp);
    printf("  max|serial-baseline| = %.4e   rel = %.4e\n",
           (double)diff_b, (double)rel_b);
    printf("  max|serial-temporal| = %.4e   rel = %.4e\n",
           (double)diff_t, (double)rel_t);

    printf("CSV,sor3d_pth_temporal,3,%d,%d,%d,baseline,strip,%.6e,%.6e,%.4e\n",
           N, iters, nthreads, t_base, cpe_base, (double)diff_b);
    printf("CSV,sor3d_pth_temporal,3,%d,%d,%d,temporal,B=%d;T=%d,%.6e,%.6e,%.4e\n",
           N, iters, nthreads, B, T, t_temp, cpe_temp, (double)diff_t);

    free(base_a); free(base_b); free(temp_a); free(temp_b);
    free(ref_a); free(ref_b);
    free(tids); free(bargs); free(targs);
    return 0;
}
