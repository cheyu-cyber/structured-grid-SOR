/*****************************************************************************

  EC527 Project — 2D SOR, pthreads with temporal blocking
  Pthreads version of sor2d_omp.c.  Same shrinking-trapezoid scheme;
  threads share work via a static partition of the tile grid (mirroring
  OMP's `collapse(2) schedule(static)`).  Baseline (one sweep per
  iteration, strip decomposition over interior rows) is in the same
  binary so the per-thread baseline/temporal comparison is taken on the
  exact same compilation and the exact same machine.

    Build: gcc -O1 -std=gnu11 -pthread sor2d_pth_temporal.c \
               -lpthread -lrt -lm -o sor2d_pth_temporal
    Run:   ./sor2d_pth_temporal <N> <iters> [B=64] [T=4]
                                [--threads NT=4] [--ppm path]

  `iters` must be a multiple of `T`.  Validates each run against an
  in-binary single-threaded reference (serial baseline run).

  Notes:
    - Persistent threads.  We pthread_create once per phase (baseline
      run, temporal run) and pthread_join once at the end.  The within-
      phase synchronisation uses pthread_barrier_t.
    - Per-thread scratch buffers.  Each temporal worker malloc()s its
      own (B+2T)x(B+2T) scratch pair on entry and free()s on exit.
      Identical to OMP's per-thread scratch inside the parallel region.

 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <pthread.h>

#define MINVAL 0.0f
#define MAXVAL 10.0f
#define OMEGA  0.9f       /* Damped Jacobi: stable for omega in (0, 1] */

typedef float data_t;

#define IDX2(i,j,ld) ((size_t)(i)*(ld) + (j))

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
    double f = (double)rand() / (double)RAND_MAX;
    return a + f * (b - a);
}

static void init_array(data_t *arr, long len, unsigned seed)
{
    srand(seed);
    for (long i = 0; i < len; i++)
        arr[i] = (data_t) fRand((double)MINVAL, (double)MAXVAL);
}

static data_t max_diff(const data_t *a, const data_t *b, long len)
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

static data_t max_val(const data_t *a, long len)
{
    data_t mx = 0.0f;
    for (long i = 0; i < len; i++) {
        data_t v = fabsf(a[i]);
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

/* ============================ Baseline worker ============================== */
typedef struct {
    int tid, nthreads;
    int N, iters;
    float omega;
    data_t **psrc, **pdst;
    pthread_barrier_t *bar;
} base_args_t;

static void *base_worker(void *p)
{
    base_args_t *a = (base_args_t*) p;
    int N = a->N, nt = a->nthreads, tid = a->tid;
    int row_lo = 1 + ((long)(N-2) * tid)       / nt;
    int row_hi = 1 + ((long)(N-2) * (tid + 1)) / nt;

    for (int k = 0; k < a->iters; k++) {
        const data_t *src = *a->psrc;
        data_t       *dst = *a->pdst;

        for (int i = row_lo; i < row_hi; i++) {
            const data_t *sm = src + IDX2(i-1, 0, N);
            const data_t *sc = src + IDX2(i,   0, N);
            const data_t *sp = src + IDX2(i+1, 0, N);
            data_t       *dc = dst + IDX2(i,   0, N);
            for (int j = 1; j < N - 1; j++) {
                data_t s = sc[j];
                data_t nb = 0.25f * (sm[j] + sp[j] + sc[j-1] + sc[j+1]);
                dc[j] = s - a->omega * (s - nb);
            }
        }
        if (tid == 0) {
            for (int j = 0; j < N; j++) {
                dst[IDX2(0, j, N)]   = src[IDX2(0, j, N)];
                dst[IDX2(N-1, j, N)] = src[IDX2(N-1, j, N)];
            }
            for (int i = 0; i < N; i++) {
                dst[IDX2(i, 0, N)]    = src[IDX2(i, 0, N)];
                dst[IDX2(i, N-1, N)]  = src[IDX2(i, N-1, N)];
            }
        }
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
    float omega;
    data_t **psrc, **pdst;
    pthread_barrier_t *bar;
} temp_args_t;

static void *temp_worker(void *p)
{
    temp_args_t *a = (temp_args_t*) p;
    const int N = a->N, B = a->B, T = a->T, S = B + 2 * T;
    const int tid = a->tid, nt = a->nthreads;
    const float omega = a->omega;

    /* Per-thread scratch pair, two ping-pong buffers. */
    data_t *sa = (data_t*) malloc((size_t)S * S * sizeof(data_t));
    data_t *sb = (data_t*) malloc((size_t)S * S * sizeof(data_t));
    if (!sa || !sb) {
        fprintf(stderr, "thread %d: scratch alloc failed (%zu)\n",
                tid, (size_t)S * S * sizeof(data_t));
        return NULL;
    }

    /* Static tile partition, mirroring OMP's collapse(2) schedule(static). */
    int ntiles_i = (N - 2 + B - 1) / B;
    int ntiles_j = (N - 2 + B - 1) / B;
    int total = ntiles_i * ntiles_j;
    int t_start = ((long)total * tid)       / nt;
    int t_end   = ((long)total * (tid + 1)) / nt;

    /* Boundary-copy partition: each thread covers a row-strip of dst's
       perimeter rows.  Row 0 and row N-1 split equally; ditto for cols. */
    int br_lo = ((long)N * tid)       / nt;
    int br_hi = ((long)N * (tid + 1)) / nt;

    for (int s = 0; s < a->super; s++) {
        const data_t *src = *a->psrc;
        data_t       *dst = *a->pdst;

        /* Copy global boundaries: top/bottom rows and left/right cols. */
        for (int j = br_lo; j < br_hi; j++) {
            dst[IDX2(0, j, N)]   = src[IDX2(0, j, N)];
            dst[IDX2(N-1, j, N)] = src[IDX2(N-1, j, N)];
        }
        for (int i = br_lo; i < br_hi; i++) {
            dst[IDX2(i, 0, N)]   = src[IDX2(i, 0, N)];
            dst[IDX2(i, N-1, N)] = src[IDX2(i, N-1, N)];
        }
        pthread_barrier_wait(a->bar);

        /* My slice of the tile grid. */
        for (int idx = t_start; idx < t_end; idx++) {
            int ti = idx / ntiles_j;
            int tj = idx % ntiles_j;

            int i0 = 1 + ti * B;
            int j0 = 1 + tj * B;
            int bi = (i0 + B <= N - 1) ? B : (N - 1 - i0);
            int bj = (j0 + B <= N - 1) ? B : (N - 1 - j0);
            int gi0 = i0 - T, gj0 = j0 - T;
            int Si = bi + 2 * T, Sj = bj + 2 * T;

            /* Load (Si x Sj) scratch from src with clamp-to-edge. */
            for (int si = 0; si < Si; si++) {
                int gi = gi0 + si;
                if (gi < 0) gi = 0; else if (gi > N - 1) gi = N - 1;
                const data_t *srow = src + (size_t)gi * N;
                data_t *arow = sa + (size_t)si * S;
                for (int sj = 0; sj < Sj; sj++) {
                    int gj = gj0 + sj;
                    if (gj < 0) gj = 0; else if (gj > N - 1) gj = N - 1;
                    arow[sj] = srow[gj];
                }
            }
            memcpy(sb, sa, (size_t)Si * S * sizeof(data_t));

            /* Run T sub-steps; ping-pong between sa and sb. */
            data_t *A = sa, *Bp = sb;
            for (int t = 0; t < T; t++) {
                int lo_i = 1 + t, hi_i = Si - 1 - t;
                int lo_j = 1 + t, hi_j = Sj - 1 - t;

                /* Clip the update range to the global interior. */
                int ulo_i = lo_i; if (gi0 + ulo_i < 1)     ulo_i = 1 - gi0;
                int uhi_i = hi_i; if (gi0 + uhi_i > N - 1) uhi_i = N - 1 - gi0;
                int ulo_j = lo_j; if (gj0 + ulo_j < 1)     ulo_j = 1 - gj0;
                int uhi_j = hi_j; if (gj0 + uhi_j > N - 1) uhi_j = N - 1 - gj0;

                memcpy(Bp, A, (size_t)Si * S * sizeof(data_t));

                for (int si = ulo_i; si < uhi_i; si++) {
                    const data_t *am = A + (size_t)(si-1) * S;
                    const data_t *ac = A + (size_t) si    * S;
                    const data_t *ap = A + (size_t)(si+1) * S;
                    data_t       *bc = Bp + (size_t) si   * S;
                    for (int sj = ulo_j; sj < uhi_j; sj++) {
                        data_t v = ac[sj];
                        data_t nb = 0.25f * (am[sj] + ap[sj] +
                                             ac[sj-1] + ac[sj+1]);
                        bc[sj] = v - omega * (v - nb);
                    }
                }
                data_t *tmp = A; A = Bp; Bp = tmp;
            }

            /* Write central B x B back to dst. */
            for (int si = T; si < T + bi; si++) {
                int gi = gi0 + si;
                memcpy(dst + (size_t)gi * N + (gj0 + T),
                       A   + (size_t)si * S + T,
                       (size_t)bj * sizeof(data_t));
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
static double run_serial(int N, int iters, float omega,
                         data_t *a, data_t *b, data_t **out)
{
    data_t *src = a, *dst = b;
    struct timespec t0, t1;
    clock_gettime(CLOCK_REALTIME, &t0);
    for (int k = 0; k < iters; k++) {
        for (int i = 1; i < N - 1; i++) {
            const data_t *sm = src + IDX2(i-1, 0, N);
            const data_t *sc = src + IDX2(i,   0, N);
            const data_t *sp = src + IDX2(i+1, 0, N);
            data_t       *dc = dst + IDX2(i,   0, N);
            for (int j = 1; j < N - 1; j++) {
                data_t s = sc[j];
                data_t nb = 0.25f * (sm[j] + sp[j] + sc[j-1] + sc[j+1]);
                dc[j] = s - omega * (s - nb);
            }
        }
        for (int j = 0; j < N; j++) {
            dst[IDX2(0, j, N)]   = src[IDX2(0, j, N)];
            dst[IDX2(N-1, j, N)] = src[IDX2(N-1, j, N)];
        }
        for (int i = 0; i < N; i++) {
            dst[IDX2(i, 0, N)]   = src[IDX2(i, 0, N)];
            dst[IDX2(i, N-1, N)] = src[IDX2(i, N-1, N)];
        }
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
            "usage: %s N iters [B=64] [T=4] [--threads NT] [--ppm path]\n",
            argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    int iters = atoi(argv[2]);
    int B = 64, T = 4, nthreads = 4;
    const char *ppm_path = NULL;
    int pos = 0;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            nthreads = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--ppm") && i + 1 < argc) {
            ppm_path = argv[++i];
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

    size_t bytes = (size_t)N * N * sizeof(data_t);
    data_t *base_a = (data_t*) malloc(bytes);
    data_t *base_b = (data_t*) malloc(bytes);
    data_t *temp_a = (data_t*) malloc(bytes);
    data_t *temp_b = (data_t*) malloc(bytes);
    data_t *ref_a  = (data_t*) malloc(bytes);
    data_t *ref_b  = (data_t*) malloc(bytes);
    if (!base_a || !base_b || !temp_a || !temp_b || !ref_a || !ref_b) {
        fprintf(stderr, "alloc failed\n"); return 1;
    }
    init_array(base_a, (long)N*N, 527u);
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
    pthread_t *tids = (pthread_t*) malloc((size_t)nthreads * sizeof(pthread_t));
    base_args_t *bargs =
        (base_args_t*) malloc((size_t)nthreads * sizeof(base_args_t));
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
    temp_args_t *targs =
        (temp_args_t*) malloc((size_t)nthreads * sizeof(temp_args_t));
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
    data_t diff_b = max_diff(ref_out, base_out, (long)N*N);
    data_t diff_t = max_diff(ref_out, temp_out, (long)N*N);
    data_t scale  = max_val(ref_out, (long)N*N);
    data_t rel_b  = (scale > 0.f) ? diff_b / scale : 0.f;
    data_t rel_t  = (scale > 0.f) ? diff_t / scale : 0.f;

    double pts = (double)(N-2) * (double)(N-2) * (double)iters;

    printf("N=%d iters=%d B=%d T=%d threads=%d OMEGA=%.3f\n",
           N, iters, B, T, nthreads, (double)OMEGA);
    printf("  serial   : %9.4f s  (%7.3f Gup/s)\n",
           t_ref, pts / t_ref / 1e9);
    printf("  baseline : %9.4f s  (%7.3f Gup/s)  speedup vs serial %5.2fx\n",
           t_base, pts / t_base / 1e9, t_ref / t_base);
    printf("  temporal : %9.4f s  (%7.3f Gup/s)  speedup vs serial %5.2fx, "
           "vs baseline %5.2fx\n",
           t_temp, pts / t_temp / 1e9, t_ref / t_temp, t_base / t_temp);
    printf("  max|serial-baseline| = %.4e   rel = %.4e\n",
           (double)diff_b, (double)rel_b);
    printf("  max|serial-temporal| = %.4e   rel = %.4e\n",
           (double)diff_t, (double)rel_t);

    /* Two CSV lines for the harness. */
    printf("CSV,sor2d_pth_temporal,2,%d,%d,%d,baseline,strip,%.6e,%.6e,%.4e\n",
           N, iters, nthreads, t_base, pts / t_base / 1e9, (double)diff_b);
    printf("CSV,sor2d_pth_temporal,2,%d,%d,%d,temporal,B=%d;T=%d,%.6e,%.6e,%.4e\n",
           N, iters, nthreads, B, T, t_temp, pts / t_temp / 1e9, (double)diff_t);

    if (ppm_path) {
        if (write_ppm_gray(ppm_path, temp_out, N) == 0)
            printf("  wrote %s\n", ppm_path);
        else
            fprintf(stderr, "failed to write %s\n", ppm_path);
    }

    free(base_a); free(base_b); free(temp_a); free(temp_b);
    free(ref_a); free(ref_b);
    free(tids); free(bargs); free(targs);
    return 0;
}
