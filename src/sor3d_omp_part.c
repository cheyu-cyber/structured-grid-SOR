/*****************************************************************************

  EC527 Project — 3D SOR, OpenMP partitioning study
  Three parallel decomposition modes for the same 7-point Laplacian sweep,
  selectable at runtime.  Validates each mode against an in-binary
  single-threaded reference.

  Modes (--mode):
    slab    `#pragma omp parallel for` over outer i.  Each thread owns a
            contiguous (Ni/nt) x N x N slab.  Largest cache footprint per
            thread, simplest dispatch.  Halo-per-chunk = 2 N^2 cells.
    pencil  `#pragma omp parallel for collapse(2)` over (i, j).  Each
            thread owns chunks of size (1 row of jk-pencils) x N.  Halo
            grows along k only; halo-per-chunk approx 4 N h cells.
    cube    Tile the interior into B^3 cubes, parallelise over the tile
            grid with `collapse(3)`.  Each tile fits in cache, but halo-
            per-tile is 6 B^2 cells — the worst surface-to-volume ratio.

  These are baseline (one sweep per `iters`) versions.  The temporal
  shrinking-trapezoid kernel already lives in sor3d_omp.c; the question
  here is purely: how does the parallel decomposition pattern affect a
  plain 3D sweep at large N?

  Build: gcc -O1 -std=gnu11 -fopenmp sor3d_omp_part.c -lrt -lm \
             -o sor3d_omp_part
  Run:   OMP_NUM_THREADS=16 ./sor3d_omp_part <N> <iters>
              [--mode slab|pencil|cube] [--block B=32]

 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define MINVAL  0.0
#define MAXVAL  10.0
#define OMEGA   0.9

typedef double data_t;

#define IDX3(i,j,k,N) (((size_t)(i)*(N) + (j))*(N) + (k))

static const data_t INV6 = 1.0 / 6.0;

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

/* ============================ Common boundary copy ========================= */
static void carry_boundaries_3d(int N, const data_t *src, data_t *dst)
{
    #pragma omp parallel for schedule(static)
    for (int a = 0; a < N; a++) {
        for (int b = 0; b < N; b++) {
            dst[IDX3(0,a,b,N)]     = src[IDX3(0,a,b,N)];
            dst[IDX3(N-1,a,b,N)]   = src[IDX3(N-1,a,b,N)];
            dst[IDX3(a,0,b,N)]     = src[IDX3(a,0,b,N)];
            dst[IDX3(a,N-1,b,N)]   = src[IDX3(a,N-1,b,N)];
            dst[IDX3(a,b,0,N)]     = src[IDX3(a,b,0,N)];
            dst[IDX3(a,b,N-1,N)]   = src[IDX3(a,b,N-1,N)];
        }
    }
}

/* ============================== Slab partition ============================= */
/* Parallel over outer i only.  Each thread does a contiguous slab of i,
 * full j and k.  Inner loops have unit-stride k accesses, full row reuse. */
static void sweep_slab(const data_t *src, data_t *dst, int N, double omega)
{
    #pragma omp parallel for schedule(static)
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
}

/* ============================ Pencil partition ============================= */
/* `collapse(2)` over (i, j).  Each thread gets pencils of unit-stride k. */
static void sweep_pencil(const data_t *src, data_t *dst, int N, double omega)
{
    #pragma omp parallel for collapse(2) schedule(static)
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
}

/* ============================== Cube partition ============================= */
/* Tile the interior into B^3 blocks, parallelise across tiles.  Each tile
 * is a per-iteration unit; threads pull tiles dynamically.  This is the
 * "max cache locality, max halo overhead" extreme. */
static void sweep_cube(const data_t *src, data_t *dst,
                       int N, int B, double omega)
{
    int nti = (N - 2 + B - 1) / B;
    int ntj = (N - 2 + B - 1) / B;
    int ntk = (N - 2 + B - 1) / B;

    #pragma omp parallel for collapse(3) schedule(static)
    for (int ti = 0; ti < nti; ti++) {
        for (int tj = 0; tj < ntj; tj++) {
            for (int tk = 0; tk < ntk; tk++) {
                int i0 = 1 + ti * B;
                int j0 = 1 + tj * B;
                int k0 = 1 + tk * B;
                int i1 = (i0 + B <= N - 1) ? i0 + B : N - 1;
                int j1 = (j0 + B <= N - 1) ? j0 + B : N - 1;
                int k1 = (k0 + B <= N - 1) ? k0 + B : N - 1;

                for (int i = i0; i < i1; i++) {
                    for (int j = j0; j < j1; j++) {
                        const data_t *sm_i = src + IDX3(i-1, j, 0, N);
                        const data_t *sc_i = src + IDX3(i,   j, 0, N);
                        const data_t *sp_i = src + IDX3(i+1, j, 0, N);
                        const data_t *sm_j = src + IDX3(i, j-1, 0, N);
                        const data_t *sp_j = src + IDX3(i, j+1, 0, N);
                        data_t       *dc   = dst + IDX3(i,   j, 0, N);
                        for (int k = k0; k < k1; k++) {
                            data_t s = sc_i[k];
                            data_t nb = INV6 * (sm_i[k] + sp_i[k] +
                                                sm_j[k] + sp_j[k] +
                                                sc_i[k-1] + sc_i[k+1]);
                            dc[k] = s - omega * (s - nb);
                        }
                    }
                }
            }
        }
    }
}

/* ============================ Driver ====================================== */
typedef enum { PART_SLAB, PART_PENCIL, PART_CUBE } part_t;

static const char *part_name(part_t p)
{
    return p == PART_SLAB   ? "slab"
         : p == PART_PENCIL ? "pencil"
         :                    "cube";
}

static double run_partitioned(part_t mode, int N, int iters, int B,
                              double omega, data_t *a, data_t *b,
                              data_t **out)
{
    data_t *src = a, *dst = b;
    struct timespec t0, t1;
    clock_gettime(CLOCK_REALTIME, &t0);
    for (int it = 0; it < iters; it++) {
        switch (mode) {
        case PART_SLAB:   sweep_slab  (src, dst, N,    omega); break;
        case PART_PENCIL: sweep_pencil(src, dst, N,    omega); break;
        case PART_CUBE:   sweep_cube  (src, dst, N, B, omega); break;
        }
        carry_boundaries_3d(N, src, dst);
        data_t *tmp = src; src = dst; dst = tmp;
    }
    clock_gettime(CLOCK_REALTIME, &t1);
    *out = src;
    return interval(t0, t1);
}

/* Single-threaded reference: same kernel as slab, OMP simply won't fire
 * with one thread.  Build with -fopenmp anyway so the reference and the
 * threaded run share the same compiler-generated stencil code. */
static double run_serial_ref(int N, int iters, double omega,
                             data_t *a, data_t *b, data_t **out)
{
    int saved = omp_get_max_threads();
    omp_set_num_threads(1);
    double t = run_partitioned(PART_SLAB, N, iters, 0, omega, a, b, out);
    omp_set_num_threads(saved);
    return t;
}

/* ===================================================================
   MAIN
   =================================================================== */
int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,
            "usage: %s N iters [--mode slab|pencil|cube] [--block B=32]\n",
            argv[0]);
        return 1;
    }
    int N     = atoi(argv[1]);
    int iters = atoi(argv[2]);
    int B     = 32;
    part_t mode = PART_SLAB;

    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--mode") && i + 1 < argc) {
            const char *s = argv[++i];
            if      (!strcmp(s, "slab"))   mode = PART_SLAB;
            else if (!strcmp(s, "pencil")) mode = PART_PENCIL;
            else if (!strcmp(s, "cube"))   mode = PART_CUBE;
            else { fprintf(stderr, "bad --mode: %s\n", s); return 1; }
        } else if (!strcmp(argv[i], "--block") && i + 1 < argc) {
            B = atoi(argv[++i]);
        } else {
            fprintf(stderr, "unknown arg: %s\n", argv[i]); return 1;
        }
    }
    if (N < 4 || iters < 1 || B < 1) {
        fprintf(stderr, "bad args\n"); return 1;
    }

    int nthreads = omp_get_max_threads();
    printf("OMP_NUM_THREADS=%d\n", nthreads);

    size_t bytes = (size_t)N * N * N * sizeof(data_t);
    data_t *omp_a = (data_t*) malloc(bytes);
    data_t *omp_b = (data_t*) malloc(bytes);
    data_t *ref_a = (data_t*) malloc(bytes);
    data_t *ref_b = (data_t*) malloc(bytes);
    if (!omp_a || !omp_b || !ref_a || !ref_b) {
        fprintf(stderr, "alloc failed (%zu bytes)\n", bytes); return 1;
    }

    init_array(omp_a, (long)N*N*N, 527u);
    memcpy(omp_b, omp_a, bytes);
    memcpy(ref_a, omp_a, bytes);
    memcpy(ref_b, omp_a, bytes);

    /* ---- Serial reference ---- */
    data_t *ref_out = NULL;
    double t_ref = run_serial_ref(N, iters, OMEGA, ref_a, ref_b, &ref_out);

    /* ---- Threaded run ---- */
    data_t *omp_out = NULL;
    double t_omp = run_partitioned(mode, N, iters, B, OMEGA,
                                   omp_a, omp_b, &omp_out);

    /* ---- Validation ---- */
    data_t diff  = max_diff(ref_out, omp_out, (long)N*N*N);
    data_t scale = max_val(ref_out, (long)N*N*N);
    data_t rel   = (scale > 0.0) ? diff / scale : 0.0;

    double pts  = (double)(N-2) * (double)(N-2) * (double)(N-2) * (double)iters;
    /* CPE (lab convention): wall-clock cycles per output cell at CPNS=2.0. */
    const double CPNS = 2.0;
    double cpe_ref = CPNS * 1.0e9 * t_ref / pts;
    double cpe_omp = CPNS * 1.0e9 * t_omp / pts;

    printf("N=%d iters=%d threads=%d mode=%s",
           N, iters, nthreads, part_name(mode));
    if (mode == PART_CUBE) printf(" B=%d", B);
    printf(" OMEGA=%.3f\n", (double)OMEGA);
    printf("  serial(1t): %9.4f s  (%10.3g cycles, %7.3f CPE)\n",
           t_ref, CPNS * 1.0e9 * t_ref, cpe_ref);
    printf("  threaded  : %9.4f s  (%10.3g cycles, %7.3f CPE)  speedup %5.2fx\n",
           t_omp, CPNS * 1.0e9 * t_omp, cpe_omp, t_ref / t_omp);
    printf("  max|serial-threaded| = %.4e   rel = %.4e\n",
           (double)diff, (double)rel);

    /* Machine-readable. */
    printf("CSV,sor3d_omp_part,3,%d,%d,%d,%s,B=%d,%.6e,%.6e,%.4e\n",
           N, iters, nthreads, part_name(mode), B,
           t_omp, cpe_omp, (double)diff);

    free(omp_a); free(omp_b); free(ref_a); free(ref_b);
    return 0;
}
