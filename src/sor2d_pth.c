/*****************************************************************************

  EC527 Project — 2D SOR, pthreads CPU
  Pthreads baseline.  Threads split the interior rows [1, N-1) into
  contiguous strips; a pthread_barrier_t synchronises between sweeps.
  Mirrors the Lab-5 part-4 "horizontal-strips" decomposition (same ADT,
  same fRand-based init, same suspect-divergence guard) wrapped around the
  Lab-7-style ping-pong stencil so cross-tier comparisons are
  apples-to-apples.

    Build: gcc -O1 -std=gnu11 -pthread sor2d_pth.c -lpthread -lrt -lm -o sor2d_pth
    Run:   ./sor2d_pth <N> <iters> [nthreads=4] [--ppm path]

 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <pthread.h>

#define CPNS    2.0
#define GHOST   2
#define MINVAL  0.0
#define MAXVAL  10.0
#define OMEGA   0.9

typedef double data_t;

typedef struct {
  long int rowlen;
  data_t *data;
} arr_rec, *arr_ptr;

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
arr_ptr new_array(long int row_len)
{
  arr_ptr result = (arr_ptr) malloc(sizeof(arr_rec));
  if (!result) return NULL;
  result->rowlen = row_len;
  if (row_len > 0) {
    data_t *data = (data_t *) calloc(row_len * row_len, sizeof(data_t));
    if (!data) { free(result); return NULL; }
    result->data = data;
  } else {
    result->data = NULL;
  }
  return result;
}

int set_arr_rowlen(arr_ptr v, long int row_len) { v->rowlen = row_len; return 1; }
long int get_arr_rowlen(arr_ptr v)              { return v->rowlen; }
data_t *get_array_start(arr_ptr v)              { return v->data; }

double fRand(double fMin, double fMax)
{
  double f = (double)random() / (double)RAND_MAX;
  return fMin + f * (fMax - fMin);
}

int init_array_rand(arr_ptr v, long int row_len)
{
  srandom(row_len);
  if (row_len <= 0) return 0;
  v->rowlen = row_len;
  for (long int i = 0; i < row_len * row_len; i++)
    v->data[i] = (data_t) fRand((double)MINVAL, (double)MAXVAL);
  return 1;
}

data_t max_diff(const data_t *a, const data_t *b, long len)
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

data_t max_val(const data_t *a, long len)
{
  data_t mx = 0.0;
  for (long i = 0; i < len; i++) {
    data_t v = fabs(a[i]);
    if (v > mx) mx = v;
  }
  return mx;
}

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
  data_t scale = (mx > mn) ? 255.0 / (mx - mn) : 0.0;
  FILE *f = fopen(path, "wb");
  if (!f) return -1;
  fprintf(f, "P5\n%d %d\n255\n", N, N);
  for (long i = 0; i < (long)N*N; i++) {
    int v = (int)((a[i] - mn) * scale);
    if (v < 0) v = 0; else if (v > 255) v = 255;
    unsigned char c = (unsigned char) v;
    fwrite(&c, 1, 1, f);
  }
  fclose(f);
  return 0;
}

/* =========================== Worker thread state =========================== */
typedef struct thread_data {
  int tid;
  int nthreads;
  int N;
  int iters;
  double omega;
  /* Double-pointers so thread 0's post-sweep buffer swap is visible to all. */
  data_t **src;
  data_t **dst;
  pthread_barrier_t *bar;
} thread_data_t;

/* =============================== Worker ================================== */
void *sor_worker(void *arg)
{
  thread_data_t *a = (thread_data_t *) arg;
  const int N   = a->N;
  const int tid = a->tid;
  const int nt  = a->nthreads;
  const double omega = a->omega;

  /* Strip decomposition over interior rows i in [1, N-1) -- Lab-5 part 4
     "horizontal strips" pattern. */
  const int start = 1 + ((N - 2) * tid)       / nt;
  const int end   = 1 + ((N - 2) * (tid + 1)) / nt;

  for (int k = 0; k < a->iters; k++) {
    const data_t *src = *a->src;
    data_t       *dst = *a->dst;

    for (int i = start; i < end; i++) {
      const data_t *sm = src + (size_t)(i-1) * N;
      const data_t *sc = src + (size_t) i    * N;
      const data_t *sp = src + (size_t)(i+1) * N;
      data_t       *dc = dst + (size_t) i    * N;
      for (int j = 1; j < N-1; j++) {
        data_t s = sc[j];
        data_t nb = 0.25 * (sm[j] + sp[j] + sc[j-1] + sc[j+1]);
        dc[j] = s - omega * (s - nb);
      }
      /* Carry through the two boundary columns for this row. */
      dc[0]   = sc[0];
      dc[N-1] = sc[N-1];
    }
    /* Thread 0 carries through the top/bottom boundary rows. */
    if (tid == 0) {
      for (int j = 0; j < N; j++) {
        dst[          j] = src[          j];
        dst[(N-1)*N + j] = src[(N-1)*N + j];
      }
    }

    pthread_barrier_wait(a->bar);
    if (tid == 0) {
      data_t *tmp = *a->src; *a->src = *a->dst; *a->dst = tmp;
    }
    pthread_barrier_wait(a->bar);
  }
  return NULL;
}

/* ============================ Serial reference ============================= */
void sor2d_serial(data_t *src, data_t *dst, int N, int iters, double omega)
{
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
    for (int j = 0; j < N; j++) {
      dst[          j] = src[          j];
      dst[(N-1)*N + j] = src[(N-1)*N + j];
    }
    for (int i = 0; i < N; i++) {
      dst[i*N        ] = src[i*N        ];
      dst[i*N + (N-1)] = src[i*N + (N-1)];
    }
    if (fabs(dst[(N-2)*N + (N-2)]) > 10.0 * (MAXVAL - MINVAL)) {
      fprintf(stderr, "sor2d_serial: SUSPECT DIVERGENCE iter = %d\n", k);
      break;
    }
    data_t *tmp = src; src = dst; dst = tmp;
  }
}

/* =====================================================================
   MAIN
   ===================================================================== */
int main(int argc, char **argv)
{
  struct timespec time_start, time_stop;

  if (argc < 3) {
    fprintf(stderr, "usage: %s N iters [nthreads=4] [--ppm path]\n", argv[0]);
    return 1;
  }
  int N        = atoi(argv[1]);
  int iters    = atoi(argv[2]);
  int nthreads = 4;
  const char *ppm_path = NULL;
  for (int i = 3; i < argc; i++) {
    if (strcmp(argv[i], "--ppm") == 0 && i + 1 < argc) ppm_path = argv[++i];
    else nthreads = atoi(argv[i]);
  }
  double omega = OMEGA;
  if (N < 4 || iters < 1 || nthreads < 1) { fprintf(stderr, "bad args\n"); return 1; }

  arr_ptr v0 = new_array(N);
  init_array_rand(v0, N);
  set_arr_rowlen(v0, N);
  const data_t *init = get_array_start(v0);

  size_t bytes = (size_t)N * N * sizeof(data_t);
  data_t *pth_a = (data_t*) malloc(bytes);
  data_t *pth_b = (data_t*) malloc(bytes);
  data_t *ref_a = (data_t*) malloc(bytes);
  data_t *ref_b = (data_t*) malloc(bytes);
  memcpy(pth_a, init, bytes);
  memcpy(pth_b, init, bytes);
  memcpy(ref_a, init, bytes);
  memcpy(ref_b, init, bytes);

  /* ---- Serial reference ---- */
  clock_gettime(CLOCK_REALTIME, &time_start);
  sor2d_serial(ref_a, ref_b, N, iters, omega);
  clock_gettime(CLOCK_REALTIME, &time_stop);
  double t_ref = interval(time_start, time_stop);
  data_t *ref_result = (iters % 2 == 0) ? ref_a : ref_b;

  /* ---- Pthreads run ---- */
  pthread_barrier_t bar;
  pthread_barrier_init(&bar, NULL, nthreads);
  pthread_t      *threads = (pthread_t*)      malloc((size_t)nthreads * sizeof(pthread_t));
  thread_data_t  *args    = (thread_data_t*)  malloc((size_t)nthreads * sizeof(thread_data_t));
  data_t *cur_src = pth_a, *cur_dst = pth_b;
  for (int t = 0; t < nthreads; t++) {
    args[t].tid = t;       args[t].nthreads = nthreads;
    args[t].N   = N;       args[t].iters    = iters;    args[t].omega = omega;
    args[t].src = &cur_src; args[t].dst     = &cur_dst; args[t].bar   = &bar;
  }

  clock_gettime(CLOCK_REALTIME, &time_start);
  for (int t = 0; t < nthreads; t++) {
    int rc = pthread_create(&threads[t], NULL, sor_worker, &args[t]);
    if (rc) { printf("ERROR; pthread_create rc=%d\n", rc); exit(-1); }
  }
  for (int t = 0; t < nthreads; t++) pthread_join(threads[t], NULL);
  clock_gettime(CLOCK_REALTIME, &time_stop);
  double t_pth = interval(time_start, time_stop);
  pthread_barrier_destroy(&bar);

  /* After every sweep thread 0 swaps (*src, *dst); the last-written buffer
     is cur_src. */
  data_t *pth_result = cur_src;

  /* ---- Validation and reporting ---- */
  data_t diff  = max_diff(ref_result, pth_result, (long)N*N);
  data_t scale = max_val(ref_result, (long)N*N);
  data_t rel   = (scale > 0.0) ? diff / scale : 0.0;

  double pts = (double)(N-2) * (double)(N-2) * (double)iters;
  printf("N=%d iters=%d nthreads=%d OMEGA=%.3f GHOST=%d\n",
         N, iters, nthreads, omega, GHOST);
  printf("  serial  : %9.4f s  (%10.3g cycles, %7.3f CPE)\n",
         t_ref, (double)CPNS * 1.0e9 * t_ref,
         (double)CPNS * 1.0e9 * t_ref / pts);
  printf("  pthread : %9.4f s  (%10.3g cycles, %7.3f CPE)  speedup %5.2fx\n",
         t_pth, (double)CPNS * 1.0e9 * t_pth,
         (double)CPNS * 1.0e9 * t_pth / pts,
         t_ref / t_pth);
  printf("  max|serial-pthread| = %.4e   rel = %.4e\n", diff, rel);

  if (ppm_path) {
    if (write_ppm_gray(ppm_path, pth_result, N) == 0)
      printf("  wrote %s\n", ppm_path);
    else
      fprintf(stderr, "failed to write %s\n", ppm_path);
  }

  free(pth_a); free(pth_b); free(ref_a); free(ref_b);
  free(threads); free(args);
  free(v0->data); free(v0);
  return 0;
}
