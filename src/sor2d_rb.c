/*****************************************************************************

  EC527 Project — 2D red-black Gauss-Seidel SOR (omega tuning)
  Classical red-black GS-SOR, in-place, convergence-to-tolerance.  Same
  pattern as Lab 5's test_SOR.c::SOR_redblack and test_SOR_OMEGA.c.
  Updates all "red" cells (i+j even) in place, then all "black" cells; the
  iteration is Gauss-Seidel-order and stable for omega in (0, 2).

  Optimal omega for an NxN Laplace problem is
     omega_opt = 2 / (1 + sin(pi / (N-1)))
  which lands around 1.8 - 1.95 for the sizes we care about.

  Iteration stops when the mean |change| per interior cell falls below TOL.

    Build: gcc -O1 -std=gnu11 sor2d_rb.c -lrt -lm -o sor2d_rb
    Run:
      ./sor2d_rb <N>                — single run at theoretical omega_opt
      ./sor2d_rb <N> --sweep        — omega in [0.50, 1.99] in 0.02 steps
      ./sor2d_rb <N> --omega <w>    — single run at a specified omega
      ./sor2d_rb <N> [...] --ppm <p>— also write final field as PPM

 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define CPNS    2.0
#define GHOST   2
#define MINVAL  0.0
#define MAXVAL  10.0

#define TOL       1.0e-5
#define MAX_ITERS 100000

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

/* ===================== One red-black half-sweep ============================ */
/* Update interior cells whose (i + j) % 2 == parity, in place.  Returns the
   sum of |change| to drive the convergence check. */
double half_sweep(data_t *u, int N, double omega, int parity)
{
  double sum_abs = 0.0;
  for (int i = 1; i < N - 1; i++) {
    /* Stagger start column so (i + j) % 2 == parity within the row. */
    int j0 = ((i + parity) & 1) ? 1 : 2;
    for (int j = j0; j < N - 1; j += 2) {
      data_t s = u[i*N + j];
      data_t nb = 0.25 * (u[(i-1)*N + j] + u[(i+1)*N + j] +
                          u[ i   *N + j-1] + u[ i   *N + j+1]);
      data_t change = s - nb;
      u[i*N + j] = s - omega * change;
      double ac = (double) change; if (ac < 0) ac = -ac;
      sum_abs += ac;
    }
  }
  return sum_abs;
}

/* =========================== Solver loop =================================== */
/* Iterate until mean |change| per interior cell < TOL, or MAX_ITERS reached.
   Returns iteration count, or -1 if the iteration diverged. */
int solve(data_t *u, int N, double omega)
{
  const double inv_interior = 1.0 / ((double)(N-2) * (double)(N-2));
  for (int it = 1; it <= MAX_ITERS; it++) {
    double s  = half_sweep(u, N, omega, 0);
           s += half_sweep(u, N, omega, 1);
    if (!isfinite(s)) return -1;
    /* Lab-5 suspect-divergence sentinel. */
    if (fabs(u[(N-2)*N + (N-2)]) > 10.0 * (MAXVAL - MINVAL)) return -1;
    if (s * inv_interior < (double) TOL) return it;
  }
  return MAX_ITERS;
}

/* =====================================================================
   MAIN
   ===================================================================== */
int main(int argc, char **argv)
{
  struct timespec time_start, time_stop;

  if (argc < 2) {
    fprintf(stderr,
            "usage: %s N [--sweep] [--omega w] [--ppm path]\n", argv[0]);
    return 1;
  }
  int N = atoi(argv[1]);
  if (N < 4) { fprintf(stderr, "N too small\n"); return 1; }

  int sweep = 0;
  double user_omega = -1.0;
  const char *ppm_path = NULL;

  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--sweep") == 0) {
      sweep = 1;
    } else if (strcmp(argv[i], "--omega") == 0 && i + 1 < argc) {
      user_omega = atof(argv[++i]);
    } else if (strcmp(argv[i], "--ppm") == 0 && i + 1 < argc) {
      ppm_path = argv[++i];
    } else {
      fprintf(stderr, "unknown arg: %s\n", argv[i]); return 1;
    }
  }

  arr_ptr v0 = new_array(N);
  init_array_rand(v0, N);
  set_arr_rowlen(v0, N);
  const data_t *init = get_array_start(v0);

  size_t bytes = (size_t)N * N * sizeof(data_t);
  data_t *u = (data_t*) malloc(bytes);

  const double pi = 3.14159265358979323846;
  const double omega_opt = 2.0 / (1.0 + sin(pi / (double)(N-1)));

  if (sweep) {
    printf("# N=%d  omega_opt(theory) = %.4f\n", N, omega_opt);
    printf("# omega, iters, seconds, cycles\n");
    for (double w = 0.50; w < 2.00; w += 0.02) {
      memcpy(u, init, bytes);
      clock_gettime(CLOCK_REALTIME, &time_start);
      int iters = solve(u, N, w);
      clock_gettime(CLOCK_REALTIME, &time_stop);
      double t = interval(time_start, time_stop);
      printf("%.4f, %d, %.6f, %.3g\n",
             w, iters, t, (double)CPNS * 1.0e9 * t);
    }
  } else {
    double omega = (user_omega > 0.0) ? user_omega : omega_opt;
    memcpy(u, init, bytes);
    clock_gettime(CLOCK_REALTIME, &time_start);
    int iters = solve(u, N, omega);
    clock_gettime(CLOCK_REALTIME, &time_stop);
    double t = interval(time_start, time_stop);
    printf("N=%d OMEGA=%.4f (omega_opt=%.4f)  iters=%d  time=%.4f s  (%.3g cycles)\n",
           N, omega, omega_opt, iters, t, (double)CPNS * 1.0e9 * t);
    if (iters < 0)               printf("  DIVERGED\n");
    else if (iters == MAX_ITERS) printf("  did not converge in %d iters\n", MAX_ITERS);
    if (ppm_path) {
      if (write_ppm_gray(ppm_path, u, N) == 0) printf("  wrote %s\n", ppm_path);
      else fprintf(stderr, "failed to write %s\n", ppm_path);
    }
  }

  free(u);
  free(v0->data); free(v0);
  return 0;
}
