/*****************************************************************************

  EC527 Project — 2D red-black Gauss-Seidel SOR
  In-place, classical red-black SOR with convergence-to-tolerance iteration.
  Matches the Lab 5 test_SOR_OMEGA.c story: sweeps ALL cells of one colour
  (updates independent within a colour) then the other, and is stable for
  omega in (0, 2).  Optimal omega for an NxN Laplace problem is

      omega_opt = 2 / (1 + sin(pi / (N-1)))

  which typically lands around 1.8 - 1.95 for the sizes we care about.

  Iteration stops when the mean |change| per interior cell falls below TOL.

  The rest of the project uses a Jacobi-like ping-pong form (read src, write
  dst) because that makes temporal blocking and GPU shared-memory scratch
  trivially correct; that form is a damped Jacobi operator, stable only for
  omega in (0, 1].  This file restores true SOR for the omega-tuning story.

    Build: gcc -O1 -std=gnu11 sor2d_rb.c -lrt -lm -o sor2d_rb
    Run:   ./sor2d_rb <N>                — single run at theoretical omega_opt
           ./sor2d_rb <N> --sweep        — omega in [0.50, 1.99] in 0.02 steps
           ./sor2d_rb <N> --omega <w>    — single run at a specified omega
           ./sor2d_rb <N> [...] --ppm <p>— also write final field as PPM

 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define CPNS    2.0         /* Cycles per nanosecond -- adjust per machine */

#define MINVAL  0.0f
#define MAXVAL  10.0f

#define TOL       1e-5f
#define MAX_ITERS 100000

typedef float data_t;

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
double fRand(double fMin, double fMax)
{
    double f = (double)rand() / (double)RAND_MAX;
    return fMin + f * (fMax - fMin);
}

void init_array(data_t *a, long len, unsigned seed)
{
    srand(seed);
    for (long i = 0; i < len; i++)
        a[i] = (data_t) fRand((double)MINVAL, (double)MAXVAL);
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

/* ===================== One red-black half-sweep ============================ */
/* Update interior cells whose (i + j) % 2 == parity, in place.  Returns the
   sum of |change| to drive the convergence check. */
double half_sweep(data_t *u, int N, float omega, int parity)
{
    double sum_abs = 0.0;
    for (int i = 1; i < N - 1; i++) {
        /* Stagger start column so (i + j) % 2 == parity within the row. */
        int j0 = ((i + parity) & 1) ? 1 : 2;
        for (int j = j0; j < N - 1; j += 2) {
            data_t s = u[i*N + j];
            data_t nb = 0.25f * (u[(i-1)*N + j] + u[(i+1)*N + j] +
                                 u[i*N + j-1]   + u[i*N + j+1]);
            data_t change = s - nb;
            u[i*N + j] = s - omega * change;
            double a = (double) change; if (a < 0) a = -a;
            sum_abs += a;
        }
    }
    return sum_abs;
}

/* =========================== Solver loop =================================== */
/* Iterate until mean |change| per interior cell < TOL, or MAX_ITERS reached.
   Returns iteration count, or -1 if the iteration diverged (non-finite). */
int solve(data_t *u, int N, float omega)
{
    const double inv_interior = 1.0 / ((double)(N-2) * (double)(N-2));
    for (int it = 1; it <= MAX_ITERS; it++) {
        double s  = half_sweep(u, N, omega, 0);
               s += half_sweep(u, N, omega, 1);
        if (!isfinite(s)) return -1;
        if (s * inv_interior < (double) TOL) return it;
    }
    return MAX_ITERS;   /* didn't converge */
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
    float user_omega = -1.0f;
    const char *ppm_path = NULL;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--sweep") == 0) {
            sweep = 1;
        } else if (strcmp(argv[i], "--omega") == 0 && i + 1 < argc) {
            user_omega = (float) atof(argv[++i]);
        } else if (strcmp(argv[i], "--ppm") == 0 && i + 1 < argc) {
            ppm_path = argv[++i];
        } else {
            fprintf(stderr, "unknown arg: %s\n", argv[i]); return 1;
        }
    }

    size_t bytes = (size_t)N * N * sizeof(data_t);
    data_t *u0 = (data_t*) malloc(bytes);
    data_t *u  = (data_t*) malloc(bytes);
    init_array(u0, (long)N*N, 527u);

    const float pi = 3.14159265358979323846f;
    const float omega_opt = 2.0f / (1.0f + sinf(pi / (float)(N-1)));

    if (sweep) {
        printf("# N=%d  omega_opt(theory) = %.4f\n", N, omega_opt);
        printf("# omega, iters, seconds, cycles\n");
        for (float w = 0.50f; w < 2.00f; w += 0.02f) {
            memcpy(u, u0, bytes);
            clock_gettime(CLOCK_REALTIME, &time_start);
            int iters = solve(u, N, w);
            clock_gettime(CLOCK_REALTIME, &time_stop);
            double t = interval(time_start, time_stop);
            printf("%.4f, %d, %.6f, %.3g\n",
                   w, iters, t, (double)CPNS * 1.0e9 * t);
        }
    } else {
        float omega = (user_omega > 0.0f) ? user_omega : omega_opt;
        memcpy(u, u0, bytes);
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

    free(u0); free(u);
    return 0;
}
