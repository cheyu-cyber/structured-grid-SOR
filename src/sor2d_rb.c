/*
 * 2D red-black Gauss-Seidel SOR — in-place, convergence-to-tolerance iteration.
 *
 * The rest of the project uses a Jacobi-like ping-pong form (read src, write
 * dst) because that makes temporal blocking and GPU shared-memory scratch
 * trivially correct. The price is that the ping-pong form is a damped Jacobi
 * operator, stable only for omega in (0, 1] — so it can't actually
 * over-relax and gives up the main benefit of SOR.
 *
 * This file restores classical red-black Gauss-Seidel SOR:
 *   1. Update all "red" cells  (i+j) % 2 == 0  in place
 *   2. Update all "black" cells (i+j) % 2 == 1 in place
 * Within a color sweep all updates are independent (no red cell depends on
 * another red cell), but red uses the neighbors' already-updated black values
 * from the previous step, so the iteration is Gauss-Seidel-order and stable
 * for omega in (0, 2). Optimal omega for an NxN Laplace problem is
 *   omega_opt = 2 / (1 + sin(pi / (N-1)))
 * which typically lands around 1.8 – 1.95 for the sizes we care about.
 *
 * Iteration stops when the mean |change| per interior cell falls below TOL.
 *
 * Usage:
 *   ./sor2d_rb <N>                — single run at the theoretical omega_opt
 *   ./sor2d_rb <N> --sweep        — sweep omega in [0.50, 1.99] in 0.02 steps,
 *                                   print (omega, iters, seconds) table
 *   ./sor2d_rb <N> --omega <w>    — single run at a specified omega
 *   ./sor2d_rb <N> [...] --ppm <p>— also write the final field as PPM
 *
 * Build:  gcc -O3 -march=native -std=c11 sor2d_rb.c -o sor2d_rb -lm
 */

#include "common.h"

#define TOL       1e-5f
#define MAX_ITERS 100000

/* One half-sweep over the interior cells whose (i + j) % 2 == parity, updated
 * in place. Returns the sum of |change| to drive the convergence check. */
static double half_sweep(float *u, int N, float omega, int parity)
{
    double sum_abs = 0.0;
    for (int i = 1; i < N - 1; i++) {
        /* Stagger the start column so (i + j) % 2 == parity within the row. */
        int j0 = ((i + parity) & 1) ? 1 : 2;
        for (int j = j0; j < N - 1; j += 2) {
            float s = u[i*N + j];
            float nb = 0.25f * (u[(i-1)*N + j] + u[(i+1)*N + j] +
                                u[i*N + j-1]   + u[i*N + j+1]);
            float change = s - nb;
            u[i*N + j] = s - omega * change;
            double a = change; if (a < 0) a = -a;
            sum_abs += a;
        }
    }
    return sum_abs;
}

/* Iterate until mean |change| per interior cell < TOL or MAX_ITERS reached.
 * Returns iteration count, or -1 if the iteration diverged (non-finite). */
static int solve(float *u, int N, float omega)
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

int main(int argc, char **argv)
{
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

    size_t bytes = (size_t)N * N * sizeof(float);
    float *u0 = (float*) malloc(bytes);
    float *u  = (float*) malloc(bytes);
    init_array(u0, (long)N*N, 527u);

    const float pi = 3.14159265358979323846f;
    const float omega_opt = 2.0f / (1.0f + sinf(pi / (float)(N-1)));

    if (sweep) {
        printf("# N=%d  omega_opt(theory) = %.4f\n", N, omega_opt);
        printf("# omega, iters, seconds\n");
        for (float w = 0.50f; w < 2.00f; w += 0.02f) {
            memcpy(u, u0, bytes);
            double t0 = wall_seconds();
            int iters = solve(u, N, w);
            double t = wall_seconds() - t0;
            printf("%.4f, %d, %.6f\n", w, iters, t);
        }
    } else {
        float omega = (user_omega > 0.0f) ? user_omega : omega_opt;
        memcpy(u, u0, bytes);
        double t0 = wall_seconds();
        int iters = solve(u, N, omega);
        double t = wall_seconds() - t0;
        printf("N=%d omega=%.4f (omega_opt=%.4f)  iters=%d  time=%.4f s\n",
               N, omega, omega_opt, iters, t);
        if (iters < 0)           printf("  DIVERGED\n");
        else if (iters == MAX_ITERS) printf("  did not converge in %d iters\n", MAX_ITERS);
        if (ppm_path) {
            if (write_ppm_gray(ppm_path, u, N) == 0) printf("  wrote %s\n", ppm_path);
            else fprintf(stderr, "failed to write %s\n", ppm_path);
        }
    }

    free(u0); free(u);
    return 0;
}
