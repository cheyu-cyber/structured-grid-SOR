/*
 * 2D SOR on GPU — baseline one-sweep kernel vs. temporally blocked shared-
 * memory kernel. Same Jacobi-like ping-pong stencil as everywhere else in
 * this project, so the three implementations (CPU, OMP, GPU) all produce the
 * same values for the same initial data.
 *
 * Baseline kernel  : one sweep per launch, ping-pong buffers. Matches the
 *                    form of Lab 7's sor_sweep_3a.
 * Temporal kernel  : each block loads a (TILE x TILE) tile into shared mem
 *                    with HALO_T cells of halo on each side, runs HALO_T
 *                    sub-steps with the shrinking trapezoid, writes the
 *                    central (TILE - 2*HALO_T)^2 cells back. Every launch
 *                    advances the grid by HALO_T sweeps, so the host loop
 *                    runs iters/HALO_T launches.
 *
 * Usage:  ./sor2d_gpu <N> <iters>
 * Build:  nvcc -O3 sor2d_gpu.cu -o sor2d_gpu
 *
 * TILE and HALO_T are compile-time constants. Default:
 *   TILE=32, HALO_T=4  ->  interior = 24 output cells per block per launch.
 * Shared-memory footprint per block: 2 * TILE * TILE * 4 bytes = 8 KB.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>

#define TILE    32
#define HALO_T   4
#define INTER   (TILE - 2*HALO_T)     /* output cells per block side */

/* See common.h for the omega stability discussion. The ping-pong form here
 * is a damped Jacobi, stable only for omega in (0, 1]. */
#define OMEGA_DEFAULT 0.9f

#define CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

static double wall_seconds(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static void init_array(float *a, long len, unsigned seed)
{
    srand(seed);
    for (long i = 0; i < len; i++)
        a[i] = (float)rand() / (float)RAND_MAX * 10.0f;
}

static float max_abs_diff(const float *a, const float *b, long len)
{
    float mx = 0.0f;
    int bad = 0;
    for (long i = 0; i < len; i++) {
        float av = a[i], bv = b[i];
        if (!isfinite(av) || !isfinite(bv)) { bad++; continue; }
        float d = fabsf(av - bv);
        if (d > mx) mx = d;
    }
    if (bad) {
        fprintf(stderr, "WARNING: %d non-finite values in max_abs_diff\n", bad);
        return INFINITY;
    }
    return mx;
}

static float max_abs(const float *a, long len)
{
    float mx = 0.0f;
    for (long i = 0; i < len; i++) {
        float v = fabsf(a[i]);
        if (v > mx) mx = v;
    }
    return mx;
}

/* Host reference: identical stencil to Lab 7 / sor2d_cpu.c baseline. */
static void cpu_sor(float *src, float *dst, int N, int iters, float omega)
{
    for (int k = 0; k < iters; k++) {
        for (int i = 1; i < N-1; i++) {
            for (int j = 1; j < N-1; j++) {
                float s = src[i*N+j];
                float nb = 0.25f * (src[(i-1)*N+j] + src[(i+1)*N+j] +
                                    src[i*N+j-1] + src[i*N+j+1]);
                dst[i*N+j] = s - omega * (s - nb);
            }
        }
        for (int j = 0; j < N; j++) {
            dst[0*N+j] = src[0*N+j];
            dst[(N-1)*N+j] = src[(N-1)*N+j];
        }
        for (int i = 0; i < N; i++) {
            dst[i*N+0] = src[i*N+0];
            dst[i*N+(N-1)] = src[i*N+(N-1)];
        }
        float *tmp = src; src = dst; dst = tmp;
    }
}

/* -------------------- Baseline GPU kernel (one sweep) -------------------- */
__global__ void sor_sweep(const float *src, float *dst, int N, float omega)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= N-1 || j >= N-1) return;

    float s = src[i*N+j];
    float nb = 0.25f * (src[(i-1)*N+j] + src[(i+1)*N+j] +
                        src[i*N+j-1] + src[i*N+j+1]);
    dst[i*N+j] = s - omega * (s - nb);
}

/* Pass-through of boundary rows/cols (separate kernel so the sweep kernel
 * stays branch-free on interior threads). */
__global__ void copy_boundary(const float *src, float *dst, int N)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < N) {
        dst[0*N+t]       = src[0*N+t];
        dst[(N-1)*N+t]   = src[(N-1)*N+t];
        dst[t*N+0]       = src[t*N+0];
        dst[t*N+(N-1)]   = src[t*N+(N-1)];
    }
}

/* -------------------- Temporal GPU kernel (HALO_T sweeps) -------------------- */
__global__ void sor_temporal(const float *src, float *dst, int N, float omega)
{
    /* Two shared-memory buffers for the in-block ping-pong. */
    __shared__ float sa[TILE][TILE];
    __shared__ float sb[TILE][TILE];

    int lx = threadIdx.x;
    int ly = threadIdx.y;

    /* Each block produces an INTERxINTER interior region starting at
     * (1 + bx*INTER, 1 + by*INTER) in global coords. Local (lx, ly) maps to
     * gx = (1 + bx*INTER) + (lx - HALO_T)  — same for y. */
    int bi = blockIdx.y * INTER + 1;
    int bj = blockIdx.x * INTER + 1;
    int gi = bi + (ly - HALO_T);
    int gj = bj + (lx - HALO_T);

    /* Clamp-to-edge load. */
    int cgi = gi; if (cgi < 0) cgi = 0; else if (cgi > N-1) cgi = N-1;
    int cgj = gj; if (cgj < 0) cgj = 0; else if (cgj > N-1) cgj = N-1;
    float v = src[cgi * N + cgj];
    sa[ly][lx] = v;
    sb[ly][lx] = v;
    __syncthreads();

    /* HALO_T sub-steps; valid update region shrinks by 1 on each side each
     * sub-step. Cells outside the update region carry through via the copy
     * of the "other" buffer from the previous sub-step (the sb[ly][lx]=v
     * init above handles t=0, subsequent sub-steps preserve untouched slots
     * because we write *either* an update or a carry-through into every
     * slot). */
    int cur = 0;
    #pragma unroll
    for (int t = 0; t < HALO_T; t++) {
        int lo = 1 + t;
        int hi = TILE - 1 - t;
        bool in_trap   = (lx >= lo && lx < hi && ly >= lo && ly < hi);
        bool in_domain = (gi >= 1 && gi <= N-2 && gj >= 1 && gj <= N-2);
        float out;
        if (in_trap && in_domain) {
            float s, l, r, u, d;
            if (cur == 0) {
                s = sa[ly][lx];
                l = sa[ly][lx-1]; r = sa[ly][lx+1];
                u = sa[ly-1][lx]; d = sa[ly+1][lx];
            } else {
                s = sb[ly][lx];
                l = sb[ly][lx-1]; r = sb[ly][lx+1];
                u = sb[ly-1][lx]; d = sb[ly+1][lx];
            }
            float nb = 0.25f * (l + r + u + d);
            out = s - omega * (s - nb);
        } else {
            out = (cur == 0) ? sa[ly][lx] : sb[ly][lx];
        }
        __syncthreads();
        if (cur == 0) sb[ly][lx] = out;
        else          sa[ly][lx] = out;
        __syncthreads();
        cur = 1 - cur;
    }

    /* Write the central INTERxINTER region back. After HALO_T swaps, cur
     * points to the buffer that now holds the advanced state. */
    bool write_slot = (lx >= HALO_T && lx < TILE - HALO_T &&
                       ly >= HALO_T && ly < TILE - HALO_T);
    bool in_domain  = (gi >= 1 && gi <= N-2 && gj >= 1 && gj <= N-2);
    if (write_slot && in_domain) {
        float v_out = (cur == 0) ? sa[ly][lx] : sb[ly][lx];
        dst[gi * N + gj] = v_out;
    }
}

/* -------------------- Driver -------------------- */
int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s N iters\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    int iters = atoi(argv[2]);
    float omega = OMEGA_DEFAULT;

    if (iters % HALO_T != 0) {
        fprintf(stderr, "iters (%d) must be a multiple of HALO_T (%d)\n", iters, HALO_T);
        return 1;
    }
    int super = iters / HALO_T;

    size_t bytes = (size_t)N * N * sizeof(float);
    float *h_init = (float*) malloc(bytes);
    float *h_cpu  = (float*) malloc(bytes);
    float *h_base = (float*) malloc(bytes);
    float *h_temp = (float*) malloc(bytes);

    init_array(h_init, (long)N*N, 527u);
    memcpy(h_cpu, h_init, bytes);

    /* Warm up the GPU. */
    { float *d; CUDA_CHECK(cudaMalloc(&d, 4)); cudaFree(d); }

    /* ---------------- Baseline GPU: one sweep per launch, ping-pong -------- */
    float *d_base[2];
    CUDA_CHECK(cudaMalloc(&d_base[0], bytes));
    CUDA_CHECK(cudaMalloc(&d_base[1], bytes));
    CUDA_CHECK(cudaMemcpy(d_base[0], h_init, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_base[1], h_init, bytes, cudaMemcpyHostToDevice));

    dim3 bblk(16, 16);
    dim3 bgrd((N-2 + 15)/16, (N-2 + 15)/16);
    int  cblk = 128;
    int  cgrd = (N + cblk - 1) / cblk;

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0, 0);
    int cur = 0;
    for (int k = 0; k < iters; k++) {
        sor_sweep<<<bgrd, bblk>>>(d_base[cur], d_base[1-cur], N, omega);
        copy_boundary<<<cgrd, cblk>>>(d_base[cur], d_base[1-cur], N);
        cur = 1 - cur;
    }
    cudaEventRecord(e1, 0);
    cudaEventSynchronize(e1);
    float base_ms;
    cudaEventElapsedTime(&base_ms, e0, e1);
    CUDA_CHECK(cudaMemcpy(h_base, d_base[cur], bytes, cudaMemcpyDeviceToHost));

    /* ---------------- Temporal GPU: HALO_T sweeps per launch --------------- */
    float *d_temp[2];
    CUDA_CHECK(cudaMalloc(&d_temp[0], bytes));
    CUDA_CHECK(cudaMalloc(&d_temp[1], bytes));
    CUDA_CHECK(cudaMemcpy(d_temp[0], h_init, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_temp[1], h_init, bytes, cudaMemcpyHostToDevice));

    dim3 tblk(TILE, TILE);
    dim3 tgrd((N - 2 + INTER - 1) / INTER, (N - 2 + INTER - 1) / INTER);

    cudaEventRecord(e0, 0);
    int tcur = 0;
    for (int s = 0; s < super; s++) {
        sor_temporal<<<tgrd, tblk>>>(d_temp[tcur], d_temp[1-tcur], N, omega);
        tcur = 1 - tcur;
    }
    cudaEventRecord(e1, 0);
    cudaEventSynchronize(e1);
    float temp_ms;
    cudaEventElapsedTime(&temp_ms, e0, e1);
    CUDA_CHECK(cudaMemcpy(h_temp, d_temp[tcur], bytes, cudaMemcpyDeviceToHost));

    /* ---------------- CPU reference ---------------------------------------- */
    float *h_cpu_alt = (float*) malloc(bytes);
    memcpy(h_cpu_alt, h_init, bytes);
    double c0 = wall_seconds();
    cpu_sor(h_cpu, h_cpu_alt, N, iters, omega);
    double cpu_ms = (wall_seconds() - c0) * 1000.0;
    float *cpu_result = (iters % 2 == 0) ? h_cpu : h_cpu_alt;

    /* ---------------- Validation ------------------------------------------- */
    float diff_base = max_abs_diff(h_base, cpu_result, (long)N*N);
    float diff_temp = max_abs_diff(h_temp, cpu_result, (long)N*N);
    float diff_gpu  = max_abs_diff(h_base, h_temp, (long)N*N);
    float scale     = max_abs(cpu_result, (long)N*N);
    float rel_base = (scale > 0) ? diff_base / scale : 0.0f;
    float rel_temp = (scale > 0) ? diff_temp / scale : 0.0f;

    double pts = (double)(N-2) * (double)(N-2) * (double)iters;
    printf("N=%d  iters=%d  TILE=%d  HALO_T=%d  INTER=%d  super=%d\n",
           N, iters, TILE, HALO_T, INTER, super);
    printf("  GPU baseline  : %8.3f ms  (%8.2f Mupdates/s)\n",
           base_ms, pts / (base_ms/1000.0) / 1e6);
    printf("  GPU temporal  : %8.3f ms  (%8.2f Mupdates/s)  speedup vs base %5.2fx\n",
           temp_ms, pts / (temp_ms/1000.0) / 1e6, base_ms / temp_ms);
    printf("  CPU reference : %8.3f ms\n", cpu_ms);
    printf("  max|base-cpu|  = %.4e  (rel %.2e)\n", diff_base, rel_base);
    printf("  max|temp-cpu|  = %.4e  (rel %.2e)\n", diff_temp, rel_temp);
    printf("  max|base-temp| = %.4e\n", diff_gpu);

    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFree(d_base[0]); cudaFree(d_base[1]);
    cudaFree(d_temp[0]); cudaFree(d_temp[1]);
    free(h_init); free(h_cpu); free(h_cpu_alt);
    free(h_base); free(h_temp);
    return 0;
}
