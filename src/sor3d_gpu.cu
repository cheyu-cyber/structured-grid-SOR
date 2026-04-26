/*****************************************************************************

  EC527 Project — 3D SOR, CUDA GPU
  Mirror of sor2d_gpu.cu, one dimension up.  Same Jacobi-like ping-pong
  stencil (7-point Laplacian) as the CPU/OpenMP/pthreads tiers, so all
  implementations produce the same values for the same initial data.

  Baseline kernel  : one sweep per launch, ping-pong buffers.  Each thread
                     produces one interior cell.
  Temporal kernel  : each block loads a TILE^3 region (with HALO_T halo on
                     each side) into shared memory, runs HALO_T sub-steps
                     with the shrinking trapezoid, and writes the central
                     INTER^3 cells back.  Two shared-memory buffers (sa,
                     sb) for the in-block ping-pong; __syncthreads()
                     between sub-steps.

  Why TILE=8, HALO_T=2:

    Block size is bounded by 1024 threads on V100, so the temporal block
    can be at most 10x10x10.  TILE=10 with HALO_T=2 gives INTER=6 cells
    per side (216 useful out of 1000 threads, 21.6%).  TILE=8 with
    HALO_T=2 gives INTER=4 (64 / 512 = 12.5%) but cleaner power-of-two
    addressing and fits the 1024-thread budget with headroom.  We use
    TILE=8 here as the simplest demonstration of the scheme; the 2.5D
    streaming variant (Micikevicius 2009) is the textbook cure for the
    halo-overhead problem in 3D and is left as future work.

    Shared-memory footprint per block: 2 * 8 * 8 * 8 * 4 bytes = 4 KB.

    Build: nvcc -O2 sor3d_gpu.cu -o sor3d_gpu
    Run:   ./sor3d_gpu <N> <iters>

 *****************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>

/* ---------- Error checking (Lab 7 style) --------------------------------- */
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA_SAFE_CALL: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/* ---------- Parameters --------------------------------------------------- */
#define BBLK   8        /* baseline thread block per axis (8x8x8 = 512) */
#define TILE   8        /* temporal: thread block per axis */
#define HALO_T 2        /* temporal sub-steps amortised per launch */
#define INTER  (TILE - 2*HALO_T)  /* output cells per block side = 4 */

#define MINVAL  0.0f
#define MAXVAL 10.0f
#define OMEGA   0.9f                 /* Damped Jacobi: stable for omega in (0,1] */

typedef float data_t;

#define IDX3(i,j,k,N) (((size_t)(i)*(N) + (j))*(N) + (k))
#define INV6 (1.0f / 6.0f)

/* ---------- Host helpers (mirror sor2d_gpu.cu) --------------------------- */
double wall_seconds()
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

double fRand(double fMin, double fMax)
{
    double f = (double)rand() / (double)RAND_MAX;
    return fMin + f * (fMax - fMin);
}

void initArray(data_t *arr, long len, unsigned seed)
{
    srand(seed);
    for (long i = 0; i < len; i++)
        arr[i] = (data_t) fRand((double)MINVAL, (double)MAXVAL);
}

float max_diff(const data_t *a, const data_t *b, long len)
{
    float mx = 0.0f;
    int bad = 0;
    for (long i = 0; i < len; i++) {
        data_t av = a[i], bv = b[i];
        if (!isfinite(av) || !isfinite(bv)) { bad++; continue; }
        float d = fabsf(av - bv);
        if (d > mx) mx = d;
    }
    if (bad) {
        fprintf(stderr, "WARNING: %d non-finite values in max_diff\n", bad);
        return INFINITY;
    }
    return mx;
}

float max_val(const data_t *a, long len)
{
    float mx = 0.0f;
    for (long i = 0; i < len; i++) {
        float v = fabsf(a[i]);
        if (v > mx) mx = v;
    }
    return mx;
}

/* Host reference: same 7-point stencil as sor3d_cpu.c. */
void cpu_sor3d(data_t *src, data_t *dst, int N, int iters, float omega)
{
    for (int it = 0; it < iters; it++) {
        for (int i = 1; i < N-1; i++) {
            for (int j = 1; j < N-1; j++) {
                for (int k = 1; k < N-1; k++) {
                    data_t s  = src[IDX3(i,j,k,N)];
                    data_t nb = INV6 * (src[IDX3(i-1,j,k,N)] + src[IDX3(i+1,j,k,N)] +
                                        src[IDX3(i,j-1,k,N)] + src[IDX3(i,j+1,k,N)] +
                                        src[IDX3(i,j,k-1,N)] + src[IDX3(i,j,k+1,N)]);
                    dst[IDX3(i,j,k,N)] = s - omega * (s - nb);
                }
            }
        }
        for (int a = 0; a < N; a++) {
            for (int b = 0; b < N; b++) {
                dst[IDX3(0,a,b,N)]   = src[IDX3(0,a,b,N)];
                dst[IDX3(N-1,a,b,N)] = src[IDX3(N-1,a,b,N)];
                dst[IDX3(a,0,b,N)]   = src[IDX3(a,0,b,N)];
                dst[IDX3(a,N-1,b,N)] = src[IDX3(a,N-1,b,N)];
                dst[IDX3(a,b,0,N)]   = src[IDX3(a,b,0,N)];
                dst[IDX3(a,b,N-1,N)] = src[IDX3(a,b,N-1,N)];
            }
        }
        data_t *tmp = src; src = dst; dst = tmp;
    }
}

/* ==========================================================================
   Baseline GPU kernel: one 7-point sweep per launch.
   Mapping: threadIdx.x -> k (unit-stride), threadIdx.y -> j,
            threadIdx.z -> i.  Each thread owns one interior cell.
   ========================================================================== */
__global__ void sor_sweep_3d(const data_t *d_old, data_t *d_new,
                             int N, float omega)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int i = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (i >= N-1 || j >= N-1 || k >= N-1) return;

    data_t s  = d_old[IDX3(i,j,k,N)];
    data_t nb = INV6 * (d_old[IDX3(i-1,j,k,N)] + d_old[IDX3(i+1,j,k,N)] +
                        d_old[IDX3(i,j-1,k,N)] + d_old[IDX3(i,j+1,k,N)] +
                        d_old[IDX3(i,j,k-1,N)] + d_old[IDX3(i,j,k+1,N)]);
    d_new[IDX3(i,j,k,N)] = s - omega * (s - nb);
}

/* Pass-through of all six boundary faces. */
__global__ void copy_boundary_3d(const data_t *d_old, data_t *d_new, int N)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= N || b >= N) return;
    d_new[IDX3(0,a,b,N)]     = d_old[IDX3(0,a,b,N)];
    d_new[IDX3(N-1,a,b,N)]   = d_old[IDX3(N-1,a,b,N)];
    d_new[IDX3(a,0,b,N)]     = d_old[IDX3(a,0,b,N)];
    d_new[IDX3(a,N-1,b,N)]   = d_old[IDX3(a,N-1,b,N)];
    d_new[IDX3(a,b,0,N)]     = d_old[IDX3(a,b,0,N)];
    d_new[IDX3(a,b,N-1,N)]   = d_old[IDX3(a,b,N-1,N)];
}

/* ==========================================================================
   Temporal GPU kernel: HALO_T sweeps per launch with shared-memory scratch.
   Each block produces an INTER^3 interior region starting at
   (1 + bz*INTER, 1 + by*INTER, 1 + bx*INTER) in (i,j,k).  Threads load a
   TILE^3 region with HALO_T halo on each side, run HALO_T sub-steps with
   the shrinking trapezoid, and write the central INTER^3 cells back.
   Two shared-memory buffers; __syncthreads() between sub-steps.
   ========================================================================== */
__global__ void sor_temporal_3d(const data_t *d_old, data_t *d_new,
                                int N, float omega)
{
    __shared__ data_t sa[TILE][TILE][TILE];
    __shared__ data_t sb[TILE][TILE][TILE];

    int lz = threadIdx.z;
    int ly = threadIdx.y;
    int lx = threadIdx.x;

    int bi = blockIdx.z * INTER + 1;
    int bj = blockIdx.y * INTER + 1;
    int bk = blockIdx.x * INTER + 1;
    int gi = bi + (lz - HALO_T);
    int gj = bj + (ly - HALO_T);
    int gk = bk + (lx - HALO_T);

    /* Clamp-to-edge load. */
    int cgi = gi; if (cgi < 0) cgi = 0; else if (cgi > N-1) cgi = N-1;
    int cgj = gj; if (cgj < 0) cgj = 0; else if (cgj > N-1) cgj = N-1;
    int cgk = gk; if (cgk < 0) cgk = 0; else if (cgk > N-1) cgk = N-1;
    data_t v = d_old[IDX3(cgi, cgj, cgk, N)];
    sa[lz][ly][lx] = v;
    sb[lz][ly][lx] = v;
    __syncthreads();

    /* HALO_T sub-steps; valid update region shrinks by 1 on each side per
       sub-step.  Cells outside the trapezoid carry through their previous
       value so the central INTER^3 is exactly HALO_T-step-advanced. */
    int cur = 0;
    #pragma unroll
    for (int t = 0; t < HALO_T; t++) {
        int lo = 1 + t;
        int hi = TILE - 1 - t;
        bool in_trap   = (lx >= lo && lx < hi &&
                          ly >= lo && ly < hi &&
                          lz >= lo && lz < hi);
        bool in_domain = (gi >= 1 && gi <= N-2 &&
                          gj >= 1 && gj <= N-2 &&
                          gk >= 1 && gk <= N-2);
        data_t out;
        if (in_trap && in_domain) {
            data_t s, xm, xp, ym, yp, zm, zp;
            if (cur == 0) {
                s  = sa[lz][ly][lx];
                xm = sa[lz][ly][lx-1]; xp = sa[lz][ly][lx+1];
                ym = sa[lz][ly-1][lx]; yp = sa[lz][ly+1][lx];
                zm = sa[lz-1][ly][lx]; zp = sa[lz+1][ly][lx];
            } else {
                s  = sb[lz][ly][lx];
                xm = sb[lz][ly][lx-1]; xp = sb[lz][ly][lx+1];
                ym = sb[lz][ly-1][lx]; yp = sb[lz][ly+1][lx];
                zm = sb[lz-1][ly][lx]; zp = sb[lz+1][ly][lx];
            }
            data_t nb = INV6 * (xm + xp + ym + yp + zm + zp);
            out = s - omega * (s - nb);
        } else {
            out = (cur == 0) ? sa[lz][ly][lx] : sb[lz][ly][lx];
        }
        __syncthreads();
        if (cur == 0) sb[lz][ly][lx] = out;
        else          sa[lz][ly][lx] = out;
        __syncthreads();
        cur = 1 - cur;
    }

    /* Write the central INTER^3 region back. */
    bool write_slot = (lx >= HALO_T && lx < TILE - HALO_T &&
                       ly >= HALO_T && ly < TILE - HALO_T &&
                       lz >= HALO_T && lz < TILE - HALO_T);
    bool in_domain  = (gi >= 1 && gi <= N-2 &&
                       gj >= 1 && gj <= N-2 &&
                       gk >= 1 && gk <= N-2);
    if (write_slot && in_domain) {
        data_t v_out = (cur == 0) ? sa[lz][ly][lx] : sb[lz][ly][lx];
        d_new[IDX3(gi, gj, gk, N)] = v_out;
    }
}

/* =====================================================================
   MAIN
   ===================================================================== */
int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s N iters\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    int iters = atoi(argv[2]);
    float omega = OMEGA;

    if (N < 4 || iters < 1) { fprintf(stderr, "bad args\n"); return 1; }
    if (iters % HALO_T != 0) {
        fprintf(stderr, "iters (%d) must be a multiple of HALO_T (%d)\n",
                iters, HALO_T);
        return 1;
    }
    int super = iters / HALO_T;

    CUDA_SAFE_CALL(cudaSetDevice(0));
    /* Warm up. */
    { float *d; CUDA_SAFE_CALL(cudaMalloc(&d, 4)); cudaFree(d); }

    size_t bytes = (size_t)N * N * N * sizeof(data_t);
    data_t *h_init = (data_t*) malloc(bytes);
    data_t *h_cpu  = (data_t*) malloc(bytes);
    data_t *h_base = (data_t*) malloc(bytes);
    data_t *h_temp = (data_t*) malloc(bytes);
    if (!h_init || !h_cpu || !h_base || !h_temp) {
        fprintf(stderr, "host alloc failed (%zu bytes each)\n", bytes);
        return 1;
    }

    initArray(h_init, (long)N*N*N, 527u);
    memcpy(h_cpu, h_init, bytes);

    /* ---------------- Baseline GPU --------------------------------------- */
    data_t *d_base[2];
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_base[0], bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_base[1], bytes));
    CUDA_SAFE_CALL(cudaMemcpy(d_base[0], h_init, bytes, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_base[1], h_init, bytes, cudaMemcpyHostToDevice));

    dim3 bblk(BBLK, BBLK, BBLK);
    dim3 bgrd((N-2 + BBLK - 1) / BBLK,
              (N-2 + BBLK - 1) / BBLK,
              (N-2 + BBLK - 1) / BBLK);
    dim3 cblk(16, 16);
    dim3 cgrd((N + 15) / 16, (N + 15) / 16);

    cudaEvent_t t_start, t_stop;
    cudaEventCreate(&t_start); cudaEventCreate(&t_stop);
    cudaEventRecord(t_start, 0);
    int cur = 0;
    for (int k = 0; k < iters; k++) {
        sor_sweep_3d  <<<bgrd, bblk>>>(d_base[cur], d_base[1-cur], N, omega);
        copy_boundary_3d<<<cgrd, cblk>>>(d_base[cur], d_base[1-cur], N);
        cur = 1 - cur;
    }
    cudaEventRecord(t_stop, 0);
    cudaEventSynchronize(t_stop);
    float base_ms;
    cudaEventElapsedTime(&base_ms, t_start, t_stop);
    CUDA_SAFE_CALL(cudaMemcpy(h_base, d_base[cur], bytes, cudaMemcpyDeviceToHost));

    /* ---------------- Temporal GPU --------------------------------------- */
    data_t *d_temp[2];
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_temp[0], bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_temp[1], bytes));
    CUDA_SAFE_CALL(cudaMemcpy(d_temp[0], h_init, bytes, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_temp[1], h_init, bytes, cudaMemcpyHostToDevice));

    dim3 tblk(TILE, TILE, TILE);
    dim3 tgrd((N - 2 + INTER - 1) / INTER,
              (N - 2 + INTER - 1) / INTER,
              (N - 2 + INTER - 1) / INTER);

    cudaEventRecord(t_start, 0);
    int tcur = 0;
    for (int s = 0; s < super; s++) {
        sor_temporal_3d<<<tgrd, tblk>>>(d_temp[tcur], d_temp[1-tcur], N, omega);
        tcur = 1 - tcur;
    }
    cudaEventRecord(t_stop, 0);
    cudaEventSynchronize(t_stop);
    float temp_ms;
    cudaEventElapsedTime(&temp_ms, t_start, t_stop);
    CUDA_SAFE_CALL(cudaMemcpy(h_temp, d_temp[tcur], bytes, cudaMemcpyDeviceToHost));

    /* ---------------- CPU reference -------------------------------------- */
    data_t *h_cpu_alt = (data_t*) malloc(bytes);
    memcpy(h_cpu_alt, h_init, bytes);
    double c0 = wall_seconds();
    cpu_sor3d(h_cpu, h_cpu_alt, N, iters, omega);
    double cpu_ms = (wall_seconds() - c0) * 1000.0;
    data_t *cpu_result = (iters % 2 == 0) ? h_cpu : h_cpu_alt;

    /* ---------------- Validation ---------------------------------------- */
    float diff_base = max_diff(h_base, cpu_result, (long)N*N*N);
    float diff_temp = max_diff(h_temp, cpu_result, (long)N*N*N);
    float diff_gpu  = max_diff(h_base, h_temp, (long)N*N*N);
    float scale     = max_val(cpu_result, (long)N*N*N);
    float rel_base  = (scale > 0) ? diff_base / scale : 0.0f;
    float rel_temp  = (scale > 0) ? diff_temp / scale : 0.0f;

    double pts = (double)(N-2) * (double)(N-2) * (double)(N-2) * (double)iters;
    /* CPE (lab convention) at CPNS=2.0 cycles/ns; cycles = ms * CPNS * 1e6. */
    const double CPNS = 2.0;
    double base_cycles = (double)base_ms * CPNS * 1.0e6;
    double temp_cycles = (double)temp_ms * CPNS * 1.0e6;
    double base_cpe    = base_cycles / pts;
    double temp_cpe    = temp_cycles / pts;

    printf("N=%d  iters=%d  TILE=%d  HALO_T=%d  INTER=%d  super=%d\n",
           N, iters, TILE, HALO_T, INTER, super);
    printf("  GPU baseline  : %8.3f ms  (%10.3g cycles, %7.4f CPE)\n",
           base_ms, base_cycles, base_cpe);
    printf("  GPU temporal  : %8.3f ms  (%10.3g cycles, %7.4f CPE)  "
           "speedup vs base %5.2fx\n",
           temp_ms, temp_cycles, temp_cpe, base_ms / temp_ms);
    printf("  CPU reference : %8.3f ms\n", cpu_ms);
    printf("  max|base-cpu|  = %.4e  (rel %.2e)\n", diff_base, rel_base);
    printf("  max|temp-cpu|  = %.4e  (rel %.2e)\n", diff_temp, rel_temp);
    printf("  max|base-temp| = %.4e\n", diff_gpu);

    /* Machine-readable: two CSV lines (one per kind) for the sweep harness. */
    printf("CSV,sor3d_gpu,3,%d,%d,0,baseline,TILE=%d;HALO=%d,%.6e,%.6e,%.4e\n",
           N, iters, TILE, HALO_T, base_ms / 1000.0, base_cpe, diff_base);
    printf("CSV,sor3d_gpu,3,%d,%d,0,temporal,TILE=%d;HALO=%d,%.6e,%.6e,%.4e\n",
           N, iters, TILE, HALO_T, temp_ms / 1000.0, temp_cpe, diff_temp);

    cudaEventDestroy(t_start); cudaEventDestroy(t_stop);
    cudaFree(d_base[0]); cudaFree(d_base[1]);
    cudaFree(d_temp[0]); cudaFree(d_temp[1]);
    free(h_init); free(h_cpu); free(h_cpu_alt);
    free(h_base); free(h_temp);
    return 0;
}
