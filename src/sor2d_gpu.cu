/*****************************************************************************

  EC527 Project — 2D SOR, CUDA GPU
  Baseline one-sweep kernel vs. temporally blocked shared-memory kernel.
  Same Jacobi-like ping-pong stencil as the CPU/OpenMP/pthreads tiers, so
  all implementations produce the same values for the same initial data.

  Baseline kernel  : one sweep per launch, ping-pong buffers; matches the
                     form of Lab 7's sor_sweep_3a.
  Temporal kernel  : each block loads a (TILE x TILE) tile into shared mem
                     with HALO_T cells of halo on each side, runs HALO_T
                     sub-steps with the shrinking trapezoid, then writes
                     the central (TILE - 2*HALO_T)^2 cells back.  Every
                     launch advances the grid by HALO_T sweeps, so the host
                     loop runs iters/HALO_T launches.

    Build: nvcc -O2 sor2d_gpu.cu -o sor2d_gpu
    Run:   ./sor2d_gpu <N> <iters> [--ppm path]

  Shared-memory footprint per block: 2 * TILE * TILE * 4 bytes = 8 KB.

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
#define BLOCK_DIM    16    /* 16x16 = 256 threads for the baseline kernel.
                              Independent of the temporal-kernel TILE. */
#define TILE_DEFAULT 32    /* defaults preserve previous behaviour */
#define HALO_DEFAULT  4
#define TILE_MAX     32    /* per-axis limit: TILE^2 <= 1024 thread budget */

#define MINVAL   0.0f
#define MAXVAL  10.0f
#define OMEGA    0.9f                   /* Damped Jacobi form: stable for
                                           omega in (0, 1] only. */

typedef float data_t;

/* ---------- Host helpers (Lab 7 style) ----------------------------------- */
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

/* Minimal grayscale PPM (P5) writer. */
int write_ppm_gray(const char *path, const data_t *a, int N)
{
    float mn =  INFINITY, mx = -INFINITY;
    for (long i = 0; i < (long)N*N; i++) {
        float v = a[i];
        if (!isfinite(v)) continue;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    if (!isfinite(mn) || mn > mx) return -1;
    float scale = (mx > mn) ? 255.0f / (mx - mn) : 0.0f;
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

/* Host reference: identical stencil to the CPU tier's baseline. */
void cpu_sor(data_t *src, data_t *dst, int N, int iters, float omega)
{
    for (int k = 0; k < iters; k++) {
        for (int i = 1; i < N-1; i++) {
            for (int j = 1; j < N-1; j++) {
                data_t s = src[i*N+j];
                data_t nb = 0.25f * (src[(i-1)*N+j] + src[(i+1)*N+j] +
                                     src[i*N+j-1]   + src[i*N+j+1]);
                dst[i*N+j] = s - omega * (s - nb);
            }
        }
        for (int j = 0; j < N; j++) {
            dst[0*N+j]     = src[0*N+j];
            dst[(N-1)*N+j] = src[(N-1)*N+j];
        }
        for (int i = 0; i < N; i++) {
            dst[i*N+0]       = src[i*N+0];
            dst[i*N+(N-1)]   = src[i*N+(N-1)];
        }
        data_t *tmp = src; src = dst; dst = tmp;
    }
}

/* ==========================================================================
   Baseline GPU kernel: one SOR sweep per launch (Lab 7 sor_sweep_3a form).
   Reads from d_old, writes to d_new -- host swaps the buffers between
   launches.  Implicit kernel-launch serialisation on the same stream gives
   a safe RAW boundary between sweeps.
   ========================================================================== */
__global__ void sor_sweep(const data_t *d_old, data_t *d_new,
                          int N, float omega)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= N-1 || j >= N-1) return;

    data_t s = d_old[i*N+j];
    data_t nb = 0.25f * (d_old[(i-1)*N+j] + d_old[(i+1)*N+j] +
                         d_old[i*N+j-1]   + d_old[i*N+j+1]);
    d_new[i*N+j] = s - omega * (s - nb);
}

/* Pass-through of boundary rows/cols so the sweep kernel can stay branch-
   free on interior threads. */
__global__ void copy_boundary(const data_t *d_old, data_t *d_new, int N)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < N) {
        d_new[0*N+t]     = d_old[0*N+t];
        d_new[(N-1)*N+t] = d_old[(N-1)*N+t];
        d_new[t*N+0]     = d_old[t*N+0];
        d_new[t*N+(N-1)] = d_old[t*N+(N-1)];
    }
}

/* ==========================================================================
   Temporal GPU kernel: HALO_T sweeps per launch with shared-memory scratch.
   Each block loads a TILE x TILE region (INTER x INTER interior plus HALO_T
   halo on each side) into shared memory, runs HALO_T sub-steps with the
   shrinking trapezoid, and writes the central INTER x INTER cells back.
   Two shared-memory buffers (sa, sb) for the in-block ping-pong;
   __syncthreads() between sub-steps.
   ========================================================================== */
__global__ void sor_temporal(const data_t *d_old, data_t *d_new,
                             int N, float omega, int tile, int halo)
{
    /* Dynamic shared memory: two ping-pong buffers laid out back-to-back.
       Caller passes 2 * tile * tile * sizeof(data_t) bytes via the
       launch's shared-mem size argument. */
    extern __shared__ data_t sshared[];
    data_t *sa = sshared;
    data_t *sb = sshared + (size_t)tile * tile;

    int lx = threadIdx.x;
    int ly = threadIdx.y;
    int inter = tile - 2 * halo;

    /* Each block produces an `inter x inter` interior region starting
       at (1 + bx*inter, 1 + by*inter) in global coords. */
    int bi = blockIdx.y * inter + 1;
    int bj = blockIdx.x * inter + 1;
    int gi = bi + (ly - halo);
    int gj = bj + (lx - halo);

    /* Clamp-to-edge load. */
    int cgi = gi; if (cgi < 0) cgi = 0; else if (cgi > N-1) cgi = N-1;
    int cgj = gj; if (cgj < 0) cgj = 0; else if (cgj > N-1) cgj = N-1;
    data_t v = d_old[cgi * N + cgj];
    sa[ly * tile + lx] = v;
    sb[ly * tile + lx] = v;
    __syncthreads();

    /* `halo` sub-steps; valid update region shrinks by 1 on each side per
       sub-step.  Cells outside the trapezoid carry through from the
       previous buffer. */
    int cur = 0;
    for (int t = 0; t < halo; t++) {
        int lo = 1 + t;
        int hi = tile - 1 - t;
        bool in_trap   = (lx >= lo && lx < hi && ly >= lo && ly < hi);
        bool in_domain = (gi >= 1 && gi <= N-2 && gj >= 1 && gj <= N-2);
        data_t out;
        if (in_trap && in_domain) {
            data_t s, l, r, u, d;
            data_t *cur_buf = (cur == 0) ? sa : sb;
            s = cur_buf[ly * tile + lx];
            l = cur_buf[ly * tile + (lx-1)];
            r = cur_buf[ly * tile + (lx+1)];
            u = cur_buf[(ly-1) * tile + lx];
            d = cur_buf[(ly+1) * tile + lx];
            data_t nb = 0.25f * (l + r + u + d);
            out = s - omega * (s - nb);
        } else {
            out = (cur == 0) ? sa[ly * tile + lx] : sb[ly * tile + lx];
        }
        __syncthreads();
        if (cur == 0) sb[ly * tile + lx] = out;
        else          sa[ly * tile + lx] = out;
        __syncthreads();
        cur = 1 - cur;
    }

    /* Write the central `inter x inter` region back. */
    bool write_slot = (lx >= halo && lx < tile - halo &&
                       ly >= halo && ly < tile - halo);
    bool in_domain  = (gi >= 1 && gi <= N-2 && gj >= 1 && gj <= N-2);
    if (write_slot && in_domain) {
        data_t v_out = (cur == 0) ? sa[ly * tile + lx]
                                  : sb[ly * tile + lx];
        d_new[gi * N + gj] = v_out;
    }
}

/* =====================================================================
   MAIN
   ===================================================================== */
int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,
            "usage: %s N iters [--tile T=%d] [--halo H=%d] [--ppm path]\n",
            argv[0], TILE_DEFAULT, HALO_DEFAULT);
        return 1;
    }
    int N = atoi(argv[1]);
    int iters = atoi(argv[2]);
    int tile = TILE_DEFAULT;
    int halo = HALO_DEFAULT;
    float omega = OMEGA;
    const char *ppm_path = NULL;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--tile") && i + 1 < argc) {
            tile = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--halo") && i + 1 < argc) {
            halo = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--ppm") && i + 1 < argc) {
            ppm_path = argv[++i];
        } else {
            fprintf(stderr, "unknown arg: %s\n", argv[i]); return 1;
        }
    }

    if (tile < 4 || tile > TILE_MAX) {
        fprintf(stderr, "tile (%d) out of range [4, %d]\n", tile, TILE_MAX);
        return 1;
    }
    if (halo < 1 || 2 * halo >= tile) {
        fprintf(stderr, "halo (%d) must satisfy 1 <= halo and 2*halo < tile (%d)\n",
                halo, tile);
        return 1;
    }
    int inter = tile - 2 * halo;
    if (iters % halo != 0) {
        fprintf(stderr, "iters (%d) must be a multiple of halo (%d)\n",
                iters, halo);
        return 1;
    }
    int super = iters / halo;

    CUDA_SAFE_CALL(cudaSetDevice(0));
    /* Warm up. */
    { float *d; CUDA_SAFE_CALL(cudaMalloc(&d, 4)); cudaFree(d); }

    size_t bytes = (size_t)N * N * sizeof(data_t);
    data_t *h_init = (data_t*) malloc(bytes);
    data_t *h_cpu  = (data_t*) malloc(bytes);
    data_t *h_base = (data_t*) malloc(bytes);
    data_t *h_temp = (data_t*) malloc(bytes);

    initArray(h_init, (long)N*N, 527u);
    memcpy(h_cpu, h_init, bytes);

    /* ---------------- Baseline GPU: one sweep per launch, ping-pong ------ */
    data_t *d_base[2];
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_base[0], bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_base[1], bytes));
    CUDA_SAFE_CALL(cudaMemcpy(d_base[0], h_init, bytes, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_base[1], h_init, bytes, cudaMemcpyHostToDevice));

    dim3 bblk(BLOCK_DIM, BLOCK_DIM);
    dim3 bgrd((N-2 + BLOCK_DIM-1)/BLOCK_DIM, (N-2 + BLOCK_DIM-1)/BLOCK_DIM);
    int  cblk = 128;
    int  cgrd = (N + cblk - 1) / cblk;

    cudaEvent_t t_start, t_stop;
    cudaEventCreate(&t_start); cudaEventCreate(&t_stop);
    cudaEventRecord(t_start, 0);
    int cur = 0;
    for (int k = 0; k < iters; k++) {
        sor_sweep<<<bgrd, bblk>>>(d_base[cur], d_base[1-cur], N, omega);
        copy_boundary<<<cgrd, cblk>>>(d_base[cur], d_base[1-cur], N);
        cur = 1 - cur;
    }
    cudaEventRecord(t_stop, 0);
    cudaEventSynchronize(t_stop);
    float base_ms;
    cudaEventElapsedTime(&base_ms, t_start, t_stop);
    CUDA_SAFE_CALL(cudaMemcpy(h_base, d_base[cur], bytes, cudaMemcpyDeviceToHost));

    /* ---------------- Temporal GPU: HALO_T sweeps per launch ------------- */
    data_t *d_temp[2];
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_temp[0], bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_temp[1], bytes));
    CUDA_SAFE_CALL(cudaMemcpy(d_temp[0], h_init, bytes, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_temp[1], h_init, bytes, cudaMemcpyHostToDevice));

    dim3 tblk(tile, tile);
    dim3 tgrd((N - 2 + inter - 1) / inter, (N - 2 + inter - 1) / inter);
    size_t shmem_bytes = (size_t)2 * tile * tile * sizeof(data_t);

    cudaEventRecord(t_start, 0);
    int tcur = 0;
    for (int s = 0; s < super; s++) {
        sor_temporal<<<tgrd, tblk, shmem_bytes>>>(
            d_temp[tcur], d_temp[1-tcur], N, omega, tile, halo);
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
    cpu_sor(h_cpu, h_cpu_alt, N, iters, omega);
    double cpu_ms = (wall_seconds() - c0) * 1000.0;
    data_t *cpu_result = (iters % 2 == 0) ? h_cpu : h_cpu_alt;

    /* ---------------- Validation ---------------------------------------- */
    float diff_base = max_diff(h_base, cpu_result, (long)N*N);
    float diff_temp = max_diff(h_temp, cpu_result, (long)N*N);
    float diff_gpu  = max_diff(h_base, h_temp, (long)N*N);
    float scale     = max_val(cpu_result, (long)N*N);
    float rel_base  = (scale > 0) ? diff_base / scale : 0.0f;
    float rel_temp  = (scale > 0) ? diff_temp / scale : 0.0f;

    double pts = (double)(N-2) * (double)(N-2) * (double)iters;
    /* CPE (lab convention) at the V100/L40S clock proxy CPNS=2.0 cycles/ns:
       cycles = ms * CPNS * 1e6, CPE = cycles / pts. */
    const double CPNS = 2.0;
    double base_cycles = (double)base_ms * CPNS * 1.0e6;
    double temp_cycles = (double)temp_ms * CPNS * 1.0e6;
    double base_cpe    = base_cycles / pts;
    double temp_cpe    = temp_cycles / pts;

    printf("N=%d  iters=%d  TILE=%d  HALO=%d  INTER=%d  super=%d\n",
           N, iters, tile, halo, inter, super);
    printf("  GPU baseline  : %8.3f ms  (%10.3g cycles, %7.4f CPE)\n",
           base_ms, base_cycles, base_cpe);
    printf("  GPU temporal  : %8.3f ms  (%10.3g cycles, %7.4f CPE)  "
           "speedup vs base %5.2fx\n",
           temp_ms, temp_cycles, temp_cpe, base_ms / temp_ms);
    printf("  CPU reference : %8.3f ms\n", cpu_ms);
    printf("  max|base-cpu|  = %.4e  (rel %.2e)\n", diff_base, rel_base);
    printf("  max|temp-cpu|  = %.4e  (rel %.2e)\n", diff_temp, rel_temp);
    printf("  max|base-temp| = %.4e\n", diff_gpu);

    /* Two CSV lines for the harness; the temporal extra encodes (tile, halo)
       so plot.py can render the (TILE, HALO) heatmap. */
    printf("CSV,sor2d_gpu,2,%d,%d,0,baseline,TILE=%d;HALO=%d,%.6e,%.6e,%.4e\n",
           N, iters, tile, halo, base_ms / 1000.0, base_cpe, diff_base);
    printf("CSV,sor2d_gpu,2,%d,%d,0,temporal,TILE=%d;HALO=%d,%.6e,%.6e,%.4e\n",
           N, iters, tile, halo, temp_ms / 1000.0, temp_cpe, diff_temp);

    if (ppm_path) {
        if (write_ppm_gray(ppm_path, h_temp, N) == 0)
            printf("  wrote %s\n", ppm_path);
        else
            fprintf(stderr, "failed to write %s\n", ppm_path);
    }

    cudaEventDestroy(t_start); cudaEventDestroy(t_stop);
    cudaFree(d_base[0]); cudaFree(d_base[1]);
    cudaFree(d_temp[0]); cudaFree(d_temp[1]);
    free(h_init); free(h_cpu); free(h_cpu_alt);
    free(h_base); free(h_temp);
    return 0;
}
