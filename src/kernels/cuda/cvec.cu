// src/kernels/cuda/cvec.cu - see include/strata/kernels/cvec.hpp.
#include "strata/kernels/cvec.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;
constexpr int MAXK = 16;   // n_embd up to 4096, held in registers between the dot and the update

Cvec g_cvec;                 // the description; its device pointers are the uploading device's
bool g_on_host = false;
// the tables on every device that holds them (a layer split applies the vector on several)
constexpr int kDevices = 64;
struct DevTables { float* dir = nullptr; float* s = nullptr; int* on = nullptr; };
DevTables g_dev[kDevices];
std::vector<float> g_dir_host, g_s_host;
// STRATA_CB_DIR: the last row of R after each steered layer, staged in pinned memory (capturable) per request
constexpr int64_t kCbLayers = 128, kCbWidth = 16384;
float* g_cb_host = nullptr;
int cur_device() {
    int d = 0;
    if (cudaGetDevice(&d) != cudaSuccess || d < 0 || d >= kDevices) d = 0;
    return d;
}
bool upload_here(std::string& err) {
    DevTables& t = g_dev[cur_device()];
    if (t.dir != nullptr) return true;
    const int flag = g_on_host ? 1 : 0;
    if (cudaMalloc(&t.dir, g_dir_host.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&t.s, g_s_host.size() * sizeof(float)) != cudaSuccess || cudaMalloc(&t.on, sizeof(int)) != cudaSuccess ||
        cudaMemcpy(t.dir, g_dir_host.data(), g_dir_host.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(t.s, g_s_host.data(), g_s_host.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(t.on, &flag, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        err = "control vector: device allocation failed";
        t = DevTables{};
        return false;
    }
    return true;
}

// the fused hyper-connection read's gate (fused_gr.cu), so a write done here is bitwise the one it would have folded
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }

// one block per (stream, token): the pending write, then h . v over the stream, then the update
__global__ void cvec_kernel(float* __restrict__ R, const float* __restrict__ dir, const float* __restrict__ s_l,
                            const int* __restrict__ on, int mode, int64_t layer, int n, int hc, int64_t r_ld,
                            const float* __restrict__ bo, int64_t bo_ld, const float* __restrict__ inj,
                            int64_t inj_ld, int write) {
    const int c = blockIdx.x;
    const int64_t t = blockIdx.y;
    float* r = R + t * r_ld + (int64_t) c * n;
    const float s = s_l[layer];
    const bool steer = *on != 0 && s != 0.0f;   // uniform over the block
    if (!steer && !write) return;
    const float* v = dir + layer * n;
    const float w = write ? 2.0f * sigmoidf_(inj[t * inj_ld + c] / (float) hc) : 0.0f;
    const float* b = write ? bo + t * bo_ld : nullptr;
    float x[MAXK];
    float dot = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
        const int d = threadIdx.x + k * THREADS;
        if (d < n) {
            float xv = r[d];
            if (write) xv = fmaf(b[d], w, xv);
            x[k] = xv;
            if (steer && mode == 0) dot = fmaf(xv, v[d], dot);
        }
    }
    if (steer && mode == 0) {
        __shared__ float part[THREADS / 32];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
        if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = dot;
        __syncthreads();
        if (threadIdx.x < 32) {
            float p = threadIdx.x < THREADS / 32 ? part[threadIdx.x] : 0.0f;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) p += __shfl_xor_sync(0xffffffffu, p, o);
            if (threadIdx.x == 0) part[0] = p;
        }
        __syncthreads();
        dot = part[0] * s;   // s (h . v)
    }
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
        const int d = threadIdx.x + k * THREADS;
        if (d < n) {
            float xv = x[k];
            if (steer) xv = mode == 0 ? fmaf(-dot, v[d], xv) : xv + v[d];
            r[d] = xv;
        }
    }
}

__global__ void cvec_delta_kernel(float* __restrict__ bo, const float* __restrict__ dir, const float* __restrict__ s_l,
                                  const int* __restrict__ on, int64_t layer, int n, int64_t ld) {
    const float s = s_l[layer];
    if (*on == 0 || s == 0.0f) return;
    float* x = bo + (int64_t) blockIdx.x * ld;
    const float* v = dir + layer * n;
    float xv[MAXK];
    float dot = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
        const int d = threadIdx.x + k * THREADS;
        xv[k] = d < n ? x[d] : 0.0f;
        if (d < n) dot = fmaf(xv[k], v[d], dot);
    }
    __shared__ float part[THREADS / 32];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = dot;
    __syncthreads();
    if (threadIdx.x < 32) {
        float p = threadIdx.x < THREADS / 32 ? part[threadIdx.x] : 0.0f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) p += __shfl_xor_sync(0xffffffffu, p, o);
        if (threadIdx.x == 0) part[0] = p;
    }
    __syncthreads();
    dot = part[0] * s;
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
        const int d = threadIdx.x + k * THREADS;
        if (d < n) x[d] = fmaf(-dot, v[d], xv[k]);
    }
}

}  // namespace

const Cvec& cvec() { return g_cvec; }

bool cvec_upload(const std::vector<float>& dir, const std::vector<float>& s, int mode, int first, int last,
                 int64_t n_embd, int64_t hc, std::string& err) {
    if (n_embd < 1 || n_embd > (int64_t) THREADS * MAXK) { err = "control vector: unsupported n_embd"; return false; }
    if (s.empty() || dir.size() != s.size() * (size_t) n_embd) { err = "control vector: bad table sizes"; return false; }
    int prev = 0;   // a new vector replaces the old one on every device
    cudaGetDevice(&prev);
    for (int d = 0; d < kDevices; ++d) {
        if (g_dev[d].dir == nullptr) continue;
        cudaSetDevice(d);
        cudaDeviceSynchronize();
        cudaFree(g_dev[d].dir);
        cudaFree(g_dev[d].s);
        cudaFree(g_dev[d].on);
        g_dev[d] = DevTables{};
    }
    cudaSetDevice(prev);
    g_dir_host = dir;
    g_s_host = s;
    g_on_host = true;
    if (!upload_here(err)) return false;
    const DevTables& t = g_dev[cur_device()];
    g_cvec.dir = t.dir;
    g_cvec.s = t.s;
    g_cvec.on = t.on;
    g_cvec.mode = mode;
    g_cvec.first = first;
    g_cvec.last = last;
    g_cvec.n_embd = n_embd;
    g_cvec.hc = hc;
    g_cvec.steered.assign(s.size(), false);
    for (size_t l = 0; l < s.size(); ++l) g_cvec.steered[l] = s[l] != 0.0f;
    if (const char* cb = std::getenv("STRATA_CB_DIR"); cb != nullptr && *cb != '\0' && g_cb_host == nullptr)
        cudaHostAlloc((void**) &g_cb_host, (size_t) kCbLayers * kCbWidth * sizeof(float), cudaHostAllocDefault);
    return true;
}

bool cvec_replicate(std::string& err) { return !g_cvec.loaded() || upload_here(err); }

void cvec_set_enabled(bool on) {
    if (!g_cvec.loaded() || on == g_on_host) return;
    int prev = 0;
    cudaGetDevice(&prev);
    const int v = on ? 1 : 0;
    for (int d = 0; d < kDevices; ++d) {
        if (g_dev[d].on == nullptr) continue;
        cudaSetDevice(d);
        cudaDeviceSynchronize();   // nothing in flight may still read the flag
        cudaMemcpy(g_dev[d].on, &v, sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaSetDevice(prev);
    g_on_host = on;
}

bool cvec_enabled() { return g_cvec.loaded() && g_on_host; }

bool cvec_tables(const float** dir, const float** s, const int** on) {
    if (!g_cvec.loaded()) return false;
    const DevTables& t = g_dev[cur_device()];
    if (t.dir == nullptr) return false;
    *dir = t.dir; *s = t.s; *on = t.on;
    return true;
}

void cvec_apply(float* R, int64_t layer, int64_t T, int64_t r_ld, const float* bo, int64_t bo_ld, const float* inj,
                int64_t inj_ld, bool write, void* stream) {
    if (!g_cvec.loaded() || T < 1) return;
    const DevTables& t = g_dev[cur_device()];
    if (t.dir == nullptr) throw std::runtime_error("cvec_apply: the control vector is not on this device (cvec_replicate)");
    const dim3 grid((unsigned) g_cvec.hc, (unsigned) T);
    cvec_kernel<<<grid, THREADS, 0, (cudaStream_t) stream>>>(R, t.dir, t.s, t.on, g_cvec.mode, layer,
                                                            (int) g_cvec.n_embd, (int) g_cvec.hc, r_ld, bo, bo_ld,
                                                            inj, inj_ld, write ? 1 : 0);
    if (cudaPeekAtLastError() != cudaSuccess) throw std::runtime_error("cvec_apply: launch failed");
    if (g_cb_host != nullptr && layer >= 0 && layer < kCbLayers && r_ld <= kCbWidth)
        cudaMemcpyAsync(g_cb_host + layer * kCbWidth, R + (T - 1) * r_ld, (size_t) r_ld * sizeof(float),
                        cudaMemcpyDeviceToHost, (cudaStream_t) stream);
}

void cvec_delta(float* bo, int64_t layer, int64_t T, int64_t ld, void* stream) {
    if (!g_cvec.covers_delta(layer) || T < 1) return;
    const DevTables& t = g_dev[cur_device()];
    if (t.dir == nullptr) throw std::runtime_error("cvec_delta: the control vector is not on this device (cvec_replicate)");
    cvec_delta_kernel<<<(unsigned) T, THREADS, 0, (cudaStream_t) stream>>>(bo, t.dir, t.s, t.on, layer,
                                                                          (int) g_cvec.n_embd, ld);
    if (cudaPeekAtLastError() != cudaSuccess) throw std::runtime_error("cvec_delta: launch failed");
}

void cvec_cb_flush() {
    static const char* dir = std::getenv("STRATA_CB_DIR");
    if (g_cb_host == nullptr || dir == nullptr) return;
    cudaDeviceSynchronize();
    for (int64_t l = 0; l < (int64_t) g_cvec.steered.size() && l < kCbLayers; ++l) {
        if (!g_cvec.covers(l)) continue;
        const std::string path = std::string(dir) + "/l_out-" + std::to_string(l) + ".f32";
        if (FILE* f = std::fopen(path.c_str(), "wb")) {
            std::fwrite(g_cb_host + l * kCbWidth, sizeof(float), (size_t) (g_cvec.hc * g_cvec.n_embd), f);
            std::fclose(f);
        }
    }
}

}  // namespace strata::kernels
