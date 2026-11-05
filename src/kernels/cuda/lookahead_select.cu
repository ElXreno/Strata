#include "strata/kernels/lookahead_select.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>

namespace strata::kernels {
namespace {

constexpr int kE = 512;
constexpr int kWarps = kE / 32;

__device__ __forceinline__ float block_reduce(float v, float* scratch, bool is_max) {
#pragma unroll
    for (int m = 16; m; m >>= 1) {
        const float o = __shfl_xor_sync(0xffffffffu, v, m, 32);
        v = is_max ? fmaxf(v, o) : v + o;
    }
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) scratch[w] = v;
    __syncthreads();
    if (w == 0) {
        float r = lane < kWarps ? scratch[lane] : (is_max ? -INFINITY : 0.0f);
#pragma unroll
        for (int m = 16; m; m >>= 1) {
            const float o = __shfl_xor_sync(0xffffffffu, r, m, 32);
            r = is_max ? fmaxf(r, o) : r + o;
        }
        if (lane == 0) scratch[kWarps] = r;
    }
    __syncthreads();
    const float r = scratch[kWarps];
    __syncthreads();
    return r;
}

__global__ void lookahead_select_k(const float* __restrict__ logits, int n_tok, const float* __restrict__ bias,
                                   const int32_t* __restrict__ res, float tau, int max_out, int32_t* __restrict__ out) {
    __shared__ float scratch[kWarps + 1];
    __shared__ float cand_p[kE];
    __shared__ int cand_e[kE];
    __shared__ int count;
    const int e = threadIdx.x;
    const float b = bias[e];
    float best = 0.0f;
    for (int t = 0; t < n_tok; ++t) {
        const float v = logits[(size_t) t * kE + e] + b;
        const float mx = block_reduce(v, scratch, true);
        const float ex = expf(v - mx);
        const float z = block_reduce(ex, scratch, false);
        best = fmaxf(best, ex / z);
    }
    if (res != nullptr && res[e] >= 0) best = 0.0f;
    if (e == 0) count = 0;
    __syncthreads();
    if (best > tau) {
        const int i = atomicAdd(&count, 1);
        cand_p[i] = best;
        cand_e[i] = e;
    }
    __syncthreads();
    const int c = count;
    if (e < c) {
        const float p = cand_p[e];
        const int me = cand_e[e];
        int rank = 0;
        for (int j = 0; j < c; ++j)
            rank += cand_p[j] > p || (cand_p[j] == p && cand_e[j] < me);
        if (rank < max_out) out[1 + rank] = me;
    }
    __syncthreads();
    if (e == 0) {
        out[0] = c < max_out ? c : max_out;
        __threadfence_system();
    }
}

}  // namespace

void lookahead_select(const float* logits, int n_tok, const float* bias, const int32_t* res_layer, float tau, int max_out,
                      int32_t* out, void* stream) {
    lookahead_select_k<<<1, kE, 0, (cudaStream_t) stream>>>(logits, n_tok, bias, res_layer, tau, max_out, out);
}

}  // namespace strata::kernels
