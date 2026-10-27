// Programmatic dependent launch for the DeepSeek engine, from Project Maya #495 (MIT, needmorevram).
#pragma once

#include <cuda_runtime.h>

#include <utility>

namespace strata::ds4 {

__device__ __forceinline__ void pdl_wait() {
#if !defined(STRATA_USE_HIP) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_trigger() {
#if !defined(STRATA_USE_HIP) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

bool pdl_on();

template <typename... P, typename... A>
inline void pdl_launch(void (*kernel)(P...), dim3 grid, dim3 block, size_t smem, cudaStream_t s, A&&... args) {
#if !defined(STRATA_USE_HIP)
    if (pdl_on()) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = grid;
        cfg.blockDim = block;
        cfg.dynamicSmemBytes = smem;
        cfg.stream = s;
        cudaLaunchAttribute at[1];
        at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        at[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = at;
        cfg.numAttrs = 1;
        (void) cudaLaunchKernelEx(&cfg, kernel, std::forward<A>(args)...);
        return;
    }
#endif
    kernel<<<grid, block, smem, s>>>(std::forward<A>(args)...);
}

}  // namespace strata::ds4
