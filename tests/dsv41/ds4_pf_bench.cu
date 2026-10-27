// tests/dsv41/ds4_pf_bench.cu - the prefetch's gate (src/dsv41, DS4_PREFETCH): can a kernel pull experts from registered
// RAM into VRAM at PCIe rate, and what does it cost the CPU lane's DRAM reads and a VRAM-bound kernel running beside it.
// Needs a GPU, no model.  ds4_pf_bench [blocks ...]   (default 8 12 16 24 32); passes at >= 40 GB/s with 16 blocks and
// < 15% loss for the concurrent CPU reader.
#include <cuda_runtime.h>
#include <immintrin.h>
#include <sys/mman.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#define CU(x)                                                                             \
    do {                                                                                  \
        cudaError_t e_ = (x);                                                             \
        if (e_ != cudaSuccess) {                                                          \
            std::printf("CUDA %s: %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__); \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

namespace {

constexpr size_t kMat = 2304000, kExp = 3 * kMat, kNExp = 270, kStage = 8;

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void* anon(size_t bytes) {
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) std::exit(2);
    madvise(p, bytes, MADV_HUGEPAGE);
    std::memset(p, 0x5a, bytes);
    return p;
}

__global__ void __launch_bounds__(512) pull(const uint4* src, uint4* dst, int first, int n) {
    for (int i = 0; i < n; ++i) {
        const uint4* s = src + (size_t) ((first + i) % kNExp) * (kExp / 16);
        uint4* d = dst + (size_t) (i % kStage) * (kExp / 16);
        for (int r = 0; r < 3; ++r) {
            const size_t nv = kMat / 16, step = (size_t) gridDim.x * blockDim.x;
            size_t o = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
            const uint4* sp = s + r * nv;
            uint4* dp = d + r * nv;
            for (; o + 3 * step < nv; o += 4 * step) {
                const uint4 a = __ldcs(sp + o), b = __ldcs(sp + o + step), c = __ldcs(sp + o + 2 * step),
                            e = __ldcs(sp + o + 3 * step);
                dp[o] = a;
                dp[o + step] = b;
                dp[o + 2 * step] = c;
                dp[o + 3 * step] = e;
            }
            for (; o < nv; o += step) dp[o] = __ldcs(sp + o);
        }
    }
}

__global__ void __launch_bounds__(512) vread(const uint4* p, size_t n, int passes, unsigned* sink) {
    unsigned acc = 0;
    for (int k = 0; k < passes; ++k)
        for (size_t o = (size_t) blockIdx.x * blockDim.x + threadIdx.x; o < n; o += (size_t) gridDim.x * blockDim.x) {
            const uint4 v = p[o];
            acc += v.x ^ v.y ^ v.z ^ v.w;
        }
    if (acc == 0x12345678u) *sink = acc;
}

__attribute__((target("avx2"))) void reader(const uint8_t* buf, size_t bytes, std::atomic<bool>* run,
                                            std::atomic<uint64_t>* got) {
    __m256i acc = _mm256_setzero_si256();
    while (run->load(std::memory_order_relaxed)) {
        for (size_t o = 0; o + (64 << 20) <= bytes && run->load(std::memory_order_relaxed); o += 64 << 20) {
            for (size_t q = o; q < o + (64 << 20); q += 128) {
                acc = _mm256_add_epi64(acc, _mm256_loadu_si256((const __m256i*) (buf + q)));
                acc = _mm256_add_epi64(acc, _mm256_loadu_si256((const __m256i*) (buf + q + 32)));
                acc = _mm256_add_epi64(acc, _mm256_loadu_si256((const __m256i*) (buf + q + 64)));
                acc = _mm256_add_epi64(acc, _mm256_loadu_si256((const __m256i*) (buf + q + 96)));
            }
            got->fetch_add(64 << 20, std::memory_order_relaxed);
        }
    }
    if (_mm256_extract_epi64(acc, 0) == 42) std::printf(" ");
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<int> blocks;
    for (int i = 1; i < argc; ++i) blocks.push_back(std::atoi(argv[i]));
    if (blocks.empty()) blocks = {8, 12, 16, 24, 32};
    const size_t bytes = (kNExp * kExp + (2 << 20) - 1) / (2 << 20) * (2 << 20);
    uint8_t* host = (uint8_t*) anon(bytes);
    double t0 = now_s();
    CU(cudaHostRegister(host, bytes, cudaHostRegisterPortable | cudaHostRegisterMapped));
    void* dp = nullptr;
    CU(cudaHostGetDevicePointer(&dp, host, 0));
    std::printf("registered %.2f GB in %.2f s, device pointer %s the host one\n", (double) bytes / 1e9, now_s() - t0,
                dp == (void*) host ? "is" : "is NOT");
    if (dp != (void*) host) return 3;
    uint8_t* stage = nullptr;
    unsigned* sink = nullptr;
    const size_t vb = 2ull << 30;
    uint8_t* vbuf = nullptr;
    CU(cudaMalloc(&stage, kStage * kExp));
    CU(cudaMalloc(&vbuf, vb));
    CU(cudaMemset(vbuf, 1, vb));
    CU(cudaMalloc(&sink, 4));
    cudaStream_t sb, sv;
    int lo = 0, hi = 0;
    CU(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    CU(cudaStreamCreateWithPriority(&sb, cudaStreamNonBlocking, hi));
    CU(cudaStreamCreateWithPriority(&sv, cudaStreamNonBlocking, lo));
    cudaEvent_t e0, e1;
    CU(cudaEventCreate(&e0));
    CU(cudaEventCreate(&e1));
    const size_t rbytes = 4ull << 30;
    uint8_t* rbuf = (uint8_t*) anon(rbytes);
    std::atomic<bool> run{true};
    std::atomic<uint64_t> got{0};
    std::vector<std::thread> th;
    auto timed_pull = [&](int nb, int n, int reps) {
        CU(cudaEventRecord(e0, sb));
        for (int r = 0; r < reps; ++r) pull<<<nb, 512, 0, sb>>>((const uint4*) dp, (uint4*) stage, r * n, n);
        CU(cudaEventRecord(e1, sb));
        CU(cudaEventSynchronize(e1));
        float ms = 0;
        CU(cudaEventElapsedTime(&ms, e0, e1));
        return (double) ms / reps;
    };
    auto cpu_rate = [&](double secs) {
        const uint64_t g0 = got.load();
        const double ta = now_s();
        std::this_thread::sleep_for(std::chrono::duration<double>(secs));
        return (double) (got.load() - g0) / 1e9 / (now_s() - ta);
    };
    auto pull_rate = [&](int nb, double secs) {
        const double ta = now_s();
        double moved = 0;
        while (now_s() - ta < secs) {
            pull<<<nb, 512, 0, sb>>>((const uint4*) dp, (uint4*) stage, 0, (int) kNExp);
            CU(cudaStreamSynchronize(sb));
            moved += (double) kNExp * kExp;
        }
        return moved / 1e9 / (now_s() - ta);
    };
    for (int t = 0; t < 16; ++t) th.emplace_back([&, t] {
        reader(rbuf + (size_t) t * (rbytes / 16), rbytes / 16, &run, &got);
    });
    const double cpu_alone = cpu_rate(1.0);
    std::printf("CPU reader alone: %.1f GB/s\n", cpu_alone);
    const int vpass = 20;
    vread<<<1024, 512, 0, sv>>>((const uint4*) vbuf, vb / 16, 1, sink);
    CU(cudaStreamSynchronize(sv));
    CU(cudaEventRecord(e0, sv));
    vread<<<1024, 512, 0, sv>>>((const uint4*) vbuf, vb / 16, vpass, sink);
    CU(cudaEventRecord(e1, sv));
    CU(cudaEventSynchronize(e1));
    float v_alone = 0;
    CU(cudaEventElapsedTime(&v_alone, e0, e1));
    std::printf("VRAM kernel alone: %.2f ms (%.0f GB/s)\n", v_alone, (double) vb * vpass / 1e6 / v_alone);
    bool pass = true;
    for (int nb : blocks) {
        run.store(false);
        for (auto& t : th) t.join();
        th.clear();
        const double ms1 = timed_pull(nb, 1, 200), ms2 = timed_pull(nb, 2, 200), ms3 = timed_pull(nb, 3, 200);
        const double plain = pull_rate(nb, 1.0);
        run.store(true);
        for (int t = 0; t < 16; ++t) th.emplace_back([&, t] {
            reader(rbuf + (size_t) t * (rbytes / 16), rbytes / 16, &run, &got);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        const uint64_t g0 = got.load();
        const double ta = now_s();
        const double with_cpu = pull_rate(nb, 1.0);
        const double cpu_with = (double) (got.load() - g0) / 1e9 / (now_s() - ta);
        CU(cudaEventRecord(e0, sv));
        vread<<<1024, 512, 0, sv>>>((const uint4*) vbuf, vb / 16, vpass, sink);
        CU(cudaEventRecord(e1, sv));
        const double tb = now_s();
        double moved = 0;
        while (cudaEventQuery(e1) == cudaErrorNotReady) {
            pull<<<nb, 512, 0, sb>>>((const uint4*) dp, (uint4*) stage, 0, 16);
            CU(cudaStreamSynchronize(sb));
            moved += 16.0 * kExp;
        }
        float v_with = 0;
        CU(cudaEventElapsedTime(&v_with, e0, e1));
        std::printf("blocks %2d: 1/2/3 experts %.3f/%.3f/%.3f ms; %.1f GB/s alone; beside the CPU reader %.1f GB/s with the "
                    "reader at %.1f (%.0f%% loss); beside the VRAM kernel %.2f ms (%.0f%% loss), pull %.1f GB/s\n",
                    nb, ms1, ms2, ms3, plain, with_cpu, cpu_with, 100.0 * (1.0 - cpu_with / cpu_alone), v_with,
                    100.0 * (v_with / v_alone - 1.0), moved / 1e9 / (now_s() - tb));
        if (nb == 16 && (plain < 40.0 || cpu_with < 0.85 * cpu_alone)) pass = false;
    }
    run.store(false);
    for (auto& t : th) t.join();
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    cudaHostUnregister(host);
    return pass ? 0 : 1;
}
