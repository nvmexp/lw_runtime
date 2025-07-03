#pragma once

static inline __device__ __host__ uint64_t divide_up(uint64_t x, uint64_t y) {
  return (x + y - 1) / y;
}

static inline __device__ __host__ uint64_t align_up(uint64_t bytes,
                                                    uint64_t align_width) {
  return divide_up(bytes, align_width) * align_width;
}

static inline __device__ void __set_smem_barrier(uint32_t addr, uint32_t val) {
  asm volatile("mbarrier.init.shared.b64 [%0], 1;" ::"r"(addr), "r"(val));
}

static inline __device__ void __cp_async_shared_global(uint32_t smem_addr,
                                                       uint64_t gmem_addr,
                                                       bool p = true) {
  uint32_t m = p ? 16u : 0u;
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_addr),
      "l"(gmem_addr), "r"(m));
}

static inline __device__ void __cp_async_mbar_arrive(uint32_t addr) {
  asm volatile("cp.async.mbarrier.arrive.shared.b64 [%0];" ::"r"(addr));
}

static inline __device__ void __cp_async_wait_defer() {
  asm volatile("cp.async.wait.defer;" ::);
}

template <uint32_t n> static inline __device__ void __cp_async_wait() {
  asm volatile("cp.async.wait %0;\n" ::"n"(n));
}