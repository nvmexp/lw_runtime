#pragma once


namespace xmma {
namespace hopper {
namespace emu {

typedef enum {
  M88,
  MT88
} layout;

template<layout L, uint32_t N = 1>
static inline __device__ void STSM(uint32_t ra, uint32_t rb) {
  const uint32_t MASK = 0xFFFFFFFF;

  uint32_t tx = threadIdx.x % 32;

  /// colwert shared memory to void*
  uint64_t p_;
  asm volatile("cvta.shared.u64 %0, %1;\n":"=l"(p_):"l"(static_cast<uint64_t>(ra)));

  if(L == M88) {

    #pragma unroll
    for(uint32_t i = 0; i < N; i++) {

      uint32_t srcLane = tx / 4 + i * 8;
      uint32_t index = tx % 4;
      uint64_t p__ = __shfl_sync(MASK, p_, srcLane);

      reinterpret_cast<uint32_t*>(p__)[index] = rb;
    }

  }
  if(L == MT88) {

    #pragma unroll
    for(uint32_t i = 0; i < N; i++) {

      uint16_t *r_b16 = reinterpret_cast<uint16_t*>(rb);

      uint32_t srcLane1 = (2 * tx) % 8 + i * 8;
      uint32_t srcLane2 = (2 * tx + 1) % 8 + i * 8;
      uint32_t index = (tx / 4);
      uint64_t p1 = __shfl_sync(MASK, p_, srcLane1);
      uint64_t p2 = __shfl_sync(MASK, p_, srcLane2);

      uint16_t *pp1 = reinterpret_cast<uint16_t*>(p1);
      uint16_t *pp2 = reinterpret_cast<uint16_t*>(p2);
      pp1[index] = r_b16[0];
      pp2[index] = r_b16[1];
    }

  }
}


template<layout L, uint32_t N = 2>
static inline __device__ void STSM(uint32_t ra, uint2 rb_) {
  const uint32_t MASK = 0xFFFFFFFF;

  uint32_t tx = threadIdx.x % 32;

  uint32_t *rb = reinterpret_cast<uint32_t*>(&rb_);

  /// colwert shared memory to void*
  uint64_t p_;
  asm volatile("cvta.shared.u64 %0, %1;\n":"=l"(p_):"l"(static_cast<uint64_t>(ra)));

  if(L == M88) {

    #pragma unroll
    for(uint32_t i = 0; i < N; i++) {

      uint32_t srcLane = tx / 4 + i * 8;
      uint32_t index = tx % 4;
      uint64_t p__ = __shfl_sync(MASK, p_, srcLane);

      reinterpret_cast<uint32_t*>(p__)[index] = rb[i];
    }

  }
  if(L == MT88) {

    #pragma unroll
    for(uint32_t i = 0; i < N; i++) {

      uint16_t *r_b16 = reinterpret_cast<uint16_t*>(&rb[i]);

      uint32_t srcLane1 = (2 * tx) % 8 + i * 8;
      uint32_t srcLane2 = (2 * tx + 1) % 8 + i * 8;
      uint32_t index = (tx / 4);
      uint64_t p1 = __shfl_sync(MASK, p_, srcLane1);
      uint64_t p2 = __shfl_sync(MASK, p_, srcLane2);

      uint16_t *pp1 = reinterpret_cast<uint16_t*>(p1);
      uint16_t *pp2 = reinterpret_cast<uint16_t*>(p2);
      pp1[index] = r_b16[0];
      pp2[index] = r_b16[1];
    }

  }
}


/**
* L -> layout (M88 or MT88)
* N -> {1, 2, 4}
* ra -> shared memory pointer
* rb[N] -> registers per thread holding matrix data
*/
template<layout L, uint32_t N = 4>
static inline __device__ void STSM(uint32_t ra, uint4 rb_) {

  const uint32_t MASK = 0xFFFFFFFF;

  uint32_t tx = threadIdx.x % 32;

  uint32_t *rb = reinterpret_cast<uint32_t*>(&rb_);

  /// colwert shared memory to void*
  uint64_t p_;
  asm volatile("cvta.shared.u64 %0, %1;\n":"=l"(p_):"l"(static_cast<uint64_t>(ra)));

  if(L == M88) {

    #pragma unroll
    for(uint32_t i = 0; i < N; i++) {

      uint32_t srcLane = tx / 4 + i * 8;
      uint32_t index = tx % 4;
      uint64_t p__ = __shfl_sync(MASK, p_, srcLane);

      reinterpret_cast<uint32_t*>(p__)[index] = rb[i];
    }

  }
  if(L == MT88) {

    #pragma unroll
    for(uint32_t i = 0; i < N; i++) {

      uint16_t *r_b16 = reinterpret_cast<uint16_t*>(&rb[i]);

      uint32_t srcLane1 = (2 * tx) % 8 + i * 8;
      uint32_t srcLane2 = (2 * tx + 1) % 8 + i * 8;
      uint32_t index = (tx / 4);
      uint64_t p1 = __shfl_sync(MASK, p_, srcLane1);
      uint64_t p2 = __shfl_sync(MASK, p_, srcLane2);

      uint16_t *pp1 = reinterpret_cast<uint16_t*>(p1);
      uint16_t *pp2 = reinterpret_cast<uint16_t*>(p2);
      pp1[index] = r_b16[0];
      pp2[index] = r_b16[1];
    }

  }

}

} // end namespace emu
} // end namespace hopper
} // end namespace xmma