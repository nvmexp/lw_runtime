/***************************************************************************************************
 * Copyright (c) 2011-2019, LWPU CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LWPU CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <xmma/xmma.h>

#define XMMA_DIV_UP(m, n) (((m) + (n)-1) / (n))

#include <xmma/helpers/gemm.h>

namespace xmma {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Gpu_arch_,
    int M_,
    int N_,
    int K_,
    int WARPS_M_,
    int WARPS_N_,
    int WARPS_K_,
    int GROUPS_ = 1
>
struct Cta_tile {

    using Gpu_arch = Gpu_arch_;

    // Make sure M and N are multiples of the group size.
    static_assert((M_ % GROUPS_ == 0) && (N_ % GROUPS_ == 0), "M/N must be multiple of GROUPS");

    // The size of the CTA tile.
    enum { M = M_, N = N_, K = K_ };
    // The number of warps.
    enum { WARPS_M = WARPS_M_, WARPS_N = WARPS_N_, WARPS_K = WARPS_K_ };
    // The number of groups.
    enum { GROUPS = GROUPS_ };
    // The number of warps per CTA.
    enum { WARPS_PER_CTA = WARPS_M * WARPS_N * WARPS_K };
    // The number of threads per warp.
    enum { THREADS_PER_WARP = Gpu_arch::THREADS_PER_WARP };
    // The number of threads per CTA.
    enum { THREADS_PER_CTA = WARPS_PER_CTA * THREADS_PER_WARP };
    // Half K dimension
    enum { HALF_K = K_ / 2 };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    typename Gpu_arch_,
    typename A_type_,
    typename B_type_,
    typename C_type_,
    typename Aclwmulator_type_,
    typename Epilogue_type_,
    int Alignment_A = 8, // Alignment of elements in A in bytes
    int Alignment_B = 8, // Alignment of elements in B in bytes
    int Alignment_C = 8  // Alignment of Elements in C in bytes
>
struct Traits {
    // The architecture.
    using Gpu_arch = Gpu_arch_;
    // The data type for A elements.
    using A_type = A_type_;
    // The data type for B elements.
    using B_type = B_type_;
    // The data type for C elements.
    using C_type = C_type_;
    // The data type for aclwmulators.
    using Aclwmulator_type = Aclwmulator_type_;
    // The data type of the math in the epilogue.
    using Epilogue_type = Epilogue_type_;

    // Do we use the output precision to run split-k?
    enum { USE_SPLIT_K_WITH_OUTPUT_PRECISION = 0 };

    // Create the description of the CTA tile from a configuration.
    template<
        int M,
        int N,
        int K,
        int WARPS_M,
        int WARPS_N,
        int WARPS_K,
        int GROUPS = 1
    >
    using Cta_tile_extd = Cta_tile<Gpu_arch,
                                   M,
                                   N,
                                   K,
                                   WARPS_M,
                                   WARPS_N,
                                   WARPS_K,
                                   GROUPS>;

    // The number of bits per element of A.
    enum { BITS_PER_ELEMENT_A = Alignment_A*sizeof(A_type) };

    // An offset in bytes for A.
    static inline __host__ __device__ int64_t offset_in_bytes_a(int64_t offset) {
        return offset * static_cast<int64_t>(sizeof(A_type));
    }

    // The number of bits per element of B.
    enum { BITS_PER_ELEMENT_B = Alignment_B*sizeof(B_type) };

    // An offset in bytes for B.
    static inline __host__ __device__ int64_t offset_in_bytes_b(int64_t offset) {
        return offset * static_cast<int64_t>(sizeof(B_type));
    }

    // The number of bits per element of C.
    enum { BITS_PER_ELEMENT_C = Alignment_C*sizeof(C_type) };

    // An offset in bytes for C.
    static inline __host__ __device__ int64_t offset_in_bytes_c(int64_t offset) {
        return offset * static_cast<int64_t>(sizeof(C_type));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Gpu_arch_base {

    // By default, architectures have 32 threads per warp.
    enum { THREADS_PER_WARP = 32 };
    // By default, architectures do not support LDGSTS.
    enum { HAS_LDGSTS = 0 };
    // By default, architecture do not support super HMMA
    enum { HAS_SUPER_HMMA = 0 };
    //
    enum { HAS_UTMALDG = 0 };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits_, typename Cta_tile_ >
using Cta_tile_with_m_with_padding =
    typename Traits_::template Cta_tile_extd<Next_power_of_two<Cta_tile_::M>::VALUE,
                                             Cta_tile_::N,
                                             Cta_tile_::K,
                                             Cta_tile_::WARPS_M,
                                             Cta_tile_::WARPS_N,
                                             Cta_tile_::WARPS_K,
                                             Cta_tile_::GROUPS>;

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits_, typename Cta_tile_ >
using Cta_tile_with_n_with_padding =
    typename Traits_::template Cta_tile_extd<Cta_tile_::M,
                                             Next_power_of_two<Cta_tile_::N>::VALUE,
                                             Cta_tile_::K,
                                             Cta_tile_::WARPS_M,
                                             Cta_tile_::WARPS_N,
                                             Cta_tile_::WARPS_K,
                                             Cta_tile_::GROUPS>;

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits_, typename Cta_tile_ >
using Cta_tile_with_k_with_padding =
    typename Traits_::template Cta_tile_extd<Cta_tile_::M,
                                             Cta_tile_::N,
                                             Next_power_of_two<Cta_tile_::K>::VALUE,
                                             Cta_tile_::WARPS_M,
                                             Cta_tile_::WARPS_N,
                                             Cta_tile_::WARPS_K,
                                             Cta_tile_::GROUPS>;

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace xmma

