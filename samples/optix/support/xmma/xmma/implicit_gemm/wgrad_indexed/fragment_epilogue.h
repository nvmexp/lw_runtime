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

namespace xmma {
namespace implicit_gemm {
namespace wgrad_indexed {

///////////////////////////////////////////////////////////////////////////////////////////////////

template<int GROUPS, bool HAS_SUPER_HMMA, typename Aclwmulators>
struct Group_Shuffle {
    inline __device__ void shuffle(Aclwmulators &acc) {
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Aclwmulators>
struct Group_Shuffle<8, false, Aclwmulators> {

    // The number of regs to hold aclwmulators
    enum { NUM_REGS = Aclwmulators::NUM_REGS };

    inline __device__ void shuffle(Aclwmulators &acc) {
        int lane = threadIdx.x % 32;
        if( lane >= 16 ) {
            #pragma unroll
            for ( int k = 0; k < NUM_REGS / 2; ++k ) {
                acc.reg(k) = acc.reg(k + NUM_REGS / 2);
            }
        }
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Aclwmulators>
struct Group_Shuffle<16, false, Aclwmulators> {

    // The number of regs to hold aclwmulators
    enum { NUM_REGS = Aclwmulators::NUM_REGS };

    inline __device__ void shuffle(Aclwmulators &acc) {
        int lane = threadIdx.x % 32;
        if ( (lane >= 16 && lane < 20)
                || (lane >= 28 && lane < 32) ) {
            #pragma unroll
            for ( int k = 0; k < NUM_REGS / 2; ++k ) {
                acc.reg(k) = acc.reg(k + NUM_REGS / 2);
            }
        }
        const unsigned mask = 0xFFFFFFFF;
        int src_lane = lane;
        if( lane % 16 / 4 % 2 == 1) {
            src_lane = lane + 8;
        }
        #pragma unroll
        for ( int k = 0; k < NUM_REGS / 2; ++k ) {
            acc.reg(k) = __shfl_sync(mask, acc.reg(k), src_lane);
        }
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Aclwmulators>
struct Group_Shuffle<8, true, Aclwmulators> {

    // The number of regs to hold aclwmulators
    enum { NUM_REGS = Aclwmulators::NUM_REGS };

    inline __device__ void shuffle(Aclwmulators &acc) {
        #pragma unroll
        for ( int k = NUM_REGS / 4; k < NUM_REGS / 2; ++k ) {
          acc.reg(k) = acc.reg(k + NUM_REGS / 2);
        }
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Aclwmulators>
struct Group_Shuffle<16, true, Aclwmulators> {

    // The number of regs to hold aclwmulators
    enum { NUM_REGS = Aclwmulators::NUM_REGS };

    inline __device__ void shuffle(Aclwmulators &acc) {
        #pragma unroll
        for ( int k = NUM_REGS / 4; k < NUM_REGS / 2; ++k ) {
          acc.reg(k) = acc.reg(k + NUM_REGS / 2);
        }
        const unsigned mask = 0xFFFFFFFF;
        int lane = threadIdx.x % 32;
        int src_lane = lane;
        if ( lane >= 16 && lane % 4 < 2 ) {
            src_lane = lane + 2;
        }
        #pragma unroll
        for ( int k = 0; k < NUM_REGS / 2; ++k) {
            acc.reg(k) = __shfl_sync(mask, acc.reg(k), src_lane);
        }
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, bool IN_CTA_SPLIT_K = (Cta_tile::WARPS_K > 1) >
struct Fragment_epilogue_pre_swizzle 
    : public xmma::Fragment_epilogue_pre_swizzle<Traits, Cta_tile, IN_CTA_SPLIT_K> {

    // The base class.
    using Base = xmma::Fragment_epilogue_pre_swizzle<Traits, Cta_tile, IN_CTA_SPLIT_K>; 

    // The aclwmulators.
    using Aclwmulators = typename Base::Aclwmulators;

    // The GPU arch
    using Gpu_arch = typename Traits::Gpu_arch;

    // Does it support super HMMA
    enum { HAS_SUPER_HMMA = Gpu_arch::HAS_SUPER_HMMA };

    // Number of groups per CTA
    enum { GROUPS = Cta_tile::GROUPS };

    inline __device__ void shuffle_groups(Aclwmulators &acc) {
        Group_Shuffle<GROUPS,HAS_SUPER_HMMA,Aclwmulators> group_shuffle;
        group_shuffle.shuffle(acc);
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace wgrad
} // namespace implicit_gemm
} // namespace xmma

