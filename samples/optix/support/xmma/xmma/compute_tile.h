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

#include <xmma/layout.h>
#include <xmma/utils.h>
#include <xmma/fragment.h>
#include <xmma/params.h>
#include <xmma/helpers/fragment.h>
#include <xmma/helpers/gemm.h>
#include <xmma/numeric_types.h>
#include <xmma/hopper/compute_tile.h>

namespace xmma {

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, typename Smem_tile_a, typename Smem_tile_b >
struct Compute_tile {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The number of XMMAs.
    enum { XMMAS_M = Xmma_tile::XMMAS_M };
    enum { XMMAS_N = Xmma_tile::XMMAS_N };
    enum { XMMAS_K = Xmma_tile::XMMAS_K };

    // Clear the aclwmulators.
    inline __device__ void clear() { helpers::clear(acc_); }

    // Compute.
    inline __device__ void compute(int ki, bool = false) {
        helpers::gemm(acc_, a_[(ki-1)&1], b_[(ki-1)&1]);
    }

    // Load from shared memory.
    inline __device__ void load(Smem_tile_a &smem_a, Smem_tile_b &smem_b, int ki, bool = false) {
        smem_a.load(a_[ki&1], ki);
        smem_b.load(b_[ki&1], ki);
    }

    template< typename Callback_fuse_a > inline __device__ void apply_fuse_a( Callback_fuse_a &a_fuse ) {}
    template< typename Callback_fuse_b > inline __device__ void apply_fuse_b( Callback_fuse_b &b_fuse ) {}

    inline __device__ void debug_print() {
        for(int n = 0; n < XMMAS_N; n++) {
            for(int m = 0; m < XMMAS_M; m++) {
                for(int r = 0; r < 1; r++) {//Base_::NUM_REGS; r++) {
                    printf("blockIdx: (%d, %d, %d) threadIdx: (%d, %d, %d) value[%d]: %x\n",
                           blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, r, acc_[m][n].regs_[r]);
                }
            }
        }
    }

    // The aclwmulators.
    Fragment_aclwmulator<Traits> acc_[XMMAS_M][XMMAS_N];
    // The fragments to load A.
    typename Smem_tile_a::Fragment a_[2][XMMAS_M];
    // The fragments to load B.
    typename Smem_tile_b::Fragment b_[2][XMMAS_N];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits,
    // The CTA descriptor.
    typename Cta_tile,
    // Shared memory for A.
    typename Smem_tile_a,
    // Shared memory for B.
    typename Smem_tile_b,
    // Operation type
    Operation_type OPERATION_TYPE
>
struct Compute_tile_with_groups {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The number of XMMAs.
    enum { XMMAS_M = Xmma_tile::XMMAS_M };
    enum { XMMAS_N = Xmma_tile::XMMAS_N };
    enum { XMMAS_K = Xmma_tile::XMMAS_K };

    // Clear the aclwmulators.
    inline __device__ void clear() { helpers::clear(acc_); }

    inline __device__ void compute(int ki, bool = false) {
        // The number of elements per group in the N dimension.
        enum { N_PER_GROUP = XMMAS_N * Cta_tile::WARPS_N / Xmma_tile::XMMAS_GROUPS };
        // The 1st element in the N dimension.
        const int offset = (ki - 1)
          * Xmma_tile::K_PER_XMMA / Xmma_tile::N_PER_XMMA / N_PER_GROUP;

        // Compute the MMAs.
        #pragma unroll
        for( int mi = 0; mi < XMMAS_M; ++mi ) {
            #pragma unroll
            for( int ni = 0; ni < N_PER_GROUP; ++ni ) {
                if ( OPERATION_TYPE == Operation_type::FPROP )
                    acc_[mi][offset+ni].mma(a_[(ki-1)&1][mi], b_[(ki-1)&1][offset+ni]);
                else if ( OPERATION_TYPE == Operation_type::DGRAD ||
                          OPERATION_TYPE == Operation_type::STRIDED_DGRAD )
                    acc_[mi][offset+ni].mma(a_[(ki-1)&1][mi], b_[(ki-1)&1][ni]);
            }
        }
    }

    // Load from shared memory.
    inline __device__ void load(Smem_tile_a &smem_a, Smem_tile_b &smem_b, int ki, bool = false) {
        smem_a.load(a_[ki&1], ki);
        smem_b.load(b_[ki&1], ki);
    }

    template< typename Callback_fuse_a > inline __device__ void apply_fuse_a( Callback_fuse_a &a_fuse ) {}
    template< typename Callback_fuse_b > inline __device__ void apply_fuse_b( Callback_fuse_b &b_fuse ) {}

    // The aclwmulators.
    Fragment_aclwmulator<Traits> acc_[XMMAS_M][XMMAS_N];
    // The fragments to load A.
    typename Smem_tile_a::Fragment a_[2][XMMAS_M];
    // The fragments to load B.
    typename Smem_tile_b::Fragment b_[2][XMMAS_N];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits,
    // The CTA descriptor.
    typename Cta_tile,
    // Shared memory for A.
    typename Smem_tile_a,
    // Shared memory for B.
    typename Smem_tile_b
>
struct Compute_tile_with_groups<Traits, Cta_tile, Smem_tile_a, Smem_tile_b, Operation_type::WGRAD> {

    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The number of XMMAs.
    enum { XMMAS_M = Xmma_tile::XMMAS_M };
    enum { XMMAS_N = Xmma_tile::XMMAS_N };
    enum { XMMAS_K = Xmma_tile::XMMAS_K };

    // Clear the aclwmulators.
    inline __device__ void clear() { helpers::clear(acc_); }

    template< typename Callback_fuse_a > inline __device__ void apply_fuse_a( Callback_fuse_a &a_fuse ) {}
    template< typename Callback_fuse_b > inline __device__ void apply_fuse_b( Callback_fuse_b &b_fuse ) {}

    inline __device__ void compute(int ki, bool = false) {
        helpers::gemm(acc_, a_[(ki-1)&1], b_[(ki-1)&1]);
    }

    // Load from shared memory.
    inline __device__ void load(Smem_tile_a &smem_a, Smem_tile_b &smem_b, int ki, bool = false) {
        smem_a.load(a_[ki&1], ki);
        smem_b.load(b_[ki&1], ki);
    }

    // The aclwmulators.
    Fragment_aclwmulator<Traits> acc_[XMMAS_M][XMMAS_N / Xmma_tile::XMMAS_GROUPS];
    // The fragments to load A.
    typename Smem_tile_a::Fragment a_[2][XMMAS_M];
    // The fragments to load B.
    typename Smem_tile_b::Fragment b_[2][XMMAS_N];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits,
    // The CTA tile.
    typename Cta_tile,
    // The shared memory tile for A.
    typename Smem_tile_a,
    // The shared memory tile for B.
    typename Smem_tile_b,
    // Operation type
    Operation_type OPERATION_TYPE,
    // Do we enable groups?
    bool = (Cta_tile::GROUPS > 1)
>
struct Compute_tile_selector {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits,
    // The CTA tile.
    typename Cta_tile,
    // The shared memory tile for A.
    typename Smem_tile_a,
    // The shared memory tile for B.
    typename Smem_tile_b,
    // Operation type
    Operation_type OPERATION_TYPE
>
struct Compute_tile_selector<Traits, Cta_tile, Smem_tile_a, Smem_tile_b, OPERATION_TYPE, false> {
    using Class = Compute_tile<Traits, Cta_tile, Smem_tile_a, Smem_tile_b>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits,
    // The CTA tile.
    typename Cta_tile,
    // The shared memory tile for A.
    typename Smem_tile_a,
    // The shared memory tile for B.
    typename Smem_tile_b,
    // Operation type
    Operation_type OPERATION_TYPE
>
struct Compute_tile_selector<Traits, Cta_tile, Smem_tile_a, Smem_tile_b, OPERATION_TYPE, true> {
    using Class = Compute_tile_with_groups<
      Traits, Cta_tile, Smem_tile_a, Smem_tile_b, OPERATION_TYPE>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace xmma
