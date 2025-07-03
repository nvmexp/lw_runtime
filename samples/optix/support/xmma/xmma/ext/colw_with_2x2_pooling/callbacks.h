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

#include <xmma/utils.h>
#include <xmma/helpers/epilogue.h>

namespace xmma {
namespace ext {
namespace colw_with_2x2_pooling {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The instruction traits.
    typename Traits, 
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile, 
    // The fragment class before writing data to shared memory for swizzling.
    typename Fragment_pre_swizzle,
    // The fragment class after reading data from shared memory for swizzling.
    typename Fragment_post_swizzle,
    // The fragment class before writing data to global memory.
    typename Fragment_c,
    // The functor to apply in the post-swizzle part.
    typename Pooling_functor,
    // The base class.
    typename Base_ = xmma::helpers::Empty_callbacks_epilogue<Traits, 
                                                                 Cta_tile, 
                                                                 Fragment_pre_swizzle,
                                                                 Fragment_post_swizzle,
                                                                 Fragment_c>
>
struct Callbacks : public Base_ {
    // The base class.
    using Base = Base_;

    // Ctor.
    template< typename Params >
    inline __device__ Callbacks(const Params &params, 
                                void *smem, 
                                int bidm, 
                                int bidn, 
                                int bidz, 
                                int tidx) 
        : Base(params, smem, bidm, bidn, bidz, tidx)
        , fct_(params, bidm, bidn, bidz, tidx) {
    }

    // Post swizzle.
    template< typename Epilogue >
    inline __device__ void pre_pack(Epilogue &epilogue, int mi, int ii, Fragment_post_swizzle &frag) {
        // Call the base member function.
        Base::template pre_pack<Epilogue>(epilogue, mi, ii, frag);

        // The number of registers after the call to reduce.
        enum { REGS_AFTER_REDUCTION = Fragment_post_swizzle::NUM_REGS / 4 / Cta_tile::WARPS_K };

        // Initialize the output.
        #pragma unroll
        for( int jj = 0; jj < REGS_AFTER_REDUCTION; ++jj ) {
            fct_.init(frag.reg(jj));
        }

        // Do the reduction.
        #pragma unroll
        for( int jj = 1; jj < 4; ++jj ) {
            #pragma unroll
            for( int kk = 0; kk < REGS_AFTER_REDUCTION; ++kk ) {
                fct_.update(frag.reg(kk), frag.reg(jj*REGS_AFTER_REDUCTION + kk));
            }
        }
    }

    // The pooling functor.
    Pooling_functor fct_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Avg_fp16 {
    // Ctor.
    template< typename Params >
    inline __device__ Avg_fp16(const Params&, int, int, int, int) {
    }

    // Initialize the result.
    inline __device__ void init(uint32_t &r0) {
        r0 = xmma::hmul2(0x34003400u, r0); // 0x3400 is FP16 representation of 0.25.
    }

    // Apply the functor.
    inline __device__ void update(uint32_t &r0, const uint32_t &r1) {
        r0 = xmma::hfma2(0x34003400u, r1, r0); // 0x3400 is FP16 representation of 0.25.
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Max_fp16 {
    // Ctor.
    template< typename Params >
    inline __device__ Max_fp16(const Params&, int, int, int, int) {
    }

    // Initialize the result.
    inline __device__ void init(uint32_t &r0) {
    }

    // Apply the functor.
    inline __device__ void update(uint32_t &r0, const uint32_t &r1) {
        r0 = xmma::hmax2(r0, r1);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace colw_with_2x2_pooling
} // namespace ext
} // namespace xmma


