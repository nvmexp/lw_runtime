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

#include <xmma/fragment.h>
#include <xmma/volta/traits.h>
#include <xmma/volta/fragment.h>
#include <xmma/turing/traits.h>
#include <xmma/turing/fragment.h>
#include <xmma/ampere/traits.h>
#include <xmma/ampere/fragment.h>

namespace xmma {
namespace ext {
namespace colw_with_2x2_pooling {

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, typename Gmem_tile_c >
struct Make_fragment_hmma_fp16_post_swizzle_base {
    // The number of elements per thread.
    enum { NUM_ELTS = Gmem_tile_c::CHANNELS_PER_THREAD * 4 * Cta_tile::WARPS_K };
    // The fragment.
    using Type = xmma::Fragment<lwtlass::half_t, NUM_ELTS>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, typename Gmem_tile_c >
struct Fragment_hmma_fp16_post_swizzle 
    : public Make_fragment_hmma_fp16_post_swizzle_base<Traits, Cta_tile, Gmem_tile_c>::Type {

    // The helper to assemble the base class.
    using Make_base = Make_fragment_hmma_fp16_post_swizzle_base<Traits, Cta_tile, Gmem_tile_c>;
    // The base class.
    using Base = typename Make_base::Type;

    // Add a previous value.
    template< typename Fragment_c, typename Fragment_beta >
    inline __device__ void add_residual(const Fragment_c&, const Fragment_beta&) {
    }

    // Do the reduction.
    template< typename Fragment_alpha >
    inline __device__ void reduce(const Fragment_alpha&) {
        enum { REGS_AFTER_REDUCTION = Base::NUM_REGS / Cta_tile::WARPS_K };
        #pragma unroll
        for( int ii = 1; ii < Cta_tile::WARPS_K; ++ii ) {
            #pragma unroll
            for( int jj = 0; jj < 4; ++jj ) {
                #pragma unroll
                for( int kk = 0; kk < REGS_AFTER_REDUCTION; ++kk ) {
                    int ri = jj  *REGS_AFTER_REDUCTION + kk;
                    int rj = ii*4*REGS_AFTER_REDUCTION + ri;
                    this->reg(ri) = xmma::hadd2(this->reg(ri), this->reg(rj)); 
                }
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, typename Gmem_tile_c >
struct Fragment_post_swizzle {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, typename Gmem_tile_c >
struct Fragment_post_swizzle<xmma::Volta_hmma_fp16_traits, Cta_tile, Gmem_tile_c>
    : public Fragment_hmma_fp16_post_swizzle<xmma::Volta_hmma_fp16_traits, 
                                             Cta_tile, 
                                             Gmem_tile_c> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, typename Gmem_tile_c >
struct Fragment_post_swizzle<xmma::Turing_hmma_fp16_traits, Cta_tile, Gmem_tile_c>
    : public Fragment_hmma_fp16_post_swizzle<xmma::Turing_hmma_fp16_traits, 
                                             Cta_tile, 
                                             Gmem_tile_c> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, typename Gmem_tile_c >
struct Fragment_post_swizzle<xmma::Ampere_hmma_fp16_traits, Cta_tile, Gmem_tile_c>
    : public Fragment_hmma_fp16_post_swizzle<xmma::Ampere_hmma_fp16_traits, 
                                             Cta_tile, 
                                             Gmem_tile_c> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, int NUM_ELTS >
struct Fragment_hmma_fp16_c : public xmma::Fragment<lwtlass::half_t, NUM_ELTS> {

    // The base class.
    using Base = xmma::Fragment<lwtlass::half_t, NUM_ELTS>;

    // Add the residual.
    template< typename Fragment_c, typename Fragment_beta >
    inline __device__ void add_residual(const Fragment_c&, const Fragment_beta&) {
    }

    // Pack from a post-swizzle fragment.
    template< typename Fragment_alpha, typename Fragment_post_swizzle >
    inline __device__ void pack(const Fragment_alpha&, const Fragment_post_swizzle &frag) {
        #pragma unroll
        for( int ii = 0; ii < Base::NUM_REGS; ++ii ) {
            this->reg(ii) = frag.reg(ii);
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, int NUM_ELTS >
struct Fragment_c {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, int NUM_ELTS >
struct Fragment_c<xmma::Volta_hmma_fp16_traits, Cta_tile, NUM_ELTS>
    : public Fragment_hmma_fp16_c<xmma::Volta_hmma_fp16_traits, Cta_tile, NUM_ELTS> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, int NUM_ELTS >
struct Fragment_c<xmma::Turing_hmma_fp16_traits, Cta_tile, NUM_ELTS>
    : public Fragment_hmma_fp16_c<xmma::Turing_hmma_fp16_traits, Cta_tile, NUM_ELTS> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Cta_tile, int NUM_ELTS >
struct Fragment_c<xmma::Ampere_hmma_fp16_traits, Cta_tile, NUM_ELTS>
    : public Fragment_hmma_fp16_c<xmma::Ampere_hmma_fp16_traits, Cta_tile, NUM_ELTS> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace colw_with_2x2_pooling
} // namespace ext
} // namespace xmma

