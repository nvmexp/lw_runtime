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

#include <xmma/ext/batchnorm/bn_apply/fprop/params.h>
#include <xmma/ext/batchnorm/bn_apply/volta/gmem_tile.h>
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {
namespace ext {
namespace batchnorm {
namespace bn_apply {

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  GMEM TILE A
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, typename Input_related, bool WITH_RESIDUAL, int STAGES, bool WITH_RELU>
struct Gmem_tile_a_volta_turing<xmma::Turing_hmma_fp32_traits, Cta_tile, Input_related,
                                WITH_RESIDUAL, STAGES, WITH_RELU>
    : public Gmem_tile_a_base<xmma::Turing_hmma_fp32_traits, Cta_tile, Input_related,
                              WITH_RESIDUAL, STAGES, WITH_RELU> {

    // The base class.
    using Base_gmem = Gmem_tile_a_base<xmma::Turing_hmma_fp32_traits, Cta_tile, Input_related,
                                       WITH_RESIDUAL, STAGES, WITH_RELU>;

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_a_volta_turing(const Params &params, void *smem, const dim3 &bidx,
                                               int tidx)
        : Base_gmem(params, smem, bidx, tidx) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  GMEM TILE B
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, bool SIMPLE_1x1x1, bool WITH_RELU>
struct Gmem_tile_b_volta_turing<xmma::Turing_hmma_fp32_traits, Cta_tile, SIMPLE_1x1x1, WITH_RELU>
    : public Gmem_tile_b_base<xmma::Turing_hmma_fp32_traits, Cta_tile, SIMPLE_1x1x1, WITH_RELU> {

    // The base class.
    using Base_gmem = Gmem_tile_b_base<xmma::Turing_hmma_fp32_traits, Cta_tile, SIMPLE_1x1x1, 
                                        WITH_RELU>;

    // Ctor.
    template <typename Params>
    inline __device__ Gmem_tile_b_volta_turing(const Params &params, void *smem, const dim3 &bidx,
                                               int tidx)
        : Base_gmem(params, smem, bidx, tidx) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace bn_apply
} // namespace batchnorm
} // namespace ext
} // namespace xmma
////////////////////////////////////////////////////////////////////////////////////////////////////
