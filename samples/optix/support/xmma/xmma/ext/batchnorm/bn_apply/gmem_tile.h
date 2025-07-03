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

#include <xmma/implicit_gemm/fprop/gmem_tile.h>
#include <xmma/implicit_gemm/dgrad/gmem_tile.h>
#include <xmma/implicit_gemm/wgrad_indexed/gmem_tile.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {
namespace ext {
namespace batchnorm {
namespace bn_apply {

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits,
          typename Cta_tile,
          typename Input_related,
          // Do we have residual tensor add in the mainloop
          bool WITH_RESIDUAL,
          // Does residual add needs its own BN_Apply
          bool WITH_BNA_RESIDUAL,
          int STAGES=1 >
struct Gmem_tile_a
    : public xmma::implicit_gemm::fprop::Gmem_tile_a_t<Traits, Cta_tile, Input_related> {
};

template< typename Traits,
        typename Cta_tile,
        typename Input_related,
        int STAGES=1 >
struct Gmem_tile_a_dbna_dgrad
    : public xmma::implicit_gemm::dgrad::Gmem_tile_a_t<Traits, Cta_tile, Input_related> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Base class for ampere
template< typename Traits,
          typename Cta_tile,
          bool SIMPLE_1x1x1>
struct Gmem_tile_b
    : public xmma::implicit_gemm::wgrad_indexed::Gmem_tile_b_t<Traits, Cta_tile, SIMPLE_1x1x1> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits,
          typename Cta_tile,
          bool SIMPLE_1x1x1,
          bool WITH_WITH_FUSED_DBNA_DGRAD>
struct Gmem_tile_a_wgrad
    : public xmma::implicit_gemm::wgrad_indexed::Gmem_tile_a_n<Traits, Cta_tile> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace bn_apply
} // namespace batchnorm
} // namespace ext
} // namespace xmma
////////////////////////////////////////////////////////////////////////////////////////////////////
