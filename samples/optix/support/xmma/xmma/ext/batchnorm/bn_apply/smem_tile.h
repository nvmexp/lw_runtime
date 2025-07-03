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
namespace ext {
namespace batchnorm{
namespace bn_apply {

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits,
        typename Cta_tile,
        typename Row_Col,
        int BYTES_PER_STS,
        int BUFFERS_PER_TILE,
        int LDGS,
        uint32_t OFFSET = 0,
        int STAGES = 1>
struct Smem_tile_a_dbna_dgrad
    : public xmma::Smem_tile_a<Traits, Cta_tile, Row_Col, BYTES_PER_STS, BUFFERS_PER_TILE> {
};

template< typename Traits,
        typename Cta_tile,
        typename Row_Col,
        int BYTES_PER_STS,
        int BUFFERS_PER_TILE,
        // Do we have residual tensor add in the mainloop
        bool WITH_RESIDUAL,
        // Does residual add needs its own BN_Apply
        bool WITH_BNA_RESIDUAL,
        int LDGS,
        uint32_t OFFSET = 0,
        int STAGES = 1,
        // Does we have RELU at the end of BNa or dual_BNa
        bool WITH_RELU = true,
        bool SIMPLE_1x1x1 = false, 
        bool WITH_BITMASK_RELU_WRITE = false >
struct Smem_tile_a
    : public xmma::Smem_tile_a<Traits, Cta_tile, Row_Col, BYTES_PER_STS, BUFFERS_PER_TILE> {
};

template< typename Traits,
        typename Cta_tile,
        typename Row_Col,
        int BYTES_PER_STS,
        int BUFFERS_PER_TILE,
        bool WITH_RELU = true >
struct Smem_tile_b
    : public xmma::Smem_tile_b<Traits, Cta_tile, Row_Col, BYTES_PER_STS, BUFFERS_PER_TILE> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, 
        typename Cta_tile, 
        typename Row_Col, 
        int BYTES_PER_STS, 
        int BUFFERS_PER_TILE,
        int LDGS, 
        bool WITH_FUSED_DBNA_DGRAD,
        uint32_t OFFSET > 
struct Smem_tile_a_wgrad
    : public xmma::Smem_tile_a<Traits, Cta_tile, Row_Col, BYTES_PER_STS, BUFFERS_PER_TILE> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////



} // namespace bn_apply
} // namespace batchnorm
} // namespace ext
} // namespace xmma
////////////////////////////////////////////////////////////////////////////////////////////////////
