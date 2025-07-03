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
#include <xmma/implicit_gemm/strided_dgrad_indexed/gmem_tile.h>

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {
namespace implicit_gemm {
namespace strided_dgrad_indexed {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits,
          typename Cta_tile,
          typename Fragment_c = xmma::Fragment_c<Traits, Cta_tile>>
struct Warp_specialized_gmem_tile_epilogue
    : public xmma::implicit_gemm::strided_dgrad_indexed::Gmem_tile_epilogue<Traits,
                                                                            Cta_tile,
                                                                            Fragment_c> {

    using Base_ = xmma::implicit_gemm::strided_dgrad_indexed::Gmem_tile_epilogue<Traits,
                                                                                 Cta_tile,
                                                                                 Fragment_c>;
    // The number of bytes per STG.
    enum { BYTES_PER_STG = Base_::BYTES_PER_STG };

    // Ctor.
    template <typename Params>
    inline __device__ Warp_specialized_gmem_tile_epilogue(
        const Params& params,
        int* cta_ndhw_indices,
        int cta_id_in_dhw_dimension,
        int bidn,
        int tidx,
        xmma::Named_barrier intra_epilog_group_sync = xmma::Named_barrier() )
        : Base_( params, cta_ndhw_indices, bidn, tidx ) {

        const int DHW_PER_CTA = Cta_tile::M;
        const int DHW_PER_THREAD =
            ( DHW_PER_CTA + Cta_tile::THREADS_PER_CTA - 1 ) / Cta_tile::THREADS_PER_CTA;

        for( int i = 0; i < DHW_PER_THREAD; ++i ) {
            int dhw_index = tidx + i * Cta_tile::THREADS_PER_CTA;
            if( dhw_index < Cta_tile::M ) {
                cta_ndhw_indices[dhw_index] =
                    params.ndhw_indices_of_each_filter_pattern_gmem[cta_id_in_dhw_dimension *
                                                                        DHW_PER_CTA +
                                                                    dhw_index];
            }
        }

        // For cta_ndhw_indices
        if( intra_epilog_group_sync.invalid() ) {
            __syncthreads();
        } else {
            intra_epilog_group_sync.wait();
        }
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace strided_dgrad_indexed
}  // namespace implicit_gemm
}  // namespace xmma
