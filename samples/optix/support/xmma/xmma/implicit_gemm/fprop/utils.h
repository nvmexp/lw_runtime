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

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {
namespace implicit_gemm {
namespace fprop {

///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Params >
int compute_grid_dimensions(dim3 &grid,
                            Params &params,
                            int tile_m,
                            int tile_n,
                            bool without_l1_replication = false) {
    if ( without_l1_replication ) {
        const int n_per_cta = 1;
        const int p_per_cta = tile_m == 256 ? 16 :  8;
        const int q_per_cta = tile_m ==  64 ?  8 : 16;

        const int cta_n = xmma::div_up(params.n, n_per_cta);
        const int cta_p = xmma::div_up(params.p, p_per_cta);
        const int cta_q = xmma::div_up(params.q, q_per_cta);

        params.tiles_m = cta_n * cta_p * cta_q;
        params.tiles_n = xmma::div_up(params.k, tile_n);
    } else {
        params.tiles_m = xmma::div_up(params.n * params.o * params.p * params.q, tile_m);
        params.tiles_n = xmma::div_up(params.k * params.g, tile_n);
    }
    params.tiles_k = params.split_k.slices;

    if ( params.use_horizontal_cta_rasterization ) {
        grid.y = params.tiles_m;
        grid.x = params.tiles_n;
        params.tiles_y = params.tiles_m;
        params.tiles_x = params.tiles_n;
    } else {
        grid.x = params.tiles_m;
        grid.y = params.tiles_n;
        params.tiles_x = params.tiles_m;
        params.tiles_y = params.tiles_n;
    }
    grid.z = params.tiles_k;
    return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fprop
} // namespace implicit_gemm
} // namespace xmma
///////////////////////////////////////////////////////////////////////////////////////////////////
