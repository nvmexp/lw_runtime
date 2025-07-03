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
namespace ext {
namespace implicit_gemm {
namespace interleaved_fprop {

///////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Params >
int compute_grid_dimensions(dim3 &grid,
                            Params &params,
                            int tile_m,
                            int tile_n,
                            bool without_l1_replication = false) {

    params.tiles_m = xmma::div_up(params.k, tile_m) * params.g;
    params.tiles_n = xmma::div_up(params.n * params.o * params.p * params.q, tile_n);

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
    grid.z = 1;
    return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace interleaved_fprop
} // namespace implicit_gemm
} // namespace ext
} // namespace xmma

///////////////////////////////////////////////////////////////////////////////////////////////////
