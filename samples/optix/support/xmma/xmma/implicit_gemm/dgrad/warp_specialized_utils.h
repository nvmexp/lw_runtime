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
namespace dgrad {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Params>
int warp_specialized_compute_grid_dimensions( dim3& grid,
                                              Params& params,
                                              int tile_m,
                                              int tile_n,
                                              int tile_group ) {
    if( params.specialize > 0 ) {
        params.tiles_n =
            xmma::div_up( params.g, tile_group ) * xmma::div_up( params.c, tile_n / tile_group );
        params.tiles_m = xmma::div_up( params.n * params.d * params.h * params.w, tile_m );
        params.tiles_k = params.split_k.slices;
        params.tiles_mn = params.tiles_m * params.tiles_n;
        params.tiles_all = params.tiles_mn * params.split_k.slices;

        // Precomputed values for fast_div in persistent tile distributrion
        xmma::find_divisor( params.mul_grid_yx, params.shr_grid_yx, params.tiles_mn );
        xmma::find_divisor( params.mul_grid_x,
                            params.shr_grid_x,
                            params.use_horizontal_cta_rasterization ? params.tiles_n
                                                                    : params.tiles_m );

        // Warp specialization allocates 1 CTA/SM.
        grid.x =
            params.tile_move_step < params.tiles_all ? params.tile_move_step : params.tiles_all;
    } else {
        params.tiles_m = xmma::div_up( params.n * params.d * params.h * params.w, tile_m );
        params.tiles_n =
            xmma::div_up( params.c, tile_n / tile_group ) * xmma::div_up( params.g, tile_group );
        params.tiles_k = params.split_k.slices;
        if( params.use_horizontal_cta_rasterization ) {
            grid.y = params.tiles_m;
            grid.x = params.tiles_n;
        } else {
            grid.x = params.tiles_m;
            grid.y = params.tiles_n;
        }
        grid.z = params.tiles_k;
    }

    if( params.use_horizontal_cta_rasterization ) {
        params.tiles_y = params.tiles_m;
        params.tiles_x = params.tiles_n;
    } else {
        params.tiles_x = params.tiles_m;
        params.tiles_y = params.tiles_n;
    }
#ifdef DEBUG
    printf( "tile(%d %d) grid(%d %d %d) tiles_all:%d (%d %d) tile_move_step:%d\n",
            tile_m,
            tile_n,
            grid.x,
            grid.y,
            grid.z,
            params.tiles_all,
            params.tiles_n,
            params.tiles_m,
            params.tile_move_step );
#endif
    return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace dgrad
}  // namespace implicit_gemm
}  // namespace xmma
///////////////////////////////////////////////////////////////////////////////////////////////////
