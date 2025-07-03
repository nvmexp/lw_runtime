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
#include <xmma/smem_tile.h>
#include <xmma/helpers/epilogue.h>
#include <xmma/ext/first_layer/fprop/params.h>
#include <xmma/ext/first_layer/fprop/gmem_tile.h>
#include <xmma/ext/first_layer/fprop/smem_tile.h>

namespace xmma {
namespace ext {
namespace first_layer {
namespace fprop {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits_,
    // The size of the output tile in the depth dimension.
    int OUT_D_,
    // The size of the output tile in the vertical dimension (height).
    int OUT_H_,
    // The size of the output tile in the horizontal dimension (width).
    int OUT_W_,
    // The number of filters.
    int FLT_K_,
    // The size of the filter in the depth dimension.
    int FLT_T_,
    // The size of the filter in the vertical dimension.
    int FLT_R_,
    // The size of the filter in the horizontal dimension.
    int FLT_S_,
    // The number of warps in the M dimension.
    int WARPS_M_
>
struct Kernel_traits_cfg : public Traits_ {

    // We do not support 3D tiles for the moment.
    static_assert(OUT_D_ == 1 && FLT_T_ == 1, "To be implemented");

    // The dimension of the output tile.
    enum { OUT_D = OUT_D_ };
    enum { OUT_H = OUT_H_ };
    enum { OUT_W = OUT_W_ };

    // The dimension of the filters.
    enum { FLT_K = FLT_K_ };
    enum { FLT_T = FLT_T_ }; 
    enum { FLT_R = FLT_R_ }; 
    enum { FLT_S = FLT_S_ };

    // The padding.
    enum { PAD_D = FLT_T / 2 };
    enum { PAD_H = FLT_R / 2 };
    enum { PAD_W = FLT_S / 2 };

    // The strides.
    enum { STRIDE_D = 2 };
    enum { STRIDE_H = 2 };
    enum { STRIDE_W = 2 };

    // The dimensions of the input tile.
    enum { IMG_D = OUT_D * STRIDE_D + 2*PAD_D - 1 };
    enum { IMG_H = OUT_H * STRIDE_H + 2*PAD_H - 1 };
    enum { IMG_W = OUT_W * STRIDE_W + 2*PAD_W - 1 };

    // The number of channels per pixel/tap is 4 (as we deal with NHW4).
    enum { CHANNELS_PER_PIXEL = 4 };
    // The number of taps per XMMA.
    enum { TAPS_PER_XMMA_K = 2 };
    // The number of filter rows consummed per iteration.
    enum { FLT_R_PER_ITERATION = 2 };
    // The number of XMMA loops to produce OUT_H output rows.
    enum { INNER_LOOPS = xmma::Div_up<FLT_R, FLT_R_PER_ITERATION>::VALUE };

    // The number of input rows loaded in the prologue.
    enum { IMG_H_IN_PROLOGUE = OUT_H * STRIDE_H };
    // The number of input rows loaded in the inner loop to prepare for the next outer loop.
    enum { IMG_H_PER_INNER_LOOP = xmma::Div_up<IMG_H_IN_PROLOGUE, INNER_LOOPS>::VALUE };

    // The dimensions of the CTA tile.
    enum { TILE_M = OUT_H * OUT_W };
    enum { TILE_N = FLT_K };
    enum { TILE_K = FLT_S * FLT_R_PER_ITERATION * CHANNELS_PER_PIXEL };

    // The number of warps in the 3 dimensions of the CTA tile.
    enum { WARPS_M = WARPS_M_ };
    enum { WARPS_N = 1 };
    enum { WARPS_K = 1 };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The instruction traits.
    typename Traits_,
    // The size of the output tile in the depth dimension.
    int OUT_D_,
    // The size of the output tile in the vertical dimension (height).
    int OUT_H_,
    // The size of the output tile in the horizontal dimension (width).
    int OUT_W_,
    // The number of filters.
    int FLT_K_,
    // The size of the filter in the depth dimension.
    int FLT_T_,
    // The size of the filter in the vertical dimension.
    int FLT_R_,
    // The size of the filter in the horizontal dimension.
    int FLT_S_,
    // The number of warps in the M dimension.
    int WARPS_M_
>
struct Kernel_traits : public Kernel_traits_cfg<Traits_, 
                                                OUT_D_,
                                                OUT_H_,
                                                OUT_W_,
                                                FLT_K_,
                                                FLT_T_,
                                                FLT_R_,
                                                FLT_S_,
                                                WARPS_M_> {
    // The base class.
    using Cfg = Kernel_traits_cfg<Traits_, 
                                  OUT_D_, 
                                  OUT_H_, 
                                  OUT_W_, 
                                  FLT_K_, 
                                  FLT_T_, 
                                  FLT_R_, 
                                  FLT_S_,
                                  WARPS_M_>; 

    // The instruction traits.
    using Traits = Traits_;
    // The Cta tile. 
    using Cta_tile = typename Traits::template Cta_tile_extd<Cfg::TILE_M,
                                                             Cfg::TILE_N,
                                                             Cfg::TILE_K,
                                                             Cfg::WARPS_M,
                                                             Cfg::WARPS_N,
                                                             Cfg::WARPS_K,
                                                             1>;
    // The XMMA tile.
    using Xmma_tile = typename Traits::template Xmma_tile<Cta_tile>;

    // The dimension of the output tile.
    enum { OUT_D = OUT_D_ };
    enum { OUT_H = OUT_H_ };
    enum { OUT_W = OUT_W_ };

    // The dimension of the filters.
    enum { FLT_K = FLT_K_ };
    enum { FLT_T = FLT_T_ }; 
    enum { FLT_R = FLT_R_ }; 
    enum { FLT_S = FLT_S_ };

    // The kernel parameters.
    using Params = Params<Traits, Cta_tile>;
    // The global memory loader to initialize shared memory in the prologue.
    using Gmem_tile_prologue_a = first_layer::fprop::Gmem_tile_a<Traits, Cta_tile, Cfg, true>;
    // The global memory loader to load data in the main while we iterate over rows.
    using Gmem_tile_loop_a = first_layer::fprop::Gmem_tile_a<Traits, Cta_tile, Cfg, false>;
    // The shared memory tile for A.
    using Smem_tile_a = first_layer::fprop::Smem_tile_a<Traits, Cta_tile, Cfg>;
    // The global memory loader for B.
    using Gmem_tile_prologue_b = first_layer::fprop::Gmem_tile_b<Traits, Cta_tile, Cfg>;
    // The shared memory tile for B.
    using Smem_tile_b = first_layer::fprop::Smem_tile_b<Traits, Cta_tile, Cfg>;
    // The global memory tile for C..
    using Gmem_tile_c = first_layer::fprop::Gmem_tile_c<Traits, Cta_tile, xmma::Row, Cfg>;
    // The shared memory epilogue tile.
    using Smem_tile_c = xmma::Swizzle_epilogue<Traits, Cta_tile, xmma::Row>;
    // The callbacks.
    using Callbacks_epilogue = xmma::helpers::Empty_callbacks_epilogue<Traits, Cta_tile>;
    // The epilogue.
    using Epilogue = xmma::helpers::Epilogue<Traits, 
                                                 Cta_tile, 
                                                 xmma::Row,
                                                 Gmem_tile_c, 
                                                 Callbacks_epilogue>;
    // The number of threads per CTA.
    static int threads_per_cta() {
        return Cta_tile::THREADS_PER_CTA;
    }

    // The amount of dynamic shared memory per CTA.
    static int dynamic_smem_size_per_cta() {
        return Smem_tile_a::BYTES_PER_TILE + 
               Smem_tile_b::BYTES_PER_TILE + 
               Smem_tile_c::BYTES_PER_TILE;
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fprop
} // namespace first_layer 
} // namespace ext
} // namespace xmma 

