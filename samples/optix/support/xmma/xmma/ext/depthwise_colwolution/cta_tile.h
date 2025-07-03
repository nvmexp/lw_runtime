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

#ifndef _XMMA_EXT_DEPTHWISE_COLWOLUTION_CTA_TILE_H
#define _XMMA_EXT_DEPTHWISE_COLWOLUTION_CTA_TILE_H

#pragma once

#include <cstdint>
#include <type_traits>

namespace xmma
{
namespace ext
{
namespace depthwise_colwolution
{
template <int32_t N> struct Dim {
    public:
    static_assert(N > 0, "");
    int32_t dim[N];
};

template <int32_t DEPTH_, int32_t HEIGHT_, int32_t WIDTH_> struct Tile_3d {
    public:
    static const int32_t DEPTH = DEPTH_;
    static const int32_t HEIGHT = HEIGHT_;
    static const int32_t WIDTH = WIDTH_;
    static const int32_t VALUE = DEPTH * HEIGHT * WIDTH;
    static const bool IS_POSITIVE = (DEPTH > 0 && HEIGHT > 0 && WIDTH > 0);
};

template <typename Tile_3d_, int32_t GROUP_> struct Tile_4d {
    public:
    using Tile_3d_t = Tile_3d_;
    static const int32_t DEPTH = Tile_3d_t::DEPTH;
    static const int32_t HEIGHT = Tile_3d_t::HEIGHT;
    static const int32_t WIDTH = Tile_3d_t::WIDTH;
    static const int32_t GROUP = GROUP_;
};

// struct Tile_3d : Tile_dhw_, Tile_opq_, Tile_trs_, Tile_stride_dhw_,
// Tile_dilation_dhw_
template <typename Tile_dhw_,
          typename Tile_opq_,
          typename Tile_trs_,
          typename Tile_stride_dhw_,
          typename Tile_dilation_dhw_,
          int32_t TILE_G_>
struct Tile_memory_per_cta {
    using Tile_dhw_t = Tile_dhw_;
    using Tile_opq_t = Tile_opq_;
    using Tile_trs_t = Tile_trs_;
    using Tile_stride_dhw_t = Tile_stride_dhw_;
    using Tile_dilation_dhw_t = Tile_dilation_dhw_;
    static_assert(Tile_stride_dhw_t::IS_POSITIVE == Tile_dilation_dhw_t::IS_POSITIVE, "");
    static const int32_t TILE_G = TILE_G_;
};

template <typename Tile_dhw_,
          typename Tile_opq_,
          typename Tile_trs_,
          typename Tile_stride_dhw_,
          typename Tile_dilation_dhw_,
          int32_t TILE_G_>
using Tile_math_per_thread = Tile_memory_per_cta<Tile_dhw_,
                                                 Tile_opq_,
                                                 Tile_trs_,
                                                 Tile_stride_dhw_,
                                                 Tile_dilation_dhw_,
                                                 TILE_G_>;

template <typename Tile_memory_per_cta_,
          typename Tile_math_per_thread_,
          int32_t STAGE_,
          int32_t THREADS_PER_WARP_,
          int32_t WARPS_PER_CTA_>
struct Cta_tile {
    using Tile_memory_per_cta_t = Tile_memory_per_cta_;
    using Tile_math_per_thread_t = Tile_math_per_thread_;
    static const int32_t STAGE = STAGE_;
    static_assert(STAGE >= 2, "");
    static const int32_t THREADS_PER_WARP = THREADS_PER_WARP_;
    static const int32_t WARPS_PER_CTA = WARPS_PER_CTA_;
    using Tile_trs_t = typename Tile_memory_per_cta_t::Tile_trs_t;
    using Tile_stride_dhw_t = typename Tile_memory_per_cta_t::Tile_stride_dhw_t;
    using Tile_dilation_dhw_t = typename Tile_memory_per_cta_t::Tile_dilation_dhw_t;
    static const int32_t BYTES_PER_SMEM_LINE = 128;
    // Derived
    static const int32_t THREADS_PER_CTA = THREADS_PER_WARP * WARPS_PER_CTA;
};

} // namespace depthwise_colwolution
} // namespace ext
} // namespace xmma

#endif
