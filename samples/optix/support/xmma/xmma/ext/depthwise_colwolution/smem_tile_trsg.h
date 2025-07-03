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

#ifndef _XMMA_EXT_DEPTHWISE_COLWOLUTION_SMEM_TILE_TRSG
#define _XMMA_EXT_DEPTHWISE_COLWOLUTION_SMEM_TILE_TRSG

#pragma once

#include "cta_tile.h"
#include "params.h"
#include "utils.h"
#include "xmma/utils.h"
#include <cstdint>
#include <lwda_runtime.h>

namespace xmma
{
namespace ext
{
namespace depthwise_colwolution
{
template <typename Tile_3d_,
          int32_t TILE_G_,
          int32_t BYTES_PER_ELEMENT_,
          int32_t TILE_G_PER_THREAD_,
          int32_t THREADS_PER_CTA_>
struct Smem_tile_trsg {
    public:
    using Tile_3d_t = Tile_3d_;
    static const int32_t TILE_G = TILE_G_;
    static const int32_t BYTES_PER_ELEMENT = BYTES_PER_ELEMENT_;
    static const int32_t TILE_G_PER_THREAD = TILE_G_PER_THREAD_;
    static const int32_t SMEM_WIDTH = Tile_3d_t::WIDTH;
    static const int32_t SMEM_HEIGHT = Tile_3d_t::HEIGHT;
    static const int32_t SMEM_DEPTH = Tile_3d_t::DEPTH;
    static const int32_t BYTES_PER_UINT32 = 4;
    static const int32_t THREADS_PER_CTA = THREADS_PER_CTA_;
    // Derived
    static const int32_t BYTES_PER_INSTRUCTION_IN_SMEM = BYTES_PER_ELEMENT * TILE_G_PER_THREAD;
    using Packed_data_type_in_smem_t =
        typename xmma::Uint_from_size_in_bytes<BYTES_PER_INSTRUCTION_IN_SMEM>::Type;
    static const int32_t UINT32_PER_LDS = BYTES_PER_INSTRUCTION_IN_SMEM / BYTES_PER_UINT32;
    static_assert(UINT32_PER_LDS > 0, "");
    static const int32_t BYTES_PER_THREAD_PER_LDS = BYTES_PER_INSTRUCTION_IN_SMEM;
    static const int32_t BYTES_PER_TILE_G = TILE_G * BYTES_PER_ELEMENT;
    static const int32_t THREADS_PER_TILE_G = BYTES_PER_TILE_G / BYTES_PER_INSTRUCTION_IN_SMEM;
    static_assert(THREADS_PER_TILE_G > 0, "");
    static const int32_t LDS_PER_THREAD = SMEM_DEPTH * SMEM_HEIGHT * SMEM_WIDTH;
    static_assert(LDS_PER_THREAD > 0, "");
    static const int32_t G_PER_THREAD =
        BYTES_PER_THREAD_PER_LDS / BYTES_PER_ELEMENT *
        ((THREADS_PER_TILE_G + THREADS_PER_CTA - 1) / THREADS_PER_CTA);
    static_assert(G_PER_THREAD > 0, "");
    static const int32_t UINT32_PER_TILE_G_PER_THREAD =
        G_PER_THREAD * BYTES_PER_ELEMENT / BYTES_PER_UINT32;
    static_assert(UINT32_PER_TILE_G_PER_THREAD > 0, "");
    static_assert(G_PER_THREAD * BYTES_PER_ELEMENT ==
                      UINT32_PER_TILE_G_PER_THREAD * BYTES_PER_UINT32,
                  "");
    static const int32_t DEPTH_PER_THREAD_PER_ITERATION = SMEM_DEPTH;
    static const int32_t HEIGHT_PER_THREAD_PER_ITERATION = SMEM_HEIGHT;
    static const int32_t WIDTH_PER_THREAD_PER_ITERATION = SMEM_WIDTH;
    static const int32_t MATH_TILE_G = G_PER_THREAD;
    static const int32_t MATH_TILE_DEPTH = DEPTH_PER_THREAD_PER_ITERATION;
    static const int32_t MATH_TILE_HEIGHT = HEIGHT_PER_THREAD_PER_ITERATION;
    static const int32_t MATH_TILE_WIDTH = WIDTH_PER_THREAD_PER_ITERATION;

    __device__ inline Smem_tile_trsg(const uint32_t smem_base_address)
    {
        int32_t tid = threadIdx_x();
        int32_t tid_in_the_g_dimention = tid % THREADS_PER_TILE_G;
        ptr_base_lds_ = smem_base_address + tid_in_the_g_dimention * BYTES_PER_INSTRUCTION_IN_SMEM;
    }

    __device__ inline void set_ptrs_lds()
    {
#pragma unroll
        for (int32_t index_depth = 0; index_depth < SMEM_DEPTH; ++index_depth) {
#pragma unroll
            for (int32_t index_height = 0; index_height < SMEM_HEIGHT; ++index_height) {
#pragma unroll
                for (int32_t index_width = 0; index_width < SMEM_WIDTH; ++index_width) {
                    int32_t index_lds = index_width + index_height * SMEM_WIDTH +
                                        index_depth * SMEM_WIDTH * SMEM_HEIGHT;
                    this->ptrs_lds_[index_lds] =
                        get_linear_index<SMEM_HEIGHT * SMEM_WIDTH * BYTES_PER_TILE_G,
                                         SMEM_WIDTH * BYTES_PER_TILE_G,
                                         BYTES_PER_TILE_G>(
                            this->ptr_base_lds_, index_depth, index_height, index_width);
                }
            }
        }
    }

    __device__ inline void load_from_smem()
    {
#pragma unroll
        for (int32_t index_lds = 0; index_lds < LDS_PER_THREAD; ++index_lds) {
            xmma::lds(data_of_smem_[index_lds], ptrs_lds_[index_lds]);
        }
        swizzle();
    }

    __device__ inline void swizzle()
    {
#pragma unroll
        for (int32_t index_depth = 0; index_depth < DEPTH_PER_THREAD_PER_ITERATION; ++index_depth) {
#pragma unroll
            for (int32_t index_height = 0; index_height < HEIGHT_PER_THREAD_PER_ITERATION;
                 ++index_height) {
#pragma unroll
                for (int32_t index_width = 0; index_width < WIDTH_PER_THREAD_PER_ITERATION;
                     ++index_width) {
#pragma unroll
                    for (int32_t index_group = 0; index_group < UINT32_PER_TILE_G_PER_THREAD;
                         ++index_group) {
                        int32_t index_linear =
                            index_group + UINT32_PER_TILE_G_PER_THREAD * index_width +
                            UINT32_PER_TILE_G_PER_THREAD * WIDTH_PER_THREAD_PER_ITERATION *
                                index_height +
                            UINT32_PER_TILE_G_PER_THREAD * WIDTH_PER_THREAD_PER_ITERATION *
                                HEIGHT_PER_THREAD_PER_ITERATION * index_depth;
                        int32_t index_in_a_lds = index_linear % UINT32_PER_LDS;
                        int32_t index_lds = index_linear / UINT32_PER_LDS;
                        fetch_math_[index_depth][index_height][index_width][index_group] =
                            select<Packed_data_type_in_smem_t>(data_of_smem_[index_lds],
                                                               index_in_a_lds);
                    }
                }
            }
        }
    }

    uint32_t ptr_base_lds_;
    uint32_t ptrs_lds_[LDS_PER_THREAD];
    Packed_data_type_in_smem_t data_of_smem_[LDS_PER_THREAD];
    uint32_t fetch_math_[DEPTH_PER_THREAD_PER_ITERATION][HEIGHT_PER_THREAD_PER_ITERATION]
                        [WIDTH_PER_THREAD_PER_ITERATION][UINT32_PER_TILE_G_PER_THREAD];
};

} // namespace depthwise_colwolution
} // namespace ext
} // namespace xmma

#endif
