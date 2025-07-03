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

#ifndef _XMMA_EXT_DEPTHWISE_COLWOLUTION_ABSTRACT_TILE_H
#define _XMMA_EXT_DEPTHWISE_COLWOLUTION_ABSTRACT_TILE_H

#pragma once

#include "utils.h"
#include <cstdint>
#include <lwda_runtime.h>

namespace xmma{
namespace ext
{
namespace depthwise_colwolution
{
// The NDHWG tensor
template <typename Tile_4d_per_cta_,
          typename Tile_4d_per_thread_,
          int32_t ELEMENTS_PER_INSTRUCTION_,
          int32_t THREADS_PER_CTA_>
struct Abstract_tile {
    public:
    using Tile_4d_per_cta_t = Tile_4d_per_cta_;
    using Tile_4d_per_thread_t = Tile_4d_per_thread_;
    static const int32_t ELEMENTS_PER_INSTRUCTION = ELEMENTS_PER_INSTRUCTION_;
    static const int32_t THREADS_PER_CTA = THREADS_PER_CTA_;
    // Derived
    static const int32_t THREAD_TILES_OF_DEPTH =
        CEIL_DIV(Tile_4d_per_cta_t::DEPTH, Tile_4d_per_thread_t::DEPTH);
    static const int32_t THREAD_TILES_OF_HEIGHT =
        CEIL_DIV(Tile_4d_per_cta_t::HEIGHT, Tile_4d_per_thread_t::HEIGHT);
    static const int32_t THREAD_TILES_OF_WIDTH =
        CEIL_DIV(Tile_4d_per_cta_t::WIDTH, Tile_4d_per_thread_t::WIDTH);
    static const int32_t THREAD_TILES_OF_GROUP =
        CEIL_DIV(Tile_4d_per_cta_t::GROUP, Tile_4d_per_thread_t::GROUP);
    static_assert(Tile_4d_per_cta_t::DEPTH % Tile_4d_per_thread_t::DEPTH == 0, "");
    static_assert(Tile_4d_per_cta_t::HEIGHT % Tile_4d_per_thread_t::HEIGHT == 0, "");
    static_assert(Tile_4d_per_cta_t::WIDTH % Tile_4d_per_thread_t::WIDTH == 0, "");
    static_assert(Tile_4d_per_cta_t::GROUP % Tile_4d_per_thread_t::GROUP == 0, "");

    static const int32_t THREAD_TILES_PER_CTA = THREAD_TILES_OF_DEPTH * THREAD_TILES_OF_HEIGHT *
                                                THREAD_TILES_OF_WIDTH * THREAD_TILES_OF_GROUP;

    static const int32_t TILES_PER_THREAD = CEIL_DIV(THREAD_TILES_PER_CTA, THREADS_PER_CTA);

    static const int32_t INSTRUCTIONS_PER_GROUP_OF_THE_THREAD_TILE =
        CEIL_DIV(Tile_4d_per_thread_t::GROUP, ELEMENTS_PER_INSTRUCTION);

    static const int32_t INSTRUCTIONS_PER_THREAD_TILE =
        Tile_4d_per_thread_t::DEPTH * Tile_4d_per_thread_t::HEIGHT * Tile_4d_per_thread_t::WIDTH *
        INSTRUCTIONS_PER_GROUP_OF_THE_THREAD_TILE;

    static const int32_t VALID_INSTRUCTIONS_PER_CTA = THREAD_TILES_OF_GROUP *
                                                      INSTRUCTIONS_PER_GROUP_OF_THE_THREAD_TILE *
                                                      Tile_4d_per_cta_t::Tile_3d_t::VALUE;

    static const int32_t INSTRUCTIONS_PER_THREAD = TILES_PER_THREAD * INSTRUCTIONS_PER_THREAD_TILE;

    __device__ inline Abstract_tile()
    {
        initialize_base_offset();
        initialize_offset();
    }

    __device__ inline void initialize_base_offset()
    {
        int32_t linear_tid = threadIdx_x();
        linear_mapping<TILES_PER_THREAD,
                       THREADS_PER_CTA,
                       THREAD_TILES_OF_HEIGHT,
                       THREAD_TILES_OF_WIDTH,
                       THREAD_TILES_OF_GROUP>(base_offset_depth_,
                                              base_offset_height_,
                                              base_offset_width_,
                                              base_offset_group_,
                                              linear_tid);
#pragma unroll
        for (int32_t i = 0; i < TILES_PER_THREAD; ++i) {
            base_offset_depth_[i] *= Tile_4d_per_thread_t::DEPTH;
            base_offset_height_[i] *= Tile_4d_per_thread_t::HEIGHT;
            base_offset_width_[i] *= Tile_4d_per_thread_t::WIDTH;
            base_offset_group_[i] *= Tile_4d_per_thread_t::GROUP;
        }
    }

    __device__ inline void initialize_offset()
    {
#pragma unroll
        for (int32_t index_tile = 0; index_tile < TILES_PER_THREAD; ++index_tile) {
            initialize_offset(base_offset_depth_,
                              base_offset_height_,
                              base_offset_width_,
                              base_offset_group_,
                              index_tile);
        }
    }

    __device__ inline void initialize_offset(int32_t base_offset_depth[TILES_PER_THREAD],
                                             int32_t base_offset_height[TILES_PER_THREAD],
                                             int32_t base_offset_width[TILES_PER_THREAD],
                                             int32_t base_offset_group[TILES_PER_THREAD],
                                             const int32_t index_tile)
    {
        for (int32_t index_depth = 0; index_depth < Tile_4d_per_thread_t::DEPTH; ++index_depth) {
            for (int32_t index_height = 0; index_height < Tile_4d_per_thread_t::HEIGHT;
                 ++index_height) {
                for (int32_t index_width = 0; index_width < Tile_4d_per_thread_t::WIDTH;
                     ++index_width) {
                    for (int32_t index_group = 0;
                         index_group < INSTRUCTIONS_PER_GROUP_OF_THE_THREAD_TILE;
                         ++index_group) {
                        int32_t index_instruction =
                            index_tile * Tile_4d_per_thread_t::DEPTH *
                                Tile_4d_per_thread_t::HEIGHT * Tile_4d_per_thread_t::WIDTH *
                                INSTRUCTIONS_PER_GROUP_OF_THE_THREAD_TILE +
                            index_depth * Tile_4d_per_thread_t::HEIGHT *
                                Tile_4d_per_thread_t::WIDTH *
                                INSTRUCTIONS_PER_GROUP_OF_THE_THREAD_TILE +
                            index_height * Tile_4d_per_thread_t::WIDTH *
                                INSTRUCTIONS_PER_GROUP_OF_THE_THREAD_TILE +
                            index_width * INSTRUCTIONS_PER_GROUP_OF_THE_THREAD_TILE + index_group;
                        offset_depth_[index_instruction] =
                            base_offset_depth[index_tile] + index_depth;
                        offset_height_[index_instruction] =
                            base_offset_height[index_tile] + index_height;
                        offset_width_[index_instruction] =
                            base_offset_width[index_tile] + index_width;
                        offset_group_[index_instruction] =
                            base_offset_group[index_tile] + index_group * ELEMENTS_PER_INSTRUCTION;
                    }
                }
            }
        }
    }

    __device__ inline void expose_base_state(int32_t (&base_offset_depth)[TILES_PER_THREAD],
                                             int32_t (&base_offset_height)[TILES_PER_THREAD],
                                             int32_t (&base_offset_width)[TILES_PER_THREAD])
    {
#pragma unroll
        for (int32_t i = 0; i < TILES_PER_THREAD; ++i) {
            base_offset_depth[i] = base_offset_depth_[i];
            base_offset_height[i] = base_offset_height_[i];
            base_offset_width[i] = base_offset_width_[i];
        }
    }

    __device__ inline void expose_state(int32_t (&offset_depth)[INSTRUCTIONS_PER_THREAD_TILE],
                                        int32_t (&offset_height)[INSTRUCTIONS_PER_THREAD_TILE],
                                        int32_t (&offset_width)[INSTRUCTIONS_PER_THREAD_TILE],
                                        const int32_t index_tile)
    {
#pragma unroll
        for (int32_t i = 0; i < INSTRUCTIONS_PER_THREAD_TILE; ++i) {
            offset_depth[i] = offset_depth_[index_tile * INSTRUCTIONS_PER_THREAD_TILE + i];
            offset_height[i] = offset_height_[index_tile * INSTRUCTIONS_PER_THREAD_TILE + i];
            offset_width[i] = offset_width_[index_tile * INSTRUCTIONS_PER_THREAD_TILE + i];
        }
    }

    __device__ inline void expose_state(int32_t (&offset_depth)[INSTRUCTIONS_PER_THREAD],
                                        int32_t (&offset_height)[INSTRUCTIONS_PER_THREAD],
                                        int32_t (&offset_width)[INSTRUCTIONS_PER_THREAD])
    {
#pragma unroll
        for (int32_t i = 0; i < INSTRUCTIONS_PER_THREAD; ++i) {
            offset_depth[i] = offset_depth_[i];
            offset_height[i] = offset_height_[i];
            offset_width[i] = offset_width_[i];
        }
    }

    __device__ inline void expose_state(int32_t (&offset_depth)[INSTRUCTIONS_PER_THREAD],
                                        int32_t (&offset_height)[INSTRUCTIONS_PER_THREAD],
                                        int32_t (&offset_width)[INSTRUCTIONS_PER_THREAD],
                                        int32_t (&offset_group)[INSTRUCTIONS_PER_THREAD])
    {
        expose_state(offset_depth, offset_height, offset_width);
#pragma unroll
        for (int32_t i = 0; i < INSTRUCTIONS_PER_THREAD; ++i) {
            offset_group[i] = offset_group_[i];
        }
    }

    int32_t base_offset_depth_[TILES_PER_THREAD];
    int32_t base_offset_height_[TILES_PER_THREAD];
    int32_t base_offset_width_[TILES_PER_THREAD];
    int32_t base_offset_group_[TILES_PER_THREAD];

    int32_t offset_depth_[INSTRUCTIONS_PER_THREAD];
    int32_t offset_height_[INSTRUCTIONS_PER_THREAD];
    int32_t offset_width_[INSTRUCTIONS_PER_THREAD];
    int32_t offset_group_[INSTRUCTIONS_PER_THREAD];
};

} // namespace depthwise_colwolution
} // namespace ext
} // namespace xmma

#endif
