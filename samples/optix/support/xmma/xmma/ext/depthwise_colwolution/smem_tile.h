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

#ifndef _XMMA_EXT_DEPTHWISE_COLWOLUTION_SMEM_TILE_H
#define _XMMA_EXT_DEPTHWISE_COLWOLUTION_SMEM_TILE_H

#pragma once

#include "abstract_tile.h"
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

template <int32_t SMEM_SIZE_IN_BYTES_PER_STAGE_,
          typename Tile_3d_,
          typename Tile_4d_per_cta_,
          typename Tile_4d_per_thread_,
          int32_t BYTES_PER_ELEMENT_,
          int32_t THREADS_PER_WARP_,
          int32_t WARPS_PER_CTA_,
          typename Tile_3d_per_thread_real_>
struct Smem_tile {
    public:
    static const int32_t SMEM_SIZE_IN_BYTES_PER_STAGE = SMEM_SIZE_IN_BYTES_PER_STAGE_;
    using Tile_3d_t = Tile_3d_;
    using Tile_4d_per_cta_t = Tile_4d_per_cta_;
    using Tile_4d_per_thread_t = Tile_4d_per_thread_;
    using Tile_3d_per_thread_real_t = Tile_3d_per_thread_real_;
    static const int32_t BYTES_PER_ELEMENT = BYTES_PER_ELEMENT_;
    static const int32_t THREADS_PER_WARP = THREADS_PER_WARP_;
    static const int32_t WARPS_PER_CTA = WARPS_PER_CTA_;
    static const int32_t NUMBER_OF_MATH_BUFFERS = 2;
    static const int32_t BYTES_PER_UINT32 = 4;
    // Derived
    static const int32_t TILE_G = Tile_4d_per_cta_t::GROUP;
    static const int32_t BYTES_PER_TILE_G = BYTES_PER_ELEMENT * TILE_G;
    static const int32_t BYTES_PER_INSTRUCTION_IN_SMEM =
        BYTES_PER_ELEMENT * Tile_4d_per_thread_t::GROUP;
    static const int32_t THREADS_PER_TILE_G = BYTES_PER_TILE_G / BYTES_PER_INSTRUCTION_IN_SMEM;
    static const int32_t THREADS_PER_CTA = THREADS_PER_WARP * WARPS_PER_CTA;
    static const int32_t ELEMENTS_PER_INSTRUCTION =
        BYTES_PER_INSTRUCTION_IN_SMEM / BYTES_PER_ELEMENT;
    using Abstract_tile_t = Abstract_tile<Tile_4d_per_cta_t,
                                          Tile_4d_per_thread_t,
                                          ELEMENTS_PER_INSTRUCTION,
                                          THREADS_PER_CTA>;

    using Tile_3d_per_thread_t = typename Tile_4d_per_thread_t::Tile_3d_t;
    static const int32_t LDSS_PER_THREAD = Abstract_tile_t::INSTRUCTIONS_PER_THREAD /
                                           Tile_3d_per_thread_t::VALUE *
                                           Tile_3d_per_thread_real_t::VALUE;
    static const int32_t ITERATION_NUMBER = Abstract_tile_t::TILES_PER_THREAD;
    static const int32_t LDSS_PER_THREAD_PER_ITERATION = LDSS_PER_THREAD / ITERATION_NUMBER;
    using Packed_data_type_in_smem_t =
        typename xmma::Uint_from_size_in_bytes<BYTES_PER_INSTRUCTION_IN_SMEM>::Type;
    static const int32_t DEPTH_PER_THREAD_PER_ITERATION = Tile_3d_per_thread_real_t::DEPTH;
    static const int32_t HEIGHT_PER_THREAD_PER_ITERATION = Tile_3d_per_thread_real_t::HEIGHT;
    static const int32_t WIDTH_PER_THREAD_PER_ITERATION = Tile_3d_per_thread_real_t::WIDTH;

    static_assert(Tile_4d_per_thread_t::GROUP * BYTES_PER_ELEMENT == BYTES_PER_INSTRUCTION_IN_SMEM,
                  "");
    static const int32_t UINT32_PER_INSTRUCTION_IN_SMEM =
        BYTES_PER_INSTRUCTION_IN_SMEM / BYTES_PER_UINT32;
    static const int32_t UINT32_PER_TILE_G_PER_THREAD = UINT32_PER_INSTRUCTION_IN_SMEM;

    static const int32_t G_PER_THREAD =
        BYTES_PER_INSTRUCTION_IN_SMEM / BYTES_PER_ELEMENT *
        ((THREADS_PER_TILE_G + THREADS_PER_CTA - 1) / THREADS_PER_CTA);
    static_assert(G_PER_THREAD > 0, "");

    static const int32_t MATH_TILE_G = G_PER_THREAD;
    static const int32_t MATH_TILE_DEPTH = DEPTH_PER_THREAD_PER_ITERATION;
    static const int32_t MATH_TILE_HEIGHT = HEIGHT_PER_THREAD_PER_ITERATION;
    static const int32_t MATH_TILE_WIDTH = WIDTH_PER_THREAD_PER_ITERATION;

    __device__ inline Smem_tile(const uint32_t smem_base_address)
    {
        int32_t tid = threadIdx_x();
        int32_t tid_in_the_g_dimention = tid % THREADS_PER_TILE_G;
        ptr_base_lds_ = smem_base_address + tid_in_the_g_dimention * BYTES_PER_INSTRUCTION_IN_SMEM;
    }

    __device__ inline void set_offsets_base_lds(int32_t (&offset_base_depth)[ITERATION_NUMBER],
                                                int32_t (&offset_base_height)[ITERATION_NUMBER],
                                                int32_t (&offset_base_width)[ITERATION_NUMBER]);

    __device__ inline void set_offsets_lds(const int32_t index_iteration);

    __device__ inline void set_ptrs_lds(const int32_t index_stage)
    {
        uint32_t ptr_base_lds_with_stage_offset =
            this->ptr_base_lds_ + index_stage * SMEM_SIZE_IN_BYTES_PER_STAGE;
        get_linear_index<LDSS_PER_THREAD_PER_ITERATION,
                         Tile_3d_t::HEIGHT * Tile_3d_t::WIDTH * BYTES_PER_TILE_G,
                         Tile_3d_t::WIDTH * BYTES_PER_TILE_G,
                         BYTES_PER_TILE_G>(this->ptrs_lds_,
                                           ptr_base_lds_with_stage_offset,
                                           this->offset_depth_,
                                           this->offset_height_,
                                           this->offset_width_);
    }

    __device__ inline void load_from_smem(const int32_t buffer_index)
    {
#pragma unroll
        for (int32_t index_lds = 0; index_lds < LDSS_PER_THREAD_PER_ITERATION; ++index_lds) {
            xmma::lds(data_of_smem_[buffer_index][index_lds], ptrs_lds_[index_lds]);
        }
        swizzle(buffer_index);
    }

    __device__ inline void swizzle(const int32_t buffer_index)
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
                        int32_t index_in_a_lds = index_linear % UINT32_PER_INSTRUCTION_IN_SMEM;
                        int32_t index_lds = index_linear / UINT32_PER_INSTRUCTION_IN_SMEM;
                        fetch_math_[buffer_index][index_depth][index_height][index_width]
                                   [index_group] = select<Packed_data_type_in_smem_t>(
                                       data_of_smem_[buffer_index][index_lds], index_in_a_lds);
                    }
                }
            }
        }
    }

    __device__ inline void
    de_swizzle(const int32_t buffer_index,
               uint32_t fetch_math[DEPTH_PER_THREAD_PER_ITERATION][HEIGHT_PER_THREAD_PER_ITERATION]
                                  [WIDTH_PER_THREAD_PER_ITERATION][UINT32_PER_TILE_G_PER_THREAD])
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
                        int32_t index_in_a_lds = index_linear % UINT32_PER_INSTRUCTION_IN_SMEM;
                        int32_t index_lds = index_linear / UINT32_PER_INSTRUCTION_IN_SMEM;
                        assign<Packed_data_type_in_smem_t>(data_of_smem_[buffer_index][index_lds],
                                                           index_in_a_lds,
                                                           fetch_math[index_depth][index_height]
                                                                     [index_width][index_group]);
                    }
                }
            }
        }
    }

    __device__ inline void store_to_smem(uint32_t fetch_math[DEPTH_PER_THREAD_PER_ITERATION]
                                                            [HEIGHT_PER_THREAD_PER_ITERATION]
                                                            [WIDTH_PER_THREAD_PER_ITERATION]
                                                            [UINT32_PER_TILE_G_PER_THREAD])
    {
        de_swizzle(0, fetch_math);
#pragma unroll
        for (int32_t index_lds = 0; index_lds < LDSS_PER_THREAD_PER_ITERATION; ++index_lds) {
            xmma::sts(ptrs_lds_[index_lds], data_of_smem_[0][index_lds]);
        }
    }

    uint32_t ptr_base_lds_;
    uint32_t ptrs_lds_[LDSS_PER_THREAD_PER_ITERATION];
    Packed_data_type_in_smem_t data_of_smem_[NUMBER_OF_MATH_BUFFERS][LDSS_PER_THREAD_PER_ITERATION];
    uint32_t fetch_math_[NUMBER_OF_MATH_BUFFERS][DEPTH_PER_THREAD_PER_ITERATION]
                        [HEIGHT_PER_THREAD_PER_ITERATION][WIDTH_PER_THREAD_PER_ITERATION]
                        [UINT32_PER_TILE_G_PER_THREAD];
    int32_t offset_base_depth_[ITERATION_NUMBER];
    int32_t offset_base_height_[ITERATION_NUMBER];
    int32_t offset_base_width_[ITERATION_NUMBER];
    int32_t offset_depth_[LDSS_PER_THREAD_PER_ITERATION];
    int32_t offset_height_[LDSS_PER_THREAD_PER_ITERATION];
    int32_t offset_width_[LDSS_PER_THREAD_PER_ITERATION];
};

template <int32_t SMEM_SIZE_IN_BYTES_PER_STAGE_,
          typename Tile_memory_per_cta_,
          typename Tile_math_per_thread_,
          int32_t BYTES_PER_ELEMENT_,
          int32_t THREADS_PER_WARP_,
          int32_t WARPS_PER_CTA_>
struct Smem_tile_ndhwg
    : public Smem_tile<
          SMEM_SIZE_IN_BYTES_PER_STAGE_,
          typename Tile_memory_per_cta_::Tile_dhw_t,
          Tile_4d<typename Tile_memory_per_cta_::Tile_opq_t, Tile_memory_per_cta_::TILE_G>,
          Tile_4d<typename Tile_math_per_thread_::Tile_opq_t, Tile_math_per_thread_::TILE_G>,
          BYTES_PER_ELEMENT_,
          THREADS_PER_WARP_,
          WARPS_PER_CTA_,
          typename Tile_math_per_thread_::Tile_dhw_t> {

    using Base_t = Smem_tile<
        SMEM_SIZE_IN_BYTES_PER_STAGE_,
        typename Tile_memory_per_cta_::Tile_dhw_t,
        Tile_4d<typename Tile_memory_per_cta_::Tile_opq_t, Tile_memory_per_cta_::TILE_G>,
        Tile_4d<typename Tile_math_per_thread_::Tile_opq_t, Tile_math_per_thread_::TILE_G>,
        BYTES_PER_ELEMENT_,
        THREADS_PER_WARP_,
        WARPS_PER_CTA_,
        typename Tile_math_per_thread_::Tile_dhw_t>;

    using Tile_memory_per_cta_t = Tile_memory_per_cta_;
    using Tile_dhw_t = typename Tile_memory_per_cta_t::Tile_dhw_t;
    using Tile_trs_t = typename Tile_memory_per_cta_t::Tile_trs_t;
    using Tile_stride_dhw_t = typename Tile_memory_per_cta_t::Tile_stride_dhw_t;
    using Tile_dilation_dhw_t = typename Tile_memory_per_cta_t::Tile_dilation_dhw_t;
    static const int32_t NUMBER_OF_MATH_BUFFERS = Base_t::NUMBER_OF_MATH_BUFFERS;
    static const int32_t ITERATION_NUMBER = Base_t::ITERATION_NUMBER;

    static const int32_t MATH_TILE_G = Base_t::MATH_TILE_G;
    static const int32_t MATH_TILE_DEPTH = Base_t::MATH_TILE_DEPTH;
    static const int32_t MATH_TILE_HEIGHT = Base_t::MATH_TILE_HEIGHT;
    static const int32_t MATH_TILE_WIDTH = Base_t::MATH_TILE_WIDTH;

    __device__ inline Smem_tile_ndhwg(const uint32_t smem_base_address,
                                      int32_t params_stride_depth,
                                      int32_t params_stride_height,
                                      int32_t params_stride_width,
                                      int32_t params_dilation_depth,
                                      int32_t params_dilation_height,
                                      int32_t params_dilation_width,
                                      bool params_is_colwolution)
        : Base_t(smem_base_address), params_stride_depth_(params_stride_depth),
          params_stride_height_(params_stride_height), params_stride_width_(params_stride_width),
          params_dilation_depth_(params_dilation_depth),
          params_dilation_height_(params_dilation_height),
          params_dilation_width_(params_dilation_width),
          params_is_colwolution_(params_is_colwolution)
    {
    }

    __device__ inline void set_offsets_base_lds(int32_t offset_base_depth[ITERATION_NUMBER],
                                                int32_t offset_base_height[ITERATION_NUMBER],
                                                int32_t offset_base_width[ITERATION_NUMBER])
    {
#pragma unroll
        for (int32_t index_iteration = 0; index_iteration < ITERATION_NUMBER; ++index_iteration) {
            this->offset_base_depth_[index_iteration] =
                multiply_and_add<Tile_stride_dhw_t::IS_POSITIVE,
                                 Tile_stride_dhw_t::DEPTH,
                                 Tile_dilation_dhw_t::DEPTH>(0,
                                                             params_stride_depth_,
                                                             offset_base_depth[index_iteration],
                                                             params_dilation_depth_,
                                                             0);
            this->offset_base_height_[index_iteration] =
                multiply_and_add<Tile_stride_dhw_t::IS_POSITIVE,
                                 Tile_stride_dhw_t::HEIGHT,
                                 Tile_dilation_dhw_t::HEIGHT>(0,
                                                              params_stride_height_,
                                                              offset_base_height[index_iteration],
                                                              params_dilation_height_,
                                                              0);
            this->offset_base_width_[index_iteration] =
                multiply_and_add<Tile_stride_dhw_t::IS_POSITIVE,
                                 Tile_stride_dhw_t::WIDTH,
                                 Tile_dilation_dhw_t::WIDTH>(0,
                                                             params_stride_width_,
                                                             offset_base_width[index_iteration],
                                                             params_dilation_width_,
                                                             0);
        }
    }

    __device__ inline void set_offsets_lds(const int32_t iteration_index)
    {
#pragma unroll
        for (int32_t index_depth = 0; index_depth < MATH_TILE_DEPTH; ++index_depth) {
            for (int32_t index_height = 0; index_height < MATH_TILE_HEIGHT; ++index_height) {
                for (int32_t index_width = 0; index_width < MATH_TILE_WIDTH; ++index_width) {
                    int32_t index_linear = index_depth * MATH_TILE_HEIGHT * MATH_TILE_WIDTH +
                                           index_height * MATH_TILE_WIDTH + index_width;
                    this->offset_depth_[index_linear] =
                        this->offset_base_depth_[iteration_index] + index_depth;
                    this->offset_height_[index_linear] =
                        this->offset_base_height_[iteration_index] + index_height;
                    this->offset_width_[index_linear] =
                        this->offset_base_width_[iteration_index] + index_width;
                }
            }
        }
    }

    __device__ inline void set_ptrs_lds(const int32_t index_stage)
    {
        Base_t::set_ptrs_lds(index_stage);
    }

    int32_t params_stride_depth_;
    int32_t params_stride_height_;
    int32_t params_stride_width_;
    int32_t params_dilation_depth_;
    int32_t params_dilation_height_;
    int32_t params_dilation_width_;
    bool params_is_colwolution_;
};

template <int32_t SMEM_SIZE_IN_BYTES_PER_STAGE_,
          typename Tile_memory_per_cta_,
          typename Tile_math_per_thread_,
          int32_t BYTES_PER_ELEMENT_,
          int32_t THREADS_PER_WARP_,
          int32_t WARPS_PER_CTA_>
struct Smem_tile_nopqg
    : public Smem_tile<
          SMEM_SIZE_IN_BYTES_PER_STAGE_,
          typename Tile_memory_per_cta_::Tile_opq_t,
          Tile_4d<typename Tile_memory_per_cta_::Tile_opq_t, Tile_memory_per_cta_::TILE_G>,
          Tile_4d<typename Tile_math_per_thread_::Tile_opq_t, Tile_math_per_thread_::TILE_G>,
          BYTES_PER_ELEMENT_,
          THREADS_PER_WARP_,
          WARPS_PER_CTA_,
          typename Tile_math_per_thread_::Tile_opq_t> {

    using Base_t = Smem_tile<
        SMEM_SIZE_IN_BYTES_PER_STAGE_,
        typename Tile_memory_per_cta_::Tile_opq_t,
        Tile_4d<typename Tile_memory_per_cta_::Tile_opq_t, Tile_memory_per_cta_::TILE_G>,
        Tile_4d<typename Tile_math_per_thread_::Tile_opq_t, Tile_math_per_thread_::TILE_G>,
        BYTES_PER_ELEMENT_,
        THREADS_PER_WARP_,
        WARPS_PER_CTA_,
        typename Tile_math_per_thread_::Tile_opq_t>;

    using Tile_memory_per_cta_t = Tile_memory_per_cta_;
    using Tile_opq_t = typename Tile_memory_per_cta_t::Tile_opq_t;
    static const int32_t NUMBER_OF_MATH_BUFFERS = Base_t::NUMBER_OF_MATH_BUFFERS;

    static const int32_t MATH_TILE_G = Base_t::MATH_TILE_G;
    static const int32_t MATH_TILE_DEPTH = Base_t::MATH_TILE_DEPTH;
    static const int32_t MATH_TILE_HEIGHT = Base_t::MATH_TILE_HEIGHT;
    static const int32_t MATH_TILE_WIDTH = Base_t::MATH_TILE_WIDTH;

    __device__ inline Smem_tile_nopqg(const uint32_t smem_base_address) : Base_t(smem_base_address)
    {
    }

    __device__ inline void set_offsets_base_lds()
    {
        abstract_tile.expose_base_state(
            this->offset_base_depth_, this->offset_base_height_, this->offset_base_width_);
    }

    __device__ inline void set_offsets_lds(const int32_t index_iteration)
    {
        abstract_tile.expose_state(
            this->offset_depth_, this->offset_height_, this->offset_width_, index_iteration);
    }

    __device__ inline void set_ptrs_lds(const int32_t index_stage)
    {
        Base_t::set_ptrs_lds(index_stage);
    }

    typename Base_t::Abstract_tile_t abstract_tile;
};

} // namespace depthwise_colwolution
} // namespace ext
} // namespace xmma

#endif
