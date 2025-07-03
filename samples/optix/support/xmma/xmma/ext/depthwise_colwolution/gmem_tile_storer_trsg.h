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

#ifndef _XMMA_EXT_DEPTHWISE_COLWOLUTION_GMEM_TILE_STORER_TRSG_H
#define _XMMA_EXT_DEPTHWISE_COLWOLUTION_GMEM_TILE_STORER_TRSG_H

#pragma once

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

template <typename Cta_tile_,
          typename Math_tile_,
          int32_t TILE_G_,
          typename Data_type_sts_,
          typename Data_type_stg_,
          int32_t THREADS_PER_WARP_,
          int32_t WARPS_PER_CTA_,
          int32_t BYTES_PER_LDS_,
          int32_t ELEMENTS_PER_INSTRUCTION_IN_GMEM_>
struct Gmem_tile_storer_trsg {
    using Cta_tile_t = Cta_tile_;
    using Math_tile_t = Math_tile_;
    using Tile_3d_t = typename Math_tile_t::Tile_3d_t;
    static const int32_t TILE_G = TILE_G_;
    static const int32_t BYTES_PER_LDS = BYTES_PER_LDS_;
    static const int32_t ELEMENTS_PER_INSTRUCTION_IN_GMEM = ELEMENTS_PER_INSTRUCTION_IN_GMEM_;
    static const int32_t UINT32_PER_TILE_G = Math_tile_t::UINT32_PER_TILE_G;
    static const int32_t BYTES_PER_INT32 = 4;
    static const int32_t BYTES_PER_STS = UINT32_PER_TILE_G * BYTES_PER_INT32;
    static const int32_t ELEMENTS_PER_LDS = BYTES_PER_LDS / static_cast<int32_t>(sizeof(typename Data_type_sts_::Type));
    static const int32_t ELEMENTS_PER_STS = Math_tile_t::TILE_G;
    using Data_type_sts_t = Data_type_sts_;
    using Data_type_stg_t = Data_type_stg_;
    static const int32_t THREADS_PER_WARP = THREADS_PER_WARP_;
    static const int32_t WARPS_PER_CTA = WARPS_PER_CTA_;
    static const int32_t BYTES_PER_ELEMENT = Data_type_stg_t::BYTES_PER_ELEMENT;
    static const int32_t BYTES_PER_INSTRUCTION_IN_GMEM =
        BYTES_PER_ELEMENT * ELEMENTS_PER_INSTRUCTION_IN_GMEM;
    static const int32_t ELEMENTS_PER_STG = ELEMENTS_PER_INSTRUCTION_IN_GMEM;

    static const int32_t THREADS_PER_CTA = WARPS_PER_CTA * THREADS_PER_WARP;

    static const int32_t SMEM_SIZE_IN_BYTES =
        Math_tile_t::UINT32_NUMBER * BYTES_PER_INT32 * THREADS_PER_CTA;

    static const int32_t BYTES_PER_TILE_G_SMEM = TILE_G * Data_type_sts_t::BYTES_PER_ELEMENT;
    static const int32_t BYTES_PER_SMEM_LINE = 128;
    static const int32_t SLICE_RATIO =
        (BYTES_PER_STS * THREADS_PER_WARP + BYTES_PER_TILE_G_SMEM - 1) / BYTES_PER_TILE_G_SMEM;
    static_assert(SLICE_RATIO > 0, "");
    static const int32_t LDSS_PER_TILE_G = BYTES_PER_TILE_G_SMEM / BYTES_PER_LDS;
    static_assert(LDSS_PER_TILE_G > 0, "");
    static const int32_t STGS_PER_TILE_G = LDSS_PER_TILE_G;

    using Tile_4d_per_cta_t = Tile_4d<Tile_3d_t, TILE_G>;
    using Tile_4d_per_thread_t = Tile_4d<Tile_3d<1, 1, 1>, ELEMENTS_PER_LDS>;
    using Abstract_tile_t =
        Abstract_tile<Tile_4d_per_cta_t, Tile_4d_per_thread_t, ELEMENTS_PER_LDS, THREADS_PER_CTA>;
    static const int32_t INSTRUCTION_PER_THREAD_IN_SMEM = Abstract_tile_t::INSTRUCTIONS_PER_THREAD;
    static const int32_t INSTRUCTION_PER_THREAD_IN_GMEM = INSTRUCTION_PER_THREAD_IN_SMEM;
    static const int32_t VALID_LDSS_PER_CTA = Abstract_tile_t::VALID_INSTRUCTIONS_PER_CTA;
    static const int32_t VALID_STGS_PER_CTA = VALID_LDSS_PER_CTA;
    static const int32_t LDSS_PER_THREAD = INSTRUCTION_PER_THREAD_IN_SMEM;
    using Packed_lds_t = typename xmma::Uint_from_size_in_bytes<BYTES_PER_LDS>::Type;
    using Packed_stg_t =
        typename xmma::Uint_from_size_in_bytes<BYTES_PER_INSTRUCTION_IN_GMEM>::Type;
    static const int32_t STGS_PER_THREAD = LDSS_PER_THREAD;
    static const int32_t LDGS_PER_THREAD = STGS_PER_THREAD;
    static const int32_t SPLIT_K_STG_BYTES_PER_CTA =
        THREADS_PER_CTA * LDSS_PER_THREAD * static_cast<int32_t>(sizeof(Packed_lds_t));

    __device__ inline Gmem_tile_storer_trsg(
        Math_tile_t math_tile,
        uint32_t smem_base_address,
        const Depthwise_colwolution_parameter<Cta_tile_t> &params,
        const int32_t cta_tile_begin_t,
        const int32_t cta_tile_begin_r,
        const int32_t cta_tile_begin_s)
    {

        set_tensor_dims(params.t, params.r, params.s, params.g);
        set_tensor_strides(params.flt_stride_t, params.flt_stride_r, params.flt_stride_s);
        set_alpha(params.alpha);
        set_beta(params.beta);
        set_split_k_related_params(params.split_k_gmem,
                                   params.split_k_gmem_buffer_counter,
                                   params.split_k_gmem_final_counter,
                                   params.split_k_slices,
                                   params.split_k_buffers,
                                   params.single_split_k_buffer_size_in_bytes);
        set_ptr_base_stg(params.gmem[Tensor_type::D]);
        set_ptr_base_ldg(params.gmem[Tensor_type::C]);

        Abstract_tile_t abstract_tile;
        abstract_tile.expose_state(offset_depth_, offset_height_, offset_width_);
        math_tile.expose(data_sts_, params.is_colwolution);

        int32_t tid = threadIdx_x();
        ptr_base_sts_ = smem_base_address + tid * BYTES_PER_STS;
        ptr_base_lds_ = smem_base_address + tid % LDSS_PER_TILE_G * BYTES_PER_LDS;
        int32_t index_in_g_dimension = tid % STGS_PER_TILE_G;
        cta_tile_begin_group_ = blockIdx_y() * TILE_G + index_in_g_dimension * ELEMENTS_PER_STG;
        valid_group_ = cta_tile_begin_group_ < params_g_;
        cta_tile_begin_width_ = cta_tile_begin_s;
        cta_tile_begin_height_ = cta_tile_begin_r;
        cta_tile_begin_depth_ = cta_tile_begin_t;
        int32_t offset =
            (cta_tile_begin_group_ + cta_tile_begin_width_ * params_tensor_stride_width_ +
             cta_tile_begin_height_ * params_tensor_stride_height_ +
             cta_tile_begin_depth_ * params_tensor_stride_depth_) *
            Data_type_stg_t::BYTES_PER_ELEMENT;
        ptr_base_stg_ = move_pointer(ptr_base_stg_, offset);
        ptr_base_ldg_ = move_pointer(ptr_base_ldg_, offset);

        int32_t index_buffer_of_the_slice =
            get_index_of_the_split_k_slice() % params_split_k_buffers_;
        split_k_buffer_offset_in_bytes_of_the_slice_ =
            index_buffer_of_the_slice * params_single_split_k_buffer_size_in_bytes_;
        int32_t index_in_the_slice = blockIdx_x() + blockIdx_y() * gridDim_x();
        ptr_split_k_gmem_buffer_counter_ = move_pointer(
            params_split_k_gmem_buffer_counter_,
            (index_in_the_slice * params_split_k_buffers_ + index_buffer_of_the_slice) *
                sizeof(int32_t));
        ptr_split_k_gmem_final_counter_ =
            move_pointer(params_split_k_gmem_final_counter_, index_in_the_slice * sizeof(int32_t));
        ptr_base_stg_split_k_ = move_pointer(params_split_k_gmem_,
                                             threadIdx_x() * sizeof(Packed_lds_t) +
                                                 index_in_the_slice * SPLIT_K_STG_BYTES_PER_CTA +
                                                 split_k_buffer_offset_in_bytes_of_the_slice_);
        set_ptrs_split_k();
    }

    __device__ inline void set_ptr_base_stg(void *in) { ptr_base_stg_ = in; }

    __device__ inline void set_ptr_base_ldg(void *in) { ptr_base_ldg_ = in; }

    __device__ inline void set_tensor_dims(const int32_t depth,
                                           const int32_t height,
                                           const int32_t width,
                                           const int32_t group)
    {
        params_t_ = depth;
        params_r_ = height;
        params_s_ = width;
        params_g_ = group;
    }

    __device__ inline void
    set_tensor_strides(const int32_t depth, const int32_t height, const int32_t width)
    {
        params_tensor_stride_depth_ = depth;
        params_tensor_stride_height_ = height;
        params_tensor_stride_width_ = width;
    }

    __device__ inline void set_alpha(const uint32_t alpha) { params_alpha_ = alpha; }

    __device__ inline void set_beta(const uint32_t beta) { params_beta_ = beta; }

    __device__ inline void
    set_split_k_related_params(void *split_k_gmem,
                               void *split_k_gmem_buffer_counter,
                               void *split_k_gmem_final_counter,
                               const int32_t split_k_slices,
                               const int32_t split_k_buffers,
                               const int32_t single_split_k_buffer_size_in_bytes)
    {
        params_split_k_gmem_ = split_k_gmem;
        params_split_k_gmem_buffer_counter_ = split_k_gmem_buffer_counter;
        params_split_k_gmem_final_counter_ = split_k_gmem_final_counter;
        params_split_k_slices_ = split_k_slices;
        params_split_k_buffers_ = split_k_buffers;
        params_single_split_k_buffer_size_in_bytes_ = single_split_k_buffer_size_in_bytes;
    }

    __device__ inline int32_t get_index_of_the_split_k_slice() { return blockIdx_z(); }

    __device__ inline int32_t get_split_k_slices() { return params_split_k_slices_; }

    __device__ inline int32_t get_split_k_buffers() { return params_split_k_buffers_; }

    __device__ inline void *get_ptr_split_k_buffer_counter()
    {
        return ptr_split_k_gmem_buffer_counter_;
    }

    __device__ inline void *get_ptr_split_k_final_counter()
    {
        return ptr_split_k_gmem_final_counter_;
    }

    __device__ inline void store_to_smem()
    {
#pragma unroll
        for (int32_t index_depth = 0; index_depth < Tile_3d_t::DEPTH; ++index_depth) {
#pragma unroll
            for (int32_t index_height = 0; index_height < Tile_3d_t::HEIGHT; ++index_height) {
#pragma unroll
                for (int32_t index_width = 0; index_width < Tile_3d_t::WIDTH; ++index_width) {
#pragma unroll
                    for (int32_t index_group = 0;
                         index_group < UINT32_PER_TILE_G / (BYTES_PER_STS / BYTES_PER_INT32);
                         ++index_group) {
                        int32_t offset =
                            WARPS_PER_CTA * THREADS_PER_WARP * BYTES_PER_STS * index_group +
                            WARPS_PER_CTA * THREADS_PER_WARP * BYTES_PER_STS *
                                (UINT32_PER_TILE_G / (BYTES_PER_STS / BYTES_PER_INT32)) *
                                index_width +
                            Tile_3d_t::WIDTH * WARPS_PER_CTA * THREADS_PER_WARP * BYTES_PER_STS *
                                (UINT32_PER_TILE_G / (BYTES_PER_STS / BYTES_PER_INT32)) *
                                index_height +
                            Tile_3d_t::HEIGHT * Tile_3d_t::WIDTH * WARPS_PER_CTA *
                                THREADS_PER_WARP * BYTES_PER_STS *
                                (UINT32_PER_TILE_G / (BYTES_PER_STS / BYTES_PER_INT32)) *
                                index_depth;

                        uint32_t non_packed_tmp[BYTES_PER_STS / BYTES_PER_INT32];

                        // static_assert(BYTES_PER_STS / BYTES_PER_INT32==4,"");
                        for (int32_t index_in_a_packed_data = 0;
                             index_in_a_packed_data < BYTES_PER_STS / BYTES_PER_INT32;
                             ++index_in_a_packed_data) {
                            non_packed_tmp[index_in_a_packed_data] =
                                data_sts_[index_depth][index_height][index_width]
                                         [index_group * (BYTES_PER_STS / BYTES_PER_INT32) +
                                          index_in_a_packed_data];
                        }

                        typename xmma::Uint_from_size_in_bytes<BYTES_PER_STS>::Type tmp;
                        make_packed_uint(tmp, non_packed_tmp);
                        xmma::sts(ptr_base_sts_ + offset, tmp);
                    }
                }
            }
        }
    }

    __device__ inline void load_from_smem()
    {
#pragma unroll
        for (int32_t index_lds = 0; index_lds < LDSS_PER_THREAD; ++index_lds) {
            int32_t linear_index = threadIdx_x() + index_lds * THREADS_PER_CTA;
            int32_t offset = linear_index / LDSS_PER_TILE_G *
                             (SLICE_RATIO * WARPS_PER_CTA * BYTES_PER_TILE_G_SMEM);
            uint32_t ptr_lds_with_g = ptr_base_lds_ + offset;
#pragma unroll
            for (int32_t index_slice = 0; index_slice < SLICE_RATIO * WARPS_PER_CTA;
                 ++index_slice) {
                uint32_t ptr_lds = ptr_lds_with_g + index_slice * BYTES_PER_TILE_G_SMEM;
                if (linear_index < VALID_LDSS_PER_CTA) {
                    xmma::lds(data_lds_[index_lds][index_slice], ptr_lds);
                }
            }
        }
    }

    __device__ inline void reduction()
    {
        Reduction<Data_type_sts_t, Packed_lds_t, LDSS_PER_THREAD, SLICE_RATIO * WARPS_PER_CTA>::
            exlwte(data_d_before_type_colwersion_, data_lds_);
    }

    __device__ inline void store_to_split_k_buffer()
    {
#pragma unroll
        for (int32_t index_lds = 0; index_lds < LDSS_PER_THREAD; ++index_lds) {
            *static_cast<Packed_lds_t *>(ptrs_stg_split_k_[index_lds]) = 
                data_d_before_type_colwersion_[index_lds];
        }
    }

    __device__ inline void atomic_add_in_the_split_k_buffer()
    {
        Atomic_add<Packed_lds_t, Data_type_sts_t, LDSS_PER_THREAD>::exlwte(
            ptrs_stg_split_k_, data_d_before_type_colwersion_);
    }

    __device__ inline void set_ptrs_split_k()
    {
#pragma unroll
        for (int32_t index_lds = 0; index_lds < LDSS_PER_THREAD; ++index_lds) {
            ptrs_stg_split_k_[index_lds] = move_pointer(
                ptr_base_stg_split_k_, index_lds * (THREADS_PER_CTA * sizeof(Packed_lds_t)));
        }
    }

    __device__ inline void set_ptr_base_split_k()
    {
        ptr_base_stg_split_k_ =
            move_pointer(ptr_base_stg_split_k_, -split_k_buffer_offset_in_bytes_of_the_slice_);
    }

    __device__ inline void update_ptr_base_split_k()
    {
        ptr_base_stg_split_k_ =
            move_pointer(ptr_base_stg_split_k_, params_single_split_k_buffer_size_in_bytes_);
    }

    __device__ inline void load_from_split_k_buffer(int32_t index_split_k_buffer)
    {
        set_ptrs_split_k();
#pragma unroll
        for (int32_t index_lds = 0; index_lds < LDSS_PER_THREAD; ++index_lds) {
            xmma::ldg(data_d_before_type_colwersion_split_k_[index_lds],
                      ptrs_stg_split_k_[index_lds]);
        }
    }

    __device__ inline void add_the_data_from_the_split_k_buffer()
    {
        Packed_lds_t tmp[LDSS_PER_THREAD][2];
#pragma unroll
        for (int32_t index_lds = 0; index_lds < LDSS_PER_THREAD; ++index_lds) {
            tmp[index_lds][0] = data_d_before_type_colwersion_[index_lds];
            tmp[index_lds][1] = data_d_before_type_colwersion_split_k_[index_lds];
        }
        Reduction<Data_type_sts_t, Packed_lds_t, LDSS_PER_THREAD, 2>::exlwte(
            data_d_before_type_colwersion_, tmp);
    }

    __device__ inline void set_ptrs_and_preds_gmem()
    {
        set_ptrs_of_instruction_in_gmem();
        set_preds_of_instruction_in_gmem();
    }

    __device__ inline void set_ptrs_of_instruction_in_gmem()
    {
#pragma unroll
        for (int32_t i = 0; i < STGS_PER_THREAD; ++i) {
            int32_t offset = (offset_depth_[i] * params_tensor_stride_depth_ +
                              offset_height_[i] * params_tensor_stride_height_ +
                              offset_width_[i] * params_tensor_stride_width_) *
                             Data_type_stg_t::BYTES_PER_ELEMENT;
            ptrs_ldg_[i] = move_pointer(ptr_base_ldg_, offset);
            ptrs_stg_[i] = move_pointer(ptr_base_stg_, offset);
            // linear_index < VALID_STGS_PER_CTA is necessary as a cta only processes
            // a tile in the
            // TRS block
        }
    }

    __device__ inline void set_preds_of_instruction_in_gmem()
    {
#pragma unroll
        for (int32_t i = 0; i < STGS_PER_THREAD; ++i) {
            int32_t linear_index = threadIdx_x() + i * THREADS_PER_CTA;
            preds_gmem_[i] = (linear_index < VALID_STGS_PER_CTA && valid_group_ &&
                              cta_tile_begin_depth_ + offset_depth_[i] < params_t_ &&
                              cta_tile_begin_height_ + offset_height_[i] < params_r_ &&
                              cta_tile_begin_width_ + offset_width_[i] < params_s_);
        }
    }

    __device__ inline void load_from_gmem()
    {
#pragma unroll
        for (int32_t i = 0; i < LDGS_PER_THREAD; ++i) {
            if (params_beta_ != 0 && preds_gmem_[i]) {
                xmma::ldg(data_ldg_[i], ptrs_ldg_[i]);
            }
        }
    }

    __device__ inline void type_colwersion_for_load()
    {
        Type_colwerter<typename Data_type_sts_t::Type,
                       typename Data_type_stg_t::Type,
                       Packed_lds_t,
                       Packed_stg_t,
                       LDGS_PER_THREAD>::exlwte(data_c_after_type_colwersion_, data_ldg_);
    }

    __device__ inline void apply_alpha()
    {
        Apply_alpha<Data_type_sts_t, Packed_lds_t, LDSS_PER_THREAD>::exlwte(
            data_d_before_type_colwersion_, params_alpha_);
    }

    __device__ inline void apply_beta()
    {
        if (params_beta_ != 0) {
            Apply_beta<Data_type_sts_t, Packed_lds_t, LDSS_PER_THREAD>::exlwte(
                data_d_before_type_colwersion_, data_c_after_type_colwersion_, params_beta_);
        }
    }

    __device__ inline void type_colwersion_for_store()
    {
        Type_colwerter<typename Data_type_stg_t::Type,
                       typename Data_type_sts_t::Type,
                       Packed_stg_t,
                       Packed_lds_t,
                       LDSS_PER_THREAD>::exlwte(data_stg_, data_d_before_type_colwersion_);
    }

    __device__ inline void store_to_gmem()
    {
#pragma unroll
        for (int32_t i = 0; i < STGS_PER_THREAD; ++i) {
            if (preds_gmem_[i]) {
                *static_cast<Packed_stg_t *>(ptrs_stg_[i]) = data_stg_[i];
            }
        }
    }

    uint32_t params_tensor_stride_width_;
    uint32_t params_tensor_stride_height_;
    uint32_t params_tensor_stride_depth_;
    int32_t params_g_;
    int32_t params_t_;
    int32_t params_r_;
    int32_t params_s_;
    uint32_t params_alpha_;
    uint32_t params_beta_;
    void *params_split_k_gmem_;
    void *params_split_k_gmem_buffer_counter_;
    void *params_split_k_gmem_final_counter_;
    int32_t params_split_k_slices_;
    int32_t params_split_k_buffers_;
    void *ptr_split_k_gmem_buffer_counter_;
    void *ptr_split_k_gmem_final_counter_;
    int32_t params_single_split_k_buffer_size_in_bytes_;
    int32_t split_k_buffer_offset_in_bytes_of_the_slice_;
    int32_t cta_tile_begin_depth_;
    int32_t cta_tile_begin_height_;
    int32_t cta_tile_begin_width_;
    int32_t cta_tile_begin_group_;
    int32_t offset_depth_[STGS_PER_THREAD];
    int32_t offset_height_[STGS_PER_THREAD];
    int32_t offset_width_[STGS_PER_THREAD];
    uint32_t ptr_base_sts_;
    uint32_t ptr_base_lds_;
    void *ptr_base_ldg_;
    void *ptr_base_stg_;
    void *ptr_base_stg_split_k_;
    Packed_lds_t data_lds_[LDSS_PER_THREAD][SLICE_RATIO * WARPS_PER_CTA];
    Packed_lds_t data_d_before_type_colwersion_[LDSS_PER_THREAD];
    Packed_lds_t data_c_after_type_colwersion_[LDSS_PER_THREAD];
    Packed_stg_t data_stg_[LDSS_PER_THREAD];
    Packed_stg_t data_ldg_[LDSS_PER_THREAD];
    Packed_lds_t data_d_before_type_colwersion_split_k_[LDSS_PER_THREAD];
    uint32_t data_sts_[Tile_3d_t::DEPTH][Tile_3d_t::HEIGHT][Tile_3d_t::WIDTH]
                      [Math_tile_t::UINT32_PER_TILE_G];
    void *ptrs_ldg_[LDGS_PER_THREAD];
    void *ptrs_stg_[STGS_PER_THREAD];
    void *ptrs_stg_split_k_[STGS_PER_THREAD];
    bool preds_gmem_[STGS_PER_THREAD];
    bool valid_group_;
};

} // namespace depthwise_colwolution
} // namespace ext
} // namespace xmma

#endif
