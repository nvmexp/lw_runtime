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

#ifndef _XMMA_EXT_DEPTHWISE_COLWOLUTION_GMEM_TILE_H
#define _XMMA_EXT_DEPTHWISE_COLWOLUTION_GMEM_TILE_H

#pragma once

#include "abstract_tile.h"
#include "cta_tile.h"
#include "params.h"
#include "type_colwerter.h"
#include "utils.h"
#include "xmma/ampere/traits.h"
#include "xmma/turing/traits.h"
#include "xmma/utils.h"
#include "xmma/volta/traits.h"
#include <cstdint>
#include <lwda_runtime.h>
#include <type_traits>

namespace xmma
{
namespace ext
{
namespace depthwise_colwolution
{
template <typename Gpu_arch_,
          typename Tile_3d_,
          int32_t TILE_G_,
          int32_t STAGE_,
          int32_t THREADS_PER_CTA_,
          typename Data_type_in_gmem_,
          typename Data_type_in_smem_,
          int32_t ELEMENTS_PER_INSTRUCTION_IN_GMEM_,
          int32_t ITERATION_NUMBER_IN_GMEM_>
struct Gmem_tile {
    public:
    static const int32_t ELEMENTS_PER_INSTRUCTION_IN_GMEM = ELEMENTS_PER_INSTRUCTION_IN_GMEM_;
    static const int32_t ITERATION_NUMBER_IN_GMEM = ITERATION_NUMBER_IN_GMEM_;
    using Gpu_arch_t = Gpu_arch_;
    using Tile_3d_t = Tile_3d_;
    using Data_type_in_gmem_t = Data_type_in_gmem_;
    using Data_type_in_smem_t = Data_type_in_smem_;
    static const int32_t TILE_G = TILE_G_;
    static const int32_t STAGE = STAGE_;
    static const int32_t THREADS_PER_CTA = THREADS_PER_CTA_;
    // Derived
    static const int32_t BYTES_PER_ELEMENT = Data_type_in_gmem_t::BYTES_PER_ELEMENT;
    static const int32_t BYTES_PER_ELEMENT_IN_GMEM = Data_type_in_gmem_t::BYTES_PER_ELEMENT;
    static const int32_t BYTES_PER_ELEMENT_IN_SMEM = Data_type_in_smem_t::BYTES_PER_ELEMENT;
    static const int32_t BYTES_PER_INSTRUCTION_IN_GMEM =
        ELEMENTS_PER_INSTRUCTION_IN_GMEM * BYTES_PER_ELEMENT_IN_GMEM;
    static const bool ENABLE_LDGSTS =
        Gpu_arch_t::HAS_LDGSTS && std::is_same<Data_type_in_gmem_t, Data_type_in_smem_t>::value &&
        (BYTES_PER_INSTRUCTION_IN_GMEM == 4 || BYTES_PER_INSTRUCTION_IN_GMEM == 8 ||
         BYTES_PER_INSTRUCTION_IN_GMEM == 16);
    static_assert(ELEMENTS_PER_INSTRUCTION_IN_GMEM > 0, "");
    static const int32_t THREADS_PER_ROW = TILE_G / ELEMENTS_PER_INSTRUCTION_IN_GMEM;
    static_assert(THREADS_PER_ROW > 0, "");
    using Tile_4d_per_cta_t = Tile_4d<Tile_3d_t, TILE_G>;
    using Tile_4d_per_thread_t = Tile_4d<Tile_3d<1, 1, 1>, ELEMENTS_PER_INSTRUCTION_IN_GMEM>;
    using Abstract_tile_t = Abstract_tile<Tile_4d_per_cta_t,
                                          Tile_4d_per_thread_t,
                                          ELEMENTS_PER_INSTRUCTION_IN_GMEM,
                                          THREADS_PER_CTA>;
    static const int32_t VALID_INSTRUCTIONS_PER_CTA_IN_GMEM =
        Abstract_tile_t::VALID_INSTRUCTIONS_PER_CTA;
    static const int32_t INSTRUCTION_PER_THREAD_IN_GMEM = Abstract_tile_t::INSTRUCTIONS_PER_THREAD;
    static const int32_t INSTRUCTION_PER_THREAD_PER_ITERATION_IN_GMEM =
        (INSTRUCTION_PER_THREAD_IN_GMEM + ITERATION_NUMBER_IN_GMEM - 1) / ITERATION_NUMBER_IN_GMEM;
    static const int32_t MAX_BYTES_PER_INSTRUCTION_IN_SMEM = 16;
    static const int32_t BYTES_PER_INSTRUCTION_IN_SMEM =
        BYTES_PER_INSTRUCTION_IN_GMEM * (BYTES_PER_ELEMENT_IN_SMEM / BYTES_PER_ELEMENT_IN_GMEM);
    static_assert(BYTES_PER_INSTRUCTION_IN_SMEM > 0, "");
    static_assert(BYTES_PER_INSTRUCTION_IN_SMEM <= MAX_BYTES_PER_INSTRUCTION_IN_SMEM, "");
    static const int32_t STSS_PER_THREAD = INSTRUCTION_PER_THREAD_IN_GMEM;
    static const int32_t BYTES_PER_TILE_G_IN_GMEM = TILE_G * BYTES_PER_ELEMENT_IN_GMEM;
    static const int32_t BYTES_PER_TILE_G_IN_SMEM = TILE_G * BYTES_PER_ELEMENT_IN_SMEM;
    using Packed_data_type_in_gmem_t =
        typename xmma::Uint_from_size_in_bytes<BYTES_PER_INSTRUCTION_IN_GMEM>::Type;
    using Packed_data_type_in_smem_t =
        typename xmma::Uint_from_size_in_bytes<BYTES_PER_INSTRUCTION_IN_SMEM>::Type;
    static const int32_t SMEM_SIZE_IN_BYTES_PER_STAGE =
        ((STSS_PER_THREAD * BYTES_PER_INSTRUCTION_IN_SMEM * THREADS_PER_CTA + 127) / 128) * 128;
    static const int32_t SMEM_SIZE_IN_BYTES = SMEM_SIZE_IN_BYTES_PER_STAGE * STAGE;

    __device__ inline Gmem_tile(Abstract_tile_t &abstract_tile)
    {
        abstract_tile.expose_state(offset_depth_, offset_height_, offset_width_);
    }

    __device__ inline void set_gmem_ptr_instruction_in_gmem_base(void *in) { params_gmem_ = in; }

    __device__ inline void set_pad(const int32_t front, const int32_t top, const int32_t left)
    {
        params_pad_front_ = front;
        params_pad_top_ = top;
        params_pad_left_ = left;
    }

    __device__ inline void set_tensor_dims(const int32_t batch,
                                           const int32_t depth,
                                           const int32_t height,
                                           const int32_t width,
                                           const int32_t group)
    {
        params_batch_ = batch;
        params_depth_ = depth;
        params_height_ = height;
        params_width_ = width;
        params_group_ = group;
    }

    __device__ inline void set_tensor_strides(const int32_t batch,
                                              const int32_t depth,
                                              const int32_t height,
                                              const int32_t width)
    {
        params_tensor_stride_batch_ = batch;
        params_tensor_stride_depth_ = depth;
        params_tensor_stride_height_ = height;
        params_tensor_stride_width_ = width;
#pragma unroll
        for (int32_t i = 0; i < INSTRUCTION_PER_THREAD_IN_GMEM; ++i) {
            offset_in_bytes_[i] = (offset_depth_[i] * params_tensor_stride_depth_ +
                                   offset_height_[i] * params_tensor_stride_height_ +
                                   offset_width_[i] * params_tensor_stride_width_) *
                                  BYTES_PER_ELEMENT_IN_GMEM;
        }
    }

    __device__ inline void set_params_tiles(const int32_t opq,
                                            const uint32_t mul_opq,
                                            const uint32_t shr_opq,
                                            const int32_t pq,
                                            const uint32_t mul_pq,
                                            const uint32_t shr_pq,
                                            const int32_t q,
                                            const uint32_t mul_q,
                                            const uint32_t shr_q,
                                            const int32_t p,
                                            const int32_t o,
                                            const int32_t delta_tiles_n,
                                            const int32_t delta_tiles_o,
                                            const int32_t delta_tiles_p,
                                            const int32_t delta_tiles_q)
    {
        params_tiles_opq_ = opq;
        params_mul_tiles_opq_ = mul_opq;
        params_shr_tiles_opq_ = shr_opq;
        params_tiles_pq_ = pq;
        params_mul_tiles_pq_ = mul_pq;
        params_shr_tiles_pq_ = shr_pq;
        params_tiles_q_ = q;
        params_mul_tiles_q_ = mul_q;
        params_shr_tiles_q_ = shr_q;
        params_tiles_o_ = o;
        params_tiles_p_ = p;
        params_delta_tiles_n_ = delta_tiles_n;
        params_delta_tiles_o_ = delta_tiles_o;
        params_delta_tiles_p_ = delta_tiles_p;
        params_delta_tiles_q_ = delta_tiles_q;
    }

    __device__ inline void set_alpha(const uint32_t in) { params_alpha_ = in; }

    __device__ inline void set_beta(const uint32_t in) { params_beta_ = in; }

    __device__ inline void initialize_ptr_base(const uint32_t smem_base_address)
    {
        int32_t tile_g_index = blockIdx_y();
        int32_t linear_tid = threadIdx_x();
        int32_t start_offset_g_of_the_cta = tile_g_index * TILE_G;
        int32_t offset_g_in_the_tile =
            (linear_tid % THREADS_PER_ROW) * ELEMENTS_PER_INSTRUCTION_IN_GMEM;
        int32_t offset_g_of_the_thread = start_offset_g_of_the_cta + offset_g_in_the_tile;
        ptr_base_instruction_in_gmem_ =
            move_pointer(params_gmem_, offset_g_of_the_thread * BYTES_PER_ELEMENT_IN_GMEM);
        valid_group_ = offset_g_of_the_thread < params_group_;
        ptr_base_sts_ = smem_base_address + offset_g_in_the_tile * BYTES_PER_ELEMENT_IN_SMEM;
    }

    __device__ inline void decompose_tile_index(int32_t &cta_tile_index_n,
                                                int32_t &cta_tile_index_o,
                                                int32_t &cta_tile_index_p,
                                                int32_t &cta_tile_index_q,
                                                const int32_t cta_tile_index,
                                                bool is_update_mode)
    {
        if (is_update_mode) {
            cta_tile_index_q += params_delta_tiles_q_;
            cta_tile_index_p += params_delta_tiles_p_;
            cta_tile_index_o += params_delta_tiles_o_;
            cta_tile_index_n += params_delta_tiles_n_;
            if (cta_tile_index_q >= params_tiles_q_) {
                cta_tile_index_q -= params_tiles_q_;
                cta_tile_index_p += 1;
            }
            if (cta_tile_index_p >= params_tiles_p_) {
                cta_tile_index_p -= params_tiles_p_;
                cta_tile_index_o += 1;
            }
            if (cta_tile_index_o >= params_tiles_o_) {
                cta_tile_index_o -= params_tiles_o_;
                cta_tile_index_n += 1;
            }
        } else {
            int32_t cta_tile_index_opq;
            xmma::fast_divmod(cta_tile_index_n,
                              cta_tile_index_opq,
                              cta_tile_index,
                              params_tiles_opq_,
                              params_mul_tiles_opq_,
                              params_shr_tiles_opq_);
            int32_t cta_tile_index_pq;
            xmma::fast_divmod(cta_tile_index_o,
                              cta_tile_index_pq,
                              cta_tile_index_opq,
                              params_tiles_pq_,
                              params_mul_tiles_pq_,
                              params_shr_tiles_pq_);
            xmma::fast_divmod(cta_tile_index_p,
                              cta_tile_index_q,
                              cta_tile_index_pq,
                              params_tiles_q_,
                              params_mul_tiles_q_,
                              params_shr_tiles_q_);
        }
    }

    __device__ inline void get_tile_begin(int32_t &cta_tile_begin_depth,
                                          int32_t &cta_tile_begin_height,
                                          int32_t &cta_tile_begin_width,
                                          const int32_t cta_tile_index_o,
                                          const int32_t cta_tile_index_p,
                                          const int32_t cta_tile_index_q,
                                          const int32_t cta_tile_begin_t,
                                          const int32_t cta_tile_begin_r,
                                          const int32_t cta_tile_begin_s);

    __device__ inline void move(const int32_t cta_tile_begin_batch,
                                const int32_t cta_tile_begin_depth,
                                const int32_t cta_tile_begin_height,
                                const int32_t cta_tile_begin_width)
    {
        int32_t offset_in_bytes = (cta_tile_begin_batch * params_tensor_stride_batch_ +
                                   cta_tile_begin_depth * params_tensor_stride_depth_ +
                                   cta_tile_begin_height * params_tensor_stride_height_ +
                                   cta_tile_begin_width * params_tensor_stride_width_) *
                                  BYTES_PER_ELEMENT_IN_GMEM;
        ptr_instruction_in_gmem_ = move_pointer(ptr_base_instruction_in_gmem_, offset_in_bytes);
    }

    __device__ inline void set_preds_of_instruction_in_gmem(const int32_t cta_tile_begin_batch,
                                                            const int32_t cta_tile_begin_depth,
                                                            const int32_t cta_tile_begin_height,
                                                            const int32_t cta_tile_begin_width,
                                                            const int32_t index_begin,
                                                            const int32_t index_end)
    {
        bool valid_batch = cta_tile_begin_batch < params_batch_;
        bool valid_group_and_batch = valid_group_ && valid_batch;
#pragma unroll
        for (int32_t i = index_begin; i < index_end; ++i) {
            int32_t depth = cta_tile_begin_depth + offset_depth_[i];
            int32_t height = cta_tile_begin_height + offset_height_[i];
            int32_t width = cta_tile_begin_width + offset_width_[i];
            bool valid_depth = ((unsigned)depth < params_depth_);
            bool valid_height = ((unsigned)height < params_height_);
            bool valid_width = ((unsigned)width < params_width_);
            // bool valid_depth = (depth >= 0 && depth < params_depth_);
            // bool valid_height = (height >= 0 && height < params_height_);
            // bool valid_width = (width >= 0 && width < params_width_);
            int32_t linear_index = threadIdx_x() + i * THREADS_PER_CTA;
            preds_of_instruction_in_gmem_[i] =
                (valid_group_and_batch && linear_index < VALID_INSTRUCTIONS_PER_CTA_IN_GMEM &&
                 valid_depth && valid_height && valid_width);
        }
    }

    __device__ inline void set_preds_of_instruction_in_gmem(const int32_t cta_tile_begin_batch,
                                                            const int32_t cta_tile_begin_depth,
                                                            const int32_t cta_tile_begin_height,
                                                            const int32_t cta_tile_begin_width)
    {
        set_preds_of_instruction_in_gmem(cta_tile_begin_batch,
                                         cta_tile_begin_depth,
                                         cta_tile_begin_height,
                                         cta_tile_begin_width,
                                         0,
                                         INSTRUCTION_PER_THREAD_IN_GMEM);
    }

    __device__ inline void set_preds_of_instruction_in_gmem(const int32_t cta_tile_begin_batch,
                                                            const int32_t cta_tile_begin_depth,
                                                            const int32_t cta_tile_begin_height,
                                                            const int32_t cta_tile_begin_width,
                                                            const int32_t index_iteration)
    {
        set_preds_of_instruction_in_gmem(
            cta_tile_begin_batch,
            cta_tile_begin_depth,
            cta_tile_begin_height,
            cta_tile_begin_width,
            INSTRUCTION_PER_THREAD_PER_ITERATION_IN_GMEM * index_iteration,
            INSTRUCTION_PER_THREAD_PER_ITERATION_IN_GMEM * (index_iteration + 1));
    }

    __device__ inline void set_ptrs_of_instruction_in_gmem(const int32_t index_begin,
                                                           const int32_t index_end)
    {
#pragma unroll
        for (int32_t i = index_begin; i < index_end; ++i) {
            ptrs_instruction_in_gmem_[i] =
                move_pointer(ptr_instruction_in_gmem_, offset_in_bytes_[i]);
        }
    }

    __device__ inline void set_ptrs_of_instruction_in_gmem()
    {
        set_ptrs_of_instruction_in_gmem(0, INSTRUCTION_PER_THREAD_IN_GMEM);
    }

    __device__ inline void set_ptrs_of_instruction_in_gmem(const int32_t index_iteration)
    {
        set_ptrs_of_instruction_in_gmem(
            INSTRUCTION_PER_THREAD_PER_ITERATION_IN_GMEM * index_iteration,
            INSTRUCTION_PER_THREAD_PER_ITERATION_IN_GMEM * (index_iteration + 1));
    }

    __device__ inline void load_from_gmem()
    {
        if (!ENABLE_LDGSTS) {
#pragma unroll
            for (int32_t i = 0; i < INSTRUCTION_PER_THREAD_IN_GMEM; ++i) {
                xmma::ldg_with_pnz(data_of_gmem_[i],
                                   ptrs_instruction_in_gmem_[i],
                                   preds_of_instruction_in_gmem_[i]);
            }
        } else {
            constexpr bool ZFILL = true;
            constexpr bool BYPASS = true;
            // We need to avoid the tmp Var here.
            const void *tmp_ptrs_instruction_in_gmem_[INSTRUCTION_PER_THREAD_IN_GMEM];
#pragma unroll
            for (int32_t i = 0; i < INSTRUCTION_PER_THREAD_IN_GMEM; ++i) {
                tmp_ptrs_instruction_in_gmem_[i] = ptrs_instruction_in_gmem_[i];
            }

            xmma::Ldgsts_functor<INSTRUCTION_PER_THREAD_IN_GMEM,
                                 BYTES_PER_INSTRUCTION_IN_GMEM,
                                 ZFILL,
                                 BYPASS>
                functor(ptrs_sts_, tmp_ptrs_instruction_in_gmem_);
#pragma unroll
            for (int32_t i = 0; i < INSTRUCTION_PER_THREAD_IN_GMEM; ++i) {
                functor.ldgsts(i, preds_of_instruction_in_gmem_[i], xmma::MEM_DESC_DEFAULT);
            }
        }
    }

    __device__ inline void type_colwersion()
    {
        if (!ENABLE_LDGSTS) {
            Type_colwerter<typename Data_type_in_smem_t::Type,
                           typename Data_type_in_gmem_t::Type,
                           Packed_data_type_in_smem_t,
                           Packed_data_type_in_gmem_t,
                           INSTRUCTION_PER_THREAD_IN_GMEM>::exlwte(data_of_smem_, data_of_gmem_);
        }
    }

    __device__ inline void get_the_data_for_storing_to_gmem(
            Packed_data_type_in_smem_t in[STSS_PER_THREAD],
            const int32_t index_iteration)
    {
#pragma unroll
        for (int32_t i = 0; i < INSTRUCTION_PER_THREAD_PER_ITERATION_IN_GMEM; ++i) {
            data_of_smem_[INSTRUCTION_PER_THREAD_PER_ITERATION_IN_GMEM * index_iteration + i] =
                in[i];
        }
    }

    __device__ inline void type_colwersion_for_storing_to_gmem()
    {
        Type_colwerter<typename Data_type_in_gmem_t::Type,
                       typename Data_type_in_smem_t::Type,
                       Packed_data_type_in_gmem_t,
                       Packed_data_type_in_smem_t,
                       INSTRUCTION_PER_THREAD_IN_GMEM>::exlwte(data_of_gmem_, data_of_smem_);
    }

    __device__ inline void set_ptrs_of_instruction_in_smem(const int32_t index_stage)
    {
        uint32_t ptr_base_sts_with_stage_offset =
            this->ptr_base_sts_ + index_stage * SMEM_SIZE_IN_BYTES_PER_STAGE;
        get_linear_index<STSS_PER_THREAD,
                         Tile_3d_t::HEIGHT * Tile_3d_t::WIDTH * BYTES_PER_TILE_G_IN_SMEM,
                         Tile_3d_t::WIDTH * BYTES_PER_TILE_G_IN_SMEM,
                         BYTES_PER_TILE_G_IN_SMEM>(this->ptrs_sts_,
                                                   ptr_base_sts_with_stage_offset,
                                                   this->offset_depth_,
                                                   this->offset_height_,
                                                   this->offset_width_);
    }

    __device__ inline void store_to_smem()
    {
        if (!ENABLE_LDGSTS) {
#pragma unroll
            for (int32_t i = 0; i < STSS_PER_THREAD; ++i) {
                xmma::sts(ptrs_sts_[i], data_of_smem_[i]);
            }
        }
    }

    __device__ inline void load_from_smem()
    {
#pragma unroll
        for (int32_t i = 0; i < STSS_PER_THREAD; ++i) {
            xmma::lds(data_of_smem_[i], ptrs_sts_[i]);
        }
    }

    __device__ inline void store_to_gmem(const int32_t index_begin, const int32_t index_end)
    {
#pragma unroll
        for (int32_t i = index_begin; i < index_end; ++i) {
            if (preds_of_instruction_in_gmem_[i]) {
                *static_cast<Packed_data_type_in_gmem_t *>(ptrs_instruction_in_gmem_[i]) = 
                    data_of_gmem_[i];
            }
        }
    }

    __device__ inline void store_to_gmem() { store_to_gmem(0, INSTRUCTION_PER_THREAD_IN_GMEM); }

    __device__ inline void store_to_gmem(const int32_t index_iteration)
    {
        store_to_gmem(INSTRUCTION_PER_THREAD_PER_ITERATION_IN_GMEM * index_iteration,
                      INSTRUCTION_PER_THREAD_PER_ITERATION_IN_GMEM * (index_iteration + 1));
    }

    void *params_gmem_;
    int32_t params_pad_front_;
    int32_t params_pad_top_;
    int32_t params_pad_left_;
    int32_t params_batch_;
    int32_t params_depth_;
    int32_t params_height_;
    int32_t params_width_;
    int32_t params_group_;
    uint32_t params_tensor_stride_batch_;
    uint32_t params_tensor_stride_depth_;
    uint32_t params_tensor_stride_height_;
    uint32_t params_tensor_stride_width_;
    int32_t params_tiles_opq_;
    uint32_t params_mul_tiles_opq_;
    uint32_t params_shr_tiles_opq_;
    int32_t params_tiles_pq_;
    uint32_t params_mul_tiles_pq_;
    uint32_t params_shr_tiles_pq_;
    int32_t params_tiles_q_;
    uint32_t params_mul_tiles_q_;
    uint32_t params_shr_tiles_q_;
    int32_t params_tiles_o_;
    int32_t params_tiles_p_;
    int32_t params_delta_tiles_n_;
    int32_t params_delta_tiles_o_;
    int32_t params_delta_tiles_p_;
    int32_t params_delta_tiles_q_;

    uint32_t params_alpha_;
    uint32_t params_beta_;

    void *ptr_base_instruction_in_gmem_;
    void *ptr_instruction_in_gmem_;
    void *ptrs_instruction_in_gmem_[INSTRUCTION_PER_THREAD_IN_GMEM];
    uint32_t ptr_base_sts_;
    uint32_t ptr_sts_;
    uint32_t ptrs_sts_[STSS_PER_THREAD];
    Packed_data_type_in_gmem_t data_of_gmem_[INSTRUCTION_PER_THREAD_IN_GMEM];
    Packed_data_type_in_smem_t data_of_smem_[STSS_PER_THREAD];
    int32_t offset_depth_[INSTRUCTION_PER_THREAD_IN_GMEM];
    int32_t offset_height_[INSTRUCTION_PER_THREAD_IN_GMEM];
    int32_t offset_width_[INSTRUCTION_PER_THREAD_IN_GMEM];
    int32_t offset_in_bytes_[INSTRUCTION_PER_THREAD_IN_GMEM];
    bool preds_of_instruction_in_gmem_[INSTRUCTION_PER_THREAD_IN_GMEM];
    bool valid_group_;
};

template <typename Gpu_arch_,
          typename Cta_tile_,
          typename Data_type_in_gmem_,
          typename Data_type_in_smem_,
          int32_t ELEMENTS_PER_INSTRUCTION_IN_GMEM,
          int32_t ITERATION_NUMBER_IN_GMEM,
          int32_t TENSOR_TYPE_>
struct Gmem_tile_ndhwg : public Gmem_tile<Gpu_arch_,
                                          typename Cta_tile_::Tile_memory_per_cta_t::Tile_dhw_t,
                                          Cta_tile_::Tile_memory_per_cta_t::TILE_G,
                                          Cta_tile_::STAGE,
                                          Cta_tile_::THREADS_PER_CTA,
                                          Data_type_in_gmem_,
                                          Data_type_in_smem_,
                                          ELEMENTS_PER_INSTRUCTION_IN_GMEM,
                                          ITERATION_NUMBER_IN_GMEM> {

    using Base_t = Gmem_tile<Gpu_arch_,
                             typename Cta_tile_::Tile_memory_per_cta_t::Tile_dhw_t,
                             Cta_tile_::Tile_memory_per_cta_t::TILE_G,
                             Cta_tile_::STAGE,
                             Cta_tile_::THREADS_PER_CTA,
                             Data_type_in_gmem_,
                             Data_type_in_smem_,
                             ELEMENTS_PER_INSTRUCTION_IN_GMEM,
                             ITERATION_NUMBER_IN_GMEM>;
    using Cta_tile_t = Cta_tile_;
    using Tile_stride_dhw_t = typename Cta_tile_t::Tile_memory_per_cta_t::Tile_stride_dhw_t;
    using Tile_trs_t = typename Cta_tile_t::Tile_memory_per_cta_t::Tile_trs_t;
    using Tile_dilation_dhw_t = typename Cta_tile_t::Tile_memory_per_cta_t::Tile_dilation_dhw_t;
    static const int32_t ENABLE_LDGSTS = Base_t::ENABLE_LDGSTS;
    static const int32_t SMEM_SIZE_IN_BYTES = Base_t::SMEM_SIZE_IN_BYTES;
    static const int32_t TENSOR_TYPE = TENSOR_TYPE_;
    static_assert((TENSOR_TYPE == Tensor_type::A || TENSOR_TYPE == Tensor_type::C ||
                   TENSOR_TYPE == Tensor_type::D),
                  "");
    using Abstract_tile_t = typename Base_t::Abstract_tile_t;

    __device__ inline Gmem_tile_ndhwg(Abstract_tile_t &abstract_tile,
                                      const uint32_t smem_base_address,
                                      const Depthwise_colwolution_parameter<Cta_tile_> &params)
        : Base_t(abstract_tile), params_stride_depth_(params.stride[0]),
          params_stride_height_(params.stride[1]), params_stride_width_(params.stride[2]),
          params_dilation_depth_(params.dilation[0]), params_dilation_height_(params.dilation[1]),
          params_dilation_width_(params.dilation[2]), params_t_(params.t), params_r_(params.r),
          params_s_(params.s), params_is_colwolution_(params.is_colwolution)
    {
        set_gmem_ptr_instruction_in_gmem_base(params.gmem[TENSOR_TYPE]);
        set_pad(params.pad[0][0], params.pad[1][0], params.pad[2][0]);
        set_tensor_dims(params.n, params.d, params.h, params.w, params.g);
        set_tensor_strides(
            params.img_stride_n, params.img_stride_d, params.img_stride_h, params.img_stride_w);
        Base_t::initialize_ptr_base(smem_base_address);
    }

    __device__ inline void get_tile_begin(int32_t &cta_tile_begin_depth,
                                          int32_t &cta_tile_begin_height,
                                          int32_t &cta_tile_begin_width,
                                          const int32_t cta_tile_index_o,
                                          const int32_t cta_tile_index_p,
                                          const int32_t cta_tile_index_q,
                                          const int32_t cta_tile_begin_t,
                                          const int32_t cta_tile_begin_r,
                                          const int32_t cta_tile_begin_s)
    {
        int32_t real_cta_tile_begin_t, real_cta_tile_begin_r, real_cta_tile_begin_s;
        if (params_is_colwolution_) {
            real_cta_tile_begin_t = (params_t_ - 1 - (cta_tile_begin_t + (Tile_trs_t::DEPTH - 1)));
            real_cta_tile_begin_r = (params_r_ - 1 - (cta_tile_begin_r + (Tile_trs_t::HEIGHT - 1)));
            real_cta_tile_begin_s = (params_s_ - 1 - (cta_tile_begin_s + (Tile_trs_t::WIDTH - 1)));
        } else {
            real_cta_tile_begin_t = cta_tile_begin_t;
            real_cta_tile_begin_r = cta_tile_begin_r;
            real_cta_tile_begin_s = cta_tile_begin_s;
        }
        cta_tile_begin_depth =
            multiply_and_add<Tile_stride_dhw_t::IS_POSITIVE,
                             Tile_stride_dhw_t::DEPTH,
                             Tile_dilation_dhw_t::DEPTH>(-this->params_pad_front_,
                                                         params_stride_depth_,
                                                         cta_tile_index_o,
                                                         params_dilation_depth_,
                                                         real_cta_tile_begin_t);
        cta_tile_begin_height =
            multiply_and_add<Tile_stride_dhw_t::IS_POSITIVE,
                             Tile_stride_dhw_t::HEIGHT,
                             Tile_dilation_dhw_t::HEIGHT>(-this->params_pad_top_,
                                                          params_stride_height_,
                                                          cta_tile_index_p,
                                                          params_dilation_height_,
                                                          real_cta_tile_begin_r);
        cta_tile_begin_width = multiply_and_add<Tile_stride_dhw_t::IS_POSITIVE,
                                                Tile_stride_dhw_t::WIDTH,
                                                Tile_dilation_dhw_t::WIDTH>(-this->params_pad_left_,
                                                                            params_stride_width_,
                                                                            cta_tile_index_q,
                                                                            params_dilation_width_,
                                                                            real_cta_tile_begin_s);
    }

    __device__ inline void set_ptrs_of_instruction_in_gmem()
    {
        Base_t::set_ptrs_of_instruction_in_gmem();
    }

    __device__ inline void set_ptrs_of_instruction_in_smem(const int32_t index_stage)
    {
        Base_t::set_ptrs_of_instruction_in_smem(index_stage);
    }

    int32_t params_stride_depth_;
    int32_t params_stride_height_;
    int32_t params_stride_width_;
    int32_t params_dilation_depth_;
    int32_t params_dilation_height_;
    int32_t params_dilation_width_;
    int32_t params_t_;
    int32_t params_r_;
    int32_t params_s_;
    bool params_is_colwolution_;
};

template <typename Gpu_arch_,
          typename Cta_tile_,
          typename Data_type_in_gmem_,
          typename Data_type_in_smem_,
          int32_t ELEMENTS_PER_INSTRUCTION_IN_GMEM,
          int32_t ITERATION_NUMBER_IN_GMEM,
          int32_t TENSOR_TYPE_,
          int32_t STAGE_ = Cta_tile_::STAGE>
struct Gmem_tile_nopqg : public Gmem_tile<Gpu_arch_,
                                          typename Cta_tile_::Tile_memory_per_cta_t::Tile_opq_t,
                                          Cta_tile_::Tile_memory_per_cta_t::TILE_G,
                                          STAGE_,
                                          Cta_tile_::THREADS_PER_CTA,
                                          Data_type_in_gmem_,
                                          Data_type_in_smem_,
                                          ELEMENTS_PER_INSTRUCTION_IN_GMEM,
                                          ITERATION_NUMBER_IN_GMEM> {

    using Base_t = Gmem_tile<Gpu_arch_,
                             typename Cta_tile_::Tile_memory_per_cta_t::Tile_opq_t,
                             Cta_tile_::Tile_memory_per_cta_t::TILE_G,
                             STAGE_,
                             Cta_tile_::THREADS_PER_CTA,
                             Data_type_in_gmem_,
                             Data_type_in_smem_,
                             ELEMENTS_PER_INSTRUCTION_IN_GMEM,
                             ITERATION_NUMBER_IN_GMEM>;
    using Packed_data_type_in_gmem_t = typename Base_t::Packed_data_type_in_gmem_t;
    using Packed_data_type_in_smem_t = typename Base_t::Packed_data_type_in_smem_t;
    using Data_type_in_smem_t = typename Base_t::Data_type_in_smem_t;
    static const int32_t STSS_PER_THREAD = Base_t::STSS_PER_THREAD;

    using Tile_3d_t = typename Base_t::Tile_3d_t;
    static const int32_t TENSOR_TYPE = TENSOR_TYPE_;
    static_assert((TENSOR_TYPE == Tensor_type::B || TENSOR_TYPE == Tensor_type::C ||
                   TENSOR_TYPE == Tensor_type::D),
                  "");
    using Abstract_tile_t = typename Base_t::Abstract_tile_t;

    __device__ inline Gmem_tile_nopqg(Abstract_tile_t &abstract_tile,
                                      const uint32_t smem_base_address,
                                      const Depthwise_colwolution_parameter<Cta_tile_> &params)
        : Base_t(abstract_tile)
    {
        set_gmem_ptr_instruction_in_gmem_base(params.gmem[TENSOR_TYPE]);
        set_pad(params.pad[0][0], params.pad[1][0], params.pad[2][0]);
        set_tensor_dims(params.n, params.o, params.p, params.q, params.g);
        set_tensor_strides(
            params.out_stride_n, params.out_stride_o, params.out_stride_p, params.out_stride_q);
        set_params_tiles(params.tiles_opq,
                         params.mul_tiles_opq,
                         params.shr_tiles_opq,
                         params.tiles_pq,
                         params.mul_tiles_pq,
                         params.shr_tiles_pq,
                         params.tiles_q,
                         params.mul_tiles_q,
                         params.shr_tiles_q,
                         params.tiles_p,
                         params.tiles_o,
                         params.delta_tiles_n,
                         params.delta_tiles_o,
                         params.delta_tiles_p,
                         params.delta_tiles_q);
        set_alpha(params.alpha);
        set_beta(params.beta);
        Base_t::initialize_ptr_base(smem_base_address);
    }

    __device__ inline void get_tile_begin(int32_t &cta_tile_begin_depth,
                                          int32_t &cta_tile_begin_height,
                                          int32_t &cta_tile_begin_width,
                                          const int32_t cta_tile_index_o,
                                          const int32_t cta_tile_index_p,
                                          const int32_t cta_tile_index_q,
                                          const int32_t cta_tile_begin_t,
                                          const int32_t cta_tile_begin_r,
                                          const int32_t cta_tile_begin_s)
    {
        cta_tile_begin_depth = cta_tile_index_o * Tile_3d_t::DEPTH;
        cta_tile_begin_height = cta_tile_index_p * Tile_3d_t::HEIGHT;
        cta_tile_begin_width = cta_tile_index_q * Tile_3d_t::WIDTH;
    }

    __device__ inline void set_ptrs_of_instruction_in_gmem()
    {
        Base_t::set_ptrs_of_instruction_in_gmem();
    }

    __device__ inline void set_ptrs_of_instruction_in_gmem(const int32_t index_iteration)
    {
        Base_t::set_ptrs_of_instruction_in_gmem(index_iteration);
    }

    __device__ inline void set_ptrs_of_instruction_in_smem(const int32_t index_stage)
    {
        Base_t::set_ptrs_of_instruction_in_smem(index_stage);
    }
};

template <typename Gpu_arch_,
          typename Cta_tile_,
          typename Data_type_in_gmem_,
          typename Data_type_in_smem_,
          int32_t ELEMENTS_PER_INSTRUCTION_IN_GMEM>
struct Gmem_tile_trsg : public Gmem_tile<Gpu_arch_,
                                         typename Cta_tile_::Tile_memory_per_cta_t::Tile_trs_t,
                                         Cta_tile_::Tile_memory_per_cta_t::TILE_G,
                                         1 /* STAGE */,
                                         Cta_tile_::THREADS_PER_CTA,
                                         Data_type_in_gmem_,
                                         Data_type_in_smem_,
                                         ELEMENTS_PER_INSTRUCTION_IN_GMEM,
                                         1 /* ITERATION_NUMBER_IN_GMEM */> {

    using Base_t = Gmem_tile<Gpu_arch_,
                             typename Cta_tile_::Tile_memory_per_cta_t::Tile_trs_t,
                             Cta_tile_::Tile_memory_per_cta_t::TILE_G,
                             1 /* STAGE */,
                             Cta_tile_::THREADS_PER_CTA,
                             Data_type_in_gmem_,
                             Data_type_in_smem_,
                             ELEMENTS_PER_INSTRUCTION_IN_GMEM,
                             1 /* ITERATION_NUMBER_IN_GMEM */>;
    using Tile_3d_t = typename Base_t::Tile_3d_t;
    static const int32_t TENSOR_TYPE = Tensor_type::B;
    static_assert(TENSOR_TYPE == Tensor_type::B, "");
    using Abstract_tile_t = typename Base_t::Abstract_tile_t;

    __device__ inline Gmem_tile_trsg(Abstract_tile_t &abstract_tile,
                                     const uint32_t smem_base_address,
                                     const Depthwise_colwolution_parameter<Cta_tile_> &params)
        : Base_t(abstract_tile)
    {
        set_gmem_ptr_instruction_in_gmem_base(params.gmem[TENSOR_TYPE]);
        set_tensor_dims(1 /* batch */, params.t, params.r, params.s, params.g);
        set_tensor_strides(
            params.flt_stride_t, params.flt_stride_t, params.flt_stride_r, params.flt_stride_s);
        set_params_tiles(params.tiles_trs,
                         params.mul_tiles_trs,
                         params.shr_tiles_trs,
                         params.tiles_rs,
                         params.mul_tiles_rs,
                         params.shr_tiles_rs,
                         params.tiles_s,
                         params.mul_tiles_s,
                         params.shr_tiles_s,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0);
        Base_t::initialize_ptr_base(smem_base_address);
    }

    __device__ inline void get_tile_begin(int32_t &cta_tile_begin_depth,
                                          int32_t &cta_tile_begin_height,
                                          int32_t &cta_tile_begin_width,
                                          const int32_t cta_tile_index_o,
                                          const int32_t cta_tile_index_p,
                                          const int32_t cta_tile_index_q,
                                          const int32_t cta_tile_begin_t,
                                          const int32_t cta_tile_begin_r,
                                          const int32_t cta_tile_begin_s)
    {
        cta_tile_begin_depth = cta_tile_begin_t;
        cta_tile_begin_height = cta_tile_begin_r;
        cta_tile_begin_width = cta_tile_begin_s;
    }

    __device__ inline void set_ptrs_of_instruction_in_gmem()
    {
        Base_t::set_ptrs_of_instruction_in_gmem();
    }

    __device__ inline void set_ptrs_of_instruction_in_smem(const int32_t index_stage)
    {
        Base_t::set_ptrs_of_instruction_in_smem(index_stage);
    }
};

} // namespace depthwise_colwolution
} // namespace ext
} // namespace xmma

#endif
