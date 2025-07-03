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

#ifndef _XMMA_EXT_DEPTHWISE_COLWOLUTION_FPROP_KERNEL_H
#define _XMMA_EXT_DEPTHWISE_COLWOLUTION_FPROP_KERNEL_H

#pragma once

#include "xmma/ext/depthwise_colwolution/cta_tile.h"
#include "xmma/ext/depthwise_colwolution/gmem_tile.h"
#include "xmma/ext/depthwise_colwolution/math_tile.h"
#include "xmma/ext/depthwise_colwolution/params.h"
#include "xmma/ext/depthwise_colwolution/smem_tile.h"
#include "xmma/ext/depthwise_colwolution/smem_tile_trsg.h"
#include "xmma/ext/depthwise_colwolution/split_k.h"
#include "xmma/ext/depthwise_colwolution/utils.h"
#include <cstdint>
#include <lwda_runtime.h>

namespace xmma
{
namespace ext
{
namespace depthwise_colwolution
{
namespace fprop
{

//#define BUFFER_IN_SMEM
#define USE_INCREASING_MODE false
template <typename Kernel_traits>
__global__ static void kernel(typename Kernel_traits::Params params)
{
    extern __shared__ char smem[];

    using Cta_tile_t = typename Kernel_traits::Cta_tile_t;
    using Tile_trs_t = typename Cta_tile_t::Tile_memory_per_cta_t::Tile_trs_t;
    using Gmem_tile_loader_ndhwg_t = typename Kernel_traits::Gmem_tile_loader_ndhwg_t;
    using Gmem_tile_loader_nopqg_t = typename Kernel_traits::Gmem_tile_loader_nopqg_t;
    using Gmem_tile_storer_nopqg_t = typename Kernel_traits::Gmem_tile_storer_nopqg_t;
    using Gmem_tile_loader_trsg_t = typename Kernel_traits::Gmem_tile_loader_trsg_t;
    using Abstract_tile_ndhwg_t = typename Kernel_traits::Abstract_tile_ndhwg_t;
    using Abstract_tile_nopqg_t = typename Kernel_traits::Abstract_tile_nopqg_t;
    using Abstract_tile_trsg_t = typename Kernel_traits::Abstract_tile_trsg_t;
    using Smem_tile_ndhwg_t = typename Kernel_traits::Smem_tile_ndhwg_t;
    using Smem_tile_nopqg_t = typename Kernel_traits::Smem_tile_nopqg_t;
    using Smem_tile_loader_nopqg_t = typename Kernel_traits::Smem_tile_loader_nopqg_t;
    using Smem_tile_trsg_t = typename Kernel_traits::Smem_tile_trsg_t;
    using Math_tile_tmp_trsg_t = typename Kernel_traits::Math_tile_tmp_trsg_t;
    using Math_tile_tmp_ndhwg_t = typename Kernel_traits::Math_tile_tmp_ndhwg_t;
    using Math_tile_tmp_nopqg_t = typename Kernel_traits::Math_tile_tmp_nopqg_t;
    using Math_tile_tmp_loader_nopqg_t = typename Kernel_traits::Math_tile_tmp_loader_nopqg_t;
    using Math_tile_trsg_t = typename Kernel_traits::Math_tile_trsg_t;
    using Math_tile_ndhwg_t = typename Kernel_traits::Math_tile_ndhwg_t;
    using Math_tile_nopqg_t = typename Kernel_traits::Math_tile_nopqg_t;
    using Math_tile_loader_nopqg_t = typename Kernel_traits::Math_tile_loader_nopqg_t;
    using Math_operation_t = typename Kernel_traits::Math_operation_t;
    static const bool SUPPORT_BETA = Kernel_traits::SUPPORT_BETA;

    int32_t cta_tile_index_t, cta_tile_index_r, cta_tile_index_s;
    int32_t cta_tile_index_rs;
    xmma::fast_divmod(cta_tile_index_t,
                      cta_tile_index_rs,
                      blockIdx_x(),
                      params.tiles_rs,
                      params.mul_tiles_rs,
                      params.shr_tiles_rs);
    xmma::fast_divmod(cta_tile_index_r,
                      cta_tile_index_s,
                      cta_tile_index_rs,
                      params.tiles_s,
                      params.mul_tiles_s,
                      params.shr_tiles_s);
    int32_t cta_tile_begin_t = cta_tile_index_t * Tile_trs_t::DEPTH;
    int32_t cta_tile_begin_r = cta_tile_index_r * Tile_trs_t::HEIGHT;
    int32_t cta_tile_begin_s = cta_tile_index_s * Tile_trs_t::WIDTH;

    uint32_t smem_base_address = xmma::get_smem_pointer(smem);
    uint32_t smem_base_address_ndhwg = smem_base_address;
    uint32_t smem_base_address_trsg =
        smem_base_address_ndhwg + Gmem_tile_loader_ndhwg_t::SMEM_SIZE_IN_BYTES;
    uint32_t smem_base_address_storer_nopqg =
        smem_base_address_trsg + Gmem_tile_loader_trsg_t::SMEM_SIZE_IN_BYTES;
    uint32_t smem_base_address_loader_nopqg =
        smem_base_address_storer_nopqg + Gmem_tile_storer_nopqg_t::SMEM_SIZE_IN_BYTES;
    Abstract_tile_ndhwg_t abstract_tile_ndhwg;
    Abstract_tile_trsg_t abstract_tile_trsg;
    Abstract_tile_nopqg_t abstract_tile_nopqg;
    Gmem_tile_loader_ndhwg_t gmem_loader_ndhwg(
        abstract_tile_ndhwg, smem_base_address_ndhwg, params);
    Gmem_tile_loader_trsg_t gmem_trsg(abstract_tile_trsg, smem_base_address_trsg, params);
    Gmem_tile_storer_nopqg_t gmem_storer_nopqg(
        abstract_tile_nopqg, smem_base_address_storer_nopqg, params);
    Gmem_tile_loader_nopqg_t gmem_loader_nopqg(
        abstract_tile_nopqg, smem_base_address_loader_nopqg, params);

    int32_t index_cta_tile = blockIdx_z();
    int32_t cta_tile_index_batch, cta_tile_index_depth, cta_tile_index_height, cta_tile_index_width;
#ifdef BUFFER_IN_SMEM
    __shared__ int32_t cta_tile_begin_batch[Cta_tile_t::STAGE];
    __shared__ int32_t cta_tile_begin_o[Cta_tile_t::STAGE];
    __shared__ int32_t cta_tile_begin_p[Cta_tile_t::STAGE];
    __shared__ int32_t cta_tile_begin_q[Cta_tile_t::STAGE];
#else
    int32_t cta_tile_begin_batch[Cta_tile_t::STAGE];
    int32_t cta_tile_begin_o[Cta_tile_t::STAGE];
    int32_t cta_tile_begin_p[Cta_tile_t::STAGE];
    int32_t cta_tile_begin_q[Cta_tile_t::STAGE];
#endif

    int32_t cta_tile_begin_d, cta_tile_begin_h, cta_tile_begin_w;
    int32_t tmp_cta_tile_index_batch, tmp_cta_tile_begin_o, tmp_cta_tile_begin_p,
        tmp_cta_tile_begin_q;
#pragma unroll
    for (int32_t index_prefetch_stage = 0; index_prefetch_stage < Cta_tile_t::STAGE;
         ++index_prefetch_stage) {
        xmma::ext::depthwise_colwolution::set_memory_no_alias();
        gmem_storer_nopqg.decompose_tile_index(cta_tile_index_batch,
                                               cta_tile_index_depth,
                                               cta_tile_index_height,
                                               cta_tile_index_width,
                                               index_cta_tile,
                                               USE_INCREASING_MODE && (index_prefetch_stage > 0));
        tmp_cta_tile_index_batch = cta_tile_index_batch;
        gmem_storer_nopqg.get_tile_begin(tmp_cta_tile_begin_o,
                                         tmp_cta_tile_begin_p,
                                         tmp_cta_tile_begin_q,
                                         cta_tile_index_depth,
                                         cta_tile_index_height,
                                         cta_tile_index_width,
                                         cta_tile_begin_t,
                                         cta_tile_begin_r,
                                         cta_tile_begin_s);
#ifdef BUFFER_IN_SMEM
        if (threadIdx_x() == 0) {
#endif
            cta_tile_begin_batch[index_prefetch_stage] = tmp_cta_tile_index_batch;
            cta_tile_begin_o[index_prefetch_stage] = tmp_cta_tile_begin_o;
            cta_tile_begin_p[index_prefetch_stage] = tmp_cta_tile_begin_p;
            cta_tile_begin_q[index_prefetch_stage] = tmp_cta_tile_begin_q;
#ifdef BUFFER_IN_SMEM
        }
#endif
        gmem_loader_ndhwg.get_tile_begin(cta_tile_begin_d,
                                         cta_tile_begin_h,
                                         cta_tile_begin_w,
                                         tmp_cta_tile_begin_o,
                                         tmp_cta_tile_begin_p,
                                         tmp_cta_tile_begin_q,
                                         cta_tile_begin_t,
                                         cta_tile_begin_r,
                                         cta_tile_begin_s);
        if (index_prefetch_stage == 0) {
            gmem_trsg.get_tile_begin(cta_tile_begin_t,
                                     cta_tile_begin_r,
                                     cta_tile_begin_s,
                                     0,
                                     0,
                                     0,
                                     cta_tile_begin_t,
                                     cta_tile_begin_r,
                                     cta_tile_begin_s);
        }
        if (SUPPORT_BETA) {
            gmem_loader_nopqg.move(tmp_cta_tile_index_batch,
                                   tmp_cta_tile_begin_o,
                                   tmp_cta_tile_begin_p,
                                   tmp_cta_tile_begin_q);
        }
        gmem_loader_ndhwg.move(
            tmp_cta_tile_index_batch, cta_tile_begin_d, cta_tile_begin_h, cta_tile_begin_w);
        if (index_prefetch_stage == 0) {
            gmem_trsg.move(0, 0, 0, 0);
        }
        if (SUPPORT_BETA) {
            gmem_loader_nopqg.set_preds_of_instruction_in_gmem(tmp_cta_tile_index_batch,
                                                               tmp_cta_tile_begin_o,
                                                               tmp_cta_tile_begin_p,
                                                               tmp_cta_tile_begin_q);
        }
        gmem_loader_ndhwg.set_preds_of_instruction_in_gmem(
            tmp_cta_tile_index_batch, cta_tile_begin_d, cta_tile_begin_h, cta_tile_begin_w);
        if (index_prefetch_stage == 0) {
            gmem_trsg.set_preds_of_instruction_in_gmem(0, 0, 0, 0);
        }
        if (SUPPORT_BETA) {
            gmem_loader_nopqg.set_ptrs_of_instruction_in_gmem();
        }
        gmem_loader_ndhwg.set_ptrs_of_instruction_in_gmem();
        if (index_prefetch_stage == 0) {
            gmem_trsg.set_ptrs_of_instruction_in_gmem();
        }
        if (SUPPORT_BETA) {
            gmem_loader_nopqg.set_ptrs_of_instruction_in_smem(index_prefetch_stage);
        }
        gmem_loader_ndhwg.set_ptrs_of_instruction_in_smem(index_prefetch_stage);
        if (index_prefetch_stage == 0) {
            gmem_trsg.set_ptrs_of_instruction_in_smem(index_prefetch_stage);
        }
        if (index_prefetch_stage < Cta_tile_t::STAGE - 1) {
            if (SUPPORT_BETA) {
                gmem_loader_nopqg.load_from_gmem();
            }
            gmem_loader_ndhwg.load_from_gmem();
            if (index_prefetch_stage == 0) {
                gmem_trsg.load_from_gmem();
            }
            xmma::ldgdepbar<Kernel_traits::ENABLE_LDGSTS>();
            if (SUPPORT_BETA) {
                gmem_loader_nopqg.type_colwersion();
            }
            gmem_loader_ndhwg.type_colwersion();
            if (index_prefetch_stage == 0) {
                gmem_trsg.type_colwersion();
            }
            if (SUPPORT_BETA) {
                gmem_loader_nopqg.store_to_smem();
            }
            gmem_loader_ndhwg.store_to_smem();
            if (index_prefetch_stage == 0) {
                gmem_trsg.store_to_smem();
            }
            index_cta_tile += params.split_m_slices;
        }
        xmma::ext::depthwise_colwolution::reset_memory_no_alias();
    }
    Smem_tile_nopqg_t smem_storer_nopqg(smem_base_address_storer_nopqg);
    Smem_tile_loader_nopqg_t smem_loader_nopqg(smem_base_address_loader_nopqg);
    Smem_tile_trsg_t smem_loader_trsg(smem_base_address_trsg);
    Smem_tile_ndhwg_t smem_loader_ndhwg(smem_base_address_ndhwg,
                                        params.stride[0],
                                        params.stride[1],
                                        params.stride[2],
                                        params.dilation[0],
                                        params.dilation[1],
                                        params.dilation[2],
                                        params.is_colwolution);
    smem_storer_nopqg.set_offsets_base_lds();
    if (SUPPORT_BETA) {
        smem_loader_nopqg.set_offsets_base_lds();
    }
    smem_loader_ndhwg.set_offsets_base_lds(smem_storer_nopqg.offset_base_depth_,
                                           smem_storer_nopqg.offset_base_height_,
                                           smem_storer_nopqg.offset_base_width_);
    int32_t index_stage_smem = 0;
#define GET_THE_INDEX_OF_THE_LWRRENT_TILE_IN_SMEM(idx)                                             \
    tmp_cta_tile_index_batch = cta_tile_begin_batch[idx];                                          \
    tmp_cta_tile_begin_o = cta_tile_begin_o[idx];                                                  \
    tmp_cta_tile_begin_p = cta_tile_begin_p[idx];                                                  \
    tmp_cta_tile_begin_q = cta_tile_begin_q[idx];
#define GET_THE_INDEX_OF_THE_LWRRENT_TILE(idx)                                                     \
    if (index_stage_smem == idx) {                                                                 \
        GET_THE_INDEX_OF_THE_LWRRENT_TILE_IN_SMEM(idx)                                             \
    }

#ifdef BUFFER_IN_SMEM
    GET_THE_INDEX_OF_THE_LWRRENT_TILE_IN_SMEM(index_stage_smem);
#else
    if (Cta_tile_t::STAGE >= 2) {
        GET_THE_INDEX_OF_THE_LWRRENT_TILE(0);
        GET_THE_INDEX_OF_THE_LWRRENT_TILE(1);
    }
#endif

    gmem_storer_nopqg.move(cta_tile_begin_batch[index_stage_smem],
                           cta_tile_begin_o[index_stage_smem],
                           cta_tile_begin_p[index_stage_smem],
                           cta_tile_begin_q[index_stage_smem]);
    gmem_storer_nopqg.set_preds_of_instruction_in_gmem(
        tmp_cta_tile_index_batch, tmp_cta_tile_begin_o, tmp_cta_tile_begin_p, tmp_cta_tile_begin_q);
    gmem_storer_nopqg.set_ptrs_of_instruction_in_gmem();
    gmem_storer_nopqg.set_ptrs_of_instruction_in_smem(0);

    smem_storer_nopqg.set_offsets_lds(0);
    if (SUPPORT_BETA) {
        smem_loader_nopqg.set_offsets_lds(0);
    }
    smem_loader_ndhwg.set_offsets_lds(0);
    smem_storer_nopqg.set_ptrs_lds(0);
    if (SUPPORT_BETA) {
        smem_loader_nopqg.set_ptrs_lds(index_stage_smem);
    }
    smem_loader_ndhwg.set_ptrs_lds(index_stage_smem);

    xmma::depbar<Kernel_traits::ENABLE_LDGSTS, Cta_tile_t::STAGE>();
    __syncthreads();
    smem_loader_trsg.set_ptrs_lds();
    smem_loader_trsg.load_from_smem();
    Math_tile_tmp_trsg_t math_tile_tmp_trsg(smem_loader_trsg.fetch_math_);
    math_tile_tmp_trsg.flip(params.is_colwolution);
    Math_tile_trsg_t math_tile_trsg(math_tile_tmp_trsg);
    Math_tile_nopqg_t math_tile_nopqg;
    math_tile_nopqg.clear();
    Math_operation_t math_operation;
    if (SUPPORT_BETA) {
        smem_loader_nopqg.load_from_smem(0);
    }
    smem_loader_ndhwg.load_from_smem(0);
    for (int32_t index_main_loop = 0; index_main_loop < params.count_main_loop; ++index_main_loop) {
        xmma::ext::depthwise_colwolution::set_memory_no_alias();
        if (SUPPORT_BETA) {
            gmem_loader_nopqg.load_from_gmem();
        }
        gmem_loader_ndhwg.load_from_gmem();
        xmma::ldgdepbar<Kernel_traits::ENABLE_LDGSTS>();
#pragma unroll
        for (int32_t index_iteration = 0; index_iteration < Kernel_traits::ITERATION_NUMBER;
             ++index_iteration) {
            Math_tile_tmp_loader_nopqg_t math_tile_tmp_loader_nopqg(
                smem_loader_nopqg.fetch_math_[index_iteration %
                                              Smem_tile_loader_nopqg_t::NUMBER_OF_MATH_BUFFERS]);
            Math_tile_loader_nopqg_t math_tile_loader_nopqg(math_tile_tmp_loader_nopqg);
            Math_tile_tmp_ndhwg_t math_tile_tmp_ndhwg(
                smem_loader_ndhwg
                    .fetch_math_[index_iteration % Smem_tile_ndhwg_t::NUMBER_OF_MATH_BUFFERS]);
            Math_tile_ndhwg_t math_tile_ndhwg(math_tile_tmp_ndhwg);
            math_operation.exlwte(math_tile_nopqg, math_tile_ndhwg, math_tile_trsg);
            math_tile_nopqg.apply_alpha(params.alpha);
            if (SUPPORT_BETA) {
                math_tile_nopqg.apply_beta(math_tile_loader_nopqg.data_, params.beta);
            }
            Math_tile_tmp_nopqg_t math_tile_tmp_nopqg(math_tile_nopqg);
            smem_storer_nopqg.store_to_smem(math_tile_tmp_nopqg.data_);
            math_tile_nopqg.clear();
            int32_t index_iteration_smem = index_iteration;

            index_iteration_smem =
                increase_and_mod<Kernel_traits::ITERATION_NUMBER>(index_iteration_smem);
            smem_storer_nopqg.set_offsets_lds(index_iteration_smem);
            if (SUPPORT_BETA) {
                smem_loader_nopqg.set_offsets_lds(index_iteration_smem);
            }
            smem_loader_ndhwg.set_offsets_lds(index_iteration_smem);

            bool is_next_stage = (index_iteration == Kernel_traits::ITERATION_NUMBER - 1);
            if (is_next_stage) {
                index_cta_tile += params.split_m_slices;
                gmem_storer_nopqg.decompose_tile_index(cta_tile_index_batch,
                                                       cta_tile_index_depth,
                                                       cta_tile_index_height,
                                                       cta_tile_index_width,
                                                       index_cta_tile,
                                                       USE_INCREASING_MODE);
                tmp_cta_tile_index_batch = cta_tile_index_batch;
                gmem_storer_nopqg.get_tile_begin(tmp_cta_tile_begin_o,
                                                 tmp_cta_tile_begin_p,
                                                 tmp_cta_tile_begin_q,
                                                 cta_tile_index_depth,
                                                 cta_tile_index_height,
                                                 cta_tile_index_width,
                                                 cta_tile_begin_t,
                                                 cta_tile_begin_r,
                                                 cta_tile_begin_s);
#define GET_THE_INDEX_OF_THE_NEXT_TILE_IN_SMEM(idx)                                                \
    cta_tile_begin_batch[idx] = tmp_cta_tile_index_batch;                                          \
    cta_tile_begin_o[idx] = tmp_cta_tile_begin_o;                                                  \
    cta_tile_begin_p[idx] = tmp_cta_tile_begin_p;                                                  \
    cta_tile_begin_q[idx] = tmp_cta_tile_begin_q;
#define GET_THE_INDEX_OF_THE_NEXT_TILE(idx)                                                        \
    if (index_stage_smem == idx) {                                                                 \
        GET_THE_INDEX_OF_THE_NEXT_TILE_IN_SMEM(idx);                                               \
    }

#ifdef BUFFER_IN_SMEM
                if (threadIdx_x() == 0) {
                    GET_THE_INDEX_OF_THE_NEXT_TILE_IN_SMEM(index_stage_smem);
                }
#else
                if (Cta_tile_t::STAGE >= 2) {
                    GET_THE_INDEX_OF_THE_NEXT_TILE(0);
                    GET_THE_INDEX_OF_THE_NEXT_TILE(1);
                }
                if (Cta_tile_t::STAGE >= 3) {
                    GET_THE_INDEX_OF_THE_NEXT_TILE(2);
                }
                if (Cta_tile_t::STAGE >= 4) {
                    GET_THE_INDEX_OF_THE_NEXT_TILE(3);
                }
                if (Cta_tile_t::STAGE >= 5) {
                    GET_THE_INDEX_OF_THE_NEXT_TILE(4);
                }
                if (Cta_tile_t::STAGE >= 6) {
                    GET_THE_INDEX_OF_THE_NEXT_TILE(5);
                }

#endif
                gmem_loader_ndhwg.get_tile_begin(cta_tile_begin_d,
                                                 cta_tile_begin_h,
                                                 cta_tile_begin_w,
                                                 tmp_cta_tile_begin_o,
                                                 tmp_cta_tile_begin_p,
                                                 tmp_cta_tile_begin_q,
                                                 cta_tile_begin_t,
                                                 cta_tile_begin_r,
                                                 cta_tile_begin_s);
                if (SUPPORT_BETA) {
                    gmem_loader_nopqg.move(tmp_cta_tile_index_batch,
                                           tmp_cta_tile_begin_o,
                                           tmp_cta_tile_begin_p,
                                           tmp_cta_tile_begin_q);
                }
                gmem_loader_ndhwg.move(
                    tmp_cta_tile_index_batch, cta_tile_begin_d, cta_tile_begin_h, cta_tile_begin_w);
                if (SUPPORT_BETA) {
                    gmem_loader_nopqg.set_preds_of_instruction_in_gmem(tmp_cta_tile_index_batch,
                                                                       tmp_cta_tile_begin_o,
                                                                       tmp_cta_tile_begin_p,
                                                                       tmp_cta_tile_begin_q);
                }
                gmem_loader_ndhwg.set_preds_of_instruction_in_gmem(
                    tmp_cta_tile_index_batch, cta_tile_begin_d, cta_tile_begin_h, cta_tile_begin_w);
                if (SUPPORT_BETA) {
                    gmem_loader_nopqg.set_ptrs_of_instruction_in_gmem();
                }
                gmem_loader_ndhwg.set_ptrs_of_instruction_in_gmem();
                if (SUPPORT_BETA) {
                    gmem_loader_nopqg.type_colwersion();
                }
                gmem_loader_ndhwg.type_colwersion();
                if (SUPPORT_BETA) {
                    gmem_loader_nopqg.store_to_smem();
                }
                gmem_loader_ndhwg.store_to_smem();

                xmma::depbar<Kernel_traits::ENABLE_LDGSTS, Cta_tile_t::STAGE>();
                __syncthreads();
                if (SUPPORT_BETA) {
                    gmem_loader_nopqg.set_ptrs_of_instruction_in_smem(index_stage_smem);
                }
                gmem_loader_ndhwg.set_ptrs_of_instruction_in_smem(index_stage_smem);
                index_stage_smem = increase_and_mod<Cta_tile_t::STAGE>(index_stage_smem);
            }
            smem_storer_nopqg.set_ptrs_lds(0);
            if (SUPPORT_BETA) {
                smem_loader_nopqg.set_ptrs_lds(index_stage_smem);
            }
            smem_loader_ndhwg.set_ptrs_lds(index_stage_smem);
            if (SUPPORT_BETA) {
                smem_loader_nopqg.load_from_smem(index_iteration_smem %
                                                 Smem_tile_loader_nopqg_t::NUMBER_OF_MATH_BUFFERS);
            }
            smem_loader_ndhwg.load_from_smem(index_iteration_smem %
                                             Smem_tile_ndhwg_t::NUMBER_OF_MATH_BUFFERS);
        }
        gmem_storer_nopqg.load_from_smem();
        gmem_storer_nopqg.type_colwersion_for_storing_to_gmem();
        gmem_storer_nopqg.store_to_gmem();
        __syncthreads();
#ifdef BUFFER_IN_SMEM
        GET_THE_INDEX_OF_THE_LWRRENT_TILE_IN_SMEM(index_stage_smem);
#else
        if (Cta_tile_t::STAGE >= 2) {
            GET_THE_INDEX_OF_THE_LWRRENT_TILE(0);
            GET_THE_INDEX_OF_THE_LWRRENT_TILE(1);
        }
        if (Cta_tile_t::STAGE >= 3) {
            GET_THE_INDEX_OF_THE_LWRRENT_TILE(2);
        }
        if (Cta_tile_t::STAGE >= 4) {
            GET_THE_INDEX_OF_THE_LWRRENT_TILE(3);
        }
        if (Cta_tile_t::STAGE >= 5) {
            GET_THE_INDEX_OF_THE_LWRRENT_TILE(4);
        }
        if (Cta_tile_t::STAGE >= 6) {
            GET_THE_INDEX_OF_THE_LWRRENT_TILE(5);
        }

#endif
#undef GET_THE_INDEX_OF_THE_LWRRENT_TILE
#undef GET_THE_INDEX_OF_THE_LWRRENT_TILE_IN_SMEM
#undef GET_THE_INDEX_OF_THE_NEXT_TILE
#undef GET_THE_INDEX_OF_THE_NEXT_TILE_IN_SMEM
        gmem_storer_nopqg.move(tmp_cta_tile_index_batch,
                               tmp_cta_tile_begin_o,
                               tmp_cta_tile_begin_p,
                               tmp_cta_tile_begin_q);
        gmem_storer_nopqg.set_preds_of_instruction_in_gmem(tmp_cta_tile_index_batch,
                                                           tmp_cta_tile_begin_o,
                                                           tmp_cta_tile_begin_p,
                                                           tmp_cta_tile_begin_q);
        gmem_storer_nopqg.set_ptrs_of_instruction_in_gmem();
        gmem_storer_nopqg.set_ptrs_of_instruction_in_smem(0);
        xmma::ext::depthwise_colwolution::reset_memory_no_alias();
    }
    return;
}

} // namespace fprop
} // namespace depthwise_colwolution
} // namespace ext
} // namespace xmma

#endif
