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

#ifndef _XMMA_EXT_DEPTHWISE_COLWOLUTION_WGRAD_KERNEL_H
#define _XMMA_EXT_DEPTHWISE_COLWOLUTION_WGRAD_KERNEL_H

#pragma once

#include "xmma/ext/depthwise_colwolution/cta_tile.h"
#include "xmma/ext/depthwise_colwolution/gmem_tile.h"
#include "xmma/ext/depthwise_colwolution/gmem_tile_storer_trsg.h"
#include "xmma/ext/depthwise_colwolution/math_tile.h"
#include "xmma/ext/depthwise_colwolution/params.h"
#include "xmma/ext/depthwise_colwolution/smem_tile.h"
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
namespace wgrad
{

#define USE_INCREASING_MODE false
template <typename Kernel_traits>
__global__ static void kernel(typename Kernel_traits::Params params)
{
    extern __shared__ char smem[];

    using Cta_tile_t = typename Kernel_traits::Cta_tile_t;
    using Tile_trs_t = typename Cta_tile_t::Tile_memory_per_cta_t::Tile_trs_t;
    using Gmem_tile_loader_ndhwg_t = typename Kernel_traits::Gmem_tile_loader_ndhwg_t;
    using Gmem_tile_loader_nopqg_t = typename Kernel_traits::Gmem_tile_loader_nopqg_t;
    using Abstract_tile_ndhwg_t = typename Kernel_traits::Abstract_tile_ndhwg_t;
    using Abstract_tile_nopqg_t = typename Kernel_traits::Abstract_tile_nopqg_t;
    using Smem_tile_ndhwg_t = typename Kernel_traits::Smem_tile_ndhwg_t;
    using Smem_tile_nopqg_t = typename Kernel_traits::Smem_tile_nopqg_t;
    using Math_tile_trsg_t = typename Kernel_traits::Math_tile_trsg_t;
    using Math_tile_ndhwg_t = typename Kernel_traits::Math_tile_ndhwg_t;
    using Math_tile_nopqg_t = typename Kernel_traits::Math_tile_nopqg_t;
    using Math_tile_tmp_trsg_t = typename Kernel_traits::Math_tile_tmp_trsg_t;
    using Math_tile_tmp_ndhwg_t = typename Kernel_traits::Math_tile_tmp_ndhwg_t;
    using Math_tile_tmp_nopqg_t = typename Kernel_traits::Math_tile_tmp_nopqg_t;
    using Math_operation_t = typename Kernel_traits::Math_operation_t;
    using Gmem_tile_trsg_t = typename Kernel_traits::Gmem_tile_trsg_t;
    using Split_k_t = typename Kernel_traits::Split_k_t;

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
    uint32_t smem_base_address_nopqg =
        smem_base_address_ndhwg + Gmem_tile_loader_ndhwg_t::SMEM_SIZE_IN_BYTES;
    Abstract_tile_ndhwg_t abstract_tile_ndhwg;
    Abstract_tile_nopqg_t abstract_tile_nopqg;
    Gmem_tile_loader_ndhwg_t gmem_ndhwg(abstract_tile_ndhwg, smem_base_address_ndhwg, params);
    Gmem_tile_loader_nopqg_t gmem_nopqg(abstract_tile_nopqg, smem_base_address_nopqg, params);

    int32_t index_cta_tile = blockIdx_z();
    int32_t cta_tile_index_batch, cta_tile_index_depth, cta_tile_index_height, cta_tile_index_width;
    int32_t cta_tile_begin_o, cta_tile_begin_p, cta_tile_begin_q;
    int32_t cta_tile_begin_d, cta_tile_begin_h, cta_tile_begin_w;
#pragma unroll
    for (int32_t index_prefetch_stage = 0; index_prefetch_stage < Cta_tile_t::STAGE;
         ++index_prefetch_stage) {
        xmma::ext::depthwise_colwolution::set_memory_no_alias();
        gmem_nopqg.decompose_tile_index(cta_tile_index_batch,
                                        cta_tile_index_depth,
                                        cta_tile_index_height,
                                        cta_tile_index_width,
                                        index_cta_tile,
                                        USE_INCREASING_MODE && (index_prefetch_stage > 0));
        gmem_nopqg.get_tile_begin(cta_tile_begin_o,
                                  cta_tile_begin_p,
                                  cta_tile_begin_q,
                                  cta_tile_index_depth,
                                  cta_tile_index_height,
                                  cta_tile_index_width,
                                  cta_tile_begin_t,
                                  cta_tile_begin_r,
                                  cta_tile_begin_s);
        gmem_ndhwg.get_tile_begin(cta_tile_begin_d,
                                  cta_tile_begin_h,
                                  cta_tile_begin_w,
                                  cta_tile_begin_o,
                                  cta_tile_begin_p,
                                  cta_tile_begin_q,
                                  cta_tile_begin_t,
                                  cta_tile_begin_r,
                                  cta_tile_begin_s);
        gmem_nopqg.move(cta_tile_index_batch, cta_tile_begin_o, cta_tile_begin_p, cta_tile_begin_q);
        gmem_ndhwg.move(cta_tile_index_batch, cta_tile_begin_d, cta_tile_begin_h, cta_tile_begin_w);
        gmem_nopqg.set_preds_of_instruction_in_gmem(
            cta_tile_index_batch, cta_tile_begin_o, cta_tile_begin_p, cta_tile_begin_q);
        gmem_ndhwg.set_preds_of_instruction_in_gmem(
            cta_tile_index_batch, cta_tile_begin_d, cta_tile_begin_h, cta_tile_begin_w);
        gmem_nopqg.set_ptrs_of_instruction_in_gmem();
        gmem_ndhwg.set_ptrs_of_instruction_in_gmem();
        gmem_nopqg.set_ptrs_of_instruction_in_smem(index_prefetch_stage);
        gmem_ndhwg.set_ptrs_of_instruction_in_smem(index_prefetch_stage);

        if (index_prefetch_stage < Cta_tile_t::STAGE - 1) {
            gmem_nopqg.load_from_gmem();
            gmem_ndhwg.load_from_gmem();
            xmma::ldgdepbar<Kernel_traits::ENABLE_LDGSTS>();
            gmem_nopqg.type_colwersion();
            gmem_ndhwg.type_colwersion();
            gmem_nopqg.store_to_smem();
            gmem_ndhwg.store_to_smem();
            index_cta_tile += params.split_k_slices;
        }
        xmma::ext::depthwise_colwolution::reset_memory_no_alias();
    }
    Smem_tile_nopqg_t smem_nopqg(smem_base_address_nopqg);
    Smem_tile_ndhwg_t smem_ndhwg(smem_base_address_ndhwg,
                                 params.stride[0],
                                 params.stride[1],
                                 params.stride[2],
                                 params.dilation[0],
                                 params.dilation[1],
                                 params.dilation[2],
                                 params.is_colwolution);
    smem_nopqg.set_offsets_base_lds();
    smem_ndhwg.set_offsets_base_lds(smem_nopqg.offset_base_depth_,
                                    smem_nopqg.offset_base_height_,
                                    smem_nopqg.offset_base_width_);
    xmma::depbar<Kernel_traits::ENABLE_LDGSTS, Cta_tile_t::STAGE>();
    __syncthreads();
    Math_tile_trsg_t math_tile_trsg;
    math_tile_trsg.clear();
    Math_operation_t math_operation;
    int32_t index_stage_smem = 0;
    smem_nopqg.set_offsets_lds(0);
    smem_nopqg.set_ptrs_lds(index_stage_smem);
    smem_nopqg.load_from_smem(0);
    smem_ndhwg.set_offsets_lds(0);
    smem_ndhwg.set_ptrs_lds(index_stage_smem);
    smem_ndhwg.load_from_smem(0);
    for (int32_t index_main_loop = 0; index_main_loop < params.count_main_loop; ++index_main_loop) {
        xmma::ext::depthwise_colwolution::set_memory_no_alias();
        gmem_nopqg.load_from_gmem();
        gmem_ndhwg.load_from_gmem();
        xmma::ldgdepbar<Kernel_traits::ENABLE_LDGSTS>();
#pragma unroll
        for (int32_t index_iteration = 0; index_iteration < Kernel_traits::ITERATION_NUMBER;
             ++index_iteration) {
            Math_tile_tmp_nopqg_t math_tile_tmp_nopqg(
                smem_nopqg
                    .fetch_math_[index_iteration % Smem_tile_nopqg_t::NUMBER_OF_MATH_BUFFERS]);
            Math_tile_tmp_ndhwg_t math_tile_tmp_ndhwg(
                smem_ndhwg
                    .fetch_math_[index_iteration % Smem_tile_ndhwg_t::NUMBER_OF_MATH_BUFFERS]);
            Math_tile_nopqg_t math_tile_nopqg(math_tile_tmp_nopqg);
            Math_tile_ndhwg_t math_tile_ndhwg(math_tile_tmp_ndhwg);
            math_operation.exlwte(math_tile_trsg, math_tile_ndhwg, math_tile_nopqg);
            int32_t index_iteration_smem = index_iteration;
            index_iteration_smem =
                increase_and_mod<Kernel_traits::ITERATION_NUMBER>(index_iteration_smem);
            smem_nopqg.set_offsets_lds(index_iteration_smem);
            smem_ndhwg.set_offsets_lds(index_iteration_smem);
            bool is_next_stage = (index_iteration == Kernel_traits::ITERATION_NUMBER - 1);
            if (is_next_stage) {
                index_stage_smem = increase_and_mod<Cta_tile_t::STAGE>(index_stage_smem);
                gmem_nopqg.type_colwersion();
                gmem_ndhwg.type_colwersion();
                gmem_nopqg.store_to_smem();
                gmem_ndhwg.store_to_smem();
                xmma::depbar<Kernel_traits::ENABLE_LDGSTS, Cta_tile_t::STAGE>();
                __syncthreads();
            }
            smem_nopqg.set_ptrs_lds(index_stage_smem);
            smem_ndhwg.set_ptrs_lds(index_stage_smem);
            smem_nopqg.load_from_smem(index_iteration_smem %
                                      Smem_tile_nopqg_t::NUMBER_OF_MATH_BUFFERS);
            smem_ndhwg.load_from_smem(index_iteration_smem %
                                      Smem_tile_ndhwg_t::NUMBER_OF_MATH_BUFFERS);
        }
        index_cta_tile += params.split_k_slices;
        gmem_nopqg.decompose_tile_index(cta_tile_index_batch,
                                        cta_tile_index_depth,
                                        cta_tile_index_height,
                                        cta_tile_index_width,
                                        index_cta_tile,
                                        USE_INCREASING_MODE);
        gmem_nopqg.get_tile_begin(cta_tile_begin_o,
                                  cta_tile_begin_p,
                                  cta_tile_begin_q,
                                  cta_tile_index_depth,
                                  cta_tile_index_height,
                                  cta_tile_index_width,
                                  cta_tile_begin_t,
                                  cta_tile_begin_r,
                                  cta_tile_begin_s);
        gmem_ndhwg.get_tile_begin(cta_tile_begin_d,
                                  cta_tile_begin_h,
                                  cta_tile_begin_w,
                                  cta_tile_begin_o,
                                  cta_tile_begin_p,
                                  cta_tile_begin_q,
                                  cta_tile_begin_t,
                                  cta_tile_begin_r,
                                  cta_tile_begin_s);
        gmem_nopqg.move(cta_tile_index_batch, cta_tile_begin_o, cta_tile_begin_p, cta_tile_begin_q);
        gmem_ndhwg.move(cta_tile_index_batch, cta_tile_begin_d, cta_tile_begin_h, cta_tile_begin_w);
        gmem_nopqg.set_preds_of_instruction_in_gmem(
            cta_tile_index_batch, cta_tile_begin_o, cta_tile_begin_p, cta_tile_begin_q);
        gmem_ndhwg.set_preds_of_instruction_in_gmem(
            cta_tile_index_batch, cta_tile_begin_d, cta_tile_begin_h, cta_tile_begin_w);
        gmem_nopqg.set_ptrs_of_instruction_in_gmem();
        gmem_ndhwg.set_ptrs_of_instruction_in_gmem();

        int32_t index_stage = index_main_loop % Cta_tile_t::STAGE;
        gmem_nopqg.set_ptrs_of_instruction_in_smem(index_stage);
        gmem_ndhwg.set_ptrs_of_instruction_in_smem(index_stage);
        xmma::ext::depthwise_colwolution::reset_memory_no_alias();
    }
    __syncthreads();
    uint32_t smem_base_address_trsg = smem_base_address;
    Math_tile_tmp_trsg_t math_tile_tmp_trsg(math_tile_trsg);
    Gmem_tile_trsg_t gmem_tile_trsg(math_tile_tmp_trsg,
                                    smem_base_address_trsg,
                                    params,
                                    cta_tile_begin_t,
                                    cta_tile_begin_r,
                                    cta_tile_begin_s);
    gmem_tile_trsg.set_ptrs_and_preds_gmem();
    if (blockIdx_z() == params.split_k_slices - 1) {
        gmem_tile_trsg.load_from_gmem();
    }
    gmem_tile_trsg.store_to_smem();
    __syncthreads();
    gmem_tile_trsg.load_from_smem();
    gmem_tile_trsg.reduction();
    Split_k_t split_k(gmem_tile_trsg);
    split_k.exlwte(gmem_tile_trsg);
    if (blockIdx_z() < params.split_k_slices - 1) {
        return;
    }
    gmem_tile_trsg.apply_alpha();
    gmem_tile_trsg.type_colwersion_for_load();
    gmem_tile_trsg.apply_beta();
    gmem_tile_trsg.type_colwersion_for_store();
    gmem_tile_trsg.store_to_gmem();
    return;
}

} // namespace wgrad
} // namespace depthwise_colwolution
} // namespace ext
} // namespace xmma

#endif
