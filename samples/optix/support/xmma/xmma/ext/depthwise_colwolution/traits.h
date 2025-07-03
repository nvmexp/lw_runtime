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

#ifndef _XMMA_EXT_DEPTHWISE_COLWOLUTION_TRAITS_H
#define _XMMA_EXT_DEPTHWISE_COLWOLUTION_TRAITS_H

#pragma once

#include "cta_tile.h"
#include "fprop/kernel.h"
#include "gmem_tile.h"
#include "gmem_tile_storer_trsg.h"
#include "math_tile.h"
#include "params.h"
#include "smem_tile.h"
#include "smem_tile_trsg.h"
#include "split_k.h"
#include "traits.h"
#include "utils.h"
#include "wgrad/kernel.h"
#include "xmma/params.h"
#include "xmma/xmma.h"
#include <cstdint>
#include <lwda_runtime.h>

namespace xmma
{
namespace ext
{
namespace depthwise_colwolution
{

template <typename Gpu_arch_,
          xmma::Operation_type Operation_,
          typename Cta_tile_,
          typename Data_type_io_,
          typename Data_type_acc_,
          int32_t ELEMENTS_PER_LDG_,
          bool SUPPORT_BETA_,
          bool DO_TYPE_COLWESION_EARLY_IN_GMEM_TILE_>
struct Kernel_traits {
    using Gpu_arch_t = Gpu_arch_;
    static const xmma::Operation_type Operation = Operation_;
    static_assert(Operation == xmma::Operation_type::FPROP ||
                      Operation == xmma::Operation_type::DGRAD,
                  "");
    using Cta_tile_t = Cta_tile_;
    using Params = Depthwise_colwolution_parameter<Cta_tile_t>;
    using Data_type_io_t = Data_type_io_;
    using Data_type_acc_t = Data_type_acc_;
    static const int32_t ELEMENTS_PER_LDG = ELEMENTS_PER_LDG_;
    static const bool SUPPORT_BETA = SUPPORT_BETA_;
    static const bool DO_TYPE_COLWESION_EARLY_IN_GMEM_TILE = DO_TYPE_COLWESION_EARLY_IN_GMEM_TILE_;
    using Data_type_ldg_t = Data_type_io_t;
    using Data_type_sts_t = typename Type_selector<DO_TYPE_COLWESION_EARLY_IN_GMEM_TILE,
                                                   Data_type_acc_t,
                                                   Data_type_ldg_t>::Type;
    using Gmem_tile_loader_ndhwg_t = Gmem_tile_ndhwg<Gpu_arch_t,
                                                     Cta_tile_t,
                                                     Data_type_ldg_t,
                                                     Data_type_sts_t,
                                                     ELEMENTS_PER_LDG,
                                                     1,
                                                     Tensor_type::A>;
    using Gmem_tile_storer_nopqg_t = Gmem_tile_nopqg<Gpu_arch_t,
                                                     Cta_tile_t,
                                                     Data_type_ldg_t,
                                                     Data_type_sts_t,
                                                     ELEMENTS_PER_LDG,
                                                     1,
                                                     Tensor_type::D,
                                                     1>;
    using Gmem_tile_loader_trsg_t =
        Gmem_tile_trsg<Gpu_arch_t, Cta_tile_t, Data_type_ldg_t, Data_type_sts_t, ELEMENTS_PER_LDG>;

    using Gmem_tile_loader_nopqg_t = Gmem_tile_nopqg<Gpu_arch_t,
                                                     Cta_tile_t,
                                                     Data_type_ldg_t,
                                                     Data_type_sts_t,
                                                     ELEMENTS_PER_LDG,
                                                     1,
                                                     Tensor_type::C>;

    using Abstract_tile_ndhwg_t = typename Gmem_tile_loader_ndhwg_t::Abstract_tile_t;
    using Abstract_tile_nopqg_t = typename Gmem_tile_storer_nopqg_t::Abstract_tile_t;
    using Abstract_tile_trsg_t = typename Gmem_tile_loader_trsg_t::Abstract_tile_t;
    using Smem_tile_ndhwg_t =
        Smem_tile_ndhwg<Gmem_tile_loader_ndhwg_t::SMEM_SIZE_IN_BYTES_PER_STAGE,
                        typename Cta_tile_t::Tile_memory_per_cta_t,
                        typename Cta_tile_t::Tile_math_per_thread_t,
                        Data_type_sts_t::BYTES_PER_ELEMENT,
                        Cta_tile_t::THREADS_PER_WARP,
                        Cta_tile_t::WARPS_PER_CTA>;
    using Smem_tile_nopqg_t =
        Smem_tile_nopqg<Gmem_tile_storer_nopqg_t::SMEM_SIZE_IN_BYTES_PER_STAGE,
                        typename Cta_tile_t::Tile_memory_per_cta_t,
                        typename Cta_tile_t::Tile_math_per_thread_t,
                        Data_type_sts_t::BYTES_PER_ELEMENT,
                        Cta_tile_t::THREADS_PER_WARP,
                        Cta_tile_t::WARPS_PER_CTA>;
    using Smem_tile_loader_nopqg_t =
        Smem_tile_nopqg<Gmem_tile_loader_nopqg_t::SMEM_SIZE_IN_BYTES_PER_STAGE,
                        typename Cta_tile_t::Tile_memory_per_cta_t,
                        typename Cta_tile_t::Tile_math_per_thread_t,
                        Data_type_sts_t::BYTES_PER_ELEMENT,
                        Cta_tile_t::THREADS_PER_WARP,
                        Cta_tile_t::WARPS_PER_CTA>;

    static const int32_t ITERATION_NUMBER = Smem_tile_nopqg_t::ITERATION_NUMBER;
    using Tile_trs_t = typename Cta_tile_t::Tile_memory_per_cta_t::Tile_trs_t;
    using Smem_tile_trsg_t = Smem_tile_trsg<Tile_trs_t,
                                            Cta_tile_t::Tile_memory_per_cta_t::TILE_G,
                                            Data_type_sts_t::BYTES_PER_ELEMENT,
                                            Cta_tile_t::Tile_math_per_thread_t::TILE_G,
                                            Cta_tile_t::THREADS_PER_CTA>;

    using Math_tile_tmp_loader_nopqg_t =
        Math_tile_3d<Smem_tile_loader_nopqg_t::MATH_TILE_G,
                     Tile_3d<Smem_tile_loader_nopqg_t::MATH_TILE_DEPTH,
                             Smem_tile_loader_nopqg_t::MATH_TILE_HEIGHT,
                             Smem_tile_loader_nopqg_t::MATH_TILE_WIDTH>,
                     Data_type_sts_t,
                     Data_type_sts_t>;

    using Math_tile_tmp_nopqg_t = Math_tile_3d<Smem_tile_nopqg_t::MATH_TILE_G,
                                               Tile_3d<Smem_tile_nopqg_t::MATH_TILE_DEPTH,
                                                       Smem_tile_nopqg_t::MATH_TILE_HEIGHT,
                                                       Smem_tile_nopqg_t::MATH_TILE_WIDTH>,
                                               Data_type_sts_t,
                                               Data_type_acc_t>;

    using Math_tile_tmp_ndhwg_t = Math_tile_3d<Smem_tile_ndhwg_t::MATH_TILE_G,
                                               Tile_3d<Smem_tile_ndhwg_t::MATH_TILE_DEPTH,
                                                       Smem_tile_ndhwg_t::MATH_TILE_HEIGHT,
                                                       Smem_tile_ndhwg_t::MATH_TILE_WIDTH>,
                                               Data_type_sts_t,
                                               Data_type_sts_t>;

    using Math_tile_tmp_trsg_t =
        Math_tile_3d<Smem_tile_ndhwg_t::MATH_TILE_G,
                     Tile_3d<Tile_trs_t::DEPTH, Tile_trs_t::HEIGHT, Tile_trs_t::WIDTH>,
                     Data_type_sts_t,
                     Data_type_sts_t>;

    using Math_tile_loader_nopqg_t =
        Math_tile_3d<Smem_tile_loader_nopqg_t::MATH_TILE_G,
                     Tile_3d<Smem_tile_loader_nopqg_t::MATH_TILE_DEPTH,
                             Smem_tile_loader_nopqg_t::MATH_TILE_HEIGHT,
                             Smem_tile_loader_nopqg_t::MATH_TILE_WIDTH>,
                     Data_type_acc_t,
                     Data_type_sts_t>;

    using Math_tile_nopqg_t = Math_tile_3d<Smem_tile_nopqg_t::MATH_TILE_G,
                                           Tile_3d<Smem_tile_nopqg_t::MATH_TILE_DEPTH,
                                                   Smem_tile_nopqg_t::MATH_TILE_HEIGHT,
                                                   Smem_tile_nopqg_t::MATH_TILE_WIDTH>,
                                           Data_type_acc_t,
                                           Data_type_acc_t>;

    using Math_tile_ndhwg_t = Math_tile_3d<Smem_tile_ndhwg_t::MATH_TILE_G,
                                           Tile_3d<Smem_tile_ndhwg_t::MATH_TILE_DEPTH,
                                                   Smem_tile_ndhwg_t::MATH_TILE_HEIGHT,
                                                   Smem_tile_ndhwg_t::MATH_TILE_WIDTH>,
                                           Data_type_acc_t,
                                           Data_type_sts_t>;

    using Math_tile_trsg_t =
        Math_tile_3d<Smem_tile_ndhwg_t::MATH_TILE_G,
                     Tile_3d<Tile_trs_t::DEPTH, Tile_trs_t::HEIGHT, Tile_trs_t::WIDTH>,
                     Data_type_acc_t,
                     Data_type_sts_t>;

    using Math_operation_t =
        Math_operation<(Operation == xmma::Operation_type::DGRAD ? xmma::Operation_type::FPROP
                                                                 : Operation),
                       Math_tile_nopqg_t,
                       Math_tile_ndhwg_t,
                       Math_tile_trsg_t,
                       typename Cta_tile_t::Tile_math_per_thread_t::Tile_stride_dhw_t,
                       typename Cta_tile_t::Tile_math_per_thread_t::Tile_dilation_dhw_t>;

    static const bool ENABLE_LDGSTS = Gmem_tile_loader_ndhwg_t::Base_t::ENABLE_LDGSTS;
    static const int32_t THREADS_PER_CTA = Cta_tile_t::THREADS_PER_CTA;
    static const int32_t SMEM_SIZE_IN_BYTES_MAIN_LOOP =
        Gmem_tile_loader_ndhwg_t::SMEM_SIZE_IN_BYTES + Gmem_tile_loader_trsg_t::SMEM_SIZE_IN_BYTES +
        Gmem_tile_storer_nopqg_t::SMEM_SIZE_IN_BYTES +
        (SUPPORT_BETA ? Gmem_tile_loader_nopqg_t::SMEM_SIZE_IN_BYTES : 0);
    static const int32_t SMEM_SIZE_IN_BYTES_EPILOGUE = 0;
    static const int32_t SMEM_SIZE_IN_BYTES =
        (SMEM_SIZE_IN_BYTES_MAIN_LOOP > SMEM_SIZE_IN_BYTES_EPILOGUE ? SMEM_SIZE_IN_BYTES_MAIN_LOOP
                                                                    : SMEM_SIZE_IN_BYTES_EPILOGUE);

#if !defined(__LWDACC_RTC__)

    typedef void (*Kernel_type)(Params params);

    // Return device kernel function pointer.
    XMMA_HOST static Kernel_type kernel_ptr(const Params params = Params())
    {
        return &xmma::ext::depthwise_colwolution::fprop::kernel<Kernel_traits>;
    }

    // Return split k kernel function pointer.
    XMMA_HOST static Kernel_type split_k_kernel_ptr() { return nullptr; }

#endif

    // The number of threads in the CTA.
    XMMA_HOST static int32_t threads_per_cta(const Params params = Params())
    {
        return Cta_tile_t::THREADS_PER_CTA;
    }

    // The amount of shared memory per CTA.
    XMMA_HOST static int32_t dynamic_smem_size_per_cta() { return SMEM_SIZE_IN_BYTES; }

    // The amount of epilogue shared memory per CTA.
    XMMA_HOST static int32_t epilogue_smem_size_per_cta() { return SMEM_SIZE_IN_BYTES_EPILOGUE; }

    XMMA_HOST static void compute_grid_size(dim3 &grid,
                                            Depthwise_colwolution_parameter<Cta_tile_t> &params)
    {
        int32_t tmp[2];
        tmp[Grid_dim::TRS] = params.tiles_trs;
        tmp[Grid_dim::G] = (params.g + Cta_tile_t::Tile_memory_per_cta_t::TILE_G - 1) /
                           Cta_tile_t::Tile_memory_per_cta_t::TILE_G;
        grid.x = tmp[Grid_dim::TRS];
        grid.y = tmp[Grid_dim::G];
        grid.z = params.split_m_slices;
        return;
    }

    XMMA_HOST static int32_t
    get_split_k_start_offset(Depthwise_colwolution_parameter<Cta_tile_t> &params)
    {
        int32_t trsg_size_in_bytes =
            params.t * params.r * params.s * params.g * Gmem_tile_loader_trsg_t::BYTES_PER_ELEMENT;
        trsg_size_in_bytes = ((trsg_size_in_bytes + 127) / 128) * 128;
        return trsg_size_in_bytes;
    }

    XMMA_HOST static int32_t
    get_split_k_total_buffer_size_in_bytes(Depthwise_colwolution_parameter<Cta_tile_t> &params)
    {
        dim3 grid;
        compute_grid_size(grid, params);
        int32_t trsg_size_in_bytes = get_split_k_start_offset(params);
        return trsg_size_in_bytes + get_split_k_gmem_size_in_bytes(grid, params.split_k_buffers) +
               get_split_k_buffer_counter_size_in_bytes(grid, params.split_k_buffers) +
               get_split_k_final_counter_size_in_bytes(grid);
    }

    XMMA_HOST static int32_t get_single_split_k_buffer_size_in_bytes(const dim3 &grid) { return 0; }

    XMMA_HOST static int32_t get_split_k_gmem_size_in_bytes(const dim3 &grid,
                                                            int32_t split_k_buffers)
    {
        return get_single_split_k_buffer_size_in_bytes(grid) * split_k_buffers;
    }

    XMMA_HOST static int32_t get_split_k_buffer_counter_size_in_bytes(const dim3 &grid,
                                                                      int32_t split_k_buffers)
    {
        return 0;
    }

    XMMA_HOST static int32_t get_split_k_final_counter_size_in_bytes(const dim3 &grid) { return 0; }
};

template <typename Gpu_arch_,
          typename Cta_tile_,
          typename Data_type_io_,
          typename Data_type_acc_,
          int32_t ELEMENTS_PER_LDG_,
          bool SUPPORT_BETA_,
          bool DO_TYPE_COLWESION_EARLY_IN_GMEM_TILE_>
struct Kernel_traits<Gpu_arch_,
                     xmma::Operation_type::WGRAD,
                     Cta_tile_,
                     Data_type_io_,
                     Data_type_acc_,
                     ELEMENTS_PER_LDG_,
                     SUPPORT_BETA_,
                     DO_TYPE_COLWESION_EARLY_IN_GMEM_TILE_> {
    static_assert(SUPPORT_BETA_ == true, "");
    using Gpu_arch_t = Gpu_arch_;
    static const xmma::Operation_type Operation = xmma::Operation_type::WGRAD;
    using Cta_tile_t = Cta_tile_;
    using Params = Depthwise_colwolution_parameter<Cta_tile_t>;
    using Data_type_io_t = Data_type_io_;
    using Data_type_acc_t = Data_type_acc_;
    static const int32_t ELEMENTS_PER_LDG = ELEMENTS_PER_LDG_;
    static const bool DO_TYPE_COLWESION_EARLY_IN_GMEM_TILE = DO_TYPE_COLWESION_EARLY_IN_GMEM_TILE_;
    using Data_type_ldg_t = Data_type_io_t;
    using Data_type_sts_t = typename Type_selector<DO_TYPE_COLWESION_EARLY_IN_GMEM_TILE,
                                                   Data_type_acc_t,
                                                   Data_type_ldg_t>::Type;
    using Gmem_tile_loader_ndhwg_t = Gmem_tile_ndhwg<Gpu_arch_t,
                                                     Cta_tile_t,
                                                     Data_type_ldg_t,
                                                     Data_type_sts_t,
                                                     ELEMENTS_PER_LDG,
                                                     1,
                                                     Tensor_type::A>;
    using Gmem_tile_loader_nopqg_t = Gmem_tile_nopqg<Gpu_arch_t,
                                                     Cta_tile_t,
                                                     Data_type_ldg_t,
                                                     Data_type_sts_t,
                                                     ELEMENTS_PER_LDG,
                                                     1,
                                                     Tensor_type::B>;
    using Abstract_tile_ndhwg_t = typename Gmem_tile_loader_ndhwg_t::Abstract_tile_t;
    using Abstract_tile_nopqg_t = typename Gmem_tile_loader_nopqg_t::Abstract_tile_t;
    using Smem_tile_ndhwg_t =
        Smem_tile_ndhwg<Gmem_tile_loader_ndhwg_t::SMEM_SIZE_IN_BYTES_PER_STAGE,
                        typename Cta_tile_t::Tile_memory_per_cta_t,
                        typename Cta_tile_t::Tile_math_per_thread_t,
                        Data_type_sts_t::BYTES_PER_ELEMENT,
                        Cta_tile_t::THREADS_PER_WARP,
                        Cta_tile_t::WARPS_PER_CTA>;
    using Smem_tile_nopqg_t =
        Smem_tile_nopqg<Gmem_tile_loader_nopqg_t::SMEM_SIZE_IN_BYTES_PER_STAGE,
                        typename Cta_tile_t::Tile_memory_per_cta_t,
                        typename Cta_tile_t::Tile_math_per_thread_t,
                        Data_type_sts_t::BYTES_PER_ELEMENT,
                        Cta_tile_t::THREADS_PER_WARP,
                        Cta_tile_t::WARPS_PER_CTA>;
    using Tile_trs_t = typename Cta_tile_t::Tile_memory_per_cta_t::Tile_trs_t;
    static const int32_t ITERATION_NUMBER = Smem_tile_ndhwg_t::ITERATION_NUMBER;

    using Math_tile_tmp_nopqg_t = Math_tile_3d<Smem_tile_nopqg_t::MATH_TILE_G,
                                               Tile_3d<Smem_tile_nopqg_t::MATH_TILE_DEPTH,
                                                       Smem_tile_nopqg_t::MATH_TILE_HEIGHT,
                                                       Smem_tile_nopqg_t::MATH_TILE_WIDTH>,
                                               Data_type_sts_t,
                                               Data_type_sts_t>;

    using Math_tile_tmp_ndhwg_t = Math_tile_3d<Smem_tile_ndhwg_t::MATH_TILE_G,
                                               Tile_3d<Smem_tile_ndhwg_t::MATH_TILE_DEPTH,
                                                       Smem_tile_ndhwg_t::MATH_TILE_HEIGHT,
                                                       Smem_tile_ndhwg_t::MATH_TILE_WIDTH>,
                                               Data_type_sts_t,
                                               Data_type_sts_t>;

    using Math_tile_tmp_trsg_t =
        Math_tile_3d<Smem_tile_ndhwg_t::MATH_TILE_G,
                     Tile_3d<Tile_trs_t::DEPTH, Tile_trs_t::HEIGHT, Tile_trs_t::WIDTH>,
                     Data_type_sts_t,
                     Data_type_acc_t>;

    using Math_tile_nopqg_t = Math_tile_3d<Smem_tile_nopqg_t::MATH_TILE_G,
                                           Tile_3d<Smem_tile_nopqg_t::MATH_TILE_DEPTH,
                                                   Smem_tile_nopqg_t::MATH_TILE_HEIGHT,
                                                   Smem_tile_nopqg_t::MATH_TILE_WIDTH>,
                                           Data_type_acc_t,
                                           Data_type_sts_t>;

    using Math_tile_ndhwg_t = Math_tile_3d<Smem_tile_ndhwg_t::MATH_TILE_G,
                                           Tile_3d<Smem_tile_ndhwg_t::MATH_TILE_DEPTH,
                                                   Smem_tile_ndhwg_t::MATH_TILE_HEIGHT,
                                                   Smem_tile_ndhwg_t::MATH_TILE_WIDTH>,
                                           Data_type_acc_t,
                                           Data_type_sts_t>;

    using Math_tile_trsg_t =
        Math_tile_3d<Smem_tile_ndhwg_t::MATH_TILE_G,
                     Tile_3d<Tile_trs_t::DEPTH, Tile_trs_t::HEIGHT, Tile_trs_t::WIDTH>,
                     Data_type_acc_t,
                     Data_type_acc_t>;

    using Math_operation_t =
        Math_operation<Operation,
                       Math_tile_trsg_t,
                       Math_tile_ndhwg_t,
                       Math_tile_nopqg_t,
                       typename Cta_tile_t::Tile_math_per_thread_t::Tile_stride_dhw_t,
                       typename Cta_tile_t::Tile_math_per_thread_t::Tile_dilation_dhw_t>;

    using Gmem_tile_trsg_t =
        Gmem_tile_storer_trsg<Cta_tile_t,
                              Math_tile_tmp_trsg_t,
                              Cta_tile_t::Tile_memory_per_cta_t::TILE_G,
                              Data_type_sts_t,
                              Data_type_ldg_t,
                              Cta_tile_t::THREADS_PER_WARP,
                              Cta_tile_t::WARPS_PER_CTA,
                              Gmem_tile_loader_ndhwg_t::BYTES_PER_INSTRUCTION_IN_SMEM,
                              ELEMENTS_PER_LDG>;

    using Split_k_t = Split_k<Gmem_tile_trsg_t>;

    static const bool ENABLE_LDGSTS = Gmem_tile_loader_ndhwg_t::Base_t::ENABLE_LDGSTS;
    static const int32_t THREADS_PER_CTA = Cta_tile_t::THREADS_PER_CTA;
    static const int32_t SMEM_SIZE_IN_BYTES_MAIN_LOOP =
        Gmem_tile_loader_ndhwg_t::SMEM_SIZE_IN_BYTES + Gmem_tile_loader_nopqg_t::SMEM_SIZE_IN_BYTES;
    static const int32_t SMEM_SIZE_IN_BYTES_EPILOGUE = Gmem_tile_trsg_t::SMEM_SIZE_IN_BYTES;
    static const int32_t SMEM_SIZE_IN_BYTES =
        (SMEM_SIZE_IN_BYTES_MAIN_LOOP > SMEM_SIZE_IN_BYTES_EPILOGUE ? SMEM_SIZE_IN_BYTES_MAIN_LOOP
                                                                    : SMEM_SIZE_IN_BYTES_EPILOGUE);

#if !defined(__LWDACC_RTC__)

    typedef void (*Kernel_type)(Params params);

    // Return device kernel function pointer.
    XMMA_HOST static Kernel_type kernel_ptr(const Params params = Params())
    {
        return &xmma::ext::depthwise_colwolution::wgrad::kernel<Kernel_traits>;
    }

    // Return split k kernel function pointer.
    XMMA_HOST static Kernel_type split_k_kernel_ptr() { return nullptr; }

#endif

    // The number of threads in the CTA.
    XMMA_HOST static int32_t threads_per_cta(const Params params = Params())
    {
        return Cta_tile_t::THREADS_PER_CTA;
    }

    // The amount of shared memory per CTA.
    XMMA_HOST static int32_t dynamic_smem_size_per_cta() { return SMEM_SIZE_IN_BYTES; }

    // The amount of epilogue shared memory per CTA.
    XMMA_HOST static int32_t epilogue_smem_size_per_cta() { return SMEM_SIZE_IN_BYTES_EPILOGUE; }

    XMMA_HOST static void compute_grid_size(dim3 &grid,
                                            Depthwise_colwolution_parameter<Cta_tile_t> &params)
    {
        int32_t tmp[2];
        tmp[Grid_dim::TRS] = params.tiles_trs;
        tmp[Grid_dim::G] = (params.g + Cta_tile_t::Tile_memory_per_cta_t::TILE_G - 1) /
                           Cta_tile_t::Tile_memory_per_cta_t::TILE_G;
        grid.x = tmp[Grid_dim::TRS];
        grid.y = tmp[Grid_dim::G];
        grid.z = params.split_k_slices;
        return;
    }

    XMMA_HOST static int32_t
    get_split_k_start_offset(Depthwise_colwolution_parameter<Cta_tile_t> &params)
    {
        int32_t trsg_size_in_bytes =
            params.t * params.r * params.s * params.g * Gmem_tile_trsg_t::BYTES_PER_ELEMENT;
        trsg_size_in_bytes = ((trsg_size_in_bytes + 127) / 128) * 128;
        return trsg_size_in_bytes;
    }

    XMMA_HOST static int32_t
    get_split_k_total_buffer_size_in_bytes(Depthwise_colwolution_parameter<Cta_tile_t> &params)
    {
        dim3 grid;
        compute_grid_size(grid, params);
        int32_t trsg_size_in_bytes = get_split_k_start_offset(params);
        return trsg_size_in_bytes + get_split_k_gmem_size_in_bytes(grid, params.split_k_buffers) +
               get_split_k_buffer_counter_size_in_bytes(grid, params.split_k_buffers) +
               get_split_k_final_counter_size_in_bytes(grid);
    }

    XMMA_HOST static int32_t get_single_split_k_buffer_size_in_bytes(const dim3 &grid)
    {
        if (grid.z > 1) {
            return Gmem_tile_trsg_t::SPLIT_K_STG_BYTES_PER_CTA * grid.x * grid.y;
        } else {
            return 0;
        }
    }

    XMMA_HOST static int32_t get_split_k_gmem_size_in_bytes(const dim3 &grid,
                                                            int32_t split_k_buffers)
    {
        return get_single_split_k_buffer_size_in_bytes(grid) * split_k_buffers;
    }

    XMMA_HOST static int32_t get_split_k_buffer_counter_size_in_bytes(const dim3 &grid,
                                                                      int32_t split_k_buffers)
    {
        if (grid.z > 1) {
            return grid.x * grid.y * split_k_buffers * static_cast<int32_t>(sizeof(int32_t));
        } else {
            return 0;
        }
    }

    XMMA_HOST static int32_t get_split_k_final_counter_size_in_bytes(const dim3 &grid)
    {
        if (grid.z > 1) {
            return grid.x * grid.y * static_cast<int32_t>(sizeof(int32_t));
        } else {
            return 0;
        }
    }
};

} // namespace depthwise_colwolution
} // namespace ext
} // namespace xmma

#endif
