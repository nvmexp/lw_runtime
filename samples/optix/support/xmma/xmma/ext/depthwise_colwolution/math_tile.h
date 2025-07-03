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

#ifndef _XMMA_EXT_DEPTHWISE_COLWOLUTION_MATH_TILE_H
#define _XMMA_EXT_DEPTHWISE_COLWOLUTION_MATH_TILE_H

#pragma once

#include "data_type.h"
#include "utils.h"
#include "xmma/params.h"
#include "xmma/utils.h"
#include <cstdint>
#include <lwda_fp16.h>
#include <lwda_runtime.h>

namespace xmma
{
namespace ext
{
namespace depthwise_colwolution
{

template <int32_t TILE_G_, typename Tile_3d_, typename Data_type_self_, typename Data_type_other_>
struct Math_tile_3d {
    public:
    static const int32_t TILE_G = TILE_G_;
    using Tile_3d_t = Tile_3d_;
    using Data_type_t = Data_type_self_;
    using Data_type_other_t = Data_type_other_;
    using Math_tile_3d_other_t =
        Math_tile_3d<TILE_G, Tile_3d_t, Data_type_other_t, Data_type_other_t>;
    static const int32_t BYTES_PER_UINT32 = 4;
    static const int32_t UINT32_PER_TILE_G =
        TILE_G * static_cast<int32_t>(sizeof(typename Data_type_t::Type)) / BYTES_PER_UINT32;
    static_assert(UINT32_PER_TILE_G > 0, "");
    // Derived
    static const int32_t UINT32_NUMBER = Tile_3d_t::VALUE * UINT32_PER_TILE_G;
    using Packed_data_type_t =
        typename xmma::Uint_from_size_in_bytes<UINT32_PER_TILE_G * BYTES_PER_UINT32>::Type;

    __device__ inline Math_tile_3d() {}

    __device__ inline void clear()
    {
#pragma unroll
        for (int32_t index_depth = 0; index_depth < Tile_3d_t::DEPTH; ++index_depth) {
#pragma unroll
            for (int32_t index_height = 0; index_height < Tile_3d_t::HEIGHT; ++index_height) {
#pragma unroll
                for (int32_t index_width = 0; index_width < Tile_3d_t::WIDTH; ++index_width) {
#pragma unroll
                    for (int32_t index_group = 0; index_group < UINT32_PER_TILE_G; ++index_group) {
                        data_[index_depth][index_height][index_width][index_group] = 0;
                    }
                }
            }
        }
    }

    __device__ inline Math_tile_3d(uint32_t in[Tile_3d_t::DEPTH][Tile_3d_t::HEIGHT]
                                              [Tile_3d_t::WIDTH][UINT32_PER_TILE_G])
    {
        copy<Tile_3d_t::DEPTH, Tile_3d_t::HEIGHT, Tile_3d_t::WIDTH, UINT32_PER_TILE_G>(data_, in);
    }

    __device__ inline Math_tile_3d(Math_tile_3d_other_t &other)
    {
#pragma unroll
        for (int32_t index_depth = 0; index_depth < Tile_3d_t::DEPTH; ++index_depth) {
#pragma unroll
            for (int32_t index_height = 0; index_height < Tile_3d_t::HEIGHT; ++index_height) {
#pragma unroll
                for (int32_t index_width = 0; index_width < Tile_3d_t::WIDTH; ++index_width) {
                    Type_colwerter<typename Data_type_t::Type,
                                   typename Data_type_other_t::Type,
                                   uint32_t,
                                   uint32_t,
                                   Math_tile_3d_other_t::UINT32_PER_TILE_G>::
                        exlwte(data_[index_depth][index_height][index_width],
                               other.data_[index_depth][index_height][index_width]);
                }
            }
        }
    }

    __device__ inline void flip(bool do_flip)
    {
        uint32_t tmp_data[Tile_3d_t::DEPTH][Tile_3d_t::HEIGHT][Tile_3d_t::WIDTH][UINT32_PER_TILE_G];
#pragma unroll
        for (int32_t index_depth = 0; index_depth < Tile_3d_t::DEPTH; ++index_depth) {
#pragma unroll
            for (int32_t index_height = 0; index_height < Tile_3d_t::HEIGHT; ++index_height) {
#pragma unroll
                for (int32_t index_width = 0; index_width < Tile_3d_t::WIDTH; ++index_width) {
#pragma unroll
                    for (int32_t index_group = 0; index_group < UINT32_PER_TILE_G; ++index_group) {
                        if (do_flip) {
                            tmp_data[index_depth][index_height][index_width][index_group] =
                                data_[Tile_3d_t::DEPTH - 1 - index_depth]
                                     [Tile_3d_t::HEIGHT - 1 - index_height]
                                     [Tile_3d_t::WIDTH - 1 - index_width][index_group];
                        } else {
                            tmp_data[index_depth][index_height][index_width][index_group] =
                                data_[index_depth][index_height][index_width][index_group];
                        }
                    }
                }
            }
        }
        copy<Tile_3d_t::DEPTH, Tile_3d_t::HEIGHT, Tile_3d_t::WIDTH, UINT32_PER_TILE_G>(data_,
                                                                                       tmp_data);
    }

    __device__ inline void expose(uint32_t (&out)[Tile_3d_t::DEPTH][Tile_3d_t::HEIGHT]
                                                 [Tile_3d_t::WIDTH][UINT32_PER_TILE_G],
                                  bool do_flip)
    {
        flip(do_flip);
        copy<Tile_3d_t::DEPTH, Tile_3d_t::HEIGHT, Tile_3d_t::WIDTH, UINT32_PER_TILE_G>(out, data_);
    }

    __device__ inline void pack()
    {
        uint32_t tmp_packed_data[Tile_3d_t::VALUE][UINT32_PER_TILE_G];
#pragma unroll
        for (int32_t index_depth = 0; index_depth < Tile_3d_t::DEPTH; ++index_depth) {
#pragma unroll
            for (int32_t index_height = 0; index_height < Tile_3d_t::HEIGHT; ++index_height) {
#pragma unroll
                for (int32_t index_width = 0; index_width < Tile_3d_t::WIDTH; ++index_width) {
#pragma unroll
                    for (int32_t index_group = 0; index_group < UINT32_PER_TILE_G; ++index_group) {
                        int32_t index_linear = index_width + index_height * Tile_3d_t::WIDTH +
                                               index_depth * Tile_3d_t::HEIGHT * Tile_3d_t::WIDTH;
                        tmp_packed_data[index_linear][index_group] =
                            data_[index_depth][index_height][index_width][index_group];
                    }
                }
            }
        }
        for (int32_t i = 0; i < Tile_3d_t::VALUE; ++i) {
            make_packed_uint(packed_data_[i], tmp_packed_data[i]);
        }
    }

    __device__ inline void apply_alpha(uint32_t alpha)
    {
#pragma unroll
        for (int32_t index_depth = 0; index_depth < Tile_3d_t::DEPTH; ++index_depth) {
#pragma unroll
            for (int32_t index_height = 0; index_height < Tile_3d_t::HEIGHT; ++index_height) {
#pragma unroll
                for (int32_t index_width = 0; index_width < Tile_3d_t::WIDTH; ++index_width) {
                    Apply_alpha<Data_type_t, uint32_t, UINT32_PER_TILE_G>::exlwte(
                        data_[index_depth][index_height][index_width], alpha);
                }
            }
        }
    }

    __device__ inline void apply_beta(uint32_t data_c[Tile_3d_t::DEPTH][Tile_3d_t::HEIGHT]
                                                     [Tile_3d_t::WIDTH][UINT32_PER_TILE_G],
                                      uint32_t beta)
    {
        if (beta != 0) {
#pragma unroll
            for (int32_t index_depth = 0; index_depth < Tile_3d_t::DEPTH; ++index_depth) {
#pragma unroll
                for (int32_t index_height = 0; index_height < Tile_3d_t::HEIGHT; ++index_height) {
#pragma unroll
                    for (int32_t index_width = 0; index_width < Tile_3d_t::WIDTH; ++index_width) {
                        Apply_beta<Data_type_t, uint32_t, UINT32_PER_TILE_G>::exlwte(
                            data_[index_depth][index_height][index_width],
                            data_c[index_depth][index_height][index_width],
                            beta);
                    }
                }
            }
        }
    }

    uint32_t data_[Tile_3d_t::DEPTH][Tile_3d_t::HEIGHT][Tile_3d_t::WIDTH][UINT32_PER_TILE_G];

    Packed_data_type_t packed_data_[Tile_3d_t::VALUE];
};

template <xmma::Operation_type Operation_,
          typename Math_tile_c_,
          typename Math_tile_a_,
          typename Math_tile_b_,
          typename Tile_stride_dhw_,
          typename Tile_dilation_dhw_>
struct Math_operation {
    public:
    static const xmma::Operation_type OPERATION = Operation_;
    using Math_tile_c_t = Math_tile_c_;
    using Math_tile_a_t = Math_tile_a_;
    using Math_tile_b_t = Math_tile_b_;
    using Tile_stride_dhw_t = Tile_stride_dhw_;
    using Tile_dilation_dhw_t = Tile_dilation_dhw_;
    __device__ inline void exlwte(Math_tile_c_t &math_tile_c,
                                  const Math_tile_a_t &math_tile_a,
                                  const Math_tile_b_t &math_tile_b,
                                  const int32_t index_t,
                                  const int32_t index_r,
                                  const int32_t index_s);
};

// Wgrad
template <typename Math_tile_c_,
          typename Math_tile_a_,
          typename Math_tile_b_,
          typename Tile_stride_dhw_,
          typename Tile_dilation_dhw_>
struct Math_operation<xmma::Operation_type::WGRAD,
                      Math_tile_c_,
                      Math_tile_a_,
                      Math_tile_b_,
                      Tile_stride_dhw_,
                      Tile_dilation_dhw_> {
    static const xmma::Operation_type OPERATION = xmma::Operation_type::WGRAD;
    using Math_tile_c_t = Math_tile_c_;
    using Math_tile_a_t = Math_tile_a_;
    using Math_tile_b_t = Math_tile_b_;
    using Tile_stride_dhw_t = Tile_stride_dhw_;
    using Tile_dilation_dhw_t = Tile_dilation_dhw_;
    // Derived
    static const int32_t UINT32_PER_TILE_G = Math_tile_c_t::UINT32_PER_TILE_G;
    using Multiply_and_add_t = Multiply_and_add<typename Math_tile_c_t::Data_type_t,
                                                typename Math_tile_a_t::Data_type_t,
                                                typename Math_tile_b_t::Data_type_t>;

    __device__ inline void exlwte(Math_tile_c_t &math_tile_c,
                                  const Math_tile_a_t &math_tile_a,
                                  const Math_tile_b_t &math_tile_b)
    {
#pragma unroll
        for (int32_t index_t = 0; index_t < Math_tile_c_t::Tile_3d_t::DEPTH; ++index_t) {
#pragma unroll
            for (int32_t index_r = 0; index_r < Math_tile_c_t::Tile_3d_t::HEIGHT; ++index_r) {
#pragma unroll
                for (int32_t index_s = 0; index_s < Math_tile_c_t::Tile_3d_t::WIDTH; ++index_s) {
#pragma unroll
                    for (int32_t index_depth = 0; index_depth < Math_tile_b_t::Tile_3d_t::DEPTH;
                         ++index_depth) {
#pragma unroll
                        for (int32_t index_height = 0;
                             index_height < Math_tile_b_t::Tile_3d_t::HEIGHT;
                             ++index_height) {
#pragma unroll
                            for (int32_t index_width = 0;
                                 index_width < Math_tile_b_t::Tile_3d_t::WIDTH;
                                 ++index_width) {
#pragma unroll
                                for (int i = 0; i < UINT32_PER_TILE_G; ++i) {
                                    math_tile_c.data_[index_t][index_r][index_s]
                                                     [i] = Multiply_and_add_t::exlwte(
                                        math_tile_c.data_[index_t][index_r][index_s][i],
                                        math_tile_a.data_[index_depth * Tile_stride_dhw_t::DEPTH +
                                                          index_t * Tile_dilation_dhw_t::DEPTH]
                                                         [index_height * Tile_stride_dhw_t::HEIGHT +
                                                          index_r * Tile_dilation_dhw_t::HEIGHT]
                                                         [index_width * Tile_stride_dhw_t::WIDTH +
                                                          index_s * Tile_dilation_dhw_t::WIDTH][i],
                                        math_tile_b.data_[index_depth][index_height][index_width]
                                                         [i]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
};

// Fprop
template <typename Math_tile_c_,
          typename Math_tile_a_,
          typename Math_tile_b_,
          typename Tile_stride_dhw_,
          typename Tile_dilation_dhw_>
struct Math_operation<xmma::Operation_type::FPROP,
                      Math_tile_c_,
                      Math_tile_a_,
                      Math_tile_b_,
                      Tile_stride_dhw_,
                      Tile_dilation_dhw_> {
    static const xmma::Operation_type OPERATION = xmma::Operation_type::FPROP;
    using Math_tile_c_t = Math_tile_c_;
    using Math_tile_a_t = Math_tile_a_;
    using Math_tile_b_t = Math_tile_b_;
    using Tile_stride_dhw_t = Tile_stride_dhw_;
    using Tile_dilation_dhw_t = Tile_dilation_dhw_;
    // Derived
    static const int32_t UINT32_PER_TILE_G = Math_tile_c_t::UINT32_PER_TILE_G;
    using Multiply_and_add_t = Multiply_and_add<typename Math_tile_c_t::Data_type_t,
                                                typename Math_tile_a_t::Data_type_t,
                                                typename Math_tile_b_t::Data_type_t>;

    __device__ inline void exlwte(Math_tile_c_t &math_tile_c,
                                  const Math_tile_a_t &math_tile_a,
                                  const Math_tile_b_t &math_tile_b)
    {
#pragma unroll
        for (int32_t index_depth = 0; index_depth < Math_tile_c_t::Tile_3d_t::DEPTH;
             ++index_depth) {
#pragma unroll
            for (int32_t index_height = 0; index_height < Math_tile_c_t::Tile_3d_t::HEIGHT;
                 ++index_height) {
#pragma unroll
                for (int32_t index_width = 0; index_width < Math_tile_c_t::Tile_3d_t::WIDTH;
                     ++index_width) {
#pragma unroll
                    for (int i = 0; i < UINT32_PER_TILE_G; ++i) {
#pragma unroll
                        for (int32_t index_t = 0; index_t < Math_tile_b_t::Tile_3d_t::DEPTH;
                             ++index_t) {
#pragma unroll
                            for (int32_t index_r = 0; index_r < Math_tile_b_t::Tile_3d_t::HEIGHT;
                                 ++index_r) {
#pragma unroll
                                for (int32_t index_s = 0; index_s < Math_tile_b_t::Tile_3d_t::WIDTH;
                                     ++index_s) {
                                    math_tile_c.data_[index_depth][index_height][index_width]
                                                     [i] = Multiply_and_add_t::exlwte(
                                        math_tile_c.data_[index_depth][index_height][index_width]
                                                         [i],
                                        math_tile_a.data_[index_depth * Tile_stride_dhw_t::DEPTH +
                                                          index_t * Tile_dilation_dhw_t::DEPTH]
                                                         [index_height * Tile_stride_dhw_t::HEIGHT +
                                                          index_r * Tile_dilation_dhw_t::HEIGHT]
                                                         [index_width * Tile_stride_dhw_t::WIDTH +
                                                          index_s * Tile_dilation_dhw_t::WIDTH][i],
                                        math_tile_b.data_[index_t][index_r][index_s][i]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
};

} // namespace depthwise_colwolution
} // namespace ext
} // namespace xmma

#endif
