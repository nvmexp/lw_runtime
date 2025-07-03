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
#pragma once

#include <xmma/gemm/gmem_tile.h>
#include <xmma/ext/batchnorm/fragment.h>
#include <xmma/implicit_gemm/dgrad/gmem_tile.h>

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace xmma {
namespace ext {
namespace batchnorm {
namespace bn_stats {
namespace dgrad {

template <
    // The instruction traits.
    typename Traits,
    // The dimensions of the tile computed by the CTA.
    typename Cta_tile,
    // The number of bytes per STG.
    int BYTES_PER_STG = 16,
    // The fragment class before writing data to global memory.
    typename Fragment_c = xmma::Fragment_hmma_fp32_c_bn_stats<Traits, Cta_tile>,
    typename Fragment_post_swizzle = xmma::Fragment_epilogue_post_swizzle<Traits, Cta_tile>,
    // Assuming tensor is packed and tensor strides won't be used.
    bool DISABLE_STRIDES = false>
struct Gmem_tile_c_t : public xmma::implicit_gemm::dgrad::Gmem_tile_epilogue_base<Traits,
                                                                                  Cta_tile,
                                                                                  xmma::Row,
                                                                                  BYTES_PER_STG,
                                                                                  Fragment_c,
                                                                                  DISABLE_STRIDES> {
    using Base = xmma::implicit_gemm::dgrad::Gmem_tile_epilogue_base<Traits,
                                                                     Cta_tile,
                                                                     xmma::Row,
                                                                     BYTES_PER_STG,
                                                                     Fragment_c,
                                                                     DISABLE_STRIDES>;

    template <typename Params>
    inline __device__ Gmem_tile_c_t( const Params &params, int bidm, int bidn, int bidz, int tidx )
        : Base( params, bidm, bidn, bidz, tidx ) {

        params_bn_fprop_mean_ = params.bn_fprop_mean_gmem;
        params_bn_fprop_ilw_stddev_ = params.bn_fprop_ilw_stddev_gmem;
        params_bn_fprop_tensor_ = params.bn_fprop_tensor_gmem;
        params_bn_drelu_bitmask_ = params.bn_drelu_bitmask;

        params_dual_bn_fprop_mean_ = params.dual_bn_fprop_mean_gmem;
        params_dual_bn_fprop_ilw_stddev_ = params.dual_bn_fprop_ilw_stddev_gmem;
        params_dual_bn_fprop_tensor_ = params.dual_bn_fprop_tensor_gmem;

        bidn_ = bidn;
        tidx_ = tidx;
    }

    // Load the data from global memory.
    inline __device__ void load_from_ptr( Fragment_c &data,
                                          int mi,
                                          int ii,
                                          int mask,
                                          const char *ptr,
                                          uint64_t mem_desc = MEM_DESC_DEFAULT ) {
        if( mask ) {
            if( BYTES_PER_STG == 16 ) {
                uint4 tmp;
                xmma::ldg( tmp, ptr + Traits::offset_in_bytes_c( Base::offsets_[ii] ), mem_desc );
                data.from_int4( tmp );
            } else if( BYTES_PER_STG == 8 ) {
                uint2 tmp;
                xmma::ldg( tmp, ptr + Traits::offset_in_bytes_c( Base::offsets_[ii] ), mem_desc );
                data.from_int2( tmp );
            } else {
                uint32_t tmp;
                xmma::ldg( tmp, ptr + Traits::offset_in_bytes_c( Base::offsets_[ii] ), mem_desc );
                data.reg( 0 ) = tmp;
            }
        }
    }

    // Load data from global memory across channels
    inline __device__ void load_from_ptr_across_channels( Fragment<float, 8> &data,
                                                          int mask,
                                                          const char *ptr,
                                                          uint64_t mem_desc = MEM_DESC_DEFAULT ) {
        if( mask ) {
            if( BYTES_PER_STG == 16 ) {
                int col = Base::Tile_distribution::compute_col( tidx_ );
                int offset = bidn_ * Cta_tile::N + col * Base::ELEMENTS_PER_STG;
                data.ldg( ptr + Traits::offset_in_bytes_c( offset ), mem_desc );
            } else {
                static_assert( BYTES_PER_STG == 16, "" );
            }
        }
    }

    // Load the data from global memory.
    inline __device__ void load_bn_fprop_mean( Fragment<float, 8> &data,
                                               int mi,
                                               int ii,
                                               int mask,
                                               uint64_t mem_desc = MEM_DESC_DEFAULT ) {
        const char *ptr = reinterpret_cast<const char *>( params_bn_fprop_mean_ );
        load_from_ptr_across_channels( data, mask, ptr, mem_desc );
    }

    inline __device__ void load_dual_bn_fprop_mean( Fragment<float, 8> &data,
                                                    int mi,
                                                    int ii,
                                                    int mask,
                                                    uint64_t mem_desc = MEM_DESC_DEFAULT ) {
        const char *ptr = reinterpret_cast<const char *>( params_dual_bn_fprop_mean_ );
        load_from_ptr_across_channels( data, mask, ptr, mem_desc );
    }

    inline __device__ void load_bn_fprop_ilw_stddev( Fragment<float, 8> &data,
                                                     int mi,
                                                     int ii,
                                                     int mask,
                                                     uint64_t mem_desc = MEM_DESC_DEFAULT ) {
        const char *ptr = reinterpret_cast<const char *>( params_bn_fprop_ilw_stddev_ );
        load_from_ptr_across_channels( data, mask, ptr, mem_desc );
    }

    inline __device__ void load_dual_bn_fprop_ilw_stddev( Fragment<float, 8> &data,
                                                          int mi,
                                                          int ii,
                                                          int mask,
                                                          uint64_t mem_desc = MEM_DESC_DEFAULT ) {
        const char *ptr = reinterpret_cast<const char *>( params_dual_bn_fprop_ilw_stddev_ );
        load_from_ptr_across_channels( data, mask, ptr, mem_desc );
    }

    inline __device__ void load_bn_fprop_tensor( Fragment_c &data,
                                                 int mi,
                                                 int ii,
                                                 int mask,
                                                 uint64_t mem_desc = MEM_DESC_DEFAULT ) {
        const char *ptr = reinterpret_cast<const char *>( params_bn_fprop_tensor_ );
        load_from_ptr( data, mi, ii, mask, ptr, mem_desc );
    }

    inline __device__ void load_dual_bn_fprop_tensor( Fragment_c &data,
                                                      int mi,
                                                      int ii,
                                                      int mask,
                                                      uint64_t mem_desc = MEM_DESC_DEFAULT ) {
        const char *ptr = reinterpret_cast<const char *>( params_dual_bn_fprop_tensor_ );
        load_from_ptr( data, mi, ii, mask, ptr, mem_desc );
    }

    inline __device__ void load_bitmask_full( Fragment_c &data,
                                              int mi,
                                              int ii,
                                              int mask,
                                              uint64_t mem_desc = MEM_DESC_DEFAULT ) {
        const char *ptr = reinterpret_cast<const char *>( params_bn_drelu_bitmask_ );
        load_from_ptr( data, mi, ii, mask, ptr, mem_desc );
    }

    // The pointer to the input residual buffer.
    const void *params_bn_fprop_mean_;
    const void *params_bn_fprop_ilw_stddev_;
    const void *params_bn_fprop_tensor_;
    const void *params_bn_drelu_bitmask_;
    // dual scale, bias and tensor gmem ptrs
    const void *params_dual_bn_fprop_mean_;
    const void *params_dual_bn_fprop_ilw_stddev_;
    const void *params_dual_bn_fprop_tensor_;

    int bidn_, tidx_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
}  // namespace dgrad
}  // namespace bn_stats
}  // namespace batchnorm
}  // namespace ext
}  // namespace xmma
