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

#include "lwda_runtime.h"
#include <xmma/hopper/emu/lwda_tma_utils_internal.h>

namespace xmma {
namespace hopper {
namespace emu {

static inline lwdaError lwdaSetTmaTileDescriptorv2( lwdaTmaDescv2 *p_desc,
                                                    const void *p_addr,
                                                    uint32_t dims,
                                                    lwdaTmaDescFormatv2 format,
                                                    lwdaTmaDescInterleavev2 interleave,
                                                    lwdaTmaDescSwizzlev2 swizzle,
                                                    lwdaTmaDescPromotiolw2 promotion,
                                                    uint32_t *p_tensor_size,
                                                    uint64_t *p_tensor_stride,
                                                    uint32_t *p_traversal_stride,
                                                    uint32_t *p_box_size,
                                                    uint32_t fill_oob,
                                                    uint32_t round_to_tf32 ) {

    set_tensor_common_0( p_desc, reinterpret_cast<uint64_t>( p_addr ) );
    set_tensor_common_1( p_desc,
                         TENSOR_TILED,
                         dims,
                         format,
                         interleave,
                         swizzle,
                         fill_oob,
                         round_to_tf32,
                         promotion );

    set_tensor_stride( p_desc, p_tensor_stride, dims );
    set_tensor_size( p_desc, p_tensor_size, dims );

    set_traversal_stride_tiled( p_desc, p_traversal_stride, dims );

    set_box_size( p_desc, p_box_size, dims );
    return lwdaSuccess;
}

static inline lwdaError lwdaSetTmaIm2ColDescriptorv2( lwdaTmaDescv2 *p_desc,
                                                      const void *p_addr,
                                                      uint32_t dims,
                                                      lwdaTmaDescFormatv2 format,
                                                      lwdaTmaDescInterleavev2 interleave,
                                                      lwdaTmaDescSwizzlev2 swizzle,
                                                      lwdaTmaDescPromotiolw2 promotion,
                                                      uint32_t *p_tensor_size,
                                                      uint64_t *p_tensor_stride,
                                                      uint32_t *p_traversal_stride,
                                                      uint32_t range_c,
                                                      uint32_t range_ndhw,
                                                      int32_t *p_box_base_corner_dhw,
                                                      int32_t *p_box_far_corner_dhw,
                                                      uint32_t fill_oob,
                                                      uint32_t round_to_f32 ) {

    set_tensor_common_0( p_desc, reinterpret_cast<uint64_t>( p_addr ) );
    set_tensor_common_1( p_desc,
                         TENSOR_IM2COL,
                         dims,
                         format,
                         interleave,
                         swizzle,
                         fill_oob,
                         round_to_f32,
                         promotion );

    set_tensor_stride( p_desc, p_tensor_stride, dims );
    set_tensor_size( p_desc, p_tensor_size, dims );

    set_traversal_stride_im2col( p_desc, p_traversal_stride, dims );

    set_range_c( p_desc, range_c );
    set_box_corner_dhw( p_desc,
                        reinterpret_cast<uint32_t *>( p_box_base_corner_dhw ),
                        reinterpret_cast<uint32_t *>( p_box_far_corner_dhw ),
                        dims );
    set_range_ndhw( p_desc, range_ndhw );
    return lwdaSuccess;
}

/**
 * @brief Set tensor parameters to TMA descriptor
 * 
 * @param p_desc Pointer to tma descriptor allocated on host
 * @param ptr Memory address to tensors in device memory
 * @param format Data type of elements in tensor
 * @param interleaved Interleaved format of elements in tensor
 * @param p_tensor_size Array of length of each dimension of tensor
 * @param p_tensor_stride Array of strides of tensor in bytes for each dimension
 * @param dims Number of dimensions of tensor
 */
static inline lwdaError
lwdaSetTmaTensorDescriptor(lwdaTmaDesc *p_desc, const void *ptr,
                           lwdaTmaFormat format, lwdaTmaInterleaved interleaved,
                           uint32_t *p_tensor_size, uint64_t *p_tensor_stride,
                           int dims) {
  if (p_desc != NULL && dims > 0 && dims <= 5 && p_tensor_size != NULL &&
      p_tensor_stride != NULL && ptr != NULL) {
    tma_descriptor_t *p_internal_desc =
        reinterpret_cast<tma_descriptor_t *>(p_desc);
    set_tensor_global_address(p_internal_desc, ptr);
    set_format(p_internal_desc, format);
    set_interleaved(p_internal_desc, interleaved);
    set_tensor_size(p_internal_desc, p_tensor_size, dims);
    set_tensor_stride(p_internal_desc, p_tensor_stride, dims);
    set_tensor_dim(p_internal_desc, static_cast<uint8_t>(dims));
    return lwdaSuccess;
  }
  return lwdaErrorIlwalidValue;
}

/**
 * @brief Set image access parameters in TMA descriptor
 * 
 * @param p_desc Pointer to tma descriptor allocated on host
 * @param swizzle swizzle pattern of how data will be stored in shared memory
 * @param p_traversal_stride Array of width of how elements are accessed along each dimension
 * @param load_range_c how many elements to load along c dimension
 * @param load_range_ndhw how many elements to load along lwmmulative ndhw dimension
 * @param p_box_base_corner array of signed offset of starting index along d, h, w dimensions. The array length should not be greater than 3
 * @param p_box_far_corner array of signed offset of ending index along d, h, w dimensions. The array length should not be greater than 3
 * @param dims number of dimensions in tensor
 */
static inline lwdaError lwdaSetTmaImageAccessDescriptor(
    lwdaTmaDesc *p_desc, lwdaTmaSwizzle swizzle,
    uint64_t *p_traversal_stride, uint64_t load_range_c,
    uint64_t load_range_ndhw, int64_t *p_box_base_corner,
    int64_t *p_box_far_corner, int dims) {
  if (p_desc != NULL && p_traversal_stride != NULL &&
      p_box_base_corner != NULL && p_box_far_corner != NULL && dims > 0 &&
      dims <= 5) {
    tma_descriptor_t *p_internal_desc =
        reinterpret_cast<tma_descriptor_t *>(p_desc);
    set_tensor_type(p_internal_desc, IM2COL);
    set_swizzle(p_internal_desc, swizzle);
    set_traversal_stride(p_internal_desc, p_traversal_stride, dims);
    set_im2col_load_range_c(p_internal_desc, load_range_c);
    set_im2col_load_range_ndhw(p_internal_desc, load_range_ndhw);
    set_im2col_box_base_corner(p_internal_desc, p_box_base_corner, dims);
    set_im2col_box_far_corner(p_internal_desc, p_box_far_corner, dims);
    return lwdaSuccess;
  }
  return lwdaErrorIlwalidValue;
}

/**
 * @brief Set tile access parameters in TMA descriptor
 * 
 * @param p_desc Pointer to tma descriptor allocated on host
 * @param swizzle swizzle pattern of how data will be stored in shared memory
 * @param p_traversal_stride array of access stride along each dimension
 * @param p_box_size array of width of elements to be accessed along each dimension
 * @param dims nubmer of dimensions in tensor
 */
static inline lwdaError
lwdaSetTmaTileAccessDescriptor(lwdaTmaDesc *p_desc, lwdaTmaSwizzle swizzle,
                               uint64_t *p_traversal_stride,
                               uint64_t *p_box_size, int dims) {
  if (p_desc != NULL && p_traversal_stride != NULL && p_box_size != NULL &&
      dims > 0 && dims <= 5) {
    tma_descriptor_t *p_internal_desc =
        reinterpret_cast<tma_descriptor_t *>(p_desc);
    set_tensor_type(p_internal_desc, TILED);
    set_swizzle(p_internal_desc, swizzle);
    set_traversal_stride(p_internal_desc, p_traversal_stride, dims);
    set_tiled_box_size(p_internal_desc, p_box_size, dims);
    return lwdaSuccess;
  }
  return lwdaErrorIlwalidValue;
}

} // end namespace emu
} // end namespace hopper
} // end namespace xmma