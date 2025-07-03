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

#include <xmma/hopper/emu/lwda_tma_get_string.h>

#include <stdio.h>

namespace xmma {
namespace hopper {
namespace emu {

static inline __device__ __host__ void print_tma_descriptor_raw(const tma_descriptor_t *p_desc) {
  assert(p_desc != NULL);
  printf("--- Raw TMA Tensor Params ---\n");
  printf("tensor.type: %s\n", get_string_desc_type(p_desc->tensor.type));
  printf("tensor.dim: %u\n", static_cast<uint32_t>(p_desc->tensor.dim));
  printf("tensor.tensor_global_address: %zx\n", p_desc->tensor.tensor_global_address);
  printf("tensor.format: %s\n", get_string_format(p_desc->tensor.format));
  printf("tensor.interleaved: %s\n", get_string_interleaved(p_desc->tensor.interleaved));
  for(uint32_t i = 0; i < 5; i++) {
    printf("tensor.tensor_size[%u]: %u\n", i, p_desc->tensor.tensor_size[i]);
  }
  for(uint32_t i = 0; i < 4; i++) {
    printf("tensor.tensor_stride[%u]: %zu\n", i, p_desc->tensor.tensor_stride[i]);
  }
  
  printf("--- Raw TMA Access Params ---\n");
  for(uint32_t i = 0; i < 5; i++) {
    printf("access.traversal_stride[%u]: %u\n", i, p_desc->access.traversal_stride[i]);
  }
  printf("access.swizzle: %s\n", get_string_swizzle(p_desc->access.swizzle));
  printf("access.oob_fill_mode: %s\n", p_desc->access.oob_fill_mode == false ? "ZERO" : "CONSTANT");
  printf("access.f32_to_tf32: %s\n", p_desc->access.f32_to_tf32 == false ? "DISABLED" : "ENABLED");
  printf("access.promotion: %s\n", get_string_promotion(p_desc->access.promotion));
  
  printf("--- Raw TMA Tile Descriptor ---\n");
  for(uint32_t i = 0; i < 5; i++) {
    printf("tiled.box_size[%u]: %u\n", i, p_desc->tiled.box_size[i]);
  }

  printf("--- Raw TMA Im2Col Descriptor ---\n");
  printf("im2col.load_range_ndhw: %u\n", p_desc->im2col.load_range_ndhw);
  printf("im2col.load_range_c: %u\n", p_desc->im2col.load_range_c);

  for(uint32_t i = 0; i < 3; i++) {
    printf("im2col.box_base_corner[%u]: %u\n", i, p_desc->im2col.box_base_corner[i]);
  }

  for(uint32_t i = 0; i < 3; i++) {
    printf("im2col.box_far_corner[%u]: %u\n", i, p_desc->im2col.box_far_corner[i]);
  }
}

static inline __device__ __host__ void print_tma_descriptor_interp(const tma_descriptor_t *p_desc) {
  printf("--- TMA Tensor Params ---\n");
  assert(p_desc != NULL);
  printf("tensor.type: %s\n", get_string_desc_type(p_desc->tensor.type));
  printf("tensor.dim: %u\n", static_cast<uint32_t>(p_desc->tensor.dim));

  void* addr;
  get_tensor_global_address(p_desc, addr);
  printf("tensor.tensor_global_address: %zx\n", reinterpret_cast<uint64_t>(addr));
  printf("tensor.format: %s\n", get_string_format(p_desc->tensor.format));
  printf("tensor.interleaved: %s\n", get_string_interleaved(p_desc->tensor.interleaved));
  uint64_t tensor_size_[5];
  get_tensor_size(p_desc, tensor_size_);
  for(uint32_t i = 0; i < 5; i++) {
    printf("tensor.tensor_size[%u]: %zu\n", i, tensor_size_[i]);
  }
  uint64_t tensor_stride_[4];
  get_tensor_stride(p_desc, tensor_stride_);
  for(uint32_t i = 0; i < 4; i++) {
    printf("tensor.tensor_stride[%u]: %zu\n", i, tensor_stride_[i]);
  }
  
  printf("--- TMA Access Params ---\n");
  uint64_t traversal_stride_[5];
  get_traversal_stride(p_desc, traversal_stride_);
  for(uint32_t i = 0; i < 5; i++) {
    printf("access.traversal_stride[%u]: %zu\n", i, traversal_stride_[i]);
  }
  printf("access.swizzle: %s\n", get_string_swizzle(p_desc->access.swizzle));
  printf("access.oob_fill_mode: %s\n", p_desc->access.oob_fill_mode == false ? "ZERO" : "CONSTANT");
  printf("access.f32_to_tf32: %s\n", p_desc->access.f32_to_tf32 == false ? "DISABLED" : "ENABLED");
  printf("access.promotion: %s\n", get_string_promotion(p_desc->access.promotion));
  
  printf("--- TMA Tile Descriptor ---\n");
  uint64_t box_size_[5];
  get_tiled_box_size(p_desc, box_size_);
  for(uint32_t i = 0; i < 5; i++) {
    printf("tiled.box_size[%u]: %zu\n", i, box_size_[i]);
  }

  printf("--- TMA Im2Col Descriptor ---\n");
  uint64_t load_range_ndhw_;
  get_im2col_load_range_ndhw(p_desc, load_range_ndhw_);
  printf("im2col.load_range_ndhw: %zu\n", load_range_ndhw_);
  uint64_t load_range_c_;
  get_im2col_load_range_c(p_desc, load_range_c_);
  printf("im2col.load_range_c: %zu\n", load_range_c_);

  int64_t box_base_corner_[3];
  get_im2col_box_base_corner(p_desc, box_base_corner_);
  for(uint32_t i = 0; i < 3; i++) {
    printf("im2col.box_base_corner[%u]: %zu\n", i, box_base_corner_[i]);
  }

  int64_t box_far_corner_[3];
  get_im2col_box_far_corner(p_desc, box_far_corner_);
  for(uint32_t i = 0; i < 3; i++) {
    printf("im2col.box_far_corner[%u]: %zu\n", i, box_far_corner_[i]);
  }
}

} // end namespace emu
} // end namespace hopper
} // end namespace xmma