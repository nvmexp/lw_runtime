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

#include <xmma/hopper/emu/lwda_tma_types.h>
#if !defined(__LWDACC_RTC__)
#include <cassert>
#include <cstdio>
#endif

constexpr uint64_t k_tensor_global_address_mask = ((uint64_t(1) << 55) - 1);
constexpr uint64_t k_tensor_stride_mask = ((uint64_t(1) << 37) - 1);
constexpr uint8_t k_traversal_stride_mask = ((uint8_t(1) << 4) - 1);
constexpr uint16_t k_load_range_ndhw_mask = ((uint16_t(1) << 11) - 1);

constexpr uint32_t k_shared_data_address_mask = ((uint32_t(1) << 25) - 1);
constexpr uint32_t k_shared_barrier_address_mask = ((uint32_t(1) << 16) - 1);

constexpr int32_t  k_cacheline_bytes = 128;
constexpr uint32_t k_warp_size = 32;
constexpr uint32_t k_align_width_bytes = 16;

namespace xmma {
namespace hopper {
namespace emu {

static inline __device__ uint32_t get_shared_data_address(uint32_t ur) {
  return (ur & k_shared_data_address_mask) << 7;
}

static inline __device__ uint32_t set_shared_data_address(uint32_t addr) {
  return (addr >> 7) << 7;
}

static inline __device__ uint32_t get_shared_barrier_address(uint32_t ur) {
  return (ur & k_shared_barrier_address_mask) << 3;
}

static inline __device__ uint32_t set_shared_barrier_address(uint32_t addr) {
  return (addr >> 3) & k_shared_barrier_address_mask;
}

static inline __device__ __host__ void get_tensor_type(const tma_descriptor_t *p_desc, desc_type_t &type) {
  type = p_desc->tensor.type;
}

static inline __device__ __host__ void set_tensor_type(tma_descriptor_t *p_desc, desc_type_t type) {
  p_desc->tensor.type = type;
}

static inline __device__ __host__ void get_tensor_dim(const tma_descriptor_t *p_desc, uint8_t &dim) {
  dim = p_desc->tensor.dim;
}

static inline __device__ __host__ void set_tensor_dim(tma_descriptor_t *p_desc, uint8_t dim) {
  p_desc->tensor.dim = dim;
}

static inline __device__ __host__ void get_tensor_global_address(const tma_descriptor_t *p_desc, void* &addr) {
  addr = reinterpret_cast<void*>(((p_desc->tensor.tensor_global_address) & k_tensor_global_address_mask) << 4);
}

static inline __device__ __host__ void set_tensor_global_address(tma_descriptor_t *p_desc, const void *addr) {
  p_desc->tensor.tensor_global_address = ((reinterpret_cast<uint64_t>(addr) >> 4) & k_tensor_global_address_mask);
}

static inline __device__ __host__ void get_format(const tma_descriptor_t *p_desc, format_t &format) {
  format = p_desc->tensor.format;
}

static inline __device__ __host__ void set_format(tma_descriptor_t *p_desc, format_t format) {
  p_desc->tensor.format = format;
}

static inline __device__ __host__ void get_interleaved(const tma_descriptor_t *p_desc, interleaved_t &interleaved) {
  interleaved = p_desc->tensor.interleaved;
}

static inline __device__ __host__ void set_interleaved(tma_descriptor_t *p_desc, interleaved_t interleaved) {
  p_desc->tensor.interleaved = interleaved;
}

static inline __device__ __host__ void get_swizzle(const tma_descriptor_t *p_desc, swizzle_t &swizzle) {
  swizzle = p_desc->access.swizzle;
}

static inline __device__ __host__ void set_swizzle(tma_descriptor_t *p_desc, swizzle_t swizzle) {
  p_desc->access.swizzle = swizzle;
}

static inline __device__ __host__ void get_tensor_size(const tma_descriptor_t *p_desc, uint32_t *p_tensor_size, uint32_t dims) {
  for(uint32_t i = 0; i < dims; i++) {
    p_tensor_size[i] = static_cast<uint32_t>(p_desc->tensor.tensor_size[i]) + 1;
  }
}

static inline __device__ __host__ void get_tensor_size(const tma_descriptor_t *p_desc, uint64_t tensor_size[5]) {
  for(uint32_t i = 0; i < p_desc->tensor.dim; i++) {
    tensor_size[i] = static_cast<uint64_t>(p_desc->tensor.tensor_size[i]) + 1;
  }
}

static inline __device__ __host__ void set_tensor_size(tma_descriptor_t *p_desc, uint32_t *p_tensor_size, uint32_t dims) {
  for(uint32_t i = 0; i < dims; i++) {
    p_desc->tensor.tensor_size[i] = static_cast<uint32_t>(p_tensor_size[i] - 1);
  }
}

static inline __device__ __host__ void set_tensor_size(tma_descriptor_t *p_desc, uint64_t tensor_size[5]) {
  for(uint32_t i = 0; i < p_desc->tensor.dim; i++) {
    p_desc->tensor.tensor_size[i] = static_cast<uint32_t>(tensor_size[i] - 1);
  }
}

static inline __device__ __host__ void get_tensor_stride(const tma_descriptor_t *p_desc, uint64_t *p_tensor_stride, uint32_t dims) {
  for(uint32_t i = 0; i < dims - 1; i++) {
    p_tensor_stride[i] = ((p_desc->tensor.tensor_stride[i]) & k_tensor_stride_mask) << 4;
  }
}

static inline __device__ __host__ void get_tensor_stride(const tma_descriptor_t *p_desc, uint64_t tensor_stride[4]) {
  for(int32_t i = 0; i < p_desc->tensor.dim - 1; i++) {
    tensor_stride[i] = (p_desc->tensor.tensor_stride[i] & k_tensor_stride_mask) << 4;
  }
}

static inline __device__ __host__ void set_tensor_stride(tma_descriptor_t *p_desc, uint64_t *p_tensor_stride, uint32_t dims) {
  for(uint32_t i = 0; i < dims - 1; i++) {
    p_desc->tensor.tensor_stride[i] = (p_tensor_stride[i] >> 4) & k_tensor_stride_mask;
  }
}

static inline __device__ __host__ void set_tensor_stride(tma_descriptor_t *p_desc, uint64_t tensor_stride[4]) {
  for(int32_t i = 0; i < p_desc->tensor.dim - 1; i++) {
    p_desc->tensor.tensor_stride[i] = (tensor_stride[i] >> 4) & k_tensor_stride_mask;
  }
}

static inline __device__ __host__ void get_traversal_stride(const tma_descriptor_t *p_desc, uint64_t *p_traversal_stride, int dims) {
  for(int32_t i = 0; i < dims; i++) {
    p_traversal_stride[i] = ((static_cast<uint64_t>(p_desc->access.traversal_stride[i]) & k_traversal_stride_mask) + 1);
  }
}

static inline __device__ __host__ void get_traversal_stride(const tma_descriptor_t *p_desc, uint64_t traversal_stride[5]) {
  for(uint32_t i = 0; i < p_desc->tensor.dim; i++) {
    traversal_stride[i] = ((static_cast<uint64_t>(p_desc->access.traversal_stride[i]) & k_traversal_stride_mask) + 1);
  }
}

static inline __device__ __host__ void set_traversal_stride(tma_descriptor_t *p_desc, uint64_t *p_traversal_stride, int dims) {
  for(int32_t i = 0; i < dims; i++) {
    p_desc->access.traversal_stride[i] = static_cast<uint8_t>((p_traversal_stride[i] - 1) & k_traversal_stride_mask);
  }
}

static inline __device__ __host__ void set_traversal_stride(tma_descriptor_t *p_desc, uint64_t traversal_stride[5]) {
  for(uint32_t i = 0; i < p_desc->tensor.dim; i++) {
    p_desc->access.traversal_stride[i] = static_cast<uint8_t>((traversal_stride[i] - 1) & k_traversal_stride_mask);
  }
}

static inline __device__ __host__ void get_tiled_box_size(const tma_descriptor_t *p_desc, uint64_t *p_box_size, int dims) {
  for(int32_t i = 0; i < dims; i++) {
    p_box_size[i] = static_cast<uint64_t>(p_desc->tiled.box_size[i]) + 1;
  }
}

static inline __device__ __host__ void get_tiled_box_size(const tma_descriptor_t *p_desc, uint64_t box_size[5]) {
  for(uint32_t i = 0; i < p_desc->tensor.dim; i++) {
    box_size[i] = static_cast<uint64_t>(p_desc->tiled.box_size[i]) + 1;
  }
}

static inline __device__ __host__ void set_tiled_box_size(tma_descriptor_t *p_desc, uint64_t *p_box_size, int dims) {
  for(int32_t i = 0; i < dims; i++) {
    p_desc->tiled.box_size[i] = static_cast<uint8_t>(p_box_size[i] - 1);
  }
}

static inline __device__ __host__ void set_tiled_box_size(tma_descriptor_t *p_desc, uint64_t box_size[5]) {
  for(uint32_t i = 0; i < p_desc->tensor.dim; i++) {
    p_desc->tiled.box_size[i] = static_cast<uint8_t>(box_size[i] - 1);
  }
}

static inline __device__ __host__ void get_im2col_load_range_ndhw(const tma_descriptor_t *p_desc, uint64_t &load_range_ndhw) {
  load_range_ndhw = (static_cast<uint64_t>(p_desc->im2col.load_range_ndhw) + 1) & k_load_range_ndhw_mask;
}

static inline __device__ __host__ void set_im2col_load_range_ndhw(tma_descriptor_t *p_desc, uint64_t load_range_ndhw) {
  p_desc->im2col.load_range_ndhw = static_cast<uint16_t>(load_range_ndhw - 1) & k_load_range_ndhw_mask;
}

static inline __device__ __host__ void get_im2col_load_range_c(const tma_descriptor_t *p_desc, uint64_t &load_range_c) {
  load_range_c = static_cast<uint64_t>(p_desc->im2col.load_range_c) + 1;
}

static inline __device__ __host__ void set_im2col_load_range_c(tma_descriptor_t *p_desc, uint64_t load_range_c) {
  p_desc->im2col.load_range_c = static_cast<uint8_t>(load_range_c - 1);
}

static inline __device__ __host__ void get_im2col_box_base_corner(const tma_descriptor_t *p_desc, int64_t *p_box_base_corner, int dims) {
  if(dims >= 2) {
    p_box_base_corner[0] = static_cast<int64_t>(p_desc->im2col.box_base_corner[0]);
  }
  if(dims >= 3) {
    p_box_base_corner[1] = static_cast<int64_t>(p_desc->im2col.box_base_corner[1]);
  }
  if(dims >= 4) {
    p_box_base_corner[2] = static_cast<int64_t>(p_desc->im2col.box_base_corner[2]);
  }
}

static inline __device__ __host__ void get_im2col_box_base_corner(const tma_descriptor_t *p_desc, int64_t p_box_base_corner[3]) {
  for(int i = 0; i < 3; i++) {
    p_box_base_corner[i] = static_cast<int64_t>(p_desc->im2col.box_base_corner[i]);
  }
}

static inline __device__ __host__ void set_im2col_box_base_corner(tma_descriptor_t *p_desc, int64_t *p_box_base_corner, int dims) {
  if(dims >= 2) {
    p_desc->im2col.box_base_corner[0] = static_cast<int16_t>(p_box_base_corner[0]);
  }
  if(dims >= 3) {
    p_desc->im2col.box_base_corner[1] = static_cast<int16_t>(p_box_base_corner[1]);
  }
  if(dims >= 4) {
    p_desc->im2col.box_base_corner[2] = static_cast<int16_t>(p_box_base_corner[2]);
  }
}

static inline __device__ __host__ void set_im2col_box_base_corner(tma_descriptor_t *p_desc, int64_t box_base_corner[3]) {
  for(uint32_t i = 0; i < 3; i++) {
    p_desc->im2col.box_base_corner[i] = static_cast<int16_t>(box_base_corner[i]);
  }
}

static inline __device__ __host__ void get_im2col_box_far_corner(const tma_descriptor_t *p_desc, int64_t *p_box_far_corner, int dims) {
  if(dims >= 2) {
    p_box_far_corner[0] = static_cast<int64_t>(p_desc->im2col.box_far_corner[0]);
  }
  if(dims >= 3) {
    p_box_far_corner[1] = static_cast<int64_t>(p_desc->im2col.box_far_corner[1]);
  }
  if(dims >= 4) {
    p_box_far_corner[2] = static_cast<int64_t>(p_desc->im2col.box_far_corner[2]);
  }
}

static inline __device__ __host__ void get_im2col_box_far_corner(const tma_descriptor_t *p_desc, int64_t box_far_corner[3]) {
  for(uint32_t i = 0; i < 3; i++) {
    box_far_corner[i] = static_cast<int64_t>(p_desc->im2col.box_far_corner[i]);
  }
}

static inline __device__ __host__ void set_im2col_box_far_corner(tma_descriptor_t *p_desc, int64_t *p_box_far_corner, int dims) {
  if(dims >= 2) {
    p_desc->im2col.box_far_corner[0] = static_cast<int16_t>(p_box_far_corner[0]);
  }
  if(dims >= 3) {
    p_desc->im2col.box_far_corner[1] = static_cast<int16_t>(p_box_far_corner[1]);
  }
  if(dims >= 4) {
    p_desc->im2col.box_far_corner[2] = static_cast<int16_t>(p_box_far_corner[2]);
  }
}

static inline __device__ __host__ void set_im2col_box_far_corner(tma_descriptor_t *p_desc, int64_t box_far_corner[3]) {
  for(uint32_t i = 0; i < 3; i++) {
    p_desc->im2col.box_far_corner[i] = static_cast<int16_t>(box_far_corner[i]);
  }
}


static inline __device__ uint64_t get_swizzle_width_bytes(swizzle_t swizzle) {
  uint64_t swizzle_width_bytes = swizzle == SWIZZLE_128B ? 128 : k_align_width_bytes;
  swizzle_width_bytes = swizzle == SWIZZLE_64B ? 64 : swizzle_width_bytes;
  swizzle_width_bytes = swizzle == SWIZZLE_32B ? 32 : swizzle_width_bytes;
  // If everything is false, default is 16 -> k_align_width_bytes -> SWIZZLE_DISABLED
  return swizzle_width_bytes;
}

template<size_t k_element_size>
static inline __device__ uint64_t get_interleaved_width_bytes(interleaved_t interleaved) {
  uint64_t interleaved_width_bytes = interleaved == INTERLEAVED_16B ? 16 : k_element_size;
  interleaved_width_bytes = interleaved == INTERLEAVED_32B ? 32 : interleaved_width_bytes;
  return interleaved_width_bytes;
}

static inline __device__ uint64_t get_bytes_from_format(format_t format) {
  switch(format) {
    case U8: return 1;
    case U16:
    case F16_RN:
    case BF16_RN: return 2;
    case U32:
    case S32:
    case F32_RN:
    case F32_FTZ_RN: return 4;
    case U64:
    case S64:
    case F64_RN: return 8;
    default: return 1;
  }
}

static inline __device__ uint64_t get_bytes_from_interleaved(interleaved_t interleaved) {
  switch(interleaved) {
    case INTERLEAVED_16B: return 16;
    case INTERLEAVED_32B: return 32;
    default: return 0;
  }
}

// We may need non byte version of get_swizzle_width_bytes


template<uint8_t k_dim, desc_type_t k_desc>
static inline __device__ __host__ void validate_tma_descriptor(const tma_descriptor_t *p_desc) {
  assert(p_desc->tensor.type == k_desc && "bad type");
  assert(p_desc->tensor.dim == k_dim && "bad dim");
  assert(p_desc->tensor.tensor_global_address < ((uint64_t(1) << 55) - 1) && "bad tensor_global_address");
  assert(p_desc->tensor.format < FORMAT_MAX && "bad format");
  assert(p_desc->tensor.interleaved < INTERLEAVED_MAX && "bad interleaved");
  assert(p_desc->tensor.tensor_stride[0] <= ((uint64_t(1) << 37) - 1) && "bad tensor_stride[0]");
  assert(p_desc->tensor.tensor_stride[1] <= ((uint64_t(1) << 37) - 1) && "bad tensor_stride[1]");
  assert(p_desc->tensor.tensor_stride[2] <= ((uint64_t(1) << 37) - 1) && "bad tensor_stride[2]");
  assert(p_desc->tensor.tensor_stride[3] <= ((uint64_t(1) << 37) - 1) && "bad tensor_stride[3]");

  assert(p_desc->access.traversal_stride[0] < 8 && "bad traversal_stride[0]");
  assert(p_desc->access.traversal_stride[1] < 8 && "bad traversal_stride[1]");
  assert(p_desc->access.traversal_stride[2] < 8 && "bad traversal_stride[2]");
  assert(p_desc->access.traversal_stride[3] < 8 && "bad traversal_stride[3]");
  assert(p_desc->access.traversal_stride[4] < 8 && "bad traversal_stride[4]");

  assert(p_desc->tiled.box_size[0] < 256 && "bad box_size[0]");
  assert(p_desc->tiled.box_size[1] < 256 && "bad box_size[1]");
  assert(p_desc->tiled.box_size[2] < 256 && "bad box_size[2]");
  assert(p_desc->tiled.box_size[3] < 256 && "bad box_size[3]");
  assert(p_desc->tiled.box_size[4] < 256 && "bad box_size[4]");

  assert(p_desc->im2col.load_range_ndhw < 1024 && "bad load_range_ndhw");
  
  assert(p_desc->im2col.load_range_c < 256 && "bad load_range_c");
  
  assert(p_desc->im2col.box_base_corner[0] < 16 && "bad box_base_corner[0]");
  assert(p_desc->im2col.box_base_corner[1] < 16 && "bad box_base_corner[1]");
  assert(p_desc->im2col.box_base_corner[2] < 16 && "bad box_base_corner[2]");

  assert(p_desc->im2col.box_far_corner[0] < 16 && "bad box_far_corner[0]");
  assert(p_desc->im2col.box_far_corner[1] < 16 && "bad box_far_corner[1]");
  assert(p_desc->im2col.box_far_corner[2] < 16 && "bad box_far_corner[2]");

  uint64_t traversal_stride[5];
  swizzle_t swizzle;

  get_swizzle(p_desc, swizzle);
  get_traversal_stride(p_desc, traversal_stride);

  if( swizzle != SWIZZLE_DISABLED ) {
      assert( traversal_stride[0] == 1 );
  }
}

} // end namespace emu
} // end namespace hopper
} // end namespace xmma

namespace xmma {
namespace fk {
namespace gh100 {
namespace tma {

constexpr int k_smem_row_bytes = 128;
constexpr int k_max_block_size_col = 256;

static inline __device__ __host__ constexpr int __div_up(int x, int y) {
  return (x + y - 1) / y;
}

static inline __device__ __host__ constexpr int __tile_row_bytes(int tile_row, int bytes) {
  return tile_row * bytes;
}

static inline __device__ __host__ constexpr int __tile_row_trunc_bytes(int tile_row, int bytes) {
  return __tile_row_bytes(tile_row, bytes) > k_smem_row_bytes ? k_smem_row_bytes : __tile_row_bytes(tile_row, bytes);
}

static inline __device__ __host__ constexpr int __tile_row_trunc(int tile_row, int bytes) {
  return __tile_row_trunc_bytes(tile_row, bytes) / bytes;
}

static inline __device__ __host__ constexpr int __tile_col_trunc(int tile_col) {
  return tile_col >= k_max_block_size_col ? k_max_block_size_col : tile_col;
}

static inline __device__ __host__ constexpr int __tile_block_row(int tile_row, int bytes) {
  return __tile_row_trunc(tile_row, bytes);
}

static inline __device__ __host__ constexpr int __tile_block_col(int tile_col) {
  return __tile_col_trunc(tile_col);
}

static inline __device__ __host__ constexpr int __num_tile_blocks_row(int tile_row, int bytes) {
  return __div_up(tile_row, __tile_block_row(tile_row, bytes));
}

static inline __device__ __host__ constexpr int __num_tile_blocks_col(int tile_col) {
  return __div_up(tile_col, __tile_block_col(tile_col));
}

static inline __device__ __host__ constexpr int __tile_block_row_aligned_bytes(int tile_row, int bytes) {
  return __tile_row_trunc_bytes(tile_row, bytes) > 64 ? 128 : (__tile_row_trunc_bytes(tile_row, bytes) > 32 ? 64 : (__tile_row_trunc_bytes(tile_row, bytes) > 16 ? 32 : 16));
}

static inline __device__ __host__ constexpr swizzle_t __tile_block_row_swizzle(int tile_row, int bytes) {
    return __tile_block_row_aligned_bytes( tile_row, bytes ) == 128
               ? SWIZZLE_128B
               : ( __tile_block_row_aligned_bytes( tile_row, bytes ) == 64
                       ? SWIZZLE_64B
                       : ( __tile_block_row_aligned_bytes( tile_row, bytes ) == 32
                               ? SWIZZLE_32B
                               : SWIZZLE_DISABLED ) );
}

static inline __device__ __host__ constexpr int __tile_block_row_aligned(int tile_row, int bytes) {
  return __tile_block_row_aligned_bytes(tile_row, bytes) / bytes;
}

static inline __device__ __host__ constexpr int __num_swizzles_in_cache_line(int tile_row, int bytes) {
  return k_smem_row_bytes / __tile_block_row_aligned_bytes(tile_row, bytes);
}

static inline __device__ __host__ constexpr int __num_swizzles_in_swizzle_width(int tile_row, int bytes) {
  return __tile_block_row_aligned_bytes(tile_row, bytes) / 16; // each sector is 16 bytes
}

static inline __device__ __host__ constexpr int __tile_block_col_aligned(int tile_row, int tile_col, int bytes) {
  return __div_up(tile_col, __num_swizzles_in_swizzle_width(tile_row, bytes)) * __num_swizzles_in_swizzle_width(tile_row, bytes);
}

static inline __device__ __host__ constexpr int __tile_row_aligned(int tile_row, int bytes) {
  return __tile_block_row_aligned(tile_row, bytes) * __num_tile_blocks_row(tile_row, bytes);
}

static inline __device__ __host__ constexpr int __tile_col_aligned(int tile_row, int tile_col, int bytes) {
  return __tile_block_col_aligned(tile_row, tile_col, bytes) * __num_tile_blocks_col(tile_col);
}

static inline __device__ __host__ constexpr int __matrix_row(int bytes) {
  return k_smem_row_bytes / bytes;
}

static inline __device__ __host__ constexpr int __matrix_col(int tile_row, int tile_col, int bytes) {
  return __tile_block_col_aligned(tile_row, tile_col, bytes) / __num_tile_blocks_col(tile_col);
}

static inline __device__ __host__ constexpr int __tile_block_row_aligned_offset_bytes(int tile_row, int tile_col, int bytes) {
  return __matrix_col(tile_row, tile_col, bytes) * k_smem_row_bytes;
}

static inline __device__ __host__ constexpr int __tile_block_col_aligned_offset_bytes(int tile_row, int tile_col, int bytes) {
  return __tile_block_row_aligned_offset_bytes(tile_row, tile_col, bytes) * __num_tile_blocks_row(tile_row, bytes);
}

static inline __device__ __host__ constexpr int __tile_shared_memory_bytes(int tile_row, int tile_col, int bytes) {
  return __tile_row_aligned(tile_row, bytes) * __tile_col_aligned(tile_row, tile_col, bytes) * bytes;
}

}
}
}
}

/**
 * CAUTION:
 * !!!! DONT CHANGE THE ORDER OF THE FUNCTIONS BELOW !!!!!
 */


#define NAMESPACE_XMMA_BEGIN namespace xmma {
#define NAMESPACE_XMMA_END } // end namespace xmma

#define NAMESPACE_TMA_BEGIN namespace tma {
#define NAMESPACE_TMA_END } // end namespace tma

#define NAMESPACE_INTERNAL_BEGIN namespace internal {
#define NAMESPACE_INTERNAL_END } // end namespace internal

#define CONST_FK_PREFIX static inline __device__ __host__ constexpr 

NAMESPACE_XMMA_BEGIN
NAMESPACE_TMA_BEGIN
NAMESPACE_INTERNAL_BEGIN

constexpr int k_max_block_size_col = 256;

CONST_FK_PREFIX int kDivideUp(int x, int y) {
  return int(long(x + y - 1) / y);
}

CONST_FK_PREFIX int kTileRowBytes(int tile_row, int bits) {
  return (tile_row * bits) / 8;
}

CONST_FK_PREFIX int kTileRowTruncBytes(int tile_row, int bits) {
  return kTileRowBytes(tile_row, bits) > k_cacheline_bytes ? k_cacheline_bytes : kTileRowBytes(tile_row, bits);
}

CONST_FK_PREFIX int kTileRowTrunc(int tile_row, int bits) {
  return (kTileRowTruncBytes(tile_row, bits) * 8) / bits;
}

CONST_FK_PREFIX int kTileColTrunc(int tile_col) {
  return tile_col > k_max_block_size_col ? k_max_block_size_col : tile_col;
}

NAMESPACE_INTERNAL_END

// External
CONST_FK_PREFIX int kTileBlockRow(int tile_row, int bits) {
  return internal::kTileRowTrunc(tile_row, bits);
}

// External
CONST_FK_PREFIX int kTileBlockCol(int tile_col) {
  return internal::kTileColTrunc(tile_col);
}

// External
CONST_FK_PREFIX int kNumTileBlocksRow(int tile_row, int bits) {
  return internal::kDivideUp(tile_row, kTileBlockRow(tile_row, bits));
}

// External
CONST_FK_PREFIX int kNumTileBlocksCol(int tile_col) {
  return internal::kDivideUp(tile_col, kTileBlockCol(tile_col));
}

NAMESPACE_INTERNAL_BEGIN

CONST_FK_PREFIX int kTileBlockRowAlignedBytes(int tile_row, int bits) {
  return kTileRowTruncBytes(tile_row, bits) > 64 ? 128 : (kTileRowTruncBytes(tile_row, bits) > 32 ? 64 : (kTileRowTruncBytes(tile_row, bits) > 16 ? 32 : 16));
}

NAMESPACE_INTERNAL_END

// External
CONST_FK_PREFIX swizzle_t kTileBlockRowSwizzle(int tile_row, int bits) {
    return xmma::tma::internal::kTileBlockRowAlignedBytes( tile_row, bits ) == 128
               ? SWIZZLE_128B
               : ( xmma::tma::internal::kTileBlockRowAlignedBytes( tile_row, bits ) == 64
                       ? SWIZZLE_64B
                       : ( xmma::tma::internal::kTileBlockRowAlignedBytes( tile_row, bits ) == 32
                               ? SWIZZLE_32B
                               : SWIZZLE_DISABLED ) );
}

NAMESPACE_INTERNAL_BEGIN

CONST_FK_PREFIX int kTileBlockRowAligned(int tile_row, int bits) {
  return (kTileBlockRowAlignedBytes(tile_row, bits) * 8) / bits;
}

CONST_FK_PREFIX int kNumSwizzlesInCacheLine(int tile_row, int bits) {
  return k_cacheline_bytes / kTileBlockRowAlignedBytes(tile_row, bits);
}

CONST_FK_PREFIX int kNumSwizzlesInSwizzleWidth(int tile_row, int bits) {
  return kTileBlockRowAlignedBytes(tile_row, bits) / 16;
}

CONST_FK_PREFIX int kTileBlockColAligned(int tile_row, int tile_col, int bits) {
  return kDivideUp(tile_col, kNumSwizzlesInSwizzleWidth(tile_row, bits)) * kNumSwizzlesInSwizzleWidth(tile_row, bits);
}

CONST_FK_PREFIX int kTileRowAligned(int tile_row, int bits) {
  return kTileBlockRowAligned(tile_row, bits) * kNumTileBlocksRow(tile_row, bits);
}

CONST_FK_PREFIX int kTileColAligned(int tile_row, int tile_col, int bits) {
  return kTileBlockColAligned(tile_row, tile_col, bits) * kNumTileBlocksCol(tile_col);
}

CONST_FK_PREFIX int kMatrixRow(int bits) {
  return (k_cacheline_bytes * 8) / bits;
}

CONST_FK_PREFIX int kMatrixCol(int tile_row, int tile_col, int bits) {
  return kTileBlockColAligned(tile_row, tile_col, bits) / kNumTileBlocksCol(tile_col);
}

NAMESPACE_INTERNAL_END

// External
CONST_FK_PREFIX int kTileBlockRowAlignedOffsetBytes(int tile_row, int tile_col, int bits) {
  return internal::kMatrixCol(tile_row, tile_col, bits) * k_cacheline_bytes;
}

// External
CONST_FK_PREFIX int kTileBlockColAlignedOffsetBytes(int tile_row, int tile_col, int bits) {
  return kTileBlockRowAlignedOffsetBytes(tile_row, tile_col, bits) * kNumTileBlocksRow(tile_row, bits);
}

// External
CONST_FK_PREFIX int kTileSharedMemoryBytes(int tile_row, int tile_col, int bits) {
  return (internal::kTileRowAligned(tile_row, bits) * internal::kTileColAligned(tile_row, tile_col, bits) * bits) / 8;
}

NAMESPACE_TMA_END
NAMESPACE_XMMA_END
