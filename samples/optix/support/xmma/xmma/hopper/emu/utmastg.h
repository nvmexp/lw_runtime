#pragma once

#include "xmma/hopper/emu/utma_utils.h"

namespace xmma {
namespace hopper {
namespace emu {
  
/**
 * If smem barriers are required, emulate it in the code
 */
template <uint8_t DIM, lwdaTmaDescType DESC_TYPE, bool MCAST>
__device__ void UTMASTG(const lwdaTmaDesc *p_desc_, uint32_t urb0,
                           uint32_t urb1, int32_t urb2, int32_t urb3,
                           int32_t urb4, int32_t urb5, int32_t urb6,
                           uint32_t urc) {
  static_assert(DIM > 0 && DIM < 6, "dimensions should be (0, 5]\n");

  using namespace xmma::hopper::emu;

  if (threadIdx.x % 32 == 0) { /// Select first thread in warp do copy

    /// TODO: Validate dim vs DIM

    const tma_descriptor_t *p_desc =
        reinterpret_cast<const tma_descriptor_t *>(p_desc_);

    /// Get shared memory data pointer from URF
    uint32_t smem_data_addr = get_shared_data_address(urb0);

    // void *p_smem_data = reinterpret_cast<void
    // *>(__cvta_shared_to_generic(static_cast<uint64_t(smem_data_addr)));
    // printf("smem_data_addr: %u\n", smem_data_addr);

    /// Get shared memory barrier pointer from URF

    /// Get coordinates
    int64_t coord0 = 0, coord1 = 0, coord2 = 0, coord3 = 0, coord4 = 0;
    if (DIM >= 1) {
      coord0 = static_cast<int64_t>(urb2);
    }
    if (DIM >= 2) {
      coord1 = static_cast<int64_t>(urb3);
    }
    if (DIM >= 3) {
      coord2 = static_cast<int64_t>(urb4);
    }
    if (DIM >= 4) {
      coord3 = static_cast<int64_t>(urb5);
    }
    if (DIM == 5) {
      coord4 = static_cast<int64_t>(urb6);
    }
    void *p_gmem_data;
    get_tensor_global_address(p_desc, p_gmem_data);
    uint64_t gmem_data_addr = reinterpret_cast<uint64_t>(p_gmem_data);

    uint64_t tensor_size[5];
    get_tensor_size(p_desc, tensor_size);
    /// Set default values.
    for (int i = DIM; i < 5; i++) {
      tensor_size[i] = 1;
    }

    format_t format;
    get_format(p_desc, format);
    uint64_t bytes = get_bytes_from_format(format);

    interleaved_t interleaved;
    get_interleaved(p_desc, interleaved);
    // bytes = (interleaved == INTERLEAVED_16B || interleaved ==
    // INTERLEAVED_32B) ?  get_bytes_from_interleaved(interleaved) : bytes;

    uint64_t align_width = k_align_width_bytes / bytes;

    swizzle_t swizzle;
    get_swizzle(p_desc, swizzle);
    uint64_t swizzle_width_bytes = get_swizzle_width_bytes(swizzle);
    uint64_t swizzle_width = swizzle_width_bytes / k_align_width_bytes;

    uint64_t traversal_stride[5];
    get_traversal_stride(p_desc, traversal_stride);

    /// swizzle is allowed only when traversal stride for inner most dimension
    /// is zero
    // assert(traversal_stride[0] == 1 && swizzle != SWIZZLE_NONE);

    /// get tensor stride
    uint64_t tensor_stride[4];
    get_tensor_stride(p_desc, tensor_stride);

    /// Set default values.
    for (int i = DIM; i < 5; i++) {
      tensor_stride[i] = tensor_stride[DIM - 1];
    }

    void *p_smem_data;
    asm("{cvta.shared.u64 %0, %1;}\n"
          : "=l"(p_smem_data)
          : "l"(static_cast<uint64_t>(smem_data_addr)));

    /**
     * For TILED mode,
     * 1. validate parameters
     * 2. get all the paramters required from registers
     * 3. get all parameters from descriptor
     */
    if (DESC_TYPE == TILED) {
      /// get box_size
      uint64_t box_size[5] = {0, 0, 0, 0, 0};
      get_tiled_box_size(p_desc, box_size);
      // printf("box_size: %lu, %lu, %lu, %lu, %lu\n", box_size[0], box_size[1],
      // box_size[2], box_size[3], box_size[4]);

      uint64_t box_stride[5] = {1, 1, 1, 1, 1};
      if (DIM >= 1)
        box_stride[0] = box_size[0] / traversal_stride[0];
      if (DIM >= 2)
        box_stride[1] = box_size[1] / traversal_stride[1];
      if (DIM >= 3)
        box_stride[2] = box_size[2] / traversal_stride[2];
      if (DIM >= 4)
        box_stride[3] = box_size[3] / traversal_stride[3];
      if (DIM == 5)
        box_stride[4] = box_size[4] / traversal_stride[4];

      /**
       * as we always use 16byte based cp.async, align box_stride[0] to 16 bytes
       */
      uint64_t box_stride_bytes = box_stride[0] * bytes;
      // printf("box_stride[0]: %lu, box_stride_bytes: %lu\n", box_stride[0],
      // box_stride_bytes);
      uint64_t num_align_widths_in_box_stride =
          __align_up(box_stride_bytes, k_align_width_bytes);

      /// Make sure you are aligned with swizzle bytes before aligning with
      /// cacheline The below code fails if box_size < 128B and
      /// swizzle_width_bytes = 128 Make sure box_size is aligned with swizzle
      /// width first and then align with cacheline Fix below:
      uint64_t aligned_box_stride_swizzle_bytes =
          __div_up(box_stride_bytes, swizzle_width_bytes) * swizzle_width_bytes;

      uint64_t aligned_box_stride_bytes = aligned_box_stride_swizzle_bytes;

      uint64_t num_box_in_cacheline =
          k_cacheline_bytes / aligned_box_stride_bytes;

      uint64_t num_box_along_col =
          __div_up(box_stride[1], num_box_in_cacheline);

      /// shared memory byte offset for a 2d tile

      uint64_t num_elements_per_align_width = k_align_width_bytes / bytes;
      // uint64_t num_elements_per_swizzle_width = swizzle_width_bytes / bytes;
      uint64_t num_align_widths_per_swizzle_width =
          swizzle_width_bytes / k_align_width_bytes;
      uint64_t num_elements_per_cacheline = k_cacheline_bytes / bytes;

      uint64_t num_swizzle_widths_per_cacheline =
          k_cacheline_bytes / swizzle_width_bytes;
      uint64_t num_elements_per_swizzle_width = swizzle_width_bytes / bytes;

      uint64_t smem_stride_box_bytes = num_box_along_col * k_cacheline_bytes;
      uint64_t smem_stride_box = smem_stride_box_bytes / bytes;

      switch (format) {
      case U16:
      case F16_RN:
      case BF16_RN: {

        typedef uint16_t element_t;
        element_t border_value = static_cast<element_t>(0);
        for (uint64_t d4 = 0; d4 < box_stride[4]; d4++) {
          for (uint64_t d3 = 0; d3 < box_stride[3]; d3++) {
            for (uint64_t d2 = 0; d2 < box_stride[2]; d2++) {
              for (uint64_t d1 = 0; d1 < box_stride[1]; d1++) {
                for (uint64_t d0 = 0; d0 < box_stride[0]; d0++) {

                  int64_t dd0 = coord0 + d0 * traversal_stride[0];
                  int64_t dd1 = coord1 + d1 * traversal_stride[1];
                  int64_t dd2 = coord2 + d2 * traversal_stride[2];
                  int64_t dd3 = coord3 + d3 * traversal_stride[3];
                  int64_t dd4 = coord4 + d4 * traversal_stride[4];

                  // printf("ddd0: %ld, ddd1: %ld\n", ddd0, ddd1);

                  int64_t gmem_index = dd0 + dd1 * (tensor_stride[0] / bytes) +
                                       dd2 * (tensor_stride[1] / bytes) +
                                       dd3 * (tensor_stride[2] / bytes) +
                                       dd4 * (tensor_stride[3] / bytes);

                  uint64_t d0x = d0 % num_elements_per_align_width;
                  uint64_t d0y = d0 / num_elements_per_align_width;
                  uint64_t d1x = d1 % num_swizzle_widths_per_cacheline;
                  uint64_t d1y = d1 / num_swizzle_widths_per_cacheline;

                  uint64_t smem_index =
                      d0x +
                      (d0y ^ (d1y % num_align_widths_per_swizzle_width)) *
                          num_elements_per_align_width +
                      d1x * num_elements_per_swizzle_width +
                      d1y * num_elements_per_cacheline + d2 * smem_stride_box +
                      d3 * smem_stride_box * box_stride[2] +
                      d4 * smem_stride_box * box_stride[2] * box_stride[3];

                  element_t value = border_value;

                  if (dd0 >= 0 && dd0 < tensor_size[0] && dd1 >= 0 &&
                      dd1 < tensor_size[1] && dd2 >= 0 &&
                      dd2 < tensor_size[2] && dd3 >= 0 &&
                      dd3 < tensor_size[3] && dd4 >= 0 &&
                      dd4 < tensor_size[4]) {
                    value = reinterpret_cast<element_t *>(p_smem_data)[smem_index];
                    reinterpret_cast<element_t *>(p_gmem_data)[gmem_index] = value;
                  }
                }
              }
            }
          }
        }
      }
      }

    } // end TILED mode

    if (DESC_TYPE == IM2COL) { // begin IM2COL mode

      /// get 5d coordinates from operands
      int64_t coord_c = coord0, coord_w = coord1, coord_h = coord2,
              coord_d = coord3, coord_n = coord4;

      /// get offsets for operand urc
      uint32_t offset_w = 0, offset_h = 0, offset_d = 0;
      get_im2col_offset<DIM>(urc, offset_d, offset_h, offset_w);

      /// get load_range_ndhw
      /// get load_range_c
      /// get box_base_corner
      /// get box_far_corner
      uint64_t load_range_c, load_range_ndhw;
      get_im2col_load_range_c(p_desc, load_range_c);
      get_im2col_load_range_ndhw(p_desc, load_range_ndhw);

      int64_t box_base_corner[3], box_far_corner[3];
      get_im2col_box_base_corner(p_desc, box_base_corner);
      get_im2col_box_far_corner(p_desc, box_far_corner);

      /// tensor_size[x] is user defined extents
      /// tensor_size_x is logical extent with filter access in mind (provided
      /// by user though _base_ and _far_ values)

      int64_t tensor_size_c = tensor_size[0];
      int64_t tensor_size_w = tensor_size[1];
      int64_t tensor_size_h = tensor_size[2];
      int64_t tensor_size_d = tensor_size[3];
      int64_t tensor_size_n = tensor_size[4];

      uint64_t load_range_c_bytes = load_range_c * bytes;
      uint64_t num_align_widths_in_load_range_c =
          __align_up(load_range_c_bytes, k_align_width_bytes);

      uint64_t aligned_load_range_c_swizzle_bytes =
          __align_up(load_range_c_bytes, swizzle_width_bytes);
      uint64_t aligned_load_range_c_bytes = aligned_load_range_c_swizzle_bytes;

      uint64_t num_box_in_cacheline =
          k_cacheline_bytes / aligned_load_range_c_bytes;
      uint64_t num_box_along_col =
          __div_up(load_range_ndhw, num_box_in_cacheline);

      int64_t tensor_start_coord_c = coord_c;
      int64_t tensor_start_coord_w = coord_w;
      int64_t tensor_start_coord_h = coord_h;
      int64_t tensor_start_coord_d = coord_d;
      int64_t tensor_start_coord_n = coord_n;

      int64_t c = tensor_start_coord_c;
      int64_t w = tensor_start_coord_w;
      int64_t h = tensor_start_coord_h;
      int64_t d = tensor_start_coord_d;
      int64_t n = tensor_start_coord_n;

      for (uint64_t range_ndhw = 0; range_ndhw < load_range_ndhw;
           range_ndhw++) {
        for (uint64_t range_c = 0; range_c < load_range_c;
             range_c += 16 / bytes) {

          int64_t c_ = c + range_c;
          /// transform logical coordinates into physical coordinates of tensor
          int64_t w_ = w + offset_w;
          int64_t h_ = h + offset_h;
          int64_t d_ = d + offset_d;
          int64_t n_ = n;

          /// units in bytes
          int64_t gmem_offset = c_ * bytes + w_ * tensor_stride[0] +
                                h_ * tensor_stride[1] + d_ * tensor_stride[2] +
                                n_ * tensor_stride[3];
          int64_t gmem_index = gmem_offset / bytes;

          uint64_t range_c_x = range_c % align_width;
          uint64_t range_c_y = range_c / align_width;

          uint64_t range_c_y_x = range_c_y % swizzle_width;
          uint64_t range_c_y_y = range_c_y / swizzle_width;

          uint64_t range_ndhw_x = range_ndhw % num_box_in_cacheline;
          uint64_t range_ndhw_y = range_ndhw / num_box_in_cacheline;

          /// units in
          uint64_t smem_index =
              range_c_x +
              (range_c_y_x ^ (range_ndhw_y % swizzle_width)) * align_width +
              range_c_y_y * swizzle_width * align_width +
              range_ndhw_x * load_range_c +
              range_ndhw_y * (k_cacheline_bytes / bytes);
          uint64_t smem_offset = smem_index * bytes;

          typedef uint16_t element_t;

          if (c_ < tensor_size_c && c_ >= 0 && w_ < tensor_size_w && w_ >= 0 &&
              h_ < tensor_size_h && h_ >= 0 && d_ < tensor_size_d && d_ >= 0 &&
              n_ < tensor_size_n && n_ >= 0) {
          element_t value = reinterpret_cast<element_t *>(p_smem_data)[smem_index];
          reinterpret_cast<element_t *>(p_gmem_data)[gmem_index] = value;
          }
        }

        /// Ilwariant for C
        w = w + traversal_stride[1];
        /// if w_ exceeds the bounds,
        if (w >= tensor_size_w + box_far_corner[0]) {
          /// round it back
          w = box_base_corner[0];
          /// increment next dimension
          h = h + traversal_stride[2];
        }
        if (h >= tensor_size_h + box_far_corner[1]) {
          /// round it back
          h = box_base_corner[1];
          /// increment next dimension
          d = d + traversal_stride[3];
        }
        if (d >= tensor_size_d + box_far_corner[2]) {
          // round it back
          d = box_base_corner[2];
          /// increment next dimension
          n = n + traversal_stride[4];
        }
      }
    } // end IM2COL mode

  } // end threadIdx.x % 32  == 0
}

}
}
}