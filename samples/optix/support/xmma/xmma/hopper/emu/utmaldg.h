#pragma once

#include <xmma/hopper/emu/utma_utils.h>
#include <xmma/hopper/emu/lwda_tma_utils_internal.h>
#include <xmma/hopper/emu/xmma_tma_print_desc.h>

template <uint8_t DIM, lwdaTmaDescType DESC_TYPE = TILED>
__device__ void __utmapf2( const lwdaTmaDescv2 *p_desc_,
                           int32_t urb0,
                           int32_t urb1,
                           int32_t urb2,
                           int32_t urb3,
                           int32_t urb4 ) {
    // Prefetch data from global memory to L2
}

template <lwdaCacheCtrl k_ctl> __device__ void __utmacctl( const lwdaTmaDescv2 *p_desc_ ) {
    const tma_descriptor_t *p_desc = reinterpret_cast<const tma_descriptor_t *>( p_desc_ );
    assert( p_desc != NULL );
}

template <uint8_t DIM, lwdaTmaDescType DESC_TYPE = TILED, bool MCAST>
__device__ void __utmaldg(const lwdaTmaDesc *p_desc_, uint32_t urb0,
                          uint32_t urb1, int32_t urb2, int32_t urb3,
                          int32_t urb4, int32_t urb5, int32_t urb6) {
  const tma_descriptor_t *p_desc =
      reinterpret_cast<const tma_descriptor_t *>(p_desc_);

  using namespace xmma::hopper::emu;

  if (threadIdx.x % 32 == 0 && threadIdx.x == 0 && blockIdx.x == 0 &&
      blockIdx.y == 0) {
    // printf("%s\n", KGRN);
    // print_tma_descriptor_raw(p_desc);
    // print_tma_descriptor_interp(p_desc);
    // validate_tma_descriptor<DIM, DESC_TYPE>(p_desc);
    // printf("%s\n", KNRM);
  }

  int64_t tensor_start_coord[5] = {0, 0, 0, 0, 0};
  if (DIM >= 1)
    tensor_start_coord[0] = static_cast<int64_t>(urb2);
  if (DIM >= 2)
    tensor_start_coord[1] = static_cast<int64_t>(urb3);
  if (DIM >= 3)
    tensor_start_coord[2] = static_cast<int64_t>(urb4);
  if (DIM >= 4)
    tensor_start_coord[3] = static_cast<int64_t>(urb5);
  if (DIM == 5)
    tensor_start_coord[4] = static_cast<int64_t>(urb6);

  // Get all members from tma descriptor
  void *p_gmem_tensor;
  get_tensor_global_address(p_desc, p_gmem_tensor);
  uint64_t gmem_data_addr = reinterpret_cast<uint64_t>(p_gmem_tensor);

  uint64_t tensor_size[5];
  get_tensor_size(p_desc, tensor_size);

  uint64_t tensor_stride[4];
  get_tensor_stride(p_desc, tensor_stride);
  /*
  if(threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
  for(uint32_t i = 0; i < 4; i++) {
    printf("tensor_stride[%u]: %lu\n", i, tensor_stride[i]);
  }
  }
  */

  uint64_t traversal_stride[5];
  get_traversal_stride(p_desc, traversal_stride);
  /*
  if(threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
  for(uint32_t i = 0; i < 5; i++) {
    printf("traversal_stride[%u]: %lu\n", i, traversal_stride[i]);
  }
  }
  */

  uint64_t box_size[5];
  get_tiled_box_size(p_desc, box_size);

  // Set default values.
  for (int i = DIM; i < 5; i++) {
    box_size[i] = 1;
    tensor_size[i] = 1;
    traversal_stride[i] = 1;
    // The strides of higher dim are the same as the the last valid dimension.
    tensor_stride[i] = tensor_stride[DIM];
  }

  format_t format;
  swizzle_t swizzle;
  interleaved_t interleaved;

  get_format(p_desc, format);
  get_swizzle(p_desc, swizzle);
  get_interleaved(p_desc, interleaved);

  uint32_t smem_data_addr = get_shared_data_address(urb0);
  uint32_t smem_barr_addr = get_shared_barrier_address(urb1);
  // printf("smem_data_addr: %u\n", smem_data_addr);

#if __LWDA_ARCH__ >= 800

  // For warp-spec, pick first thread in warp (tx % 32 == 0)
  if (threadIdx.x % 32 == 0 && threadIdx.x == 0) {

    switch (format) {
    case U16:
    case F16_RN:
    case BF16_RN: {

      typedef uint16_t element_t;
      // Disabled interleaved for ldgsts (bit ilwolved)
      // uint64_t interleaved_width_bytes = interleaved == INTERLEAVED_16B ? 16
      // : sizeof(element_t); interleaved_width_bytes = interleaved ==
      // INTERLEAVED_32B ? 32 : interleaved_width_bytes;

      uint64_t swizzle_width_bytes = get_swizzle_width_bytes(swizzle);
      // uint64_t num_swizzle_widths_in_cacheline = k_cacheline_bytes /
      // swizzle_width_bytes;
      uint64_t num_align_widths_in_swizzle_width =
          swizzle_width_bytes / k_align_width_bytes;
      // constexpr uint64_t k_max_elements_per_cacheline = k_cacheline_bytes /
      // sizeof(element_t); uint64_t num_elements_per_swizzle_width =
      // swizzle_width_bytes / sizeof(element_t); uint64_t num_cp_per_swizzle =
      // swizzle_width_bytes / k_align_width_bytes;

      // If swizzle is disabled, do a regular copy
      // Else, restrictions apply
      if( swizzle != SWIZZLE_DISABLED ) {
          uint64_t num_align_16bytes_rect_size_x =
              divide_up( box_size[0] * sizeof( element_t ), 16 );
          // uint64_t num_align_16bytes_rect_size_x_bytes =
          // num_align_16bytes_rect_size_x * 16; uint64_t
          // num_align_16bytes_rect_size_x_elem = num_align_16bytes_rect_size_x *
          // (16 / sizeof(element_t));
          uint64_t num_align_swizzle_rect_size_x =
              divide_up( box_size[0] * sizeof( element_t ), swizzle_width_bytes );
          uint64_t num_box_size_x_in_cacheline =
              k_cacheline_bytes / ( box_size[0] * sizeof( element_t ) );

          int coord_v = 0;
          for( uint64_t d4 = 0; d4 < box_size[4]; d4 += traversal_stride[4] ) {
              for( uint64_t d3 = 0; d3 < box_size[3]; d3 += traversal_stride[3] ) {
                  for( uint64_t d2 = 0; d2 < box_size[2]; d2 += traversal_stride[2] ) {
                      for( uint64_t d1 = 0; d1 < box_size[1]; d1 += traversal_stride[1] ) {
                          for( uint64_t d0 = 0; d0 < num_align_16bytes_rect_size_x; d0++ ) {

                              // colwert to element coordinate system
                              int64_t dd0 =
                                  tensor_start_coord[0] + d0 * ( 16 / sizeof( element_t ) );
                              int64_t dd1 = tensor_start_coord[1] + d1;
                              int64_t dd2 = tensor_start_coord[2] + d2;
                              int64_t dd3 = tensor_start_coord[3] + d3;
                              int64_t dd4 = tensor_start_coord[4] + d4;
                              uint64_t gmem_offset = static_cast<uint64_t>(
                                  dd4 * tensor_stride[3] + dd3 * tensor_stride[2] +
                                  dd2 * tensor_stride[1] + dd1 * tensor_stride[0] +
                                  dd0 * sizeof( element_t ) );
                              // uint64_t smem_offset = static_cast<uint64_t>((d0 + (d1 *
                              // num_align_16bytes_rect_size_x)) * 16);

                              // Callwlates index of align_width (16B) in a swizzle width
                              // (swizzle_t)
                              uint64_t d0x = d0 % num_align_widths_in_swizzle_width;
                              uint64_t d0y = d0 / num_align_widths_in_swizzle_width;

                              // Callwlates index of d1 along the row
                              uint64_t d1x = coord_v % num_box_size_x_in_cacheline;
                              uint64_t d1y = coord_v / num_box_size_x_in_cacheline;

                              uint64_t smem_offset =
                                  ( d0x ^ ( d1y % num_align_widths_in_swizzle_width ) ) * 16 +
                                  d0y * num_align_widths_in_swizzle_width * 16 +
                                  d1x * box_size[0] * sizeof( element_t ) + d1y * k_cacheline_bytes;

                              if( ( dd0 >= 0 && dd0 < tensor_size[0] ) &&
                                  ( dd1 >= 0 && dd1 < tensor_size[1] ) &&
                                  ( dd2 >= 0 && dd2 < tensor_size[2] ) &&
                                  ( dd3 >= 0 && dd3 < tensor_size[3] ) &&
                                  ( dd4 >= 0 && dd4 < tensor_size[4] ) ) {
                                  printf( "ldgsts(%x, %llx)\n",
                                          smem_data_addr + static_cast<uint32_t>( smem_offset ),
                                          gmem_data_addr + gmem_offset );
                                  __cp_async_shared_global(
                                      smem_data_addr + static_cast<uint32_t>( smem_offset ),
                                      gmem_data_addr + gmem_offset );
                              }
                          }
                          coord_v++;
                      }
                  }
              }
          }
      }

      break;
    }
    }
  }

#else

  if (threadIdx.x == 0) {

    switch (format) {
    case U16:
    case F16_RN:
    case BF16_RN: {

      typedef uint16_t element_t;

      element_t border_value = static_cast<element_t>(0);

      uint64_t interleaved_width_bytes =
          get_interleaved_width_bytes<sizeof(element_t)>(interleaved);
      uint64_t interleaved_vector_width =
          interleaved_width_bytes / sizeof(element_t);

      uint64_t swizzle_width_bytes = get_swizzle_width_bytes(swizzle);

      constexpr uint64_t k_num_elements_per_cacheline =
          k_cacheline_bytes / sizeof(element_t);
      uint64_t num_elements_per_swizzle_width =
          swizzle_width_bytes / sizeof(element_t);
      constexpr uint64_t k_num_elements_per_align_width =
          k_align_width_bytes / sizeof(element_t);

      uint64_t num_swizzle_widths_per_cacheline =
          k_cacheline_bytes / swizzle_width_bytes;
      uint64_t num_align_widths_per_swizzle_width =
          swizzle_width_bytes / k_align_width_bytes;

      element_t *gmem_data_addr_elem =
          reinterpret_cast<element_t *>(gmem_data_addr);
      element_t *smem_data_addr_elem = NULL;
#if LWDART_VERSION >= 11000
      smem_data_addr_elem = reinterpret_cast<element_t *>(
          __cvta_shared_to_generic(smem_data_addr));
#endif

      // If swizzle is disabled, do a regular copy
      // Else, restrictions apply
      if( swizzle != SWIZZLE_DISABLED ) {
          uint64_t box_stride[5] = { 1, 1, 1, 1, 1 };
          // Make default values as 1. When multiplying, it gets easier
          box_stride[0] = box_size[0];
          // rect_size[0] = box_stride[0];
          // printf("box_stride[%u]: %lu rect_size[0]: %lu\n", 0, box_stride[0],
          // rect_size[0]);

          for( uint32_t i = 1; i < DIM; i++ ) {
              box_stride[i] = box_size[i] / traversal_stride[i];
              // rect_size[1] *= box_stride[i];
              // printf("box_stride[%u]: %lu, rect_size[1]: %lu\n", i,
              // box_stride[i], rect_size[1]);
          }

          for( uint64_t d1 = 0; d1 < box_stride[1]; d1++ ) {
              for( uint64_t d0 = 0; d0 < box_stride[0]; d0++ ) {
                  int64_t dd0 = tensor_start_coord[0] + d0 * traversal_stride[0];
                  int64_t dd1 = tensor_start_coord[1] + d1 * traversal_stride[1];

                  int64_t ddd0 = dd0 * interleaved_vector_width;
                  int64_t ddd1 = ( dd1 % interleaved_vector_width ) +
                                 ( dd1 / interleaved_vector_width ) * interleaved_vector_width;

                  // printf("ddd0: %ld, ddd1: %ld\n", ddd0, ddd1);

                  int64_t gmem_offset = ddd0 + ddd1 * ( tensor_stride[0] / sizeof( element_t ) );

                  element_t value = border_value;

                  if( dd0 >= 0 && dd0 < tensor_size[0] && dd1 >= 0 && dd1 < tensor_size[1] ) {
                      value = gmem_data_addr_elem[gmem_offset];
                  }

                  uint64_t ddd0x = d0 % k_num_elements_per_align_width;
                  uint64_t ddd0y = d0 / k_num_elements_per_align_width;
                  uint64_t ddd1x = d1 % num_swizzle_widths_per_cacheline;
                  uint64_t ddd1y = d1 / num_swizzle_widths_per_cacheline;

                  uint64_t smem_offset =
                      ddd0x +
                      ( ddd0y ^ ( ddd1y % num_align_widths_per_swizzle_width ) ) *
                          k_num_elements_per_align_width +
                      ddd1x * num_elements_per_swizzle_width + ddd1y * k_num_elements_per_cacheline;

                  smem_data_addr_elem[smem_offset] = value;

                  printf( "gmem_offset: %lu, smem_offset: %lu, (d0, d1): (%lu, %lu), "
                          "(ddd0x, ddd0y, ddd1x, ddd1y): (%lu, %lu, %lu, %lu) value: "
                          "%lu\n",
                          gmem_offset,
                          smem_offset,
                          d0,
                          d1,
                          ddd0x,
                          ddd0y,
                          ddd1x,
                          ddd1y,
                          static_cast<uint64_t>( value ) );
              }
          }
      }
    }
    }
  }
#endif
}

template <uint8_t DIM>
XMMA_HOST_DEVICE uint32_t set_im2col_offset(uint32_t t, uint32_t r,
                                            uint32_t s) {
  uint32_t offset = 0;
  if (DIM == 5) {
    offset = (s & 0x1F);
    offset = offset | ((r & 0x1F) << 5);
    offset = offset | ((t & 0x1F) << 10);
    return offset;
  }
}

template <uint8_t DIM>
XMMA_HOST_DEVICE void get_im2col_offset(uint32_t offset, uint32_t &t,
                                        uint32_t &r, uint32_t &s) {
  if (DIM == 5) {
    s = offset & 0x1F;
    r = (offset >> 5) & 0x1F;
    t = (offset >> 10) & 0x1F;
  }
}

static inline __device__ uint64_t __div_up(uint64_t x, uint64_t y) {
  return (x + y - 1) / y;
}

static inline __device__ uint64_t __align_up(uint64_t x, uint64_t y) {
  return ((x + y - 1) / y) * y;
}

/**
 * If smem barriers are required, emulate it in the code
 */
template <uint8_t DIM, lwdaTmaDescType DESC_TYPE, bool MCAST>
__device__ void UTMALDG(const lwdaTmaDesc *p_desc_, uint32_t urb0,
                           uint32_t urb1, int32_t urb2, int32_t urb3,
                           int32_t urb4, int32_t urb5, int32_t urb6,
                           uint32_t urc) {
  static_assert(DIM > 0 && DIM < 6, "dimensions should be (0, 5]\n");

  using namespace xmma::hopper::emu;

  if (threadIdx.x % 32 == 0) { /// Select first thread in warp do copy

    /// TODO: Validate dim vs DIM

    const tma_descriptor_t *p_desc =
        reinterpret_cast<const tma_descriptor_t *>(p_desc_);
    // print_tma_descriptor_interp(p_desc);

    /// Get shared memory data pointer from URF
    uint32_t smem_data_addr = get_shared_data_address(urb0);
    /// printf("smem_data_addr: %u\n", smem_data_addr);

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
    // assert(traversal_stride[0] == 1 && swizzle != SWIZZLE_DISABLED);

    /// get tensor stride
    uint64_t tensor_stride[4];
    get_tensor_stride(p_desc, tensor_stride);

    /// Set default values.
    for (int i = DIM; i < 5; i++) {
      tensor_stride[i] = tensor_stride[DIM - 1];
    }

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
#if defined(__LWDA_ARCH__) && (__LWDA_ARCH__ >= 800)

      void *p_smem_data;
      asm( "{cvta.shared.u64 %0, %1;}\n"
           : "=l"( p_smem_data )
           : "l"( static_cast<uint64_t>( smem_data_addr ) ) );

      uint64_t coord_vertical = 0;
      for (uint64_t d4 = 0; d4 < box_stride[4]; d4++) {
        for (uint64_t d3 = 0; d3 < box_stride[3]; d3++) {
          for (uint64_t d2 = 0; d2 < box_stride[2]; d2++) {
            for (uint64_t d1 = 0; d1 < box_stride[1]; d1++) {
              for (uint64_t d0 = 0; d0 < (box_stride[0] * bytes) / 16; d0++) {

                int64_t dd0 = coord0 + d0 * traversal_stride[0] *
                                           num_elements_per_align_width;
                int64_t dd1 = coord1 + d1 * traversal_stride[1];
                int64_t dd2 = coord2 + d2 * traversal_stride[2];
                int64_t dd3 = coord3 + d3 * traversal_stride[3];
                int64_t dd4 = coord4 + d4 * traversal_stride[4];

                int64_t gmem_offset = dd0 * bytes + dd1 * tensor_stride[0] +
                                      dd2 * tensor_stride[1] +
                                      dd3 * tensor_stride[2] +
                                      dd4 * tensor_stride[3];

                uint64_t d0x = d0 % 1;
                uint64_t d0y = d0 / 1;
                uint64_t d1x =
                    coord_vertical % num_swizzle_widths_per_cacheline;
                uint64_t d1y =
                    coord_vertical / num_swizzle_widths_per_cacheline;

                uint64_t smem_offset =
                    d0x * k_align_width_bytes +
                    (d0y ^ (d1y % num_align_widths_per_swizzle_width)) *
                        k_align_width_bytes +
                    d1x * num_elements_per_swizzle_width * bytes +
                    d1y * num_elements_per_cacheline * bytes;

                if (dd0 >= 0 && dd0 < tensor_size[0] && dd1 >= 0 &&
                    dd1 < tensor_size[1] && dd2 >= 0 && dd2 < tensor_size[2] &&
                    dd3 >= 0 && dd3 < tensor_size[3] && dd4 >= 0 &&
                    dd4 < tensor_size[4]) {
                  __cp_async_shared_global(
                      smem_data_addr + static_cast<uint32_t>(smem_offset),
                      gmem_data_addr + gmem_offset);
                } else {
                  __cp_async_shared_global(
                      smem_data_addr + static_cast<uint32_t>(smem_offset),
                      gmem_data_addr + gmem_offset, false);
                }
              }
              coord_vertical++;
            }
          }
        }
      }
#else
      void *p_smem_data;
      asm("{cvta.shared.u64 %0, %1;}\n"
          : "=l"(p_smem_data)
          : "l"(static_cast<uint64_t>(smem_data_addr)));
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
                    value =
                        reinterpret_cast<element_t *>(p_gmem_data)[gmem_index];
                    reinterpret_cast<element_t *>(p_smem_data)[smem_index] =
                        value;
                  }
                }
              }
            }
          }
        }
      }
      }
#endif

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

          bool pred = false;

          if (c_ < tensor_size_c && c_ >= 0 && w_ < tensor_size_w && w_ >= 0 &&
              h_ < tensor_size_h && h_ >= 0 && d_ < tensor_size_d && d_ >= 0 &&
              n_ < tensor_size_n && n_ >= 0) {
            pred = true;
          }
          __cp_async_shared_global(smem_data_addr +
                                       static_cast<uint32_t>(smem_offset),
                                   gmem_data_addr + gmem_offset, pred);
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

///
///
/// END
///
///
