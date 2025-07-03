/***************************************************************************************************
 * Copyright (c) 2011-2019, LWPU CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 *modification, are not permit- ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *DISCLAIMED. IN NO EVENT SHALL LWPU CORPORATION BE LIABLE FOR ANY DIRECT,
 *INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 *OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 *NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 *EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <cstdint>

#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define KBLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"

typedef enum { TILED, IM2COL } desc_type_t;

typedef enum {
  INTERLEAVED_NONE,
  INTERLEAVED_16B,
  INTERLEAVED_32B,
  INTERLEAVED_MAX
} interleaved_t;

typedef enum { SWIZZLE_DISABLED, SWIZZLE_32B, SWIZZLE_64B, SWIZZLE_128B, SWIZZLE_MAX } swizzle_t;

typedef enum { BARRIER64, BARRIER128 } barrier_t;

typedef enum { L2_NONE, L2_64B, L2_128B, L2_256B, L2_MAX } promotion_t;

typedef enum {
    U8 = 0,
    U16,
    U32,
    S32,
    U64,
    S64,
    F16_RN,
    F32_RN,
    F32_FTZ_RN,
    F64_RN,
    BF16_RN,
    FORMAT_MAX
} format_t;

typedef enum { TENSOR_ZFILL, TENSOR_CFILL } oob_fill_mode_t;


//
typedef struct {
  desc_type_t type;
  uint8_t dim;
  // 54b, value contains address without 4 lsbs
  uint64_t tensor_global_address;
  format_t format;
  interleaved_t interleaved;
  // Range [1, 2^32]
  uint32_t tensor_size[5];
  // 36b, values containts stride without 4 lsbs
  uint64_t tensor_stride[4];
} tma_tensor_params_t;

typedef struct {
  // 3b, range [1:8]
  uint8_t traversal_stride[5];
  swizzle_t swizzle;
  // false = zero, true = constant
  bool oob_fill_mode;
  // false = disable, true = enable
  bool f32_to_tf32;
  promotion_t promotion;
} tma_access_params_t;

typedef struct {
  // 8b, [1:256]
  uint8_t box_size[5];
} tma_tile_descriptor_t;

// TODO: Change precision of base and far corner to the one in FD
typedef struct {
  // 10b, range [1:1024]
  uint16_t load_range_ndhw;
  // 8b, range [1:256]
  uint8_t load_range_c;
  // 4b, [0:15]
  int16_t box_base_corner[3];
  // 4b, [0:15]
  int16_t box_far_corner[3];
} tma_im2col_descriptor_t;

typedef struct {
  tma_tensor_params_t tensor;
  tma_access_params_t access;
  tma_tile_descriptor_t tiled;
  tma_im2col_descriptor_t im2col;
} tma_descriptor_t;

typedef enum {
  PREFETCH,      // Prefetch tma descriptor using global memory address
  ILWALIDATE,    // Ilwalidate tma descriptor in l2 cache
  ILWALIDATE_ALL // Ilwalidate tma descriptor and all elements in l2 cache line
} lwdaCacheCtrl;

constexpr uint64_t k_max_tensor_size = (1llu << 36);
constexpr uint64_t k_max_tensor_stride = (1llu << 36);
constexpr uint64_t k_max_block_size = 256llu;
constexpr uint64_t k_max_traversal_stride = (1llu << 3);

constexpr uint64_t k_min_tensor_size = 1llu;
constexpr uint64_t k_min_tensor_stride = 0llu;
constexpr uint64_t k_min_block_size = 1llu;
constexpr uint64_t k_min_traversal_stride = 1llu;

constexpr uint32_t k_max_cta_id = (1 << 6) - 1;

//
// LWCA specific structures
//
typedef desc_type_t lwdaTmaDescType;
typedef swizzle_t lwdaTmaSwizzle;
typedef format_t lwdaTmaFormat;
typedef interleaved_t lwdaTmaInterleaved;
typedef barrier_t lwdaTmaBarrier;
typedef promotion_t lwdaTmaPromotion;

// 128 byte tma descriptor
typedef struct alignas(8) {
  uint8_t data[128];
} lwdaTmaDesc;

typedef format_t lwdaTmaDescFormatv2;
typedef swizzle_t lwdaTmaDescSwizzlev2;

typedef struct {
    uint64_t tensor_common0;
    uint32_t tensor_common1;

    uint32_t tensor_stride_lower[4];  //< 36b of 64b with 4B aligned
    uint32_t tensor_stride_upper;
    uint32_t tensor_size[5];          //< value -1
    uint32_t traversal_stride_box_0;  //< packed 3b (-1)

    uint32_t box_size_end;
} lwdaTmaDescTiled;

typedef struct {
    uint64_t tensor_common0;
    uint32_t tensor_common1;

    uint32_t tensor_stride_lower[4];
    uint32_t tensor_stride_upper;
    uint32_t tensor_size[5];
    uint32_t traversal_stride_range_c;

    uint32_t box_corner_dhw;
    uint32_t range_ndhw;
} lwdaTmaDescIm2Col;

typedef struct {
    uint64_t data[8];
} lwdaTmaDescv2;

typedef enum { TENSOR_TILED = 0, TENSOR_IM2COL } lwdaTmaDescTypev2;

typedef enum { INTERLEAVE_DISABLED = 0, INTERLEAVE_16B, INTERLEAVE_32B } lwdaTmaDescInterleavev2;

typedef enum {
    PROMOTION_DISABLED = 0,
    PROMOTION_64B,
    PROMOTION_128B,
    PROMOTION_256B
} lwdaTmaDescPromotiolw2;

/*
#define PRINT_VAR( var ) std::cerr << __func__ << " " << #var << " " << var << std::endl;
*/

static inline void set_tensor_common_0( lwdaTmaDescv2 *p_desc, uint64_t addr ) {
    lwdaTmaDescTiled *desc = reinterpret_cast<lwdaTmaDescTiled *>( p_desc );
    desc->tensor_common0 = 0;
    desc->tensor_common0 |= ( addr );
/*
#ifdef PRINT_TMA_DESC
    std::bitset<64> b_( desc->tensor_common0 );
    std::bitset<64> c_( addr );
    PRINT_VAR( b_ );
    PRINT_VAR( c_ );
    PRINT_VAR(addr);
#endif
*/
}

static inline void set_tensor_common_1( lwdaTmaDescv2 *p_desc,
                                        lwdaTmaDescTypev2 desc_type,
                                        uint32_t dims,
                                        lwdaTmaDescFormatv2 format,
                                        lwdaTmaDescInterleavev2 interleave,
                                        lwdaTmaDescSwizzlev2 swizzle,
                                        uint32_t fill,
                                        uint32_t f32_to_tf32,
                                        lwdaTmaDescPromotiolw2 promotion ) {
    lwdaTmaDescTiled *desc = reinterpret_cast<lwdaTmaDescTiled *>( p_desc );

    desc->tensor_common1 = 0;
    desc->tensor_common1 |= desc_type == TENSOR_TILED ? 0x0 : 0x1;

    constexpr uint32_t VERSION_SHIFT = 1;
    constexpr uint32_t VERSION_BITS = 3;
    desc->tensor_common1 |= ( 1u << VERSION_SHIFT );

    constexpr uint32_t DIM_BITS = 3;
    constexpr uint32_t DIM_SHIFT = VERSION_SHIFT + VERSION_BITS;
    constexpr uint32_t DIM_MASK = ( 1u << DIM_BITS ) - 1;
    desc->tensor_common1 |= ( ( dims - 1 ) & DIM_MASK ) << DIM_SHIFT;

    constexpr uint32_t FORMAT_BITS = 4;
    constexpr uint32_t FORMAT_SHIFT = DIM_SHIFT + DIM_BITS;
    constexpr uint32_t FORMAT_MASK = ( 1u << FORMAT_BITS ) - 1;
    desc->tensor_common1 |= ( static_cast<uint32_t>( format ) & FORMAT_MASK ) << FORMAT_SHIFT;

    constexpr uint32_t INTERLEAVE_BITS = 2;
    constexpr uint32_t INTERLEAVE_SHIFT = FORMAT_SHIFT + FORMAT_BITS;
    constexpr uint32_t INTERLEAVE_MASK = ( 1u << INTERLEAVE_BITS ) - 1;
    desc->tensor_common1 |= ( static_cast<uint32_t>( interleave ) & INTERLEAVE_MASK )
                            << INTERLEAVE_SHIFT;

    constexpr uint32_t SWIZZLE_BITS = 2;
    constexpr uint32_t SWIZZLE_SHIFT = INTERLEAVE_SHIFT + INTERLEAVE_BITS;
    constexpr uint32_t SWIZZLE_MASK = ( 1u << SWIZZLE_BITS ) - 1;
    desc->tensor_common1 |= ( static_cast<uint32_t>( swizzle ) & SWIZZLE_MASK ) << SWIZZLE_SHIFT;

    constexpr uint32_t FILL_BITS = 1;
    constexpr uint32_t FILL_SHIFT = SWIZZLE_SHIFT + SWIZZLE_BITS;
    constexpr uint32_t FILL_MASK = ( 1u << FILL_BITS ) - 1;
    desc->tensor_common1 |= ( static_cast<uint32_t>( fill ) & FILL_MASK ) << FILL_SHIFT;

    constexpr uint32_t F32_TO_TF32_BITS = 1;
    constexpr uint32_t F32_TO_TF32_SHIFT = FILL_SHIFT + FILL_BITS;
    constexpr uint32_t F32_TO_TF32_MASK = ( 1u << F32_TO_TF32_BITS ) - 1;
    desc->tensor_common1 |= ( static_cast<uint32_t>( f32_to_tf32 ) & F32_TO_TF32_MASK )
                            << F32_TO_TF32_SHIFT;

    constexpr uint32_t PROMOTION_BITS = 2;
    constexpr uint32_t PROMOTION_SHIFT = F32_TO_TF32_SHIFT + F32_TO_TF32_BITS;
    constexpr uint32_t PROMOTION_MASK = ( 1u << PROMOTION_BITS ) - 1;
    desc->tensor_common1 |= ( static_cast<uint32_t>( promotion ) & PROMOTION_MASK )
                            << PROMOTION_SHIFT;
/*
#ifdef PRINT_TMA_DESC
    b_ = desc->tensor_common1;
    PRINT_VAR( b_ );
#endif
*/
}

static inline void
set_tensor_stride( lwdaTmaDescv2 *p_desc, uint64_t *p_tensor_stride, uint32_t dims ) {
    lwdaTmaDescTiled *desc = reinterpret_cast<lwdaTmaDescTiled *>( p_desc );

    constexpr uint32_t TENSOR_STRIDE_UPPER_BITS = 4;
    constexpr uint32_t TENSOR_STRIDE_UPPER_MASK = ( 1u << TENSOR_STRIDE_UPPER_BITS ) - 1;

    for( uint32_t i = 0; i < dims - 1; i++ ) {
        desc->tensor_stride_lower[i] = 0u;
        uint64_t tensor_stride_lower_64b = (p_tensor_stride[i] >> 4) & 0xFFFFFFFFlu;
        desc->tensor_stride_lower[i] = static_cast<uint32_t>(tensor_stride_lower_64b);
/*
#ifdef PRINT_TMA_DESC
        std::bitset<32> b_( desc->tensor_stride_lower[i] );
        PRINT_VAR( b_ );
#endif
*/
    }
    desc->tensor_stride_upper = 0u;

    for( uint32_t i = 0; i < dims - 1; i++ ) {
        uint64_t tensor_stride = p_tensor_stride[i];
        tensor_stride = tensor_stride >> 4;
        uint64_t tensor_stride_upper = tensor_stride >> 32;
        uint32_t tensor_stride_upper_32b = static_cast<uint32_t>( tensor_stride_upper );
        desc->tensor_stride_upper |= ( ( tensor_stride_upper_32b & TENSOR_STRIDE_UPPER_MASK )
                                       << ( i * TENSOR_STRIDE_UPPER_BITS ) );
    }
/*
#ifdef PRINT_TMA_DESC
    std::bitset<64> b_( desc->tensor_stride_upper );
    PRINT_VAR( b_ );
#endif
*/
}

static inline void
set_tensor_size( lwdaTmaDescv2 *p_desc, uint32_t *p_tensor_size, uint32_t dims ) {
    lwdaTmaDescTiled *desc = reinterpret_cast<lwdaTmaDescTiled *>( p_desc );
    for( uint32_t dim = 0; dim < dims; dim++ ) {
        desc->tensor_size[dim] = p_tensor_size[dim] - 1;
/*
#ifdef PRINT_TMA_DESC
        PRINT_VAR( desc->tensor_size[dim] );
#endif
*/
    }
}

static inline void
set_traversal_stride_tiled( lwdaTmaDescv2 *p_desc, uint32_t *p_traversal_stride, uint32_t dims ) {
    lwdaTmaDescTiled *desc = reinterpret_cast<lwdaTmaDescTiled *>( p_desc );

    desc->traversal_stride_box_0 = 0;

    constexpr uint32_t TRAVERSAL_STRIDE_BITS = 3;
    constexpr uint32_t TRAVERSAL_STRIDE_MASK = ( 1u << TRAVERSAL_STRIDE_BITS ) - 1;

    for( uint32_t dim = 0; dim < dims; dim++ ) {
        uint32_t traversal_stride = p_traversal_stride[dim] - 1;
        traversal_stride = ( traversal_stride & TRAVERSAL_STRIDE_MASK )
                           << ( dim * TRAVERSAL_STRIDE_BITS );
        desc->traversal_stride_box_0 |= traversal_stride;
    }
/*
#ifdef PRINT_TMA_DESC
    std::bitset<32> b_( desc->traversal_stride_box_0 );
    PRINT_VAR( b_ );
#endif
*/
}

static inline void set_box_size( lwdaTmaDescv2 *p_desc, uint32_t *p_box_size, uint32_t dims ) {
    lwdaTmaDescTiled *desc = reinterpret_cast<lwdaTmaDescTiled *>( p_desc );

    desc->box_size_end = 0;

    constexpr uint32_t BOX_SIZE_BITS = 8;
    constexpr uint32_t BOX_SIZE_MASK = ( 1 << BOX_SIZE_BITS ) - 1;

    if( dims > 1 ) {
        uint32_t box_size_0 = p_box_size[0] - 1;
        box_size_0 = box_size_0 & BOX_SIZE_MASK;
        box_size_0 = box_size_0 << 24;
        desc->traversal_stride_box_0 |= box_size_0;
    }

    for( uint32_t dim = 1; dim < dims; dim++ ) {
        uint32_t box_size = p_box_size[dim] - 1;
        box_size = box_size & BOX_SIZE_MASK;
        box_size = box_size << ( ( dim - 1 ) * BOX_SIZE_BITS );
        desc->box_size_end |= box_size;
    }
/*
#ifdef PRINT_TMA_DESC
    std::bitset<32> b_( desc->traversal_stride_box_0 );
    PRINT_VAR( b_ );
    std::bitset<32> c_( desc->box_size_end );
    PRINT_VAR( c_ );
#endif
*/
}

static inline void
set_traversal_stride_im2col( lwdaTmaDescv2 *p_desc, uint32_t *p_traversal_stride, uint32_t dims ) {

    lwdaTmaDescIm2Col *desc = reinterpret_cast<lwdaTmaDescIm2Col *>( p_desc );

    desc->traversal_stride_range_c = 0;

    constexpr uint32_t TRAVERSAL_STRIDE_BITS = 3;
    constexpr uint32_t TRAVERSAL_STRIDE_MASK = ( 1u << ( TRAVERSAL_STRIDE_BITS + 1 ) ) - 1;

    for( uint32_t dim = 0; dim < dims; dim++ ) {
        uint32_t traversal_stride = p_traversal_stride[dim] - 1;
        traversal_stride = ( traversal_stride & TRAVERSAL_STRIDE_MASK )
                           << ( dim * TRAVERSAL_STRIDE_BITS );
        desc->traversal_stride_range_c |= traversal_stride;
    }
/*
#ifdef PRINT_TMA_DESC
    std::bitset<32> b_( desc->traversal_stride_range_c );
    PRINT_VAR( b_ );
#endif
*/
}

static inline void set_range_c( lwdaTmaDescv2 *p_desc, uint32_t range_c ) {
    lwdaTmaDescIm2Col *desc = reinterpret_cast<lwdaTmaDescIm2Col *>( p_desc );

    constexpr uint32_t RANGE_C_BITS = 8;
    constexpr uint32_t RANGE_C_MASK = ( 1u << RANGE_C_BITS ) - 1;

    range_c = range_c & RANGE_C_MASK;
    desc->traversal_stride_range_c |= ( range_c << 24 );
/*
#ifdef PRINT_TMA_DESC
    std::bitset<32> b_( desc->traversal_stride_range_c );
    PRINT_VAR( b_ );
#endif
*/
}

static inline void set_box_corner_dhw( lwdaTmaDescv2 *p_desc,
                                       uint32_t *p_base_corner,
                                       uint32_t *p_far_corner,
                                       uint32_t dims ) {
    lwdaTmaDescIm2Col *desc = reinterpret_cast<lwdaTmaDescIm2Col *>( p_desc );

    desc->box_corner_dhw = 0;

    uint32_t box_base_corner = 0, box_far_corner = 0;
    uint32_t box_corner_dhw = 0;

    if( dims == 3 ) {
        constexpr uint32_t BOX_CORNER_BITS = 16;
        constexpr uint32_t BOX_CORNER_MASK = ( 1u << BOX_CORNER_BITS ) - 1;

        box_base_corner = p_base_corner[0] & BOX_CORNER_MASK;
        box_far_corner = p_far_corner[0] & BOX_CORNER_MASK;
    }

    if( dims == 4 ) {
        constexpr uint32_t BOX_CORNER_BITS = 8;
        constexpr uint32_t BOX_CORNER_MASK = ( 1u << BOX_CORNER_BITS ) - 1;

        box_base_corner = p_base_corner[0] & BOX_CORNER_MASK;
        box_base_corner |= ( ( p_base_corner[1] & BOX_CORNER_MASK ) << BOX_CORNER_BITS );

        box_far_corner = p_far_corner[0] & BOX_CORNER_MASK;
        box_far_corner |= ( ( p_far_corner[1] & BOX_CORNER_MASK ) << BOX_CORNER_BITS );
    }

    if( dims == 5 ) {
        constexpr uint32_t BOX_CORNER_BITS = 5;
        constexpr uint32_t BOX_CORNER_MASK = ( 1u << BOX_CORNER_BITS ) - 1;

        box_base_corner = p_base_corner[0] & BOX_CORNER_MASK;
        box_base_corner |= ( ( p_base_corner[1] & BOX_CORNER_MASK ) << BOX_CORNER_BITS );
        box_base_corner |= ( ( p_base_corner[2] & BOX_CORNER_MASK ) << ( 2 * BOX_CORNER_BITS ) );

        box_far_corner = p_far_corner[0] & BOX_CORNER_MASK;
        box_far_corner |= ( ( p_far_corner[1] & BOX_CORNER_MASK ) << BOX_CORNER_BITS );
        box_far_corner |= ( ( p_far_corner[2] & BOX_CORNER_MASK ) << ( 2 * BOX_CORNER_BITS ) );
    }

    box_corner_dhw = box_base_corner;
    box_corner_dhw |= ( box_far_corner << 16 );

    desc->box_corner_dhw = box_corner_dhw;
/*
#ifdef PRINT_TMA_DESC
    std::bitset<32> b_( desc->box_corner_dhw );
    PRINT_VAR( b_ );
#endif
*/
}

static inline void set_range_ndhw( lwdaTmaDescv2 *p_desc, uint32_t ndhw ) {
    lwdaTmaDescIm2Col *desc = reinterpret_cast<lwdaTmaDescIm2Col *>( p_desc );

    desc->range_ndhw = 0;

    constexpr uint32_t RANGE_NDHW_BITS = 10;
    constexpr uint32_t RANGE_NDHW_MASK = ( 1u << RANGE_NDHW_BITS ) - 1;

    desc->range_ndhw = ( ( ndhw - 1 ) & RANGE_NDHW_MASK );

/*
#ifdef PRINT_TMA_DESC
    std::bitset<32> b_( desc->range_ndhw );
    PRINT_VAR( b_ );
#endif
*/
}