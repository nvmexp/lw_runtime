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

namespace xmma {
namespace hopper {
namespace emu {

static inline __host__ __device__ const char* get_string_desc_type(desc_type_t desc_type) {
  switch(desc_type) {
    case TILED: return "TILED";
    case IM2COL: return "IM2COL";
    default: return "Invalid Descriptor Type";
  }
}

static inline __host__ __device__ const char* get_string_format(format_t format) {
  switch(format) {
    case U8: return "U8";
    case U16: return "U16";
    case F16_RN: return "F16_RN";
    case BF16_RN: return "BF16_RN";
    case U32: return "U32";
    case S32: return "S32";
    case F32_RN: return "F32_RN";
    case F32_FTZ_RN: return "F32_FTZ_RN";
    case U64: return "U64";
    case S64: return "S64";
    case F64_RN: return "F64_RN";
    default: return "Invalid data type";
  }
}

static inline __host__ __device__ const char* get_string_interleaved(interleaved_t interleaved) {
  switch(interleaved) {
    case INTERLEAVED_NONE: return "INTERLEAVED_NONE";
    case INTERLEAVED_16B: return "INTERLEAVED_16B";
    case INTERLEAVED_32B: return "INTERLEAVED_32B";
    default: return "Invalid interleaved pattern";
  }
}

static inline __host__ __device__ const char* get_string_swizzle(swizzle_t swizzle) {
  switch(swizzle) {
  case SWIZZLE_DISABLED:
      return "SWIZZLE_DISABLED";
  case SWIZZLE_32B:
      return "SWIZZLE_32B";
  case SWIZZLE_64B:
      return "SWIZZLE_64B";
  case SWIZZLE_128B:
      return "SWIZZLE_128B";
  default:
      return "Invalid swizzle pattern";
  }
}

static inline __host__ __device__ const char* get_string_promotion(promotion_t promotion) {
  switch(promotion) {
    case L2_NONE: return "L2_NONE";
    case L2_64B: return "L2_64B";
    case L2_128B: return "L2_128B";
    case L2_256B: return "L2_256B";
    default: return "Invalid L2 Sector Promotion";
  }
}

} // end namespace emu
} // end namespace hopper
} // end namespace xmma