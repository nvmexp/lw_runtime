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

#include <type_traits>

namespace xmma {
namespace hopper {
namespace emu {
/**
 * Data type is used here because BYTES_PER_ELEMENT_A/B only provide
 * bit-width but not format of data type
 */
template<typename T>
static inline lwdaTmaFormat get_data_type_tma_desc() {
  if(std::is_same<T, uint8_t>::value) return U8;
  if(std::is_same<T, uint16_t>::value) return U16;
  if(std::is_same<T, __half>::value) return F16_RN;
  //if(std::is_same<T, bf16>::value) return BF16_RN;
  if(std::is_same<T, uint32_t>::value) return U32;
  if(std::is_same<T, int32_t>::value) return S32;
  if(std::is_same<T, float>::value) return F32_RN;
  //if(std::is_same<T, tf32>::value) return TF32;
  if(std::is_same<T, uint64_t>::value) return U64;
  if(std::is_same<T, int64_t>::value) return S64;
  if(std::is_same<T, double>::value) return F64_RN;
  return FORMAT_MAX;
}

} // end namespace emu
} // end namespace hopper
} // end namespace xmma
