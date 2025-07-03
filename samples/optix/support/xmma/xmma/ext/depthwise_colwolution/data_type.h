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

#ifndef _XMMA_EXT_DEPTHWISE_COLWOLUTION_DATA_TYPE_H
#define _XMMA_EXT_DEPTHWISE_COLWOLUTION_DATA_TYPE_H

#pragma once

#include <cstdint>
#include <lwda_fp16.h>
#include <lwda_runtime.h>

namespace xmma{
namespace ext
{
namespace depthwise_colwolution
{
struct Data_type_fp16 {
    using Type = __half;
    static const int32_t BYTES_PER_ELEMENT = sizeof(Type);
};

struct Data_type_fp32 {
    using Type = float;
    static const int32_t BYTES_PER_ELEMENT = sizeof(Type);
};

struct Data_type_int32 {
    using Type = int32_t;
    static const int32_t BYTES_PER_ELEMENT = sizeof(Type);
};

} // namespace depthwise
} // namespace ext
} // namespace xmma

#endif
