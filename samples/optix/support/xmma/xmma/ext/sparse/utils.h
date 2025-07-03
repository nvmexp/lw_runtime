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

#include <xmma/utils.h>

namespace xmma {

enum { RELU = 0, GELU = 1, RT_ACT = 2 };

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldgsts32(uint32_t (&smem_ptr)[1], const char* (&gmem_ptr)[1], uint32_t preds) {
    asm volatile( \
        "{\n" \
        "\t.reg .pred p<1>;\n" \
        "\t.reg .u32  src<1>;\n" \
        "\tr2p.b32 {p0}, %2.b0, 0x01;\n" \
        "\tselp.u32 src0, 0x4, 0, p0;\n" \
        "\tcp.async.ca.shared.global [%0], [%1], 0x04, src0;\n" \
        "}\n"
            :: "r"(smem_ptr[0])
            ,  "l"(gmem_ptr[0])
            ,  "r"(preds)
            );
}
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldgsts32(uint32_t (&smem_ptr)[2], const char* (&gmem_ptr)[2], uint32_t preds) {
    
    asm volatile( \
        "{\n" \
        "\t.reg .pred p<2>;\n" \
        "\t.reg .u32  src<2>;\n" \
        "\tr2p.b32 {p0, p1}, %4.b0, 0x03;\n" \
        "\tselp.u32 src0, 0x4, 0, p0;\n" \
        "\tselp.u32 src1, 0x4, 0, p1;\n" \
        "\tcp.async.ca.shared.global [%0], [%2], 0x04, src0;\n" \
        "\tcp.async.ca.shared.global [%1], [%3], 0x04, src1;\n" \
        "}\n"
            :: "r"(smem_ptr[0]), "r"(smem_ptr[1])
            ,  "l"(gmem_ptr[0]), "l"(gmem_ptr[1])
            ,  "r"(preds));
         
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldgsts64(uint32_t (&smem_ptr)[1], const char* (&gmem_ptr)[1], uint32_t preds) {
    
    asm volatile( \
        "{\n" \
        "\t.reg .pred p<1>;\n" \
        "\t.reg .u32  src<1>;\n" \
        "\tr2p.b32 {p0}, %2.b0, 0x01;\n" \
        "\tselp.u32 src0, 0x8, 0, p0;\n" \
        "\tcp.async.ca.shared.global [%0], [%1], 0x08, src0;\n" \
        "}\n"
            :: "r"(smem_ptr[0])
            ,  "l"(gmem_ptr[0])
            ,  "r"(preds));
         
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldgsts128(uint32_t dst[1],
                                 const char *src[1],
                                 uint32_t preds) {
    asm volatile( \
        "{\n" \
        "\t.reg .pred p<1>;\n" \
        "\t.reg .u32  src<1>;\n" \
        "\tr2p.b32 {p0}, %2.b0, 0x01;\n" \
        "\tselp.u32 src0, 16, 0, p0;\n" \
        "\n" \
        "\tcp.async.cg.shared.global [%0], [%1], 16, src0;\n" \
        "}\n"
            :: "r"(dst[0])
            ,  "l"(src[0])
            ,  "r"(preds));

}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldgsts128_nopreds(uint32_t dst[1],
                              const char *src[1]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" \
        :: "r"(dst[0]), "l"(src[0]));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldgsts32(uint32_t dst,
                              const char *src) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" \
        :: "r"(dst), "l"(src));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldgsts64_nopreds(uint32_t dst,
                              const char *src) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" \
        :: "r"(dst), "l"(src));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldgsts128_nopreds(uint32_t dst,
                              const char *src) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" \
        :: "r"(dst), "l"(src));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace xmma

