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

#include <xmma/xmma.h>

namespace xmma {

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// GMMAs with fp16 Aclwmulator
//
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64xNx16 TN with fp16 Aclwmulator
template<
  bool transa, // GMMA expects data to be in TN format. if A is column major, transa should be set
  bool transb, // GMMA expects data to be in TN format. if B is row major, transb should be set
  int GMMA_N,
  bool INCREMENT_SCORE_BOARD // whether or not the scoreboard will be incremented
>
inline __device__
void hgmma_fp16(const uint64_t &desc_a,
                const uint64_t &desc_b,
                uint32_t (&acc)[GMMA_N / 4]) {

}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x8x16
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x8x16 TN with fp16 Aclwmulator, where scoreboard is not incremented
template<>
inline __device__
void hgmma_fp16<false, false, 8, false>(const uint64_t &desc_a,
                                        const uint64_t &desc_b,
                                        uint32_t (&acc)[2]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n8k16.f16.f16.f16.f16 \n"
        "{%0, %1}, \n"
        "%2, \n"
        "%3, \n"
        "{%0, %1}; \n"
        "}\n"
        :  "+r"(acc[0]), "+r"(acc[1])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x8x16 TN with fp16 Aclwmulator, where scoreboard is incremented
template<>
inline __device__
void hgmma_fp16<false, false, 8, true>(const uint64_t &desc_a,
                                       const uint64_t &desc_b,
                                       uint32_t (&acc)[2]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n8k16.f16.f16.f16.f16 \n"
        "{%0, %1}, \n"
        "%2, \n"
        "%3, \n"
        "{%0, %1}, \n"
        "0; \n"
        "}\n"
        :  "+r"(acc[0]), "+r"(acc[1])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x128x16
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x128x16 TN with fp16 Aclwmulator, where scoreboard is not incremented
template<>
inline __device__
void hgmma_fp16<false, false, 128, false>(const uint64_t &desc_a,
                                          const uint64_t &desc_b,
                                          uint32_t (&acc)[32]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n128k16.f16.f16.f16.f16 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31},\n"
        "%32, \n"
        "%33, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31};\n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x128x16 TN with fp16 Aclwmulator, where scoreboard is incremented
template<>
inline __device__
void hgmma_fp16<false, false, 128, true>(const uint64_t &desc_a,
                                         const uint64_t &desc_b,
                                         uint32_t (&acc)[32]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n128k16.f16.f16.f16.f16 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31},\n"
        "%32, \n"
        "%33, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31},\n"
        "0; \n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x128x16 NN with fp16 Aclwmulator, where scoreboard is not incremented
template<>
inline __device__
void hgmma_fp16<true, false, 128, false>(const uint64_t &desc_a,
                                         const uint64_t &desc_b,
                                         uint32_t (&acc)[32]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n128k16.transA.f16.f16.f16.f16 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31},\n"
        "%32, \n"
        "%33, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31};\n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x128x16 NN with fp16 Aclwmulator, where scoreboard is incremented
template<>
inline __device__
void hgmma_fp16<true, false, 128, true>(const uint64_t &desc_a,
                                        const uint64_t &desc_b,
                                        uint32_t (&acc)[32]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n128k16.transA.f16.f16.f16.f16 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31},\n"
        "%32, \n"
        "%33, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31},\n"
        "0; \n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x128x16 TT with fp16 Aclwmulator, where scoreboard is not incremented
template<>
inline __device__
void hgmma_fp16<false, true, 128, false>(const uint64_t &desc_a,
                                         const uint64_t &desc_b,
                                         uint32_t (&acc)[32]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n128k16.transB.f16.f16.f16.f16 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31},\n"
        "%32, \n"
        "%33, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31};\n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x128x16 TT with fp16 Aclwmulator, where scoreboard is incremented
template<>
inline __device__
void hgmma_fp16<false, true, 128, true>(const uint64_t &desc_a,
                                        const uint64_t &desc_b,
                                        uint32_t (&acc)[32]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n128k16.transB.f16.f16.f16.f16 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31},\n"
        "%32, \n"
        "%33, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31},\n"
        "0; \n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x128x16 NT with fp16 Aclwmulator, where scoreboard is not incremented
template<>
inline __device__
void hgmma_fp16<true, true, 128, false>(const uint64_t &desc_a,
                                        const uint64_t &desc_b,
                                        uint32_t (&acc)[32]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n128k16.transA.transB.f16.f16.f16.f16 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31},\n"
        "%32, \n"
        "%33, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31};\n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x128x16 NT with fp16 Aclwmulator, where scoreboard is incremented
template<>
inline __device__
void hgmma_fp16<true, true, 128, true>(const uint64_t &desc_a,
                                       const uint64_t &desc_b,
                                       uint32_t (&acc)[32]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n128k16.transA.transB.f16.f16.f16.f16 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31},\n"
        "%32, \n"
        "%33, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31},\n"
        "0; \n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x64x16
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x64x16 TN with fp16 Aclwmulator, where scoreboard is not incremented
template<>
inline __device__
void hgmma_fp16<false, false, 64, false>(const uint64_t &desc_a,
                                         const uint64_t &desc_b,
                                         uint32_t (&acc)[16]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n64k16.f16.f16.f16.f16 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15},\n"
        "%16, \n"
        "%17, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15};\n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x64x16 TN with fp16 Aclwmulator, where scoreboard is incremented
template<>
inline __device__
void hgmma_fp16<false, false, 64, true>(const uint64_t &desc_a,
                                        const uint64_t &desc_b,
                                        uint32_t (&acc)[16]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n64k16.f16.f16.f16.f16 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15},\n"
        "%16, \n"
        "%17, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15}, \n"
        "0; \n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x64x16 NN with fp16 Aclwmulator, where scoreboard is not incremented
template<>
inline __device__
void hgmma_fp16<true, false, 64, false>(const uint64_t &desc_a,
                                        const uint64_t &desc_b,
                                        uint32_t (&acc)[16]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n64k16.transA.f16.f16.f16.f16 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15},\n"
        "%16, \n"
        "%17, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15};\n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x64x16 NN with fp16 Aclwmulator, where scoreboard is incremented
template<>
inline __device__
void hgmma_fp16<true, false, 64, true>(const uint64_t &desc_a,
                                       const uint64_t &desc_b,
                                       uint32_t (&acc)[16]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n64k16.transA.f16.f16.f16.f16 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15},\n"
        "%16, \n"
        "%17, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15}, \n"
        "0; \n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x64x16 TT with fp16 Aclwmulator, where scoreboard is not incremented
template<>
inline __device__
void hgmma_fp16<false, true, 64, false>(const uint64_t &desc_a,
                                        const uint64_t &desc_b,
                                        uint32_t (&acc)[16]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n64k16.transB.f16.f16.f16.f16 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15},\n"
        "%16, \n"
        "%17, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15};\n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x64x16 TT with fp16 Aclwmulator, where scoreboard is incremented
template<>
inline __device__
void hgmma_fp16<false, true, 64, true>(const uint64_t &desc_a,
                                       const uint64_t &desc_b,
                                       uint32_t (&acc)[16]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n64k16.transB.f16.f16.f16.f16 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15},\n"
        "%16, \n"
        "%17, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15}, \n"
        "0; \n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x64x16 NT with fp16 Aclwmulator, where scoreboard is not incremented
template<>
inline __device__
void hgmma_fp16<true, true, 64, false>(const uint64_t &desc_a,
                                       const uint64_t &desc_b,
                                       uint32_t (&acc)[16]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n64k16.transA.transB.f16.f16.f16.f16 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15},\n"
        "%16, \n"
        "%17, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15};\n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x64x16 NT with fp16 Aclwmulator, where scoreboard is incremented
template<>
inline __device__
void hgmma_fp16<true, true, 64, true>(const uint64_t &desc_a,
                                      const uint64_t &desc_b,
                                      uint32_t (&acc)[16]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n64k16.transA.transB.f16.f16.f16.f16 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15},\n"
        "%16, \n"
        "%17, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15}, \n"
        "0; \n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x256x16
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x256x16 TN with fp16 Aclwmulator, where scoreboard is not incremented
template<>
inline __device__
void hgmma_fp16<false, false, 256, false>(const uint64_t &desc_a,
                                          const uint64_t &desc_b,
                                          uint32_t (&acc)[64]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n256k16.f16.f16.f16.f16 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63},\n"
        "%64, \n"
        "%65, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63};\n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        ,  "+r"(acc[32]), "+r"(acc[33]), "+r"(acc[34]), "+r"(acc[35])
        ,  "+r"(acc[36]), "+r"(acc[37]), "+r"(acc[38]), "+r"(acc[39])
        ,  "+r"(acc[40]), "+r"(acc[41]), "+r"(acc[42]), "+r"(acc[43])
        ,  "+r"(acc[44]), "+r"(acc[45]), "+r"(acc[46]), "+r"(acc[47])
        ,  "+r"(acc[48]), "+r"(acc[49]), "+r"(acc[50]), "+r"(acc[51])
        ,  "+r"(acc[52]), "+r"(acc[53]), "+r"(acc[54]), "+r"(acc[55])
        ,  "+r"(acc[56]), "+r"(acc[57]), "+r"(acc[58]), "+r"(acc[59])
        ,  "+r"(acc[60]), "+r"(acc[61]), "+r"(acc[62]), "+r"(acc[63])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x256x16 TN with fp16 Aclwmulator, where scoreboard is incremented
template<>
inline __device__
void hgmma_fp16<false, false, 256, true>(const uint64_t &desc_a,
                                         const uint64_t &desc_b,
                                         uint32_t (&acc)[64]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n256k16.f16.f16.f16.f16 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63},\n"
        "%64, \n"
        "%65, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63},\n"
        "0;"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        ,  "+r"(acc[32]), "+r"(acc[33]), "+r"(acc[34]), "+r"(acc[35])
        ,  "+r"(acc[36]), "+r"(acc[37]), "+r"(acc[38]), "+r"(acc[39])
        ,  "+r"(acc[40]), "+r"(acc[41]), "+r"(acc[42]), "+r"(acc[43])
        ,  "+r"(acc[44]), "+r"(acc[45]), "+r"(acc[46]), "+r"(acc[47])
        ,  "+r"(acc[48]), "+r"(acc[49]), "+r"(acc[50]), "+r"(acc[51])
        ,  "+r"(acc[52]), "+r"(acc[53]), "+r"(acc[54]), "+r"(acc[55])
        ,  "+r"(acc[56]), "+r"(acc[57]), "+r"(acc[58]), "+r"(acc[59])
        ,  "+r"(acc[60]), "+r"(acc[61]), "+r"(acc[62]), "+r"(acc[63])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x256x16 NN with fp16 Aclwmulator, where scoreboard is not incremented
template<>
inline __device__
void hgmma_fp16<true, false, 256, false>(const uint64_t &desc_a,
                                         const uint64_t &desc_b,
                                         uint32_t (&acc)[64]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n256k16.transA.f16.f16.f16.f16 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63},\n"
        "%64, \n"
        "%65, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63};\n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        ,  "+r"(acc[32]), "+r"(acc[33]), "+r"(acc[34]), "+r"(acc[35])
        ,  "+r"(acc[36]), "+r"(acc[37]), "+r"(acc[38]), "+r"(acc[39])
        ,  "+r"(acc[40]), "+r"(acc[41]), "+r"(acc[42]), "+r"(acc[43])
        ,  "+r"(acc[44]), "+r"(acc[45]), "+r"(acc[46]), "+r"(acc[47])
        ,  "+r"(acc[48]), "+r"(acc[49]), "+r"(acc[50]), "+r"(acc[51])
        ,  "+r"(acc[52]), "+r"(acc[53]), "+r"(acc[54]), "+r"(acc[55])
        ,  "+r"(acc[56]), "+r"(acc[57]), "+r"(acc[58]), "+r"(acc[59])
        ,  "+r"(acc[60]), "+r"(acc[61]), "+r"(acc[62]), "+r"(acc[63])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x256x16 NN with fp16 Aclwmulator, where scoreboard is incremented
template<>
inline __device__
void hgmma_fp16<true, false, 256, true>(const uint64_t &desc_a,
                                        const uint64_t &desc_b,
                                        uint32_t (&acc)[64]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n256k16.transA.f16.f16.f16.f16 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63},\n"
        "%64, \n"
        "%65, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63},\n"
        "0;"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        ,  "+r"(acc[32]), "+r"(acc[33]), "+r"(acc[34]), "+r"(acc[35])
        ,  "+r"(acc[36]), "+r"(acc[37]), "+r"(acc[38]), "+r"(acc[39])
        ,  "+r"(acc[40]), "+r"(acc[41]), "+r"(acc[42]), "+r"(acc[43])
        ,  "+r"(acc[44]), "+r"(acc[45]), "+r"(acc[46]), "+r"(acc[47])
        ,  "+r"(acc[48]), "+r"(acc[49]), "+r"(acc[50]), "+r"(acc[51])
        ,  "+r"(acc[52]), "+r"(acc[53]), "+r"(acc[54]), "+r"(acc[55])
        ,  "+r"(acc[56]), "+r"(acc[57]), "+r"(acc[58]), "+r"(acc[59])
        ,  "+r"(acc[60]), "+r"(acc[61]), "+r"(acc[62]), "+r"(acc[63])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x256x16 TT with fp16 Aclwmulator, where scoreboard is not incremented
template<>
inline __device__
void hgmma_fp16<false, true, 256, false>(const uint64_t &desc_a,
                                         const uint64_t &desc_b,
                                         uint32_t (&acc)[64]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n256k16.transB.f16.f16.f16.f16 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63},\n"
        "%64, \n"
        "%65, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63};\n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        ,  "+r"(acc[32]), "+r"(acc[33]), "+r"(acc[34]), "+r"(acc[35])
        ,  "+r"(acc[36]), "+r"(acc[37]), "+r"(acc[38]), "+r"(acc[39])
        ,  "+r"(acc[40]), "+r"(acc[41]), "+r"(acc[42]), "+r"(acc[43])
        ,  "+r"(acc[44]), "+r"(acc[45]), "+r"(acc[46]), "+r"(acc[47])
        ,  "+r"(acc[48]), "+r"(acc[49]), "+r"(acc[50]), "+r"(acc[51])
        ,  "+r"(acc[52]), "+r"(acc[53]), "+r"(acc[54]), "+r"(acc[55])
        ,  "+r"(acc[56]), "+r"(acc[57]), "+r"(acc[58]), "+r"(acc[59])
        ,  "+r"(acc[60]), "+r"(acc[61]), "+r"(acc[62]), "+r"(acc[63])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x256x16 TT with fp16 Aclwmulator, where scoreboard is incremented
template<>
inline __device__
void hgmma_fp16<false, true, 256, true>(const uint64_t &desc_a,
                                        const uint64_t &desc_b,
                                        uint32_t (&acc)[64]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n256k16.transB.f16.f16.f16.f16 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63},\n"
        "%64, \n"
        "%65, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63},\n"
        "0;"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        ,  "+r"(acc[32]), "+r"(acc[33]), "+r"(acc[34]), "+r"(acc[35])
        ,  "+r"(acc[36]), "+r"(acc[37]), "+r"(acc[38]), "+r"(acc[39])
        ,  "+r"(acc[40]), "+r"(acc[41]), "+r"(acc[42]), "+r"(acc[43])
        ,  "+r"(acc[44]), "+r"(acc[45]), "+r"(acc[46]), "+r"(acc[47])
        ,  "+r"(acc[48]), "+r"(acc[49]), "+r"(acc[50]), "+r"(acc[51])
        ,  "+r"(acc[52]), "+r"(acc[53]), "+r"(acc[54]), "+r"(acc[55])
        ,  "+r"(acc[56]), "+r"(acc[57]), "+r"(acc[58]), "+r"(acc[59])
        ,  "+r"(acc[60]), "+r"(acc[61]), "+r"(acc[62]), "+r"(acc[63])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x256x16 NT with fp16 Aclwmulator, where scoreboard is not incremented
template<>
inline __device__
void hgmma_fp16<true, true, 256, false>(const uint64_t &desc_a,
                                        const uint64_t &desc_b,
                                        uint32_t (&acc)[64]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n256k16.transA.transB.f16.f16.f16.f16 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63},\n"
        "%64, \n"
        "%65, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63};\n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        ,  "+r"(acc[32]), "+r"(acc[33]), "+r"(acc[34]), "+r"(acc[35])
        ,  "+r"(acc[36]), "+r"(acc[37]), "+r"(acc[38]), "+r"(acc[39])
        ,  "+r"(acc[40]), "+r"(acc[41]), "+r"(acc[42]), "+r"(acc[43])
        ,  "+r"(acc[44]), "+r"(acc[45]), "+r"(acc[46]), "+r"(acc[47])
        ,  "+r"(acc[48]), "+r"(acc[49]), "+r"(acc[50]), "+r"(acc[51])
        ,  "+r"(acc[52]), "+r"(acc[53]), "+r"(acc[54]), "+r"(acc[55])
        ,  "+r"(acc[56]), "+r"(acc[57]), "+r"(acc[58]), "+r"(acc[59])
        ,  "+r"(acc[60]), "+r"(acc[61]), "+r"(acc[62]), "+r"(acc[63])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x256x16 TT with fp16 Aclwmulator, where scoreboard is incremented
template<>
inline __device__
void hgmma_fp16<true, true, 256, true>(const uint64_t &desc_a,
                                       const uint64_t &desc_b,
                                       uint32_t (&acc)[64]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n256k16.transA.transB.f16.f16.f16.f16 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63},\n"
        "%64, \n"
        "%65, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63},\n"
        "0;"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        ,  "+r"(acc[32]), "+r"(acc[33]), "+r"(acc[34]), "+r"(acc[35])
        ,  "+r"(acc[36]), "+r"(acc[37]), "+r"(acc[38]), "+r"(acc[39])
        ,  "+r"(acc[40]), "+r"(acc[41]), "+r"(acc[42]), "+r"(acc[43])
        ,  "+r"(acc[44]), "+r"(acc[45]), "+r"(acc[46]), "+r"(acc[47])
        ,  "+r"(acc[48]), "+r"(acc[49]), "+r"(acc[50]), "+r"(acc[51])
        ,  "+r"(acc[52]), "+r"(acc[53]), "+r"(acc[54]), "+r"(acc[55])
        ,  "+r"(acc[56]), "+r"(acc[57]), "+r"(acc[58]), "+r"(acc[59])
        ,  "+r"(acc[60]), "+r"(acc[61]), "+r"(acc[62]), "+r"(acc[63])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// GMMAs with fp32 Aclwmulator
//
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64xNx16 TN with fp32 Aclwmulator
template<
  bool transa, // GMMA expects data to be in TN format. if A is column major, transa should be set
  bool transb, // GMMA expects data to be in TN format. if B is row major, transb should be set
  int GMMA_N,
  bool INCREMENT_SCORE_BOARD // whether or not the scoreboard will be incremented
>
inline __device__
void hgmma_fp32(const uint64_t &desc_a,
                const uint64_t &desc_b,
                uint32_t (&acc)[GMMA_N / 2]) {

}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x8x16
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x8x16 TN with fp32 Aclwmulator, where scoreboard is not incremented
template<>
inline __device__
void hgmma_fp32<false, false, 8, false>(const uint64_t &desc_a,
                                        const uint64_t &desc_b,
                                        uint32_t (&acc)[4]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n8k16.f32.f16.f16.f32 \n"
        "{%0, %1, %2, %3}, \n"
        "%4, \n"
        "%5, \n"
        "{%0, %1, %2, %3}; \n"
        "}\n"
        :  "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x8x16 TN with fp32 Aclwmulator, where scoreboard is incremented
template<>
inline __device__
void hgmma_fp32<false, false, 8, true>(const uint64_t &desc_a,
                                       const uint64_t &desc_b,
                                       uint32_t (&acc)[4]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n8k16.f32.f16.f16.f32 \n"
        "{%0, %1, %2, %3}, \n"
        "%4, \n"
        "%5, \n"
        "{%0, %1, %2, %3}, \n"
        "0; \n"
        "}\n"
        :  "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x128x16
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x128x16 TN with fp32 Aclwmulator, where scoreboard is not incremented
template<>
inline __device__
void hgmma_fp32<false, false, 128, false>(const uint64_t &desc_a,
                                          const uint64_t &desc_b,
                                          uint32_t (&acc)[64]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n128k16.f32.f16.f16.f32 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63},\n"
        "%64, \n"
        "%65, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63};\n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        ,  "+r"(acc[32]), "+r"(acc[33]), "+r"(acc[34]), "+r"(acc[35])
        ,  "+r"(acc[36]), "+r"(acc[37]), "+r"(acc[38]), "+r"(acc[39])
        ,  "+r"(acc[40]), "+r"(acc[41]), "+r"(acc[42]), "+r"(acc[43])
        ,  "+r"(acc[44]), "+r"(acc[45]), "+r"(acc[46]), "+r"(acc[47])
        ,  "+r"(acc[48]), "+r"(acc[49]), "+r"(acc[50]), "+r"(acc[51])
        ,  "+r"(acc[52]), "+r"(acc[53]), "+r"(acc[54]), "+r"(acc[55])
        ,  "+r"(acc[56]), "+r"(acc[57]), "+r"(acc[58]), "+r"(acc[59])
        ,  "+r"(acc[60]), "+r"(acc[61]), "+r"(acc[62]), "+r"(acc[63])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x128x16 TN with fp32 Aclwmulator, where scoreboard is incremented
template<>
inline __device__
void hgmma_fp32<false, false, 128, true>(const uint64_t &desc_a,
                                         const uint64_t &desc_b,
                                         uint32_t (&acc)[64]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n128k16.f32.f16.f16.f32 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63},\n"
        "%64, \n"
        "%65, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63},\n"
        " 0;\n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        ,  "+r"(acc[32]), "+r"(acc[33]), "+r"(acc[34]), "+r"(acc[35])
        ,  "+r"(acc[36]), "+r"(acc[37]), "+r"(acc[38]), "+r"(acc[39])
        ,  "+r"(acc[40]), "+r"(acc[41]), "+r"(acc[42]), "+r"(acc[43])
        ,  "+r"(acc[44]), "+r"(acc[45]), "+r"(acc[46]), "+r"(acc[47])
        ,  "+r"(acc[48]), "+r"(acc[49]), "+r"(acc[50]), "+r"(acc[51])
        ,  "+r"(acc[52]), "+r"(acc[53]), "+r"(acc[54]), "+r"(acc[55])
        ,  "+r"(acc[56]), "+r"(acc[57]), "+r"(acc[58]), "+r"(acc[59])
        ,  "+r"(acc[60]), "+r"(acc[61]), "+r"(acc[62]), "+r"(acc[63])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x128x16 NN with fp32 Aclwmulator, where scoreboard is not incremented
template<>
inline __device__
void hgmma_fp32<true, false, 128, false>(const uint64_t &desc_a,
                                         const uint64_t &desc_b,
                                         uint32_t (&acc)[64]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n128k16.transA.f32.f16.f16.f32 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63},\n"
        "%64, \n"
        "%65, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63};\n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        ,  "+r"(acc[32]), "+r"(acc[33]), "+r"(acc[34]), "+r"(acc[35])
        ,  "+r"(acc[36]), "+r"(acc[37]), "+r"(acc[38]), "+r"(acc[39])
        ,  "+r"(acc[40]), "+r"(acc[41]), "+r"(acc[42]), "+r"(acc[43])
        ,  "+r"(acc[44]), "+r"(acc[45]), "+r"(acc[46]), "+r"(acc[47])
        ,  "+r"(acc[48]), "+r"(acc[49]), "+r"(acc[50]), "+r"(acc[51])
        ,  "+r"(acc[52]), "+r"(acc[53]), "+r"(acc[54]), "+r"(acc[55])
        ,  "+r"(acc[56]), "+r"(acc[57]), "+r"(acc[58]), "+r"(acc[59])
        ,  "+r"(acc[60]), "+r"(acc[61]), "+r"(acc[62]), "+r"(acc[63])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x128x16 NN with fp32 Aclwmulator, where scoreboard is incremented
template<>
inline __device__
void hgmma_fp32<true, false, 128, true>(const uint64_t &desc_a,
                                        const uint64_t &desc_b,
                                        uint32_t (&acc)[64]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n128k16.transA.f32.f16.f16.f32 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63},\n"
        "%64, \n"
        "%65, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63},\n"
        " 0;\n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        ,  "+r"(acc[32]), "+r"(acc[33]), "+r"(acc[34]), "+r"(acc[35])
        ,  "+r"(acc[36]), "+r"(acc[37]), "+r"(acc[38]), "+r"(acc[39])
        ,  "+r"(acc[40]), "+r"(acc[41]), "+r"(acc[42]), "+r"(acc[43])
        ,  "+r"(acc[44]), "+r"(acc[45]), "+r"(acc[46]), "+r"(acc[47])
        ,  "+r"(acc[48]), "+r"(acc[49]), "+r"(acc[50]), "+r"(acc[51])
        ,  "+r"(acc[52]), "+r"(acc[53]), "+r"(acc[54]), "+r"(acc[55])
        ,  "+r"(acc[56]), "+r"(acc[57]), "+r"(acc[58]), "+r"(acc[59])
        ,  "+r"(acc[60]), "+r"(acc[61]), "+r"(acc[62]), "+r"(acc[63])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x128x16 TT with fp32 Aclwmulator, where scoreboard is not incremented
template<>
inline __device__
void hgmma_fp32<false, true, 128, false>(const uint64_t &desc_a,
                                         const uint64_t &desc_b,
                                         uint32_t (&acc)[64]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n128k16.transB.f32.f16.f16.f32 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63},\n"
        "%64, \n"
        "%65, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63};\n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        ,  "+r"(acc[32]), "+r"(acc[33]), "+r"(acc[34]), "+r"(acc[35])
        ,  "+r"(acc[36]), "+r"(acc[37]), "+r"(acc[38]), "+r"(acc[39])
        ,  "+r"(acc[40]), "+r"(acc[41]), "+r"(acc[42]), "+r"(acc[43])
        ,  "+r"(acc[44]), "+r"(acc[45]), "+r"(acc[46]), "+r"(acc[47])
        ,  "+r"(acc[48]), "+r"(acc[49]), "+r"(acc[50]), "+r"(acc[51])
        ,  "+r"(acc[52]), "+r"(acc[53]), "+r"(acc[54]), "+r"(acc[55])
        ,  "+r"(acc[56]), "+r"(acc[57]), "+r"(acc[58]), "+r"(acc[59])
        ,  "+r"(acc[60]), "+r"(acc[61]), "+r"(acc[62]), "+r"(acc[63])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x128x16 TT with fp32 Aclwmulator, where scoreboard is incremented
template<>
inline __device__
void hgmma_fp32<false, true, 128, true>(const uint64_t &desc_a,
                                        const uint64_t &desc_b,
                                        uint32_t (&acc)[64]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n128k16.transB.f32.f16.f16.f32 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63},\n"
        "%64, \n"
        "%65, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63},\n"
        " 0;\n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        ,  "+r"(acc[32]), "+r"(acc[33]), "+r"(acc[34]), "+r"(acc[35])
        ,  "+r"(acc[36]), "+r"(acc[37]), "+r"(acc[38]), "+r"(acc[39])
        ,  "+r"(acc[40]), "+r"(acc[41]), "+r"(acc[42]), "+r"(acc[43])
        ,  "+r"(acc[44]), "+r"(acc[45]), "+r"(acc[46]), "+r"(acc[47])
        ,  "+r"(acc[48]), "+r"(acc[49]), "+r"(acc[50]), "+r"(acc[51])
        ,  "+r"(acc[52]), "+r"(acc[53]), "+r"(acc[54]), "+r"(acc[55])
        ,  "+r"(acc[56]), "+r"(acc[57]), "+r"(acc[58]), "+r"(acc[59])
        ,  "+r"(acc[60]), "+r"(acc[61]), "+r"(acc[62]), "+r"(acc[63])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x128x16 NT with fp32 Aclwmulator, where scoreboard is not incremented
template<>
inline __device__
void hgmma_fp32<true, true, 128, false>(const uint64_t &desc_a,
                                        const uint64_t &desc_b,
                                        uint32_t (&acc)[64]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n128k16.transA.transB.f32.f16.f16.f32 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63},\n"
        "%64, \n"
        "%65, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63};\n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        ,  "+r"(acc[32]), "+r"(acc[33]), "+r"(acc[34]), "+r"(acc[35])
        ,  "+r"(acc[36]), "+r"(acc[37]), "+r"(acc[38]), "+r"(acc[39])
        ,  "+r"(acc[40]), "+r"(acc[41]), "+r"(acc[42]), "+r"(acc[43])
        ,  "+r"(acc[44]), "+r"(acc[45]), "+r"(acc[46]), "+r"(acc[47])
        ,  "+r"(acc[48]), "+r"(acc[49]), "+r"(acc[50]), "+r"(acc[51])
        ,  "+r"(acc[52]), "+r"(acc[53]), "+r"(acc[54]), "+r"(acc[55])
        ,  "+r"(acc[56]), "+r"(acc[57]), "+r"(acc[58]), "+r"(acc[59])
        ,  "+r"(acc[60]), "+r"(acc[61]), "+r"(acc[62]), "+r"(acc[63])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x128x16 NT with fp32 Aclwmulator, where scoreboard is incremented
template<>
inline __device__
void hgmma_fp32<true, true, 128, true>(const uint64_t &desc_a,
                                       const uint64_t &desc_b,
                                       uint32_t (&acc)[64]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n128k16.transA.transB.f32.f16.f16.f32 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63},\n"
        "%64, \n"
        "%65, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63},\n"
        " 0;\n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        ,  "+r"(acc[32]), "+r"(acc[33]), "+r"(acc[34]), "+r"(acc[35])
        ,  "+r"(acc[36]), "+r"(acc[37]), "+r"(acc[38]), "+r"(acc[39])
        ,  "+r"(acc[40]), "+r"(acc[41]), "+r"(acc[42]), "+r"(acc[43])
        ,  "+r"(acc[44]), "+r"(acc[45]), "+r"(acc[46]), "+r"(acc[47])
        ,  "+r"(acc[48]), "+r"(acc[49]), "+r"(acc[50]), "+r"(acc[51])
        ,  "+r"(acc[52]), "+r"(acc[53]), "+r"(acc[54]), "+r"(acc[55])
        ,  "+r"(acc[56]), "+r"(acc[57]), "+r"(acc[58]), "+r"(acc[59])
        ,  "+r"(acc[60]), "+r"(acc[61]), "+r"(acc[62]), "+r"(acc[63])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x64x16
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x64x16 TN with fp32 Aclwmulator, where scoreboard is not incremented
template<>
inline __device__
void hgmma_fp32<false, false, 64, false>(const uint64_t &desc_a,
                                         const uint64_t &desc_b,
                                         uint32_t (&acc)[32]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n64k16.f32.f16.f16.f32 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31},\n"
        "%32, \n"
        "%33, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31};\n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x64x16 TN with fp32 Aclwmulator, where scoreboard is incremented
template<>
inline __device__
void hgmma_fp32<false, false, 64, true>(const uint64_t &desc_a,
                                        const uint64_t &desc_b,
                                        uint32_t (&acc)[32]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n64k16.f32.f16.f16.f32 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31},\n"
        "%32, \n"
        "%33, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31},\n"
        " 0; \n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x64x16 NN with fp32 Aclwmulator, where scoreboard is not incremented
template<>
inline __device__
void hgmma_fp32<true, false, 64, false>(const uint64_t &desc_a,
                                        const uint64_t &desc_b,
                                        uint32_t (&acc)[32]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n64k16.transA.f32.f16.f16.f32 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31},\n"
        "%32, \n"
        "%33, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31};\n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x64x16 NN with fp32 Aclwmulator, where scoreboard is incremented
template<>
inline __device__
void hgmma_fp32<true, false, 64, true>(const uint64_t &desc_a,
                                       const uint64_t &desc_b,
                                       uint32_t (&acc)[32]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n64k16.transA.f32.f16.f16.f32 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31},\n"
        "%32, \n"
        "%33, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31},\n"
        " 0; \n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x64x16 TT with fp32 Aclwmulator, where scoreboard is not incremented
template<>
inline __device__
void hgmma_fp32<false, true, 64, false>(const uint64_t &desc_a,
                                        const uint64_t &desc_b,
                                        uint32_t (&acc)[32]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n64k16.transB.f32.f16.f16.f32 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31},\n"
        "%32, \n"
        "%33, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31};\n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x64x16 TT with fp32 Aclwmulator, where scoreboard is incremented
template<>
inline __device__
void hgmma_fp32<false, true, 64, true>(const uint64_t &desc_a,
                                       const uint64_t &desc_b,
                                       uint32_t (&acc)[32]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n64k16.transB.f32.f16.f16.f32 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31},\n"
        "%32, \n"
        "%33, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31},\n"
        " 0; \n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x64x16 NT with fp32 Aclwmulator, where scoreboard is not incremented
template<>
inline __device__
void hgmma_fp32<true, true, 64, false>(const uint64_t &desc_a,
                                       const uint64_t &desc_b,
                                       uint32_t (&acc)[32]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n64k16.transA.transB.f32.f16.f16.f32 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31},\n"
        "%32, \n"
        "%33, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31};\n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x64x16 NT with fp32 Aclwmulator, where scoreboard is incremented
template<>
inline __device__
void hgmma_fp32<true, true, 64, true>(const uint64_t &desc_a,
                                      const uint64_t &desc_b,
                                      uint32_t (&acc)[32]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n64k16.transA.transB.f32.f16.f16.f32 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31},\n"
        "%32, \n"
        "%33, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31},\n"
        " 0; \n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x256x16
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x256x16 TN with fp32 Aclwmulator, where scoreboard is not incremented
template<>
inline __device__
void hgmma_fp32<false, false, 256, false>(const uint64_t &desc_a,
                                          const uint64_t &desc_b,
                                          uint32_t (&acc)[128]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n256k16.f32.f16.f16.f32 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63,\n"
        " %64, %65, %66, %67, %68, %69, %70, %71,\n"
        " %72, %73, %74, %75, %76, %77, %78, %79,\n"
        " %80, %81, %82, %83, %84, %85, %86, %87,\n"
        " %88, %89, %90, %91, %92, %93, %94, %95,\n"
        " %96, %97, %98, %99, %100, %101, %102, %103,\n"
        " %104, %105, %106, %107, %108, %109, %110, %111,\n"
        " %112, %113, %114, %115, %116, %117, %118, %119,\n"
        " %120, %121, %122, %123, %124, %125, %126, %127},\n"
        "%128, \n"
        "%129, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63,\n"
        " %64, %65, %66, %67, %68, %69, %70, %71,\n"
        " %72, %73, %74, %75, %76, %77, %78, %79,\n"
        " %80, %81, %82, %83, %84, %85, %86, %87,\n"
        " %88, %89, %90, %91, %92, %93, %94, %95,\n"
        " %96, %97, %98, %99, %100, %101, %102, %103,\n"
        " %104, %105, %106, %107, %108, %109, %110, %111,\n"
        " %112, %113, %114, %115, %116, %117, %118, %119,\n"
        " %120, %121, %122, %123, %124, %125, %126, %127};\n"
        "}\n"
        :  "+r"(acc[  0]), "+r"(acc[  1]), "+r"(acc[  2]), "+r"(acc[  3])
        ,  "+r"(acc[  4]), "+r"(acc[  5]), "+r"(acc[  6]), "+r"(acc[  7])
        ,  "+r"(acc[  8]), "+r"(acc[  9]), "+r"(acc[ 10]), "+r"(acc[ 11])
        ,  "+r"(acc[ 12]), "+r"(acc[ 13]), "+r"(acc[ 14]), "+r"(acc[ 15])
        ,  "+r"(acc[ 16]), "+r"(acc[ 17]), "+r"(acc[ 18]), "+r"(acc[ 19])
        ,  "+r"(acc[ 20]), "+r"(acc[ 21]), "+r"(acc[ 22]), "+r"(acc[ 23])
        ,  "+r"(acc[ 24]), "+r"(acc[ 25]), "+r"(acc[ 26]), "+r"(acc[ 27])
        ,  "+r"(acc[ 28]), "+r"(acc[ 29]), "+r"(acc[ 30]), "+r"(acc[ 31])
        ,  "+r"(acc[ 32]), "+r"(acc[ 33]), "+r"(acc[ 34]), "+r"(acc[ 35])
        ,  "+r"(acc[ 36]), "+r"(acc[ 37]), "+r"(acc[ 38]), "+r"(acc[ 39])
        ,  "+r"(acc[ 40]), "+r"(acc[ 41]), "+r"(acc[ 42]), "+r"(acc[ 43])
        ,  "+r"(acc[ 44]), "+r"(acc[ 45]), "+r"(acc[ 46]), "+r"(acc[ 47])
        ,  "+r"(acc[ 48]), "+r"(acc[ 49]), "+r"(acc[ 50]), "+r"(acc[ 51])
        ,  "+r"(acc[ 52]), "+r"(acc[ 53]), "+r"(acc[ 54]), "+r"(acc[ 55])
        ,  "+r"(acc[ 56]), "+r"(acc[ 57]), "+r"(acc[ 58]), "+r"(acc[ 59])
        ,  "+r"(acc[ 60]), "+r"(acc[ 61]), "+r"(acc[ 62]), "+r"(acc[ 63])
        ,  "+r"(acc[ 64]), "+r"(acc[ 65]), "+r"(acc[ 66]), "+r"(acc[ 67])
        ,  "+r"(acc[ 68]), "+r"(acc[ 69]), "+r"(acc[ 70]), "+r"(acc[ 71])
        ,  "+r"(acc[ 72]), "+r"(acc[ 73]), "+r"(acc[ 74]), "+r"(acc[ 75])
        ,  "+r"(acc[ 76]), "+r"(acc[ 77]), "+r"(acc[ 78]), "+r"(acc[ 79])
        ,  "+r"(acc[ 80]), "+r"(acc[ 81]), "+r"(acc[ 82]), "+r"(acc[ 83])
        ,  "+r"(acc[ 84]), "+r"(acc[ 85]), "+r"(acc[ 86]), "+r"(acc[ 87])
        ,  "+r"(acc[ 88]), "+r"(acc[ 89]), "+r"(acc[ 90]), "+r"(acc[ 91])
        ,  "+r"(acc[ 92]), "+r"(acc[ 93]), "+r"(acc[ 94]), "+r"(acc[ 95])
        ,  "+r"(acc[ 96]), "+r"(acc[ 97]), "+r"(acc[ 98]), "+r"(acc[ 99])
        ,  "+r"(acc[100]), "+r"(acc[101]), "+r"(acc[102]), "+r"(acc[103])
        ,  "+r"(acc[104]), "+r"(acc[105]), "+r"(acc[106]), "+r"(acc[107])
        ,  "+r"(acc[108]), "+r"(acc[109]), "+r"(acc[110]), "+r"(acc[111])
        ,  "+r"(acc[112]), "+r"(acc[113]), "+r"(acc[114]), "+r"(acc[115])
        ,  "+r"(acc[116]), "+r"(acc[117]), "+r"(acc[118]), "+r"(acc[119])
        ,  "+r"(acc[120]), "+r"(acc[121]), "+r"(acc[122]), "+r"(acc[123])
        ,  "+r"(acc[124]), "+r"(acc[125]), "+r"(acc[126]), "+r"(acc[127])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x256x16 TN with fp32 Aclwmulator, where scoreboard is incremented
template<>
inline __device__
void hgmma_fp32<false, false, 256, true>(const uint64_t &desc_a,
                                         const uint64_t &desc_b,
                                         uint32_t (&acc)[128]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n256k16.f32.f16.f16.f32 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63,\n"
        " %64, %65, %66, %67, %68, %69, %70, %71,\n"
        " %72, %73, %74, %75, %76, %77, %78, %79,\n"
        " %80, %81, %82, %83, %84, %85, %86, %87,\n"
        " %88, %89, %90, %91, %92, %93, %94, %95,\n"
        " %96, %97, %98, %99, %100, %101, %102, %103,\n"
        " %104, %105, %106, %107, %108, %109, %110, %111,\n"
        " %112, %113, %114, %115, %116, %117, %118, %119,\n"
        " %120, %121, %122, %123, %124, %125, %126, %127},\n"
        "%128, \n"
        "%129, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63,\n"
        " %64, %65, %66, %67, %68, %69, %70, %71,\n"
        " %72, %73, %74, %75, %76, %77, %78, %79,\n"
        " %80, %81, %82, %83, %84, %85, %86, %87,\n"
        " %88, %89, %90, %91, %92, %93, %94, %95,\n"
        " %96, %97, %98, %99, %100, %101, %102, %103,\n"
        " %104, %105, %106, %107, %108, %109, %110, %111,\n"
        " %112, %113, %114, %115, %116, %117, %118, %119,\n"
        " %120, %121, %122, %123, %124, %125, %126, %127},\n"
        " 0;"
        "}\n"
        :  "+r"(acc[  0]), "+r"(acc[  1]), "+r"(acc[  2]), "+r"(acc[  3])
        ,  "+r"(acc[  4]), "+r"(acc[  5]), "+r"(acc[  6]), "+r"(acc[  7])
        ,  "+r"(acc[  8]), "+r"(acc[  9]), "+r"(acc[ 10]), "+r"(acc[ 11])
        ,  "+r"(acc[ 12]), "+r"(acc[ 13]), "+r"(acc[ 14]), "+r"(acc[ 15])
        ,  "+r"(acc[ 16]), "+r"(acc[ 17]), "+r"(acc[ 18]), "+r"(acc[ 19])
        ,  "+r"(acc[ 20]), "+r"(acc[ 21]), "+r"(acc[ 22]), "+r"(acc[ 23])
        ,  "+r"(acc[ 24]), "+r"(acc[ 25]), "+r"(acc[ 26]), "+r"(acc[ 27])
        ,  "+r"(acc[ 28]), "+r"(acc[ 29]), "+r"(acc[ 30]), "+r"(acc[ 31])
        ,  "+r"(acc[ 32]), "+r"(acc[ 33]), "+r"(acc[ 34]), "+r"(acc[ 35])
        ,  "+r"(acc[ 36]), "+r"(acc[ 37]), "+r"(acc[ 38]), "+r"(acc[ 39])
        ,  "+r"(acc[ 40]), "+r"(acc[ 41]), "+r"(acc[ 42]), "+r"(acc[ 43])
        ,  "+r"(acc[ 44]), "+r"(acc[ 45]), "+r"(acc[ 46]), "+r"(acc[ 47])
        ,  "+r"(acc[ 48]), "+r"(acc[ 49]), "+r"(acc[ 50]), "+r"(acc[ 51])
        ,  "+r"(acc[ 52]), "+r"(acc[ 53]), "+r"(acc[ 54]), "+r"(acc[ 55])
        ,  "+r"(acc[ 56]), "+r"(acc[ 57]), "+r"(acc[ 58]), "+r"(acc[ 59])
        ,  "+r"(acc[ 60]), "+r"(acc[ 61]), "+r"(acc[ 62]), "+r"(acc[ 63])
        ,  "+r"(acc[ 64]), "+r"(acc[ 65]), "+r"(acc[ 66]), "+r"(acc[ 67])
        ,  "+r"(acc[ 68]), "+r"(acc[ 69]), "+r"(acc[ 70]), "+r"(acc[ 71])
        ,  "+r"(acc[ 72]), "+r"(acc[ 73]), "+r"(acc[ 74]), "+r"(acc[ 75])
        ,  "+r"(acc[ 76]), "+r"(acc[ 77]), "+r"(acc[ 78]), "+r"(acc[ 79])
        ,  "+r"(acc[ 80]), "+r"(acc[ 81]), "+r"(acc[ 82]), "+r"(acc[ 83])
        ,  "+r"(acc[ 84]), "+r"(acc[ 85]), "+r"(acc[ 86]), "+r"(acc[ 87])
        ,  "+r"(acc[ 88]), "+r"(acc[ 89]), "+r"(acc[ 90]), "+r"(acc[ 91])
        ,  "+r"(acc[ 92]), "+r"(acc[ 93]), "+r"(acc[ 94]), "+r"(acc[ 95])
        ,  "+r"(acc[ 96]), "+r"(acc[ 97]), "+r"(acc[ 98]), "+r"(acc[ 99])
        ,  "+r"(acc[100]), "+r"(acc[101]), "+r"(acc[102]), "+r"(acc[103])
        ,  "+r"(acc[104]), "+r"(acc[105]), "+r"(acc[106]), "+r"(acc[107])
        ,  "+r"(acc[108]), "+r"(acc[109]), "+r"(acc[110]), "+r"(acc[111])
        ,  "+r"(acc[112]), "+r"(acc[113]), "+r"(acc[114]), "+r"(acc[115])
        ,  "+r"(acc[116]), "+r"(acc[117]), "+r"(acc[118]), "+r"(acc[119])
        ,  "+r"(acc[120]), "+r"(acc[121]), "+r"(acc[122]), "+r"(acc[123])
        ,  "+r"(acc[124]), "+r"(acc[125]), "+r"(acc[126]), "+r"(acc[127])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x256x16 NN with fp32 Aclwmulator, where scoreboard is not incremented
template<>
inline __device__
void hgmma_fp32<true, false, 256, false>(const uint64_t &desc_a,
                                         const uint64_t &desc_b,
                                         uint32_t (&acc)[128]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n256k16.transA.f32.f16.f16.f32 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63,\n"
        " %64, %65, %66, %67, %68, %69, %70, %71,\n"
        " %72, %73, %74, %75, %76, %77, %78, %79,\n"
        " %80, %81, %82, %83, %84, %85, %86, %87,\n"
        " %88, %89, %90, %91, %92, %93, %94, %95,\n"
        " %96, %97, %98, %99, %100, %101, %102, %103,\n"
        " %104, %105, %106, %107, %108, %109, %110, %111,\n"
        " %112, %113, %114, %115, %116, %117, %118, %119,\n"
        " %120, %121, %122, %123, %124, %125, %126, %127},\n"
        "%128, \n"
        "%129, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63,\n"
        " %64, %65, %66, %67, %68, %69, %70, %71,\n"
        " %72, %73, %74, %75, %76, %77, %78, %79,\n"
        " %80, %81, %82, %83, %84, %85, %86, %87,\n"
        " %88, %89, %90, %91, %92, %93, %94, %95,\n"
        " %96, %97, %98, %99, %100, %101, %102, %103,\n"
        " %104, %105, %106, %107, %108, %109, %110, %111,\n"
        " %112, %113, %114, %115, %116, %117, %118, %119,\n"
        " %120, %121, %122, %123, %124, %125, %126, %127};\n"
        "}\n"
        :  "+r"(acc[  0]), "+r"(acc[  1]), "+r"(acc[  2]), "+r"(acc[  3])
        ,  "+r"(acc[  4]), "+r"(acc[  5]), "+r"(acc[  6]), "+r"(acc[  7])
        ,  "+r"(acc[  8]), "+r"(acc[  9]), "+r"(acc[ 10]), "+r"(acc[ 11])
        ,  "+r"(acc[ 12]), "+r"(acc[ 13]), "+r"(acc[ 14]), "+r"(acc[ 15])
        ,  "+r"(acc[ 16]), "+r"(acc[ 17]), "+r"(acc[ 18]), "+r"(acc[ 19])
        ,  "+r"(acc[ 20]), "+r"(acc[ 21]), "+r"(acc[ 22]), "+r"(acc[ 23])
        ,  "+r"(acc[ 24]), "+r"(acc[ 25]), "+r"(acc[ 26]), "+r"(acc[ 27])
        ,  "+r"(acc[ 28]), "+r"(acc[ 29]), "+r"(acc[ 30]), "+r"(acc[ 31])
        ,  "+r"(acc[ 32]), "+r"(acc[ 33]), "+r"(acc[ 34]), "+r"(acc[ 35])
        ,  "+r"(acc[ 36]), "+r"(acc[ 37]), "+r"(acc[ 38]), "+r"(acc[ 39])
        ,  "+r"(acc[ 40]), "+r"(acc[ 41]), "+r"(acc[ 42]), "+r"(acc[ 43])
        ,  "+r"(acc[ 44]), "+r"(acc[ 45]), "+r"(acc[ 46]), "+r"(acc[ 47])
        ,  "+r"(acc[ 48]), "+r"(acc[ 49]), "+r"(acc[ 50]), "+r"(acc[ 51])
        ,  "+r"(acc[ 52]), "+r"(acc[ 53]), "+r"(acc[ 54]), "+r"(acc[ 55])
        ,  "+r"(acc[ 56]), "+r"(acc[ 57]), "+r"(acc[ 58]), "+r"(acc[ 59])
        ,  "+r"(acc[ 60]), "+r"(acc[ 61]), "+r"(acc[ 62]), "+r"(acc[ 63])
        ,  "+r"(acc[ 64]), "+r"(acc[ 65]), "+r"(acc[ 66]), "+r"(acc[ 67])
        ,  "+r"(acc[ 68]), "+r"(acc[ 69]), "+r"(acc[ 70]), "+r"(acc[ 71])
        ,  "+r"(acc[ 72]), "+r"(acc[ 73]), "+r"(acc[ 74]), "+r"(acc[ 75])
        ,  "+r"(acc[ 76]), "+r"(acc[ 77]), "+r"(acc[ 78]), "+r"(acc[ 79])
        ,  "+r"(acc[ 80]), "+r"(acc[ 81]), "+r"(acc[ 82]), "+r"(acc[ 83])
        ,  "+r"(acc[ 84]), "+r"(acc[ 85]), "+r"(acc[ 86]), "+r"(acc[ 87])
        ,  "+r"(acc[ 88]), "+r"(acc[ 89]), "+r"(acc[ 90]), "+r"(acc[ 91])
        ,  "+r"(acc[ 92]), "+r"(acc[ 93]), "+r"(acc[ 94]), "+r"(acc[ 95])
        ,  "+r"(acc[ 96]), "+r"(acc[ 97]), "+r"(acc[ 98]), "+r"(acc[ 99])
        ,  "+r"(acc[100]), "+r"(acc[101]), "+r"(acc[102]), "+r"(acc[103])
        ,  "+r"(acc[104]), "+r"(acc[105]), "+r"(acc[106]), "+r"(acc[107])
        ,  "+r"(acc[108]), "+r"(acc[109]), "+r"(acc[110]), "+r"(acc[111])
        ,  "+r"(acc[112]), "+r"(acc[113]), "+r"(acc[114]), "+r"(acc[115])
        ,  "+r"(acc[116]), "+r"(acc[117]), "+r"(acc[118]), "+r"(acc[119])
        ,  "+r"(acc[120]), "+r"(acc[121]), "+r"(acc[122]), "+r"(acc[123])
        ,  "+r"(acc[124]), "+r"(acc[125]), "+r"(acc[126]), "+r"(acc[127])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x256x16 NN with fp32 Aclwmulator, where scoreboard is incremented
template<>
inline __device__
void hgmma_fp32<true, false, 256, true>(const uint64_t &desc_a,
                                        const uint64_t &desc_b,
                                        uint32_t (&acc)[128]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n256k16.transA.f32.f16.f16.f32 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63,\n"
        " %64, %65, %66, %67, %68, %69, %70, %71,\n"
        " %72, %73, %74, %75, %76, %77, %78, %79,\n"
        " %80, %81, %82, %83, %84, %85, %86, %87,\n"
        " %88, %89, %90, %91, %92, %93, %94, %95,\n"
        " %96, %97, %98, %99, %100, %101, %102, %103,\n"
        " %104, %105, %106, %107, %108, %109, %110, %111,\n"
        " %112, %113, %114, %115, %116, %117, %118, %119,\n"
        " %120, %121, %122, %123, %124, %125, %126, %127},\n"
        "%128, \n"
        "%129, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63,\n"
        " %64, %65, %66, %67, %68, %69, %70, %71,\n"
        " %72, %73, %74, %75, %76, %77, %78, %79,\n"
        " %80, %81, %82, %83, %84, %85, %86, %87,\n"
        " %88, %89, %90, %91, %92, %93, %94, %95,\n"
        " %96, %97, %98, %99, %100, %101, %102, %103,\n"
        " %104, %105, %106, %107, %108, %109, %110, %111,\n"
        " %112, %113, %114, %115, %116, %117, %118, %119,\n"
        " %120, %121, %122, %123, %124, %125, %126, %127},\n"
        " 0;"
        "}\n"
        :  "+r"(acc[  0]), "+r"(acc[  1]), "+r"(acc[  2]), "+r"(acc[  3])
        ,  "+r"(acc[  4]), "+r"(acc[  5]), "+r"(acc[  6]), "+r"(acc[  7])
        ,  "+r"(acc[  8]), "+r"(acc[  9]), "+r"(acc[ 10]), "+r"(acc[ 11])
        ,  "+r"(acc[ 12]), "+r"(acc[ 13]), "+r"(acc[ 14]), "+r"(acc[ 15])
        ,  "+r"(acc[ 16]), "+r"(acc[ 17]), "+r"(acc[ 18]), "+r"(acc[ 19])
        ,  "+r"(acc[ 20]), "+r"(acc[ 21]), "+r"(acc[ 22]), "+r"(acc[ 23])
        ,  "+r"(acc[ 24]), "+r"(acc[ 25]), "+r"(acc[ 26]), "+r"(acc[ 27])
        ,  "+r"(acc[ 28]), "+r"(acc[ 29]), "+r"(acc[ 30]), "+r"(acc[ 31])
        ,  "+r"(acc[ 32]), "+r"(acc[ 33]), "+r"(acc[ 34]), "+r"(acc[ 35])
        ,  "+r"(acc[ 36]), "+r"(acc[ 37]), "+r"(acc[ 38]), "+r"(acc[ 39])
        ,  "+r"(acc[ 40]), "+r"(acc[ 41]), "+r"(acc[ 42]), "+r"(acc[ 43])
        ,  "+r"(acc[ 44]), "+r"(acc[ 45]), "+r"(acc[ 46]), "+r"(acc[ 47])
        ,  "+r"(acc[ 48]), "+r"(acc[ 49]), "+r"(acc[ 50]), "+r"(acc[ 51])
        ,  "+r"(acc[ 52]), "+r"(acc[ 53]), "+r"(acc[ 54]), "+r"(acc[ 55])
        ,  "+r"(acc[ 56]), "+r"(acc[ 57]), "+r"(acc[ 58]), "+r"(acc[ 59])
        ,  "+r"(acc[ 60]), "+r"(acc[ 61]), "+r"(acc[ 62]), "+r"(acc[ 63])
        ,  "+r"(acc[ 64]), "+r"(acc[ 65]), "+r"(acc[ 66]), "+r"(acc[ 67])
        ,  "+r"(acc[ 68]), "+r"(acc[ 69]), "+r"(acc[ 70]), "+r"(acc[ 71])
        ,  "+r"(acc[ 72]), "+r"(acc[ 73]), "+r"(acc[ 74]), "+r"(acc[ 75])
        ,  "+r"(acc[ 76]), "+r"(acc[ 77]), "+r"(acc[ 78]), "+r"(acc[ 79])
        ,  "+r"(acc[ 80]), "+r"(acc[ 81]), "+r"(acc[ 82]), "+r"(acc[ 83])
        ,  "+r"(acc[ 84]), "+r"(acc[ 85]), "+r"(acc[ 86]), "+r"(acc[ 87])
        ,  "+r"(acc[ 88]), "+r"(acc[ 89]), "+r"(acc[ 90]), "+r"(acc[ 91])
        ,  "+r"(acc[ 92]), "+r"(acc[ 93]), "+r"(acc[ 94]), "+r"(acc[ 95])
        ,  "+r"(acc[ 96]), "+r"(acc[ 97]), "+r"(acc[ 98]), "+r"(acc[ 99])
        ,  "+r"(acc[100]), "+r"(acc[101]), "+r"(acc[102]), "+r"(acc[103])
        ,  "+r"(acc[104]), "+r"(acc[105]), "+r"(acc[106]), "+r"(acc[107])
        ,  "+r"(acc[108]), "+r"(acc[109]), "+r"(acc[110]), "+r"(acc[111])
        ,  "+r"(acc[112]), "+r"(acc[113]), "+r"(acc[114]), "+r"(acc[115])
        ,  "+r"(acc[116]), "+r"(acc[117]), "+r"(acc[118]), "+r"(acc[119])
        ,  "+r"(acc[120]), "+r"(acc[121]), "+r"(acc[122]), "+r"(acc[123])
        ,  "+r"(acc[124]), "+r"(acc[125]), "+r"(acc[126]), "+r"(acc[127])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x256x16 TT with fp32 Aclwmulator, where scoreboard is not incremented
template<>
inline __device__
void hgmma_fp32<false, true, 256, false>(const uint64_t &desc_a,
                                         const uint64_t &desc_b,
                                         uint32_t (&acc)[128]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n256k16.transB.f32.f16.f16.f32 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63,\n"
        " %64, %65, %66, %67, %68, %69, %70, %71,\n"
        " %72, %73, %74, %75, %76, %77, %78, %79,\n"
        " %80, %81, %82, %83, %84, %85, %86, %87,\n"
        " %88, %89, %90, %91, %92, %93, %94, %95,\n"
        " %96, %97, %98, %99, %100, %101, %102, %103,\n"
        " %104, %105, %106, %107, %108, %109, %110, %111,\n"
        " %112, %113, %114, %115, %116, %117, %118, %119,\n"
        " %120, %121, %122, %123, %124, %125, %126, %127},\n"
        "%128, \n"
        "%129, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63,\n"
        " %64, %65, %66, %67, %68, %69, %70, %71,\n"
        " %72, %73, %74, %75, %76, %77, %78, %79,\n"
        " %80, %81, %82, %83, %84, %85, %86, %87,\n"
        " %88, %89, %90, %91, %92, %93, %94, %95,\n"
        " %96, %97, %98, %99, %100, %101, %102, %103,\n"
        " %104, %105, %106, %107, %108, %109, %110, %111,\n"
        " %112, %113, %114, %115, %116, %117, %118, %119,\n"
        " %120, %121, %122, %123, %124, %125, %126, %127};\n"
        "}\n"
        :  "+r"(acc[  0]), "+r"(acc[  1]), "+r"(acc[  2]), "+r"(acc[  3])
        ,  "+r"(acc[  4]), "+r"(acc[  5]), "+r"(acc[  6]), "+r"(acc[  7])
        ,  "+r"(acc[  8]), "+r"(acc[  9]), "+r"(acc[ 10]), "+r"(acc[ 11])
        ,  "+r"(acc[ 12]), "+r"(acc[ 13]), "+r"(acc[ 14]), "+r"(acc[ 15])
        ,  "+r"(acc[ 16]), "+r"(acc[ 17]), "+r"(acc[ 18]), "+r"(acc[ 19])
        ,  "+r"(acc[ 20]), "+r"(acc[ 21]), "+r"(acc[ 22]), "+r"(acc[ 23])
        ,  "+r"(acc[ 24]), "+r"(acc[ 25]), "+r"(acc[ 26]), "+r"(acc[ 27])
        ,  "+r"(acc[ 28]), "+r"(acc[ 29]), "+r"(acc[ 30]), "+r"(acc[ 31])
        ,  "+r"(acc[ 32]), "+r"(acc[ 33]), "+r"(acc[ 34]), "+r"(acc[ 35])
        ,  "+r"(acc[ 36]), "+r"(acc[ 37]), "+r"(acc[ 38]), "+r"(acc[ 39])
        ,  "+r"(acc[ 40]), "+r"(acc[ 41]), "+r"(acc[ 42]), "+r"(acc[ 43])
        ,  "+r"(acc[ 44]), "+r"(acc[ 45]), "+r"(acc[ 46]), "+r"(acc[ 47])
        ,  "+r"(acc[ 48]), "+r"(acc[ 49]), "+r"(acc[ 50]), "+r"(acc[ 51])
        ,  "+r"(acc[ 52]), "+r"(acc[ 53]), "+r"(acc[ 54]), "+r"(acc[ 55])
        ,  "+r"(acc[ 56]), "+r"(acc[ 57]), "+r"(acc[ 58]), "+r"(acc[ 59])
        ,  "+r"(acc[ 60]), "+r"(acc[ 61]), "+r"(acc[ 62]), "+r"(acc[ 63])
        ,  "+r"(acc[ 64]), "+r"(acc[ 65]), "+r"(acc[ 66]), "+r"(acc[ 67])
        ,  "+r"(acc[ 68]), "+r"(acc[ 69]), "+r"(acc[ 70]), "+r"(acc[ 71])
        ,  "+r"(acc[ 72]), "+r"(acc[ 73]), "+r"(acc[ 74]), "+r"(acc[ 75])
        ,  "+r"(acc[ 76]), "+r"(acc[ 77]), "+r"(acc[ 78]), "+r"(acc[ 79])
        ,  "+r"(acc[ 80]), "+r"(acc[ 81]), "+r"(acc[ 82]), "+r"(acc[ 83])
        ,  "+r"(acc[ 84]), "+r"(acc[ 85]), "+r"(acc[ 86]), "+r"(acc[ 87])
        ,  "+r"(acc[ 88]), "+r"(acc[ 89]), "+r"(acc[ 90]), "+r"(acc[ 91])
        ,  "+r"(acc[ 92]), "+r"(acc[ 93]), "+r"(acc[ 94]), "+r"(acc[ 95])
        ,  "+r"(acc[ 96]), "+r"(acc[ 97]), "+r"(acc[ 98]), "+r"(acc[ 99])
        ,  "+r"(acc[100]), "+r"(acc[101]), "+r"(acc[102]), "+r"(acc[103])
        ,  "+r"(acc[104]), "+r"(acc[105]), "+r"(acc[106]), "+r"(acc[107])
        ,  "+r"(acc[108]), "+r"(acc[109]), "+r"(acc[110]), "+r"(acc[111])
        ,  "+r"(acc[112]), "+r"(acc[113]), "+r"(acc[114]), "+r"(acc[115])
        ,  "+r"(acc[116]), "+r"(acc[117]), "+r"(acc[118]), "+r"(acc[119])
        ,  "+r"(acc[120]), "+r"(acc[121]), "+r"(acc[122]), "+r"(acc[123])
        ,  "+r"(acc[124]), "+r"(acc[125]), "+r"(acc[126]), "+r"(acc[127])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x256x16 TT with fp32 Aclwmulator, where scoreboard is incremented
template<>
inline __device__
void hgmma_fp32<false, true, 256, true>(const uint64_t &desc_a,
                                        const uint64_t &desc_b,
                                        uint32_t (&acc)[128]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n256k16.transB.f32.f16.f16.f32 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63,\n"
        " %64, %65, %66, %67, %68, %69, %70, %71,\n"
        " %72, %73, %74, %75, %76, %77, %78, %79,\n"
        " %80, %81, %82, %83, %84, %85, %86, %87,\n"
        " %88, %89, %90, %91, %92, %93, %94, %95,\n"
        " %96, %97, %98, %99, %100, %101, %102, %103,\n"
        " %104, %105, %106, %107, %108, %109, %110, %111,\n"
        " %112, %113, %114, %115, %116, %117, %118, %119,\n"
        " %120, %121, %122, %123, %124, %125, %126, %127},\n"
        "%128, \n"
        "%129, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63,\n"
        " %64, %65, %66, %67, %68, %69, %70, %71,\n"
        " %72, %73, %74, %75, %76, %77, %78, %79,\n"
        " %80, %81, %82, %83, %84, %85, %86, %87,\n"
        " %88, %89, %90, %91, %92, %93, %94, %95,\n"
        " %96, %97, %98, %99, %100, %101, %102, %103,\n"
        " %104, %105, %106, %107, %108, %109, %110, %111,\n"
        " %112, %113, %114, %115, %116, %117, %118, %119,\n"
        " %120, %121, %122, %123, %124, %125, %126, %127},\n"
        " 0;"
        "}\n"
        :  "+r"(acc[  0]), "+r"(acc[  1]), "+r"(acc[  2]), "+r"(acc[  3])
        ,  "+r"(acc[  4]), "+r"(acc[  5]), "+r"(acc[  6]), "+r"(acc[  7])
        ,  "+r"(acc[  8]), "+r"(acc[  9]), "+r"(acc[ 10]), "+r"(acc[ 11])
        ,  "+r"(acc[ 12]), "+r"(acc[ 13]), "+r"(acc[ 14]), "+r"(acc[ 15])
        ,  "+r"(acc[ 16]), "+r"(acc[ 17]), "+r"(acc[ 18]), "+r"(acc[ 19])
        ,  "+r"(acc[ 20]), "+r"(acc[ 21]), "+r"(acc[ 22]), "+r"(acc[ 23])
        ,  "+r"(acc[ 24]), "+r"(acc[ 25]), "+r"(acc[ 26]), "+r"(acc[ 27])
        ,  "+r"(acc[ 28]), "+r"(acc[ 29]), "+r"(acc[ 30]), "+r"(acc[ 31])
        ,  "+r"(acc[ 32]), "+r"(acc[ 33]), "+r"(acc[ 34]), "+r"(acc[ 35])
        ,  "+r"(acc[ 36]), "+r"(acc[ 37]), "+r"(acc[ 38]), "+r"(acc[ 39])
        ,  "+r"(acc[ 40]), "+r"(acc[ 41]), "+r"(acc[ 42]), "+r"(acc[ 43])
        ,  "+r"(acc[ 44]), "+r"(acc[ 45]), "+r"(acc[ 46]), "+r"(acc[ 47])
        ,  "+r"(acc[ 48]), "+r"(acc[ 49]), "+r"(acc[ 50]), "+r"(acc[ 51])
        ,  "+r"(acc[ 52]), "+r"(acc[ 53]), "+r"(acc[ 54]), "+r"(acc[ 55])
        ,  "+r"(acc[ 56]), "+r"(acc[ 57]), "+r"(acc[ 58]), "+r"(acc[ 59])
        ,  "+r"(acc[ 60]), "+r"(acc[ 61]), "+r"(acc[ 62]), "+r"(acc[ 63])
        ,  "+r"(acc[ 64]), "+r"(acc[ 65]), "+r"(acc[ 66]), "+r"(acc[ 67])
        ,  "+r"(acc[ 68]), "+r"(acc[ 69]), "+r"(acc[ 70]), "+r"(acc[ 71])
        ,  "+r"(acc[ 72]), "+r"(acc[ 73]), "+r"(acc[ 74]), "+r"(acc[ 75])
        ,  "+r"(acc[ 76]), "+r"(acc[ 77]), "+r"(acc[ 78]), "+r"(acc[ 79])
        ,  "+r"(acc[ 80]), "+r"(acc[ 81]), "+r"(acc[ 82]), "+r"(acc[ 83])
        ,  "+r"(acc[ 84]), "+r"(acc[ 85]), "+r"(acc[ 86]), "+r"(acc[ 87])
        ,  "+r"(acc[ 88]), "+r"(acc[ 89]), "+r"(acc[ 90]), "+r"(acc[ 91])
        ,  "+r"(acc[ 92]), "+r"(acc[ 93]), "+r"(acc[ 94]), "+r"(acc[ 95])
        ,  "+r"(acc[ 96]), "+r"(acc[ 97]), "+r"(acc[ 98]), "+r"(acc[ 99])
        ,  "+r"(acc[100]), "+r"(acc[101]), "+r"(acc[102]), "+r"(acc[103])
        ,  "+r"(acc[104]), "+r"(acc[105]), "+r"(acc[106]), "+r"(acc[107])
        ,  "+r"(acc[108]), "+r"(acc[109]), "+r"(acc[110]), "+r"(acc[111])
        ,  "+r"(acc[112]), "+r"(acc[113]), "+r"(acc[114]), "+r"(acc[115])
        ,  "+r"(acc[116]), "+r"(acc[117]), "+r"(acc[118]), "+r"(acc[119])
        ,  "+r"(acc[120]), "+r"(acc[121]), "+r"(acc[122]), "+r"(acc[123])
        ,  "+r"(acc[124]), "+r"(acc[125]), "+r"(acc[126]), "+r"(acc[127])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x256x16 NT with fp32 Aclwmulator, where scoreboard is not incremented
template<>
inline __device__
void hgmma_fp32<true, true, 256, false>(const uint64_t &desc_a,
                                        const uint64_t &desc_b,
                                        uint32_t (&acc)[128]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n256k16.transA.transB.f32.f16.f16.f32 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63,\n"
        " %64, %65, %66, %67, %68, %69, %70, %71,\n"
        " %72, %73, %74, %75, %76, %77, %78, %79,\n"
        " %80, %81, %82, %83, %84, %85, %86, %87,\n"
        " %88, %89, %90, %91, %92, %93, %94, %95,\n"
        " %96, %97, %98, %99, %100, %101, %102, %103,\n"
        " %104, %105, %106, %107, %108, %109, %110, %111,\n"
        " %112, %113, %114, %115, %116, %117, %118, %119,\n"
        " %120, %121, %122, %123, %124, %125, %126, %127},\n"
        "%128, \n"
        "%129, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63,\n"
        " %64, %65, %66, %67, %68, %69, %70, %71,\n"
        " %72, %73, %74, %75, %76, %77, %78, %79,\n"
        " %80, %81, %82, %83, %84, %85, %86, %87,\n"
        " %88, %89, %90, %91, %92, %93, %94, %95,\n"
        " %96, %97, %98, %99, %100, %101, %102, %103,\n"
        " %104, %105, %106, %107, %108, %109, %110, %111,\n"
        " %112, %113, %114, %115, %116, %117, %118, %119,\n"
        " %120, %121, %122, %123, %124, %125, %126, %127};\n"
        "}\n"
        :  "+r"(acc[  0]), "+r"(acc[  1]), "+r"(acc[  2]), "+r"(acc[  3])
        ,  "+r"(acc[  4]), "+r"(acc[  5]), "+r"(acc[  6]), "+r"(acc[  7])
        ,  "+r"(acc[  8]), "+r"(acc[  9]), "+r"(acc[ 10]), "+r"(acc[ 11])
        ,  "+r"(acc[ 12]), "+r"(acc[ 13]), "+r"(acc[ 14]), "+r"(acc[ 15])
        ,  "+r"(acc[ 16]), "+r"(acc[ 17]), "+r"(acc[ 18]), "+r"(acc[ 19])
        ,  "+r"(acc[ 20]), "+r"(acc[ 21]), "+r"(acc[ 22]), "+r"(acc[ 23])
        ,  "+r"(acc[ 24]), "+r"(acc[ 25]), "+r"(acc[ 26]), "+r"(acc[ 27])
        ,  "+r"(acc[ 28]), "+r"(acc[ 29]), "+r"(acc[ 30]), "+r"(acc[ 31])
        ,  "+r"(acc[ 32]), "+r"(acc[ 33]), "+r"(acc[ 34]), "+r"(acc[ 35])
        ,  "+r"(acc[ 36]), "+r"(acc[ 37]), "+r"(acc[ 38]), "+r"(acc[ 39])
        ,  "+r"(acc[ 40]), "+r"(acc[ 41]), "+r"(acc[ 42]), "+r"(acc[ 43])
        ,  "+r"(acc[ 44]), "+r"(acc[ 45]), "+r"(acc[ 46]), "+r"(acc[ 47])
        ,  "+r"(acc[ 48]), "+r"(acc[ 49]), "+r"(acc[ 50]), "+r"(acc[ 51])
        ,  "+r"(acc[ 52]), "+r"(acc[ 53]), "+r"(acc[ 54]), "+r"(acc[ 55])
        ,  "+r"(acc[ 56]), "+r"(acc[ 57]), "+r"(acc[ 58]), "+r"(acc[ 59])
        ,  "+r"(acc[ 60]), "+r"(acc[ 61]), "+r"(acc[ 62]), "+r"(acc[ 63])
        ,  "+r"(acc[ 64]), "+r"(acc[ 65]), "+r"(acc[ 66]), "+r"(acc[ 67])
        ,  "+r"(acc[ 68]), "+r"(acc[ 69]), "+r"(acc[ 70]), "+r"(acc[ 71])
        ,  "+r"(acc[ 72]), "+r"(acc[ 73]), "+r"(acc[ 74]), "+r"(acc[ 75])
        ,  "+r"(acc[ 76]), "+r"(acc[ 77]), "+r"(acc[ 78]), "+r"(acc[ 79])
        ,  "+r"(acc[ 80]), "+r"(acc[ 81]), "+r"(acc[ 82]), "+r"(acc[ 83])
        ,  "+r"(acc[ 84]), "+r"(acc[ 85]), "+r"(acc[ 86]), "+r"(acc[ 87])
        ,  "+r"(acc[ 88]), "+r"(acc[ 89]), "+r"(acc[ 90]), "+r"(acc[ 91])
        ,  "+r"(acc[ 92]), "+r"(acc[ 93]), "+r"(acc[ 94]), "+r"(acc[ 95])
        ,  "+r"(acc[ 96]), "+r"(acc[ 97]), "+r"(acc[ 98]), "+r"(acc[ 99])
        ,  "+r"(acc[100]), "+r"(acc[101]), "+r"(acc[102]), "+r"(acc[103])
        ,  "+r"(acc[104]), "+r"(acc[105]), "+r"(acc[106]), "+r"(acc[107])
        ,  "+r"(acc[108]), "+r"(acc[109]), "+r"(acc[110]), "+r"(acc[111])
        ,  "+r"(acc[112]), "+r"(acc[113]), "+r"(acc[114]), "+r"(acc[115])
        ,  "+r"(acc[116]), "+r"(acc[117]), "+r"(acc[118]), "+r"(acc[119])
        ,  "+r"(acc[120]), "+r"(acc[121]), "+r"(acc[122]), "+r"(acc[123])
        ,  "+r"(acc[124]), "+r"(acc[125]), "+r"(acc[126]), "+r"(acc[127])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x256x16 TT with fp32 Aclwmulator, where scoreboard is incremented
template<>
inline __device__
void hgmma_fp32<true, true, 256, true>(const uint64_t &desc_a,
                                       const uint64_t &desc_b,
                                       uint32_t (&acc)[128]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n256k16.transA.transB.f32.f16.f16.f32 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63,\n"
        " %64, %65, %66, %67, %68, %69, %70, %71,\n"
        " %72, %73, %74, %75, %76, %77, %78, %79,\n"
        " %80, %81, %82, %83, %84, %85, %86, %87,\n"
        " %88, %89, %90, %91, %92, %93, %94, %95,\n"
        " %96, %97, %98, %99, %100, %101, %102, %103,\n"
        " %104, %105, %106, %107, %108, %109, %110, %111,\n"
        " %112, %113, %114, %115, %116, %117, %118, %119,\n"
        " %120, %121, %122, %123, %124, %125, %126, %127},\n"
        "%128, \n"
        "%129, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63,\n"
        " %64, %65, %66, %67, %68, %69, %70, %71,\n"
        " %72, %73, %74, %75, %76, %77, %78, %79,\n"
        " %80, %81, %82, %83, %84, %85, %86, %87,\n"
        " %88, %89, %90, %91, %92, %93, %94, %95,\n"
        " %96, %97, %98, %99, %100, %101, %102, %103,\n"
        " %104, %105, %106, %107, %108, %109, %110, %111,\n"
        " %112, %113, %114, %115, %116, %117, %118, %119,\n"
        " %120, %121, %122, %123, %124, %125, %126, %127},\n"
        " 0;"
        "}\n"
        :  "+r"(acc[  0]), "+r"(acc[  1]), "+r"(acc[  2]), "+r"(acc[  3])
        ,  "+r"(acc[  4]), "+r"(acc[  5]), "+r"(acc[  6]), "+r"(acc[  7])
        ,  "+r"(acc[  8]), "+r"(acc[  9]), "+r"(acc[ 10]), "+r"(acc[ 11])
        ,  "+r"(acc[ 12]), "+r"(acc[ 13]), "+r"(acc[ 14]), "+r"(acc[ 15])
        ,  "+r"(acc[ 16]), "+r"(acc[ 17]), "+r"(acc[ 18]), "+r"(acc[ 19])
        ,  "+r"(acc[ 20]), "+r"(acc[ 21]), "+r"(acc[ 22]), "+r"(acc[ 23])
        ,  "+r"(acc[ 24]), "+r"(acc[ 25]), "+r"(acc[ 26]), "+r"(acc[ 27])
        ,  "+r"(acc[ 28]), "+r"(acc[ 29]), "+r"(acc[ 30]), "+r"(acc[ 31])
        ,  "+r"(acc[ 32]), "+r"(acc[ 33]), "+r"(acc[ 34]), "+r"(acc[ 35])
        ,  "+r"(acc[ 36]), "+r"(acc[ 37]), "+r"(acc[ 38]), "+r"(acc[ 39])
        ,  "+r"(acc[ 40]), "+r"(acc[ 41]), "+r"(acc[ 42]), "+r"(acc[ 43])
        ,  "+r"(acc[ 44]), "+r"(acc[ 45]), "+r"(acc[ 46]), "+r"(acc[ 47])
        ,  "+r"(acc[ 48]), "+r"(acc[ 49]), "+r"(acc[ 50]), "+r"(acc[ 51])
        ,  "+r"(acc[ 52]), "+r"(acc[ 53]), "+r"(acc[ 54]), "+r"(acc[ 55])
        ,  "+r"(acc[ 56]), "+r"(acc[ 57]), "+r"(acc[ 58]), "+r"(acc[ 59])
        ,  "+r"(acc[ 60]), "+r"(acc[ 61]), "+r"(acc[ 62]), "+r"(acc[ 63])
        ,  "+r"(acc[ 64]), "+r"(acc[ 65]), "+r"(acc[ 66]), "+r"(acc[ 67])
        ,  "+r"(acc[ 68]), "+r"(acc[ 69]), "+r"(acc[ 70]), "+r"(acc[ 71])
        ,  "+r"(acc[ 72]), "+r"(acc[ 73]), "+r"(acc[ 74]), "+r"(acc[ 75])
        ,  "+r"(acc[ 76]), "+r"(acc[ 77]), "+r"(acc[ 78]), "+r"(acc[ 79])
        ,  "+r"(acc[ 80]), "+r"(acc[ 81]), "+r"(acc[ 82]), "+r"(acc[ 83])
        ,  "+r"(acc[ 84]), "+r"(acc[ 85]), "+r"(acc[ 86]), "+r"(acc[ 87])
        ,  "+r"(acc[ 88]), "+r"(acc[ 89]), "+r"(acc[ 90]), "+r"(acc[ 91])
        ,  "+r"(acc[ 92]), "+r"(acc[ 93]), "+r"(acc[ 94]), "+r"(acc[ 95])
        ,  "+r"(acc[ 96]), "+r"(acc[ 97]), "+r"(acc[ 98]), "+r"(acc[ 99])
        ,  "+r"(acc[100]), "+r"(acc[101]), "+r"(acc[102]), "+r"(acc[103])
        ,  "+r"(acc[104]), "+r"(acc[105]), "+r"(acc[106]), "+r"(acc[107])
        ,  "+r"(acc[108]), "+r"(acc[109]), "+r"(acc[110]), "+r"(acc[111])
        ,  "+r"(acc[112]), "+r"(acc[113]), "+r"(acc[114]), "+r"(acc[115])
        ,  "+r"(acc[116]), "+r"(acc[117]), "+r"(acc[118]), "+r"(acc[119])
        ,  "+r"(acc[120]), "+r"(acc[121]), "+r"(acc[122]), "+r"(acc[123])
        ,  "+r"(acc[124]), "+r"(acc[125]), "+r"(acc[126]), "+r"(acc[127])
        :  "l"(desc_a)
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// GMMAs with fp16 Aclwmulator, where A is coming from RF
//
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64xNx16 TN with fp16 Aclwmulator
template<
  bool transb, // GMMA expects data to be in TN format. if B is row major, transb should be set 
  int GMMA_N,
  bool INCREMENT_SCORE_BOARD // whether or not the scoreboard will be incremented
>
inline __device__
void hgmma_rfa_fp16(const uint32_t (&a)[4], 
                    const uint64_t &desc_b,
                    uint32_t (&acc)[GMMA_N / 4]) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x8x16
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x8x16 TN with fp16 Aclwmulator, where scoreboard is not incremented
template<>
inline __device__
void hgmma_rfa_fp16<false, 8, false>(const uint32_t (&a)[4], 
                                     const uint64_t &desc_b,
                                     uint32_t (&acc)[2]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n8k16.f16.f16.f16.f16 \n"
        "{%0, %1}, \n"
        "{%2, %3, %4, %5}, \n" 
        "%6, \n"
        "{%0, %1}; \n"
        "}\n"
        :  "+r"(acc[0]), "+r"(acc[1])
        :  "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3])
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x8x16 TN with fp16 Aclwmulator, where scoreboard is incremented
template<>
inline __device__
void hgmma_rfa_fp16<false, 8, true>(const uint32_t (&a)[4], 
                                    const uint64_t &desc_b,
                                    uint32_t (&acc)[2]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n8k16.f16.f16.f16.f16 \n"
        "{%0, %1}, \n"
        "{%2, %3, %4, %5}, \n" 
        "%6, \n"
        "{%0, %1}, \n"
        "0; \n"
        "}\n"
        :  "+r"(acc[0]), "+r"(acc[1])
        :  "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3])
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x128x16
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x128x16 TN with fp16 Aclwmulator, where scoreboard is not incremented
template<>
inline __device__
void hgmma_rfa_fp16<false, 128, false>(const uint32_t (&a)[4],
                                       const uint64_t &desc_b,
                                       uint32_t (&acc)[32]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n128k16.f16.f16.f16.f16 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31},\n"
        "{%32, %33, %34, %35}, \n"  
        "%36, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31};\n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        :  "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3])
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x128x16 TN with fp16 Aclwmulator, where scoreboard is incremented
template<>
inline __device__
void hgmma_rfa_fp16<false, 128, true>(const uint32_t (&a)[4],
                                      const uint64_t &desc_b,
                                      uint32_t (&acc)[32]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n128k16.f16.f16.f16.f16 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31},\n"
        "{%32, %33, %34, %35}, \n"  
        "%36, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31},\n"
        "0; \n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        :  "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3])
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x64x16
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x64x16 TN with fp16 Aclwmulator, where scoreboard is not incremented
template<>
inline __device__
void hgmma_rfa_fp16<false, 64, false>(const uint32_t (&a)[4],
                                      const uint64_t &desc_b,
                                      uint32_t (&acc)[16]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n64k16.f16.f16.f16.f16 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15},\n"
        "{%16, %17, %18, %19}, \n"  
        "%20, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15};\n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        :  "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3])
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x64x16 TN with fp16 Aclwmulator, where scoreboard is incremented
template<>
inline __device__
void hgmma_rfa_fp16<false, 64, true>(const uint32_t (&a)[4],
                                     const uint64_t &desc_b,
                                     uint32_t (&acc)[16]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n64k16.f16.f16.f16.f16 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15},\n"
        "{%16, %17, %18, %19}, \n"  
        "%20, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15}, \n"
        "0; \n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        :  "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3])
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x256x16
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x256x16 TN with fp16 Aclwmulator, where scoreboard is not incremented
template<>
inline __device__
void hgmma_rfa_fp16<false, 256, false>(const uint32_t (&a)[4],
                                       const uint64_t &desc_b,
                                       uint32_t (&acc)[64]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n256k16.f16.f16.f16.f16 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63},\n"
        "{%64, %65, %66, %67}, \n"
        "%68, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63};\n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        ,  "+r"(acc[32]), "+r"(acc[33]), "+r"(acc[34]), "+r"(acc[35])
        ,  "+r"(acc[36]), "+r"(acc[37]), "+r"(acc[38]), "+r"(acc[39])
        ,  "+r"(acc[40]), "+r"(acc[41]), "+r"(acc[42]), "+r"(acc[43])
        ,  "+r"(acc[44]), "+r"(acc[45]), "+r"(acc[46]), "+r"(acc[47])
        ,  "+r"(acc[48]), "+r"(acc[49]), "+r"(acc[50]), "+r"(acc[51])
        ,  "+r"(acc[52]), "+r"(acc[53]), "+r"(acc[54]), "+r"(acc[55])
        ,  "+r"(acc[56]), "+r"(acc[57]), "+r"(acc[58]), "+r"(acc[59])
        ,  "+r"(acc[60]), "+r"(acc[61]), "+r"(acc[62]), "+r"(acc[63])
        :  "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3])
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x256x16 TN with fp16 Aclwmulator, where scoreboard is incremented
template<>
inline __device__
void hgmma_rfa_fp16<false, 256, true>(const uint32_t (&a)[4],
                                      const uint64_t &desc_b,
                                      uint32_t (&acc)[64]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n256k16.f16.f16.f16.f16 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63},\n"
        "{%64, %65, %66, %67}, \n"
        "%68, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63},\n"
        "0;"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        ,  "+r"(acc[32]), "+r"(acc[33]), "+r"(acc[34]), "+r"(acc[35])
        ,  "+r"(acc[36]), "+r"(acc[37]), "+r"(acc[38]), "+r"(acc[39])
        ,  "+r"(acc[40]), "+r"(acc[41]), "+r"(acc[42]), "+r"(acc[43])
        ,  "+r"(acc[44]), "+r"(acc[45]), "+r"(acc[46]), "+r"(acc[47])
        ,  "+r"(acc[48]), "+r"(acc[49]), "+r"(acc[50]), "+r"(acc[51])
        ,  "+r"(acc[52]), "+r"(acc[53]), "+r"(acc[54]), "+r"(acc[55])
        ,  "+r"(acc[56]), "+r"(acc[57]), "+r"(acc[58]), "+r"(acc[59])
        ,  "+r"(acc[60]), "+r"(acc[61]), "+r"(acc[62]), "+r"(acc[63])
        :  "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3])
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// GMMAs with fp32 Aclwmulator, where A is coming from RF
//
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64xNx16 TN with fp32 Aclwmulator
template<
  bool transb, // GMMA expects data to be in TN format. if B is row major, transb should be set 
  int GMMA_N,
  bool INCREMENT_SCORE_BOARD // whether or not the scoreboard will be incremented
>
inline __device__
void hgmma_rfa_fp32(const uint32_t (&a)[4], 
                    const uint64_t &desc_b,
                    uint32_t (&acc)[GMMA_N / 2]) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x8x16
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x8x16 TN with fp32 Aclwmulator, where scoreboard is not incremented
template<>
inline __device__
void hgmma_rfa_fp32<false, 8, false>(const uint32_t (&a)[4], 
                   const uint64_t &desc_b,
                   uint32_t (&acc)[4]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n8k16.f32.f16.f16.f32 \n"
        "{%0, %1, %2, %3}, \n"
        "{%4, %5, %6, %7}, \n" 
        "%8, \n"
        "{%0, %1, %2, %3}; \n"
        "}\n"
        :  "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3])
        :  "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3])
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x8x16 TN with fp32 Aclwmulator, where scoreboard is incremented
template<>
inline __device__
void hgmma_rfa_fp32<false, 8, true>(const uint32_t (&a)[4], 
                   const uint64_t &desc_b,
                   uint32_t (&acc)[4]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n8k16.f32.f16.f16.f32 \n"
        "{%0, %1, %2, %3}, \n"
        "{%4, %5, %6, %7}, \n" 
        "%8, \n"
        "{%0, %1, %2, %3}, \n"
        "0; \n"
        "}\n"
        :  "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3])
        :  "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3])
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x128x16
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x128x16 TN with fp32 Aclwmulator, where scoreboard is not incremented
template<>
inline __device__
void hgmma_rfa_fp32<false, 128, false>(const uint32_t (&a)[4],
                   const uint64_t &desc_b,
                   uint32_t (&acc)[64]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n128k16.f32.f16.f16.f32 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63},\n"
        "{%64, %65, %66, %67}, \n"  
        "%68, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63};\n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        ,  "+r"(acc[32]), "+r"(acc[33]), "+r"(acc[34]), "+r"(acc[35])
        ,  "+r"(acc[36]), "+r"(acc[37]), "+r"(acc[38]), "+r"(acc[39])
        ,  "+r"(acc[40]), "+r"(acc[41]), "+r"(acc[42]), "+r"(acc[43])
        ,  "+r"(acc[44]), "+r"(acc[45]), "+r"(acc[46]), "+r"(acc[47])
        ,  "+r"(acc[48]), "+r"(acc[49]), "+r"(acc[50]), "+r"(acc[51])
        ,  "+r"(acc[52]), "+r"(acc[53]), "+r"(acc[54]), "+r"(acc[55])
        ,  "+r"(acc[56]), "+r"(acc[57]), "+r"(acc[58]), "+r"(acc[59])
        ,  "+r"(acc[60]), "+r"(acc[61]), "+r"(acc[62]), "+r"(acc[63])
        :  "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3])
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x128x16 TN with fp32 Aclwmulator, where scoreboard is incremented
template<>
inline __device__
void hgmma_rfa_fp32<false, 128, true>(const uint32_t (&a)[4],
                   const uint64_t &desc_b,
                   uint32_t (&acc)[64]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n128k16.f32.f16.f16.f32 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63},\n"
        "{%64, %65, %66, %67}, \n"  
        "%68, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63},\n"
        "0; \n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        ,  "+r"(acc[32]), "+r"(acc[33]), "+r"(acc[34]), "+r"(acc[35])
        ,  "+r"(acc[36]), "+r"(acc[37]), "+r"(acc[38]), "+r"(acc[39])
        ,  "+r"(acc[40]), "+r"(acc[41]), "+r"(acc[42]), "+r"(acc[43])
        ,  "+r"(acc[44]), "+r"(acc[45]), "+r"(acc[46]), "+r"(acc[47])
        ,  "+r"(acc[48]), "+r"(acc[49]), "+r"(acc[50]), "+r"(acc[51])
        ,  "+r"(acc[52]), "+r"(acc[53]), "+r"(acc[54]), "+r"(acc[55])
        ,  "+r"(acc[56]), "+r"(acc[57]), "+r"(acc[58]), "+r"(acc[59])
        ,  "+r"(acc[60]), "+r"(acc[61]), "+r"(acc[62]), "+r"(acc[63])
        :  "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3])
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x64x16
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x64x16 TN with fp32 Aclwmulator, where scoreboard is not incremented
template<>
inline __device__
void hgmma_rfa_fp32<false, 64, false>(const uint32_t (&a)[4],
                   const uint64_t &desc_b,
                   uint32_t (&acc)[32]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n64k16.f32.f16.f16.f32 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31},\n"
        "{%32, %33, %34, %35}, \n"  
        "%36, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31};\n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        :  "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3])
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x64x16 TN with fp32 Aclwmulator, where scoreboard is incremented
template<>
inline __device__
void hgmma_rfa_fp32<false, 64, true>(const uint32_t (&a)[4],
                   const uint64_t &desc_b,
                   uint32_t (&acc)[32]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n64k16.f32.f16.f16.f32 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31},\n"
        "{%32, %33, %34, %35}, \n"  
        "%36, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31},\n"
        "0; \n"
        "}\n"
        :  "+r"(acc[ 0]), "+r"(acc[ 1]), "+r"(acc[ 2]), "+r"(acc[ 3])
        ,  "+r"(acc[ 4]), "+r"(acc[ 5]), "+r"(acc[ 6]), "+r"(acc[ 7])
        ,  "+r"(acc[ 8]), "+r"(acc[ 9]), "+r"(acc[10]), "+r"(acc[11])
        ,  "+r"(acc[12]), "+r"(acc[13]), "+r"(acc[14]), "+r"(acc[15])
        ,  "+r"(acc[16]), "+r"(acc[17]), "+r"(acc[18]), "+r"(acc[19])
        ,  "+r"(acc[20]), "+r"(acc[21]), "+r"(acc[22]), "+r"(acc[23])
        ,  "+r"(acc[24]), "+r"(acc[25]), "+r"(acc[26]), "+r"(acc[27])
        ,  "+r"(acc[28]), "+r"(acc[29]), "+r"(acc[30]), "+r"(acc[31])
        :  "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3])
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// 64x256x16
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x256x16 TN with fp32 Aclwmulator, where scoreboard is not incremented
template<>
inline __device__
void hgmma_rfa_fp32<false, 256, false>(const uint32_t (&a)[4],
                   const uint64_t &desc_b,
                   uint32_t (&acc)[128]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n256k16.f32.f16.f16.f32 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63,\n"
        " %64, %65, %66, %67, %68, %69, %70, %71,\n"
        " %72, %73, %74, %75, %76, %77, %78, %79,\n"
        " %80, %81, %82, %83, %84, %85, %86, %87,\n"
        " %88, %89, %90, %91, %92, %93, %94, %95,\n"
        " %96, %97, %98, %99, %100, %101, %102, %103,\n"
        " %104, %105, %106, %107, %108, %109, %110, %111,\n"
        " %112, %113, %114, %115, %116, %117, %118, %119,\n"
        " %120, %121, %122, %123, %124, %125, %126, %127},\n"
        "{%128, %129, %130, %131}, \n"
        "%132, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63,\n"
        " %64, %65, %66, %67, %68, %69, %70, %71,\n"
        " %72, %73, %74, %75, %76, %77, %78, %79,\n"
        " %80, %81, %82, %83, %84, %85, %86, %87,\n"
        " %88, %89, %90, %91, %92, %93, %94, %95,\n"
        " %96, %97, %98, %99, %100, %101, %102, %103,\n"
        " %104, %105, %106, %107, %108, %109, %110, %111,\n"
        " %112, %113, %114, %115, %116, %117, %118, %119,\n"
        " %120, %121, %122, %123, %124, %125, %126, %127};\n"
        "}\n"
        :  "+r"(acc[  0]), "+r"(acc[  1]), "+r"(acc[  2]), "+r"(acc[  3])
        ,  "+r"(acc[  4]), "+r"(acc[  5]), "+r"(acc[  6]), "+r"(acc[  7])
        ,  "+r"(acc[  8]), "+r"(acc[  9]), "+r"(acc[ 10]), "+r"(acc[ 11])
        ,  "+r"(acc[ 12]), "+r"(acc[ 13]), "+r"(acc[ 14]), "+r"(acc[ 15])
        ,  "+r"(acc[ 16]), "+r"(acc[ 17]), "+r"(acc[ 18]), "+r"(acc[ 19])
        ,  "+r"(acc[ 20]), "+r"(acc[ 21]), "+r"(acc[ 22]), "+r"(acc[ 23])
        ,  "+r"(acc[ 24]), "+r"(acc[ 25]), "+r"(acc[ 26]), "+r"(acc[ 27])
        ,  "+r"(acc[ 28]), "+r"(acc[ 29]), "+r"(acc[ 30]), "+r"(acc[ 31])
        ,  "+r"(acc[ 32]), "+r"(acc[ 33]), "+r"(acc[ 34]), "+r"(acc[ 35])
        ,  "+r"(acc[ 36]), "+r"(acc[ 37]), "+r"(acc[ 38]), "+r"(acc[ 39])
        ,  "+r"(acc[ 40]), "+r"(acc[ 41]), "+r"(acc[ 42]), "+r"(acc[ 43])
        ,  "+r"(acc[ 44]), "+r"(acc[ 45]), "+r"(acc[ 46]), "+r"(acc[ 47])
        ,  "+r"(acc[ 48]), "+r"(acc[ 49]), "+r"(acc[ 50]), "+r"(acc[ 51])
        ,  "+r"(acc[ 52]), "+r"(acc[ 53]), "+r"(acc[ 54]), "+r"(acc[ 55])
        ,  "+r"(acc[ 56]), "+r"(acc[ 57]), "+r"(acc[ 58]), "+r"(acc[ 59])
        ,  "+r"(acc[ 60]), "+r"(acc[ 61]), "+r"(acc[ 62]), "+r"(acc[ 63])
        ,  "+r"(acc[ 64]), "+r"(acc[ 65]), "+r"(acc[ 66]), "+r"(acc[ 67])
        ,  "+r"(acc[ 68]), "+r"(acc[ 69]), "+r"(acc[ 70]), "+r"(acc[ 71])
        ,  "+r"(acc[ 72]), "+r"(acc[ 73]), "+r"(acc[ 74]), "+r"(acc[ 75])
        ,  "+r"(acc[ 76]), "+r"(acc[ 77]), "+r"(acc[ 78]), "+r"(acc[ 79])
        ,  "+r"(acc[ 80]), "+r"(acc[ 81]), "+r"(acc[ 82]), "+r"(acc[ 83])
        ,  "+r"(acc[ 84]), "+r"(acc[ 85]), "+r"(acc[ 86]), "+r"(acc[ 87])
        ,  "+r"(acc[ 88]), "+r"(acc[ 89]), "+r"(acc[ 90]), "+r"(acc[ 91])
        ,  "+r"(acc[ 92]), "+r"(acc[ 93]), "+r"(acc[ 94]), "+r"(acc[ 95])
        ,  "+r"(acc[ 96]), "+r"(acc[ 97]), "+r"(acc[ 98]), "+r"(acc[ 99])
        ,  "+r"(acc[100]), "+r"(acc[101]), "+r"(acc[102]), "+r"(acc[103])
        ,  "+r"(acc[104]), "+r"(acc[105]), "+r"(acc[106]), "+r"(acc[107])
        ,  "+r"(acc[108]), "+r"(acc[109]), "+r"(acc[110]), "+r"(acc[111])
        ,  "+r"(acc[112]), "+r"(acc[113]), "+r"(acc[114]), "+r"(acc[115])
        ,  "+r"(acc[116]), "+r"(acc[117]), "+r"(acc[118]), "+r"(acc[119])
        ,  "+r"(acc[120]), "+r"(acc[121]), "+r"(acc[122]), "+r"(acc[123])
        ,  "+r"(acc[124]), "+r"(acc[125]), "+r"(acc[126]), "+r"(acc[127])
        :  "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3])
        ,  "l"(desc_b));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// HGMMA 64x256x16 TN with fp32 Aclwmulator, where scoreboard is incremented
template<>
inline __device__
void hgmma_rfa_fp32<false, 256, true>(const uint32_t (&a)[4],
                   const uint64_t &desc_b,
                   uint32_t (&acc)[128]) {
#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
    asm volatile( \
        "{\n" \
        "_mma.warpgroup.m64n256k16.f32.f16.f16.f32 \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63,\n"
        " %64, %65, %66, %67, %68, %69, %70, %71,\n"
        " %72, %73, %74, %75, %76, %77, %78, %79,\n"
        " %80, %81, %82, %83, %84, %85, %86, %87,\n"
        " %88, %89, %90, %91, %92, %93, %94, %95,\n"
        " %96, %97, %98, %99, %100, %101, %102, %103,\n"
        " %104, %105, %106, %107, %108, %109, %110, %111,\n"
        " %112, %113, %114, %115, %116, %117, %118, %119,\n"
        " %120, %121, %122, %123, %124, %125, %126, %127},\n"
        "{%128, %129, %130, %131}, \n"
        "%132, \n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, \n"
        " %8, %9, %10, %11, %12, %13, %14, %15,\n"
        " %16, %17, %18, %19, %20, %21, %22, %23,\n"
        " %24, %25, %26, %27, %28, %29, %30, %31,\n"
        " %32, %33, %34, %35, %36, %37, %38, %39,\n"
        " %40, %41, %42, %43, %44, %45, %46, %47,\n"
        " %48, %49, %50, %51, %52, %53, %54, %55,\n"
        " %56, %57, %58, %59, %60, %61, %62, %63,\n"
        " %64, %65, %66, %67, %68, %69, %70, %71,\n"
        " %72, %73, %74, %75, %76, %77, %78, %79,\n"
        " %80, %81, %82, %83, %84, %85, %86, %87,\n"
        " %88, %89, %90, %91, %92, %93, %94, %95,\n"
        " %96, %97, %98, %99, %100, %101, %102, %103,\n"
        " %104, %105, %106, %107, %108, %109, %110, %111,\n"
        " %112, %113, %114, %115, %116, %117, %118, %119,\n"
        " %120, %121, %122, %123, %124, %125, %126, %127},\n"
        "0; \n"
        "}\n"
        :  "+r"(acc[  0]), "+r"(acc[  1]), "+r"(acc[  2]), "+r"(acc[  3])
        ,  "+r"(acc[  4]), "+r"(acc[  5]), "+r"(acc[  6]), "+r"(acc[  7])
        ,  "+r"(acc[  8]), "+r"(acc[  9]), "+r"(acc[ 10]), "+r"(acc[ 11])
        ,  "+r"(acc[ 12]), "+r"(acc[ 13]), "+r"(acc[ 14]), "+r"(acc[ 15])
        ,  "+r"(acc[ 16]), "+r"(acc[ 17]), "+r"(acc[ 18]), "+r"(acc[ 19])
        ,  "+r"(acc[ 20]), "+r"(acc[ 21]), "+r"(acc[ 22]), "+r"(acc[ 23])
        ,  "+r"(acc[ 24]), "+r"(acc[ 25]), "+r"(acc[ 26]), "+r"(acc[ 27])
        ,  "+r"(acc[ 28]), "+r"(acc[ 29]), "+r"(acc[ 30]), "+r"(acc[ 31])
        ,  "+r"(acc[ 32]), "+r"(acc[ 33]), "+r"(acc[ 34]), "+r"(acc[ 35])
        ,  "+r"(acc[ 36]), "+r"(acc[ 37]), "+r"(acc[ 38]), "+r"(acc[ 39])
        ,  "+r"(acc[ 40]), "+r"(acc[ 41]), "+r"(acc[ 42]), "+r"(acc[ 43])
        ,  "+r"(acc[ 44]), "+r"(acc[ 45]), "+r"(acc[ 46]), "+r"(acc[ 47])
        ,  "+r"(acc[ 48]), "+r"(acc[ 49]), "+r"(acc[ 50]), "+r"(acc[ 51])
        ,  "+r"(acc[ 52]), "+r"(acc[ 53]), "+r"(acc[ 54]), "+r"(acc[ 55])
        ,  "+r"(acc[ 56]), "+r"(acc[ 57]), "+r"(acc[ 58]), "+r"(acc[ 59])
        ,  "+r"(acc[ 60]), "+r"(acc[ 61]), "+r"(acc[ 62]), "+r"(acc[ 63])
        ,  "+r"(acc[ 64]), "+r"(acc[ 65]), "+r"(acc[ 66]), "+r"(acc[ 67])
        ,  "+r"(acc[ 68]), "+r"(acc[ 69]), "+r"(acc[ 70]), "+r"(acc[ 71])
        ,  "+r"(acc[ 72]), "+r"(acc[ 73]), "+r"(acc[ 74]), "+r"(acc[ 75])
        ,  "+r"(acc[ 76]), "+r"(acc[ 77]), "+r"(acc[ 78]), "+r"(acc[ 79])
        ,  "+r"(acc[ 80]), "+r"(acc[ 81]), "+r"(acc[ 82]), "+r"(acc[ 83])
        ,  "+r"(acc[ 84]), "+r"(acc[ 85]), "+r"(acc[ 86]), "+r"(acc[ 87])
        ,  "+r"(acc[ 88]), "+r"(acc[ 89]), "+r"(acc[ 90]), "+r"(acc[ 91])
        ,  "+r"(acc[ 92]), "+r"(acc[ 93]), "+r"(acc[ 94]), "+r"(acc[ 95])
        ,  "+r"(acc[ 96]), "+r"(acc[ 97]), "+r"(acc[ 98]), "+r"(acc[ 99])
        ,  "+r"(acc[100]), "+r"(acc[101]), "+r"(acc[102]), "+r"(acc[103])
        ,  "+r"(acc[104]), "+r"(acc[105]), "+r"(acc[106]), "+r"(acc[107])
        ,  "+r"(acc[108]), "+r"(acc[109]), "+r"(acc[110]), "+r"(acc[111])
        ,  "+r"(acc[112]), "+r"(acc[113]), "+r"(acc[114]), "+r"(acc[115])
        ,  "+r"(acc[116]), "+r"(acc[117]), "+r"(acc[118]), "+r"(acc[119])
        ,  "+r"(acc[120]), "+r"(acc[121]), "+r"(acc[122]), "+r"(acc[123])
        ,  "+r"(acc[124]), "+r"(acc[125]), "+r"(acc[126]), "+r"(acc[127])
        :  "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3])
        ,  "l"(desc_b));
#endif
}


////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace xmma
