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
// include instructions for HGMMA
#include <xmma/hopper/instructions_gmma.h>

namespace xmma {
////////////////////////////////////////////////////////////////////////////////////////////////////

//#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870
inline __device__ void warpgroup_arrive() {
#if defined( __LWDA_ARCH__ ) && __LWDA_ARCH__ >= 870
    asm volatile( "_warpgroup.arrive; \n" );
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N> inline __device__ void warpgroup_wait() {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <> inline __device__ void warpgroup_wait<0>() {
#if defined( __LWDA_ARCH__ ) && __LWDA_ARCH__ >= 870
    asm volatile( "_warpgroup.wait; \n" );
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <> inline __device__ void warpgroup_wait<1>() {
#if defined( __LWDA_ARCH__ ) && __LWDA_ARCH__ >= 870
    asm volatile( "_warpgroup.wait 1; \n" );
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <> inline __device__ void warpgroup_wait<2>() {
#if defined( __LWDA_ARCH__ ) && __LWDA_ARCH__ >= 870
    asm volatile( "_warpgroup.wait 2; \n" );
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <> inline __device__ void warpgroup_wait<3>() {
#if defined( __LWDA_ARCH__ ) && __LWDA_ARCH__ >= 870
    asm volatile( "_warpgroup.wait 3; \n" );
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <> inline __device__ void warpgroup_wait<4>() {
#if defined( __LWDA_ARCH__ ) && __LWDA_ARCH__ >= 870
    asm volatile( "_warpgroup.wait 4; \n" );
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <> inline __device__ void warpgroup_wait<5>() {
#if defined( __LWDA_ARCH__ ) && __LWDA_ARCH__ >= 870
    asm volatile( "_warpgroup.wait 5; \n" );
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <> inline __device__ void warpgroup_wait<6>() {
#if defined( __LWDA_ARCH__ ) && __LWDA_ARCH__ >= 870
    asm volatile( "_warpgroup.wait 6; \n" );
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <> inline __device__ void warpgroup_wait<7>() {
#if defined( __LWDA_ARCH__ ) && __LWDA_ARCH__ >= 870
    asm volatile( "_warpgroup.wait 7; \n" );
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <> inline __device__ void warpgroup_wait<8>() {
#if defined( __LWDA_ARCH__ ) && __LWDA_ARCH__ >= 870
    asm volatile( "_warpgroup.wait 8; \n" );
#endif
}

//#endif //#if defined(__LWDA_ARCH__) && __LWDA_ARCH__ >= 870

////////////////////////////////////////////////////////////////////////////////////////////////////
}  // namespace xmma
