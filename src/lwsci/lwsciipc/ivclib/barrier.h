// Copyright (c) 2019, LWPU CORPORATION.  All rights reserved.
//
// LWPU CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from LWPU CORPORATION is strictly prohibited.

#ifndef SIVCLIB_BARRIER_H
#define SIVCLIB_BARRIER_H

#ifdef __cplusplus
extern "C" {
#endif

// These are memory barrier instructions for x86-64, AArch32 and AArch64
// platforms.

#if defined(__x86_64__)

static inline void sivc_mb(void) {
    __asm__ __volatile__("mfence" ::: "memory");
}

static inline void sivc_rmb(void) {
    __asm__ __volatile__("lfence" ::: "memory");
}

static inline void sivc_wmb(void) {
    __asm__ __volatile__("sfence" ::: "memory");
}

#else  // !__x86_64__

static inline void sivc_mb(void) {
    __asm__ __volatile__("dmb sy" ::: "memory");
}

static inline void sivc_rmb(void) {
// GCC and ARM compiler define __aarch64__
// GHS compiler defines __ARM64__
#if defined(__aarch64__) || defined(__ARM64__)
    __asm__ __volatile__("dmb ld" ::: "memory");
#else
    __asm__ __volatile__("dmb" ::: "memory");
#endif
}

static inline void sivc_wmb(void) {
    __asm__ __volatile__("dmb st" ::: "memory");
}

#endif  // __x86_64__

#ifdef __cplusplus
} // extern "C"
#endif

#endif  // SIVCLIB_BARRIER_H
