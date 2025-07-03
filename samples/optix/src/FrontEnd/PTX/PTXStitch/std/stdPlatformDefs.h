/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2007-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifndef __stdPlatformDefs_h__
#define __stdPlatformDefs_h__

#ifndef STD_ARCH
 #if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
   #define STD_ARCH x86_64
   #define STD_ARCH_x86_64 STD_ARCH_x86_64
   #define STD_64_BIT_ARCH 1
 #elif defined(__aarch64__) || defined(_M_ARM64)
   #define STD_ARCH_aarch64 STD_ARCH_aarch64
   #define STD_ARCH aarch64
   #define STD_64_BIT_ARCH 1
 #elif defined(__powerpc64__) && defined(__LITTLE_ENDIAN__)
   #define STD_ARCH ppc64le
   #define STD_ARCH_ppc64le STD_ARCH_ppc64le
   #define STD_64_BIT_ARCH 1
 #elif defined(__arm__) || defined(_M_ARM)
   #define STD_ARCH ARMv7
   #define STD_ARCH_ARMv7 STD_ARCH_ARMv7
 #elif defined(_M_IX86) || defined(__i686__)
   #define STD_ARCH i686
   #define STD_ARCH_i686 STD_ARCH_i686
 #elif defined(__i386__)
   #define STD_ARCH i386
   #define STD_ARCH_i386 STD_ARCH_i386
 #endif
#endif

#ifndef STD_OS
 #if defined(_WIN32) || defined(_WIN16)
   #define STD_OS win32
   #define STD_OS_win32 STD_OS_win32
 // __ANDROID__ needs to be tested before __linux__. The toolchain used for
 // Android defines both.
 #elif defined(__ANDROID__)
   #define STD_OS Android
   #define STD_OS_Android STD_OS_Android
 #elif defined(__linux__)
   #define STD_OS Linux
   #define STD_OS_Linux STD_OS_Linux
 #elif defined(__QNX__) || defined(__QNXNTO__)
   #define STD_OS QNX
   #define STD_OS_QNX STD_OS_QNX
 #elif defined(__HORIZON__)
   #define STD_OS Hos
   #define STD_OS_Hos STD_OS_Hos
 #elif defined(__APPLE__)
   #define STD_OS Darwin
   #define STD_OS_Darwin STD_OS_Darwin
 #elif defined(__FreeBSD__)
   #define STD_OS FreeBSD
   #define STD_OS_FreeBSD STD_OS_FreeBSD
 #endif
#endif

// Note: STD_OS_FAMILY_Unix can't be defined in the conditional above because
// there are Makefiles that still pass -DSTD_OS to the compiler.
#if defined(STD_OS_Android) || defined(STD_OS_Hos) || defined(STD_OS_Linux) || defined(STD_OS_QNX) || defined(STD_OS_FreeBSD)
  #define STD_OS_FAMILY_Unix STD_OS_FAMILY_Unix
#endif

#endif // __stdPlatformDefs_h__
