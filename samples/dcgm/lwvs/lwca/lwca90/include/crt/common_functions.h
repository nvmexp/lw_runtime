/*
 * Copyright 1993-2017 LWPU Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to LWPU intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to LWPU and is being provided under the terms and
 * conditions of a form of LWPU software license agreement by and
 * between LWPU and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of LWPU is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, LWPU MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * LWPU DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL LWPU BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#if !defined(__COMMON_FUNCTIONS_H__)
#define __COMMON_FUNCTIONS_H__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__cplusplus) && defined(__LWDACC__)

#include "builtin_types.h"
#include "host_defines.h"

#define __LWDACC_VER__ "__LWDACC_VER__ is no longer supported.  Use __LWDACC_VER_MAJOR__, __LWDACC_VER_MINOR__, and __LWDACC_VER_BUILD__ instead."

#if !defined(__LWDACC_RTC__)
#include <string.h>
#include <time.h>

extern "C"
{
#endif /* !__LWDACC_RTC__ */
extern _CRTIMP __host__ __device__ __device_builtin__ __lwdart_builtin__ clock_t __cdecl clock(void)
#if defined(__QNX__)
asm("clock32")
#endif
__THROW;
extern         __host__ __device__ __device_builtin__ __lwdart_builtin__ void*   __cdecl memset(void*, int, size_t) __THROW;
extern         __host__ __device__ __device_builtin__ __lwdart_builtin__ void*   __cdecl memcpy(void*, const void*, size_t) __THROW;
#if !defined(__LWDACC_RTC__)
}
#endif /* !__LWDACC_RTC__ */

#if defined(__LWDA_ARCH__)

#if defined(__LWDACC_RTC__)
inline __host__ __device__ void* operator new(size_t, void *p) { return p; }
inline __host__ __device__ void* operator new[](size_t, void *p) { return p; }
inline __host__ __device__ void operator delete(void*, void*) { }
inline __host__ __device__ void operator delete[](void*, void*) { }
#else /* !__LWDACC_RTC__ */
#ifndef __LWDA_INTERNAL_SKIP_CPP_HEADERS__
#include <new>
#endif

#if defined (__GNUC__)

#define STD \
        std::
        
#else /* __GNUC__ */

#define STD

#endif /* __GNUC__ */

extern         __host__ __device__ __lwdart_builtin__ void*   __cdecl operator new(STD size_t, void*) throw();
extern         __host__ __device__ __lwdart_builtin__ void*   __cdecl operator new[](STD size_t, void*) throw();
extern         __host__ __device__ __lwdart_builtin__ void    __cdecl operator delete(void*, void*) throw();
extern         __host__ __device__ __lwdart_builtin__ void    __cdecl operator delete[](void*, void*) throw();
# if __cplusplus >= 201402L || (defined(_MSC_VER) && _MSC_VER >= 1900)
extern         __host__ __device__ __lwdart_builtin__ void    __cdecl operator delete(void*, STD size_t) throw();
extern         __host__ __device__ __lwdart_builtin__ void    __cdecl operator delete[](void*, STD size_t) throw();
#endif /* __cplusplus >= 201402L || (defined(_MSC_VER) && _MSC_VER >= 1900) */
#endif /* __LWDACC_RTC__ */

#if !defined(__LWDACC_RTC__)
#include <stdio.h>
#include <stdlib.h>
#endif /* !__LWDACC_RTC__ */

#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
namespace std {
#endif
extern "C"
{
extern
#if !defined(_MSC_VER) || _MSC_VER < 1900
_CRTIMP
#endif
__host__ __device__ __device_builtin__ __lwdart_builtin__ int     __cdecl printf(const char*, ...);
#if !defined(__LWDACC_RTC__)
extern
#if !defined(_MSC_VER) || _MSC_VER < 1900
_CRTIMP
#endif
__host__ __device__ __device_builtin__ __lwdart_builtin__ int     __cdecl fprintf(FILE*, const char*, ...);
#endif /* !__LWDACC_RTC__ */

extern _CRTIMP __host__ __device__ __lwdart_builtin__ void*   __cdecl malloc(size_t) __THROW;
extern _CRTIMP __host__ __device__ __lwdart_builtin__ void    __cdecl free(void*) __THROW;

}
#if defined(__QNX__) && !defined(_LIBCPP_VERSION)
} /* std */
#endif

#if !defined(__LWDACC_RTC__)
#include <assert.h>
#endif /* !__LWDACC_RTC__ */

extern "C"
{
#if defined(__LWDACC_RTC__)
extern __host__ __device__ void __assertfail(const char * __assertion, 
                                             const char *__file,
                                             unsigned int __line,
                                             const char *__function,
                                             size_t charsize);
#elif defined(__APPLE__)
#define __builtin_expect(exp,c) (exp)
extern __host__ __device__ __lwdart_builtin__ void __assert_rtn(
  const char *, const char *, int, const char *);
#elif defined(__ANDROID__)
extern __host__ __device__ __lwdart_builtin__ void __assert2(
  const char *, int, const char *, const char *);
#elif defined(__QNX__)
#if !defined(_LIBCPP_VERSION)
namespace std {
#endif
extern __host__ __device__ __lwdart_builtin__ void __assert(
  const char *, const char *, unsigned int, const char *);
#if !defined(_LIBCPP_VERSION)
}
#endif
#elif defined(__HORIZON__)
extern __host__ __device__ __lwdart_builtin__ void __assert_fail(
  const char *, const char *, int, const char *);
#elif defined(__GNUC__)
extern __host__ __device__ __lwdart_builtin__ void __assert_fail(
  const char *, const char *, unsigned int, const char *)
  __THROW; 
#elif defined(_WIN32)
extern __host__ __device__ __lwdart_builtin__ _CRTIMP void __cdecl _wassert(
  const wchar_t *, const wchar_t *, unsigned);
#endif
}

#if defined(__LWDACC_RTC__)
#ifdef NDEBUG
#define assert(e) (static_cast<void>(0))
#else /* !NDEBUG */
#define __ASSERT_STR_HELPER(x) #x
#define assert(e) ((e) ? static_cast<void>(0)\
                       : __assertfail(__ASSERT_STR_HELPER(e), __FILE__,\
                                      __LINE__, __PRETTY_FUNCTION__,\
                                      sizeof(char)))
#endif /* NDEBUG */
inline  __host__ __device__  void* operator new(size_t in) {  return malloc(in); }
inline  __host__ __device__  void* operator new[](size_t in) { return malloc(in); }
inline __host__ __device__  void operator delete(void* in) { return free(in); }
inline __host__ __device__  void operator delete[](void* in) {  return free(in); }
# if __cplusplus >= 201402L
inline __host__ __device__  void operator delete(void* in, size_t) { return free(in); }
inline __host__ __device__  void operator delete[](void* in, size_t) {  return free(in); }
#endif /* __cplusplus >= 201402L */
#else /* !__LWDACC_RTC__ */
#if defined (__GNUC__)

#define __LW_GLIBCXX_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) 

# if (__cplusplus >= 201103L)  && !(defined(__QNX__) && defined(_LIBCPP_VERSION))
#define THROWBADALLOC 
#else
#if defined(__ANDROID__) && (defined(__BIONIC__) || __LW_GLIBCXX_VERSION < 40900)
#define THROWBADALLOC
#else
#define THROWBADALLOC  throw(STD bad_alloc)
#endif
#endif
#define __DELETE_THROW throw()

#undef __LW_GLIBCXX_VERSION

#else /* __GNUC__ */

#define THROWBADALLOC  throw(...)

#endif /* __GNUC__ */

extern         __host__ __device__ __lwdart_builtin__ void*   __cdecl operator new(STD size_t) THROWBADALLOC;
extern         __host__ __device__ __lwdart_builtin__ void*   __cdecl operator new[](STD size_t) THROWBADALLOC;
extern         __host__ __device__ __lwdart_builtin__ void    __cdecl operator delete(void*) throw();
extern         __host__ __device__ __lwdart_builtin__ void    __cdecl operator delete[](void*) throw();
# if __cplusplus >= 201402L || (defined(_MSC_VER) && _MSC_VER >= 1900)
extern         __host__ __device__ __lwdart_builtin__ void    __cdecl operator delete(void*, STD size_t) throw();
extern         __host__ __device__ __lwdart_builtin__ void    __cdecl operator delete[](void*, STD size_t) throw();
#endif /* __cplusplus >= 201402L || (defined(_MSC_VER) && _MSC_VER >= 1900) */

#undef THROWBADALLOC
#undef STD
#endif /* __LWDACC_RTC__ */

#endif /* __LWDA_ARCH__ */

#endif /* __cplusplus && __LWDACC__ */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__LWDACC_RTC__) && (__LWDA_ARCH__ >= 350)
#include "lwda_device_runtime_api.h"
#endif

#include "math_functions.h"

#endif /* !__COMMON_FUNCTIONS_H__ */

