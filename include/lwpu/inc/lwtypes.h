/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2020 LWPU CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef LWTYPES_INCLUDED
#define LWTYPES_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

#include "cpuopsys.h"

#ifndef LWTYPES_USE_STDINT
#define LWTYPES_USE_STDINT 0
#endif

#if LWTYPES_USE_STDINT
#ifdef __cplusplus
#include <cstdint>
#include <cinttypes>
#else
#include <stdint.h>
#include <inttypes.h>
#endif // __cplusplus
#endif // LWTYPES_USE_STDINT

#ifndef __cplusplus
// Header includes to make sure wchar_t is defined for C-file compilation
// (C++ is not affected as it is a fundamental type there)
// _MSC_VER is a hack to avoid  failures for old setup of UEFI builds which are
//  lwrrently set to msvc100 but do not properly set the include paths
#if defined(LW_WINDOWS) && (!defined(_MSC_VER) || (_MSC_VER > 1600))
#include <stddef.h>
#define LW_HAS_WCHAR_T_TYPEDEF 1
#endif
#endif // __cplusplus

 /***************************************************************************\
|*                                 Typedefs                                  *|
 \***************************************************************************/

#ifdef LW_MISRA_COMPLIANCE_REQUIRED
//Typedefs for MISRA COMPLIANCE
typedef unsigned long long   UInt64;
typedef   signed long long    Int64;
typedef unsigned int         UInt32;
typedef   signed int          Int32;
typedef unsigned short       UInt16;
typedef   signed short        Int16;
typedef unsigned char        UInt8 ;
typedef   signed char         Int8 ;

typedef          void          Void;
typedef          float    float32_t;
typedef          double   float64_t;
#endif


// Floating point types
#ifdef LW_MISRA_COMPLIANCE_REQUIRED
typedef float32_t          LwF32; /* IEEE Single Precision (S1E8M23)         */
typedef float64_t          LwF64; /* IEEE Double Precision (S1E11M52)        */
#else
typedef float              LwF32; /* IEEE Single Precision (S1E8M23)         */
typedef double             LwF64; /* IEEE Double Precision (S1E11M52)        */
#endif


// 8-bit: 'char' is the only 8-bit in the C89 standard and after.
#if LWTYPES_USE_STDINT
typedef uint8_t            LwV8; /* "void": enumerated or multiple fields    */
typedef uint8_t            LwU8; /* 0 to 255                                 */
typedef  int8_t            LwS8; /* -128 to 127                              */
#else
#ifdef LW_MISRA_COMPLIANCE_REQUIRED
typedef UInt8              LwV8; /* "void": enumerated or multiple fields    */
typedef UInt8              LwU8; /* 0 to 255                                 */
typedef  Int8              LwS8; /* -128 to 127                              */
#else
typedef unsigned char      LwV8; /* "void": enumerated or multiple fields    */
typedef unsigned char      LwU8; /* 0 to 255                                 */
typedef   signed char      LwS8; /* -128 to 127                              */
#endif
#endif // LWTYPES_USE_STDINT


#if LWTYPES_USE_STDINT
typedef uint16_t           LwV16; /* "void": enumerated or multiple fields   */
typedef uint16_t           LwU16; /* 0 to 65535                              */
typedef  int16_t           LwS16; /* -32768 to 32767                         */
#else
// 16-bit: If the compiler tells us what we can use, then use it.
#ifdef __INT16_TYPE__
typedef unsigned __INT16_TYPE__ LwV16; /* "void": enumerated or multiple fields */
typedef unsigned __INT16_TYPE__ LwU16; /* 0 to 65535                            */
typedef   signed __INT16_TYPE__ LwS16; /* -32768 to 32767                       */

// The minimal standard for C89 and after
#else       // __INT16_TYPE__
#ifdef LW_MISRA_COMPLIANCE_REQUIRED
typedef UInt16             LwV16; /* "void": enumerated or multiple fields   */
typedef UInt16             LwU16; /* 0 to 65535                              */
typedef  Int16             LwS16; /* -32768 to 32767                         */
#else
typedef unsigned short     LwV16; /* "void": enumerated or multiple fields   */
typedef unsigned short     LwU16; /* 0 to 65535                              */
typedef   signed short     LwS16; /* -32768 to 32767                         */
#endif
#endif      // __INT16_TYPE__
#endif // LWTYPES_USE_STDINT

// wchar type (fixed size types consistent across Linux/Windows boundaries)
#if defined(LW_HAS_WCHAR_T_TYPEDEF)
 typedef wchar_t LwWchar;
#else
 typedef LwV16   LwWchar;
#endif

// Macro to build an LwU32 from four bytes, listed from msb to lsb
#define LwU32_BUILD(a, b, c, d) (((a) << 24) | ((b) << 16) | ((c) << 8) | (d))

#if LWTYPES_USE_STDINT
typedef uint32_t           LwV32; /* "void": enumerated or multiple fields   */
typedef uint32_t           LwU32; /* 0 to 4294967295                         */
typedef  int32_t           LwS32; /* -2147483648 to 2147483647               */
#else
// 32-bit: If the compiler tells us what we can use, then use it.
#ifdef __INT32_TYPE__
typedef unsigned __INT32_TYPE__ LwV32; /* "void": enumerated or multiple fields */
typedef unsigned __INT32_TYPE__ LwU32; /* 0 to 4294967295                       */
typedef   signed __INT32_TYPE__ LwS32; /* -2147483648 to 2147483647             */

// Older compilers
#else       // __INT32_TYPE__

// For historical reasons, LwU32/LwV32 are defined to different base intrinsic
// types than LwS32 on some platforms.
// Mainly for 64-bit linux, where long is 64 bits and win9x, where int is 16 bit.
#if (defined(LW_UNIX) || defined(vxworks) || defined(LW_WINDOWS_CE) ||  \
     defined(__arm) || defined(__IAR_SYSTEMS_ICC__) || defined(LW_QNX) || \
     defined(LW_INTEGRITY) || defined(LW_MODS) || \
     defined(__GNUC__) || defined(__clang__) || defined(LW_MACINTOSH_64)) && \
    (!defined(LW_MACINTOSH) || defined(LW_MACINTOSH_64))
#ifdef LW_MISRA_COMPLIANCE_REQUIRED
typedef UInt32             LwV32; /* "void": enumerated or multiple fields   */
typedef UInt32             LwU32; /* 0 to 4294967295                         */
#else
typedef unsigned int       LwV32; /* "void": enumerated or multiple fields   */
typedef unsigned int       LwU32; /* 0 to 4294967295                         */
#endif

// The minimal standard for C89 and after
#else       // (defined(LW_UNIX) || defined(vxworks) || ...
typedef unsigned long      LwV32; /* "void": enumerated or multiple fields   */
typedef unsigned long      LwU32; /* 0 to 4294967295                         */
#endif      // (defined(LW_UNIX) || defined(vxworks) || ...

// Mac OS 32-bit still needs this
#if defined(LW_MACINTOSH) && !defined(LW_MACINTOSH_64)
typedef   signed long      LwS32; /* -2147483648 to 2147483647               */
#else
#ifdef LW_MISRA_COMPLIANCE_REQUIRED
typedef   Int32            LwS32; /* -2147483648 to 2147483647               */
#else
typedef   signed int       LwS32; /* -2147483648 to 2147483647               */
#endif
#endif      // defined(LW_MACINTOSH) && !defined(LW_MACINTOSH_64)
#endif      // __INT32_TYPE__
#endif // LWTYPES_USE_STDINT



#if LWTYPES_USE_STDINT
typedef uint64_t           LwU64; /* 0 to 18446744073709551615                      */
typedef  int64_t           LwS64; /* -9223372036854775808 to 9223372036854775807    */

#define LwU64_fmtX PRIX64
#define LwU64_fmtx PRIx64
#define LwU64_fmtu PRIu64
#define LwU64_fmto PRIo64
#define LwS64_fmtd PRId64
#define LwS64_fmti PRIi64
#else
// 64-bit types for compilers that support them, plus some obsolete variants
#if defined(__GNUC__) || defined(__clang__) || defined(__arm) || \
    defined(__IAR_SYSTEMS_ICC__) || defined(__ghs__) || defined(_WIN64) || \
    defined(__SUNPRO_C) || defined(__SUNPRO_CC) || defined (__xlC__)
#ifdef LW_MISRA_COMPLIANCE_REQUIRED
typedef UInt64             LwU64; /* 0 to 18446744073709551615                      */
typedef  Int64             LwS64; /* -9223372036854775808 to 9223372036854775807    */
#else
typedef unsigned long long LwU64; /* 0 to 18446744073709551615                      */
typedef          long long LwS64; /* -9223372036854775808 to 9223372036854775807    */
#endif

#define LwU64_fmtX "llX"
#define LwU64_fmtx "llx"
#define LwU64_fmtu "llu"
#define LwU64_fmto "llo"
#define LwS64_fmtd "lld"
#define LwS64_fmti "lli"

// Microsoft since 2003 -- https://msdn.microsoft.com/en-us/library/29dh1w7z.aspx
#else
typedef unsigned __int64   LwU64; /* 0 to 18446744073709551615                      */
typedef          __int64   LwS64; /* -9223372036854775808 to 9223372036854775807    */

#define LwU64_fmtX "I64X"
#define LwU64_fmtx "I64x"
#define LwU64_fmtu "I64u"
#define LwU64_fmto "I64o"
#define LwS64_fmtd "I64d"
#define LwS64_fmti "I64i"

#endif
#endif // LWTYPES_USE_STDINT

#ifdef LW_TYPESAFE_HANDLES
/*
 * Can't use opaque pointer as clients might be compiled with mismatched
 * pointer sizes. TYPESAFE check will eventually be removed once all clients
 * have transistioned safely to LwHandle.
 * The plan is to then eventually scale up the handle to be 64-bits.
 */
typedef struct
{
    LwU32 val;
} LwHandle;
#else
/*
 * For compatibility with modules that haven't moved typesafe handles.
 */
typedef LwU32 LwHandle;
#endif // LW_TYPESAFE_HANDLES

/* Boolean type */
typedef LwU8 LwBool;
#define LW_TRUE           ((LwBool)(0 == 0))
#define LW_FALSE          ((LwBool)(0 != 0))

/* Tristate type: LW_TRISTATE_FALSE, LW_TRISTATE_TRUE, LW_TRISTATE_INDETERMINATE */
typedef LwU8 LwTristate;
#define LW_TRISTATE_FALSE           ((LwTristate) 0)
#define LW_TRISTATE_TRUE            ((LwTristate) 1)
#define LW_TRISTATE_INDETERMINATE   ((LwTristate) 2)

/* Macros to extract the low and high parts of a 64-bit unsigned integer */
/* Also designed to work if someone happens to pass in a 32-bit integer */
#ifdef LW_MISRA_COMPLIANCE_REQUIRED
#define LwU64_HI32(n)     ((LwU32)((((LwU64)(n)) >> 32) & 0xffffffffU))
#define LwU64_LO32(n)     ((LwU32)(( (LwU64)(n))        & 0xffffffffU))
#else
#define LwU64_HI32(n)     ((LwU32)((((LwU64)(n)) >> 32) & 0xffffffff))
#define LwU64_LO32(n)     ((LwU32)(( (LwU64)(n))        & 0xffffffff))
#endif
#define LwU40_HI32(n)     ((LwU32)((((LwU64)(n)) >>  8) & 0xffffffffU))
#define LwU40_HI24of32(n) ((LwU32)(  (LwU64)(n)         & 0xffffff00U))

/* Macros to get the MSB and LSB of a 32 bit unsigned number */
#define LwU32_HI16(n)     ((LwU16)((((LwU32)(n)) >> 16) & 0xffffU))
#define LwU32_LO16(n)     ((LwU16)(( (LwU32)(n))        & 0xffffU))

 /***************************************************************************\
|*                                                                           *|
|*  64 bit type definitions for use in interface structures.                 *|
|*                                                                           *|
 \***************************************************************************/

#if defined(LW_64_BITS)

typedef void*              LwP64; /* 64 bit void pointer                     */
typedef LwU64             LwUPtr; /* pointer sized unsigned int              */
typedef LwS64             LwSPtr; /* pointer sized signed int                */
typedef LwU64           LwLength; /* length to agree with sizeof             */

#define LwP64_VALUE(n)        (n)
#define LwP64_fmt "%p"

#define KERNEL_POINTER_FROM_LwP64(p,v) ((p)(v))
#define LwP64_PLUS_OFFSET(p,o) (LwP64)((LwU64)(p) + (LwU64)(o))

#define LwUPtr_fmtX LwU64_fmtX
#define LwUPtr_fmtx LwU64_fmtx
#define LwUPtr_fmtu LwU64_fmtu
#define LwUPtr_fmto LwU64_fmto
#define LwSPtr_fmtd LwS64_fmtd
#define LwSPtr_fmti LwS64_fmti

#else

typedef LwU64              LwP64; /* 64 bit void pointer                     */
typedef LwU32             LwUPtr; /* pointer sized unsigned int              */
typedef LwS32             LwSPtr; /* pointer sized signed int                */
typedef LwU32           LwLength; /* length to agree with sizeof             */

#define LwP64_VALUE(n)        ((void *)(LwUPtr)(n))
#define LwP64_fmt "0x%llx"

#define KERNEL_POINTER_FROM_LwP64(p,v) ((p)(LwUPtr)(v))
#define LwP64_PLUS_OFFSET(p,o) ((p) + (LwU64)(o))

#define LwUPtr_fmtX "X"
#define LwUPtr_fmtx "x"
#define LwUPtr_fmtu "u"
#define LwUPtr_fmto "o"
#define LwSPtr_fmtd "d"
#define LwSPtr_fmti "i"

#endif

#define LwP64_NULL       (LwP64)0

/*!
 * Helper macro to pack an @ref LwU64_ALIGN32 structure from a @ref LwU64.
 *
 * @param[out] pDst   Pointer to LwU64_ALIGN32 structure to pack
 * @param[in]  pSrc   Pointer to LwU64 with which to pack
 */
#define LwU64_ALIGN32_PACK(pDst, pSrc)                                         \
do {                                                                           \
    (pDst)->lo = LwU64_LO32(*(pSrc));                                          \
    (pDst)->hi = LwU64_HI32(*(pSrc));                                          \
} while (LW_FALSE)

/*!
 * Helper macro to unpack a @ref LwU64_ALIGN32 structure into a @ref LwU64.
 *
 * @param[out] pDst   Pointer to LwU64 in which to unpack
 * @param[in]  pSrc   Pointer to LwU64_ALIGN32 structure from which to unpack
 */
#define LwU64_ALIGN32_UNPACK(pDst, pSrc)                                       \
do {                                                                           \
    (*(pDst)) = LwU64_ALIGN32_VAL(pSrc);                                       \
} while (LW_FALSE)

/*!
 * Helper macro to unpack a @ref LwU64_ALIGN32 structure as a @ref LwU64.
 *
 * @param[in]  pSrc   Pointer to LwU64_ALIGN32 structure to unpack
 */
#define LwU64_ALIGN32_VAL(pSrc)                                                \
    ((LwU64) ((LwU64)((pSrc)->lo) | (((LwU64)(pSrc)->hi) << 32U)))

/*!
 * Helper macro to check whether the 32 bit aligned 64 bit number is zero.
 *
 * @param[in]  _pU64   Pointer to LwU64_ALIGN32 structure.
 *
 * @return
 *  LW_TRUE     _pU64 is zero.
 *  LW_FALSE    otherwise.
 */
#define LwU64_ALIGN32_IS_ZERO(_pU64)                                          \
    (((_pU64)->lo == 0U) && ((_pU64)->hi == 0U))

/*!
 * Helper macro to sub two 32 aligned 64 bit numbers on 64 bit processor.
 *
 * @param[in]       pSrc1   Pointer to LwU64_ALIGN32 source 1 structure.
 * @param[in]       pSrc2   Pointer to LwU64_ALIGN32 source 2 structure.
 * @param[in/out]   pDst    Pointer to LwU64_ALIGN32 dest. structure.
 */
#define LwU64_ALIGN32_ADD(pDst, pSrc1, pSrc2)                                 \
do {                                                                          \
    LwU64 __dst, __src1, __scr2;                                              \
                                                                              \
    LwU64_ALIGN32_UNPACK(&__src1, (pSrc1));                                   \
    LwU64_ALIGN32_UNPACK(&__scr2, (pSrc2));                                   \
    __dst = __src1 + __scr2;                                                  \
    LwU64_ALIGN32_PACK((pDst), &__dst);                                       \
} while (LW_FALSE)

/*!
 * Helper macro to sub two 32 aligned 64 bit numbers on 64 bit processor.
 *
 * @param[in]       pSrc1   Pointer to LwU64_ALIGN32 source 1 structure.
 * @param[in]       pSrc2   Pointer to LwU64_ALIGN32 source 2 structure.
 * @param[in/out]   pDst    Pointer to LwU64_ALIGN32 dest. structure.
 */
#define LwU64_ALIGN32_SUB(pDst, pSrc1, pSrc2)                                  \
do {                                                                           \
    LwU64 __dst, __src1, __scr2;                                               \
                                                                               \
    LwU64_ALIGN32_UNPACK(&__src1, (pSrc1));                                    \
    LwU64_ALIGN32_UNPACK(&__scr2, (pSrc2));                                    \
    __dst = __src1 - __scr2;                                                   \
    LwU64_ALIGN32_PACK((pDst), &__dst);                                        \
} while (LW_FALSE)

/*!
 * Structure for representing 32 bit aligned LwU64 (64-bit unsigned integer)
 * structures. This structure must be used because the 32 bit processor and
 * 64 bit processor compilers will pack/align LwU64 differently.
 *
 * One use case is RM being 64 bit proc whereas PMU being 32 bit proc, this
 * alignment difference will result in corrupted transactions between the RM
 * and PMU.
 *
 * See the @ref LwU64_ALIGN32_PACK and @ref LwU64_ALIGN32_UNPACK macros for
 * packing and unpacking these structures.
 *
 * @note The intention of this structure is to provide a datatype which will
 *       packed/aligned consistently and efficiently across all platforms.
 *       We don't want to use "LW_DECLARE_ALIGNED(LwU64, 8)" because that
 *       leads to memory waste on our 32-bit uprocessors (e.g. FALCONs) where
 *       DMEM efficiency is vital.
 */
typedef struct
{
    /*!
     * Low 32 bits.
     */
    LwU32 lo;
    /*!
     * High 32 bits.
     */
    LwU32 hi;
} LwU64_ALIGN32;

/* Useful macro to hide required double cast */
#define LW_PTR_TO_LwP64(n) (LwP64)(LwUPtr)(n)
#define LW_SIGN_EXT_PTR_TO_LwP64(p) ((LwP64)(LwS64)(LwSPtr)(p))
#define KERNEL_POINTER_TO_LwP64(p) ((LwP64)(uintptr_t)(p))

 /***************************************************************************\
|*                                                                           *|
|*  Limits for common types.                                                 *|
|*                                                                           *|
 \***************************************************************************/

/* Explanation of the current form of these limits:
 *
 * - Decimal is used, as hex values are by default positive.
 * - Casts are not used, as usage in the preprocessor itself (#if) ends poorly.
 * - The subtraction of 1 for some MIN values is used to get around the fact
 *   that the C syntax actually treats -x as NEGATE(x) instead of a distinct
 *   number.  Since 214748648 isn't a valid positive 32-bit signed value, we
 *   take the largest valid positive signed number, negate it, and subtract 1.
 */
#define LW_S8_MIN       (-128)
#define LW_S8_MAX       (+127)
#define LW_U8_MIN       (0U)
#define LW_U8_MAX       (+255U)
#define LW_S16_MIN      (-32768)
#define LW_S16_MAX      (+32767)
#define LW_U16_MIN      (0U)
#define LW_U16_MAX      (+65535U)
#define LW_S32_MIN      (-2147483647 - 1)
#define LW_S32_MAX      (+2147483647)
#define LW_U32_MIN      (0U)
#define LW_U32_MAX      (+4294967295U)
#define LW_S64_MIN      (-9223372036854775807LL - 1LL)
#define LW_S64_MAX      (+9223372036854775807LL)
#define LW_U64_MIN      (0ULL)
#define LW_U64_MAX      (+18446744073709551615ULL)

/* Aligns fields in structs  so they match up between 32 and 64 bit builds */
#if defined(__GNUC__) || defined(__clang__) || defined(LW_QNX)
#define LW_ALIGN_BYTES(size) __attribute__ ((aligned (size)))
#elif defined(__arm)
#define LW_ALIGN_BYTES(size) __align(ALIGN)
#else
// XXX This is dangerously nonportable!  We really shouldn't provide a default
// version of this that doesn't do anything.
#define LW_ALIGN_BYTES(size)
#endif

// LW_DECLARE_ALIGNED() can be used on all platforms.
// This macro form accounts for the fact that __declspec on Windows is required
// before the variable type,
// and LW_ALIGN_BYTES is required after the variable name.
#if defined(__GNUC__) || defined(__clang__) || defined(LW_QNX)
#define LW_DECLARE_ALIGNED(TYPE_VAR, ALIGN) TYPE_VAR __attribute__ ((aligned (ALIGN)))
#elif defined(_MSC_VER)
#define LW_DECLARE_ALIGNED(TYPE_VAR, ALIGN) __declspec(align(ALIGN)) TYPE_VAR
#elif defined(__arm)
#define LW_DECLARE_ALIGNED(TYPE_VAR, ALIGN) __align(ALIGN) TYPE_VAR
#endif

 /***************************************************************************\
|*                       Function Declaration Types                          *|
 \***************************************************************************/

// stretching the meaning of "lwtypes", but this seems to least offensive
// place to re-locate these from lwos.h which cannot be included by a number
// of builds that need them

#if defined(_MSC_VER)

    #if _MSC_VER >= 1310
    #define LW_NOINLINE __declspec(noinline)
    #else
    #define LW_NOINLINE
    #endif

    #define LW_INLINE __inline

    #if _MSC_VER >= 1200
    #define LW_FORCEINLINE __forceinline
    #else
    #define LW_FORCEINLINE __inline
    #endif

    #define LW_APIENTRY  __stdcall
    #define LW_FASTCALL  __fastcall
    #define LW_CDECLCALL __cdecl
    #define LW_STDCALL   __stdcall

    #define LW_FORCERESULTCHECK

    #define LW_ATTRIBUTE_UNUSED

    #define LW_FORMAT_PRINTF(_f, _a)

#else // ! defined(_MSC_VER)

    #if defined(__GNUC__)
        #if (__GNUC__ > 3) || \
            ((__GNUC__ == 3) && (__GNUC_MINOR__ >= 1) && (__GNUC_PATCHLEVEL__ >= 1))
            #define LW_NOINLINE __attribute__((__noinline__))
        #endif
    #elif defined(__clang__)
        #if __has_attribute(noinline)
        #define LW_NOINLINE __attribute__((__noinline__))
        #endif
    #elif defined(__arm) && (__ARMCC_VERSION >= 300000)
        #define LW_NOINLINE __attribute__((__noinline__))
    #elif (defined(__SUNPRO_C) && (__SUNPRO_C >= 0x590)) ||\
            (defined(__SUNPRO_CC) && (__SUNPRO_CC >= 0x590))
        #define LW_NOINLINE __attribute__((__noinline__))
    #elif defined (__INTEL_COMPILER)
        #define LW_NOINLINE __attribute__((__noinline__))
    #endif

    #if !defined(LW_NOINLINE)
    #define LW_NOINLINE
    #endif

    /* GreenHills compiler defines __GNUC__, but doesn't support
     * __inline__ keyword. */
    #if defined(__ghs__)
    #define LW_INLINE inline
    #elif defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
    #define LW_INLINE __inline__
    #elif defined (macintosh) || defined(__SUNPRO_C) || defined(__SUNPRO_CC)
    #define LW_INLINE inline
    #elif defined(__arm)
    #define LW_INLINE __inline
    #else
    #define LW_INLINE
    #endif

    /* Don't force inline on DEBUG builds -- it's annoying for debuggers. */
    #if !defined(DEBUG)
        /* GreenHills compiler defines __GNUC__, but doesn't support
         * __attribute__ or __inline__ keyword. */
        #if defined(__ghs__)
            #define LW_FORCEINLINE inline
        #elif defined(__GNUC__)
            // GCC 3.1 and beyond support the always_inline function attribute.
            #if (__GNUC__ > 3) || ((__GNUC__ == 3) && (__GNUC_MINOR__ >= 1))
            #define LW_FORCEINLINE __attribute__((__always_inline__)) __inline__
            #else
            #define LW_FORCEINLINE __inline__
            #endif
        #elif defined(__clang__)
            #if __has_attribute(always_inline)
            #define LW_FORCEINLINE __attribute__((__always_inline__)) __inline__
            #else
            #define LW_FORCEINLINE __inline__
            #endif
        #elif defined(__arm) && (__ARMCC_VERSION >= 220000)
            // RVDS 2.2 also supports forceinline, but ADS 1.2 does not
            #define LW_FORCEINLINE __forceinline
        #else /* defined(__GNUC__) */
            #define LW_FORCEINLINE LW_INLINE
        #endif
    #else
        #define LW_FORCEINLINE LW_INLINE
    #endif

    #define LW_APIENTRY
    #define LW_FASTCALL
    #define LW_CDECLCALL
    #define LW_STDCALL

    /*
     * The 'warn_unused_result' function attribute prompts GCC to issue a
     * warning if the result of a function tagged with this attribute
     * is ignored by a caller.  In combination with '-Werror', it can be
     * used to enforce result checking in RM code; at this point, this
     * is only done on UNIX.
     */
    #if defined(__GNUC__) && defined(LW_UNIX)
        #if (__GNUC__ > 3) || ((__GNUC__ == 3) && (__GNUC_MINOR__ >= 4))
        #define LW_FORCERESULTCHECK __attribute__((__warn_unused_result__))
        #else
        #define LW_FORCERESULTCHECK
        #endif
    #elif defined(__clang__)
        #if __has_attribute(warn_unused_result)
        #define LW_FORCERESULTCHECK __attribute__((__warn_unused_result__))
        #else
        #define LW_FORCERESULTCHECK
        #endif
    #else /* defined(__GNUC__) */
        #define LW_FORCERESULTCHECK
    #endif

    #if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
        #define LW_ATTRIBUTE_UNUSED __attribute__((__unused__))
    #else
        #define LW_ATTRIBUTE_UNUSED
    #endif

    /*
     * Functions decorated with LW_FORMAT_PRINTF(f, a) have a format string at
     * parameter number 'f' and variadic arguments start at parameter number 'a'.
     * (Note that for C++ methods, there is an implicit 'this' parameter so
     * explicit parameters are numbered from 2.)
     */
    #if defined(__GNUC__)
        #define LW_FORMAT_PRINTF(_f, _a) __attribute__((format(printf, _f, _a)))
    #else
        #define LW_FORMAT_PRINTF(_f, _a)
    #endif

#endif  // defined(_MSC_VER)

#ifdef __cplusplus
}
#endif

#endif /* LWTYPES_INCLUDED */
