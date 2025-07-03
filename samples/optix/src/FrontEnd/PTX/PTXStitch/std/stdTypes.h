/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2006-2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */
/*
 *  Module name              : stdTypes.h
 *
 *  Description              :
 *
 *     Instead of the native C/C++ types int, short, and char, with their
 *     type qualifiers, software uses the basic types defined in the common 
 *     include file stdTypes.h. The rationale for these basic types 
 *     is as follows:
 *
 *        - The basic type names follow the naming convention of the
 *          coding style, thereby enhancing source code uniformity.
 *        - Explicit types such as String, Bool, Byte are more meaningful
 *          than the usual char*, char, char, hence enhancing source code
 *          clarity/readability.
 *        - Different pointer type definitions hide the address width,
 *          hence enhancing source code portability towards architectures
 *          with 64 bit address spaces. 
 */

#ifndef stdTypes_INCLUDED
#define stdTypes_INCLUDED

#include "stdPlatformDefs.h"

#ifdef __cplusplus
extern "C" {
#endif

/*---------------------------------- Types -----------------------------------*/

/*
 * The following mappings to integer types 
 * of specified sizes (e.g. uInt32, etc) are
 * based on nonstandardized behavior of
 * the Gnu compiler and the Windows Visual
 * Studio compiler, respectively.
 */
typedef          char        Char;
typedef unsigned char        Byte;

typedef unsigned char        Bool;

#ifdef STD_OS_win32
typedef unsigned __int64   uInt64;
typedef          __int64    Int64;
#define S64_CONST(x) (x##L)
#define U64_CONST(x) (x##UL)
#else
typedef unsigned long long uInt64;
typedef   signed long long  Int64;
#define S64_CONST(x) (x##LL)
#define U64_CONST(x) (x##ULL)
#endif

typedef unsigned int       uInt32;
typedef   signed int        Int32;
typedef unsigned short     uInt16;
typedef   signed short      Int16;
typedef unsigned char       uInt8;
typedef   signed char        Int8;

typedef          float      Float;
typedef          double    Double;

/*
 * The following gives deterministic 
 * results across 32/64 and Win/Linux 
 * platforms:
 */
typedef       uInt32         uInt;
typedef        Int32          Int;

typedef          Char     *String;

#ifdef STD_64_BIT_ARCH
typedef          uInt64    Address;
#else
typedef          uInt32    Address;
#endif

typedef          Address   SizeT;
typedef          void     *Pointer;
typedef    const void     *cPointer;


typedef enum { stdLittleEndian = 0, stdBigEndian } stdEndian;

typedef union {
    Float      f;
    uInt32     i;
} stdFloatColw;

typedef union {
    Double     d;
    uInt64     i;
} stdDoubleColw;


typedef enum {
    stdOverlapsNone,
    stdOverlapsGT,
    stdOverlapsLT,
    stdOverlapsEqual,
    stdOverlapsSome,
} stdOverlapKind; 
 
 
typedef enum {
    stdUnEvaluatedMark,
    stdEvaluatingMark,
    stdEvaluatedMark
} stdEvaluationMarkType;


/*--------------------------------- Constants --------------------------------*/

#define Nil         NULL
#define False           0
#define True            1

typedef const Char    *cString;
#ifdef __cplusplus
#define S(s)           (const_cast<String>((s)))
#else
#define S(s)           ((String)(s))
#endif

#if defined(STD_OS_win32) && defined(BUILD_STD_CDECL)
    #define STD_CDECL      __cdecl
    #define STDCALL        __stdcall
#else
    #define STD_CDECL
    #define STDCALL
#endif

#ifdef __cplusplus
}

#endif

#endif
