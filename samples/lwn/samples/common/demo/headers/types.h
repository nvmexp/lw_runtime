/*
** This file contains copyrighted code for the LWN 3D API provided by a
** partner company.  The contents of this file should not be used for
** any purpose other than LWN API development and testing.
*/

// There are many types.h files; this one comes from src/*/include/types.h

/*
 Expected possible predefinitions are 
  __ghs__            - Green Hills
  __CWCC__           - CodeWarrior (replaces __MWERKS__)
  __GNUC__           - gcc
  _WIN32 || _WIN64   - msvc/visual studio
*/

#ifndef __TYPES_H__
#define __TYPES_H__

#define _CHECK_TYPES_H_  // temporary; to test that this types.h was included

// Compiler-specific sections -------------------------------------------------

#ifdef __ghs__  // for Green Hills compiler -----------------------------------
#include <ghs_null.h>
typedef signed char          s8;
typedef signed short        s16;
typedef signed int          s32;
typedef signed long long    s64;
typedef unsigned char        u8;
typedef unsigned short      u16;
typedef unsigned int        u32;
typedef unsigned long long  u64;

#define PACKED_STRUCT_BEGIN     _Pragma("pack(1)")
#define PACKED_STRUCT_END       _Pragma("pack()")
#define PACKED_STRUCT_ATTRIBUTE 

#define PRAGMA(...)  _Pragma(#__VA_ARGS__ )
#define ALIGLWAR(x)   PRAGMA( aliglwar(x) )
#define ALIGNED_VAR(varType, varAlign, varDef) \
                      ALIGLWAR(varAlign)       \
                      varType varDef

// typedef name has already been declared (with same type)
#define FORWARD_DECLARE_STRUCT_TYPE(__declname__) \
    PRAGMA(ghs nowarning 301)         \
    typedef struct __declname__ __declname__; \
    PRAGMA(ghs endnowarning 301)

#define CHANGE_SEC(sec, name) PRAGMA( ghs section sec = name )

#define SET_STRUCT_ALIGN(x)   PRAGMA(ghs struct_min_alignment(x))
#define ATTRIB_STRUCT_ALIGN(x)  

// the following definitions should be removed after addressing all the compiler warnings
#ifdef GHS_DISABLE_NON_CRITICAL_WARNINGS
#pragma ghs nowarning 1822  // "multibyte is implementation defined" (yes we know)
#pragma ghs nowarning 177   // "static symbol was declared but never referenced" (remove at cleanup)
#pragma ghs nowarning 68    // "integer colwersion resulted in a change of sign"
                            // (happens with pointer arithmetic using hex constants)
//#pragma ghs nowarning 228   // "trailing comma is nonstandard" (in enum defs)
//#pragma ghs nowarning 1545  // "address taken of a packed structure member with insufficient alignment"
                            // (Per the GH Multi document, this warning will always
                            // appear if address is taken of a packed member.
                            // Developer should understand what it means to pack a
                            // structure and potential access considerations.) 
#endif

#define __cntlzw __CLZ32

#if !defined(__cplusplus) && !defined(_WCHAR_T) && !defined(_WCHAR_T_DEFINED)
#define _WCHAR_T
#define _WCHAR_T_DEFINED
typedef unsigned short wchar_t;
#endif

#ifndef __CHAR16_DEFINED
#define __CHAR16_DEFINED
typedef wchar_t char16;
#endif

#else   // else from: #ifdef __ghs__ // leaving CW, GCC, WIN32/WIN64
#ifdef __CWCC__  // for CodeWarrior compiler ----------------------------------
typedef signed char          s8;
typedef signed short        s16;
typedef signed int          s32;
typedef signed long long    s64;
typedef unsigned char        u8;
typedef unsigned short      u16;
typedef unsigned int        u32;
typedef unsigned long long  u64;

#define PACKED_STRUCT_BEGIN _Pragma("pack(1)")  /* This pragma syntax is C99 compliant */
#define PACKED_STRUCT_END   _Pragma("pack()")
#define PACKED_STRUCT_ATTRIBUTE /* CW does not use an attribute for packing */
#define ALIGNED_VAR(varType, varAlign, varDef)  varType __attribute((aligned (varAlign))) varDef
#define CHANGE_SEC(sec, name)
#define SET_STRUCT_ALIGN(x)     
#define ATTRIB_STRUCT_ALIGN(x)  __attribute((aligned (x)))

#pragma warn_unusedvar off  // turned off while colwerting tree to GHS
#pragma warn_unusedarg off

#else   // else from: #ifdef __CWCC__ // leaving GCC, WIN32/WIN64
#ifdef __GNUC__  // for GCC ---------------------------------------------------
typedef unsigned long long  u64;
typedef signed long long    s64;
typedef unsigned int        u32;
typedef signed int          s32;
typedef unsigned short      u16;
typedef signed short        s16;
typedef unsigned char        u8;
typedef signed char          s8;

#define PACKED_STRUCT_BEGIN /* GNU does not use pragma style packing */
#define PACKED_STRUCT_END   /* GNU does not use pragma style packing */
#define PACKED_STRUCT_ATTRIBUTE __attribute__((packed)) 
#define ALIGNED_VAR(varType, varAlign, varDef)  varType __attribute((aligned (varAlign))) varDef

#define SET_STRUCT_ALIGN(x)     
#define ATTRIB_STRUCT_ALIGN(x)  __attribute((aligned (x)))

#else   // else from: #ifdef __GNUC__ // leaving WIN32/WIN64
#if defined _WIN32 || defined _WIN64  // for MSVC -----------------------------
typedef __int8               s8;
typedef __int16             s16;
typedef __int32             s32;
typedef __int64             s64;
typedef unsigned __int8      u8;
typedef unsigned __int16    u16;
typedef unsigned __int32    u32;
typedef unsigned __int64    u64;

#define PACKED_STRUCT_BEGIN     /* WIN32 does not use pragma style packing */
#define PACKED_STRUCT_END       /* WIN32 does not use pragma style packing */
#define PACKED_STRUCT_ATTRIBUTE /* WIN32 does not use an attribute for packing */
#define ALIGNED_VAR(varType, varAlign, varDef)  varType varDef
#define CHANGE_SEC(sec, name)
#define SET_STRUCT_ALIGN(x)
#define ATTRIB_STRUCT_ALIGN(x)

#else   // --------------------------------------------------------------------
#error Unknown build system
#endif  // #if defined _WIN32 || defined _WIN64
#endif  // #ifdef __GNUC__
#endif  // #ifdef __MWERKS__
#endif  // #ifdef __ghs__

// Non-compiler-specific section ----------------------------------------------

typedef volatile u8          vu8;
typedef volatile u16        vu16;
typedef volatile u32        vu32;
typedef volatile u64        vu64;
typedef volatile s8          vs8;
typedef volatile s16        vs16;
typedef volatile s32        vs32;
typedef volatile s64        vs64;

typedef float                f32;
typedef double               f64;
typedef volatile f32        vf32;
typedef volatile f64        vf64;

#ifndef BOOL
typedef int                 BOOL;
#endif  // BOOL

#ifndef TRUE
#define TRUE                1   // Any non zero value is considered TRUE
#endif  // TRUE

#ifndef FALSE
#define FALSE               0
#endif  // FALSE

#ifndef NULL
#ifdef  __cplusplus
#define NULL                0
#else   // __cplusplus
#define NULL                ((void *)0)
#endif  // __cplusplus
#endif  // NULL

// SN-Phil: AT ADDRESS MACRO
// Use the following pragma wherever a fixed address is required for
// static variables.
#define AT_ADDRESS(xyz) : (xyz)

#endif  // __TYPES_H__
