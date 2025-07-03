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
 *  Module name              : stdStdFun.h
 *
 *  Description              :
 *
 *               This module defines hash-, equality- and comparison
 *               functions on the standard types, and it defines
 *               the corresponding function types.
 * 
 *               These functions and function types are colwenient
 *               for building various hash tables, and for traversing
 *               data structures.
 */

#ifndef stdStdFun_INCLUDED
#define stdStdFun_INCLUDED

/*--------------------------------- Includes ---------------------------------*/

#include "stdTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

/*----------------------------------- Types ----------------------------------*/

/*
 * These function pointers are passed into *Traverse routines,
 * which then call the function with a data parameter.
 * But some clients do a trick where they pass in a function without the data
 * parameter, and then pass in Nil for the data.
 * This only works with __cdecl calling convention on 32-bit windows,
 * where the caller reserves the stack space.
 * Because we don't have control over some of the client uses (i.e. sasslib),
 * instead we force __cdecl to be used for all of std (and sasslib)
 * and then any other clients of std that use these function types
 * (e.g. stdEltFun) must declare their local function that is cast to this
 * to also be __cdecl.
 */
typedef uInt32 (STD_CDECL *stdHashFun)     ( Pointer e);
typedef uInt32 (STD_CDECL *stdHashDFun)    ( Pointer e,  Pointer data);
typedef Bool   (STD_CDECL *stdPropertyFun) ( Pointer e,               Pointer data );
typedef Bool   (STD_CDECL *stdEqualFun)    ( Pointer e1, Pointer e2);
typedef Bool   (STD_CDECL *stdEqualDFun)   ( Pointer e1, Pointer e2,  Pointer data );
typedef Bool   (STD_CDECL *stdLessEqFun)   ( Pointer e1, Pointer e2);
typedef Bool   (STD_CDECL *stdLessEqDFun)  ( Pointer e1, Pointer e2,  Pointer data );
typedef void   (STD_CDECL *stdPairFun)     ( Pointer e1, Pointer e2,  Pointer data );
typedef void   (STD_CDECL *stdTripleFun)   ( Pointer e1, Pointer e2,  Pointer data );
typedef void   (STD_CDECL *stdQuadFun)     ( Pointer e1, Pointer e2,  Pointer e3, Pointer data );
typedef void   (STD_CDECL *stdEltFun)      ( Pointer e,               Pointer data );
typedef void   (STD_CDECL *stdDataFun)     ( Pointer data );


typedef struct {
    stdEltFun      traverse;
    Pointer        data;
} stdEltTraversalRec;

typedef struct {
    stdPairFun     traverse;
    Pointer        data;
} stdPairTraversalRec;

typedef struct {
    stdHashFun     hash;
    stdEqualFun    equal;
    uInt           nrofBuckets;
    Pointer        data;
    stdHashDFun    hashD;
    stdEqualDFun   equalD;
} stdHashTableParameters;


/*--------------------------------- Functions --------------------------------*/

uInt32 STD_CDECL stdIntHash       ( Int32 e );
Bool   STD_CDECL stdIntEqual      ( Int32 e1, Int32 e2 );
Bool   STD_CDECL stdIntLessEq     ( Int32 e1, Int32 e2 );
uInt32 STD_CDECL stdInt64Hash     ( Int64 e );
Bool   STD_CDECL stdInt64Equal    ( Int64 e1,  Int64 e2 );
Bool   STD_CDECL stdInt64LessEq   ( Int64 e1,  Int64 e2 );


uInt32 STD_CDECL stdStringHash    ( cString s );
Bool   STD_CDECL stdStringEqual   ( cString e1, cString e2 );
Bool   STD_CDECL stdStringLessEq  ( cString e1, cString e2 );


uInt32 STD_CDECL stdFloatHash     ( Float e );
Bool   STD_CDECL stdFloatEqual    ( Float e1, Float e2 );
Bool   STD_CDECL stdFloatLessEq   ( Float e1, Float e2 );


//#define DETERMINISTIC_POINTER_HASHING
#ifdef DETERMINISTIC_POINTER_HASHING
#define  stdPointerHash  memspPointerHash
#define _stdPointerHash  memspPointerHash
#else
uInt32   STD_CDECL stdPointerHash ( Pointer e );
#define _stdPointerHash STD_POINTER_HASH
#endif

Bool   STD_CDECL stdPointerEqual     ( Pointer e1, Pointer e2 );
Bool   STD_CDECL stdPointerLessEq    ( Pointer e1, Pointer e2 );


uInt32 STD_CDECL stdpInt32Hash       ( Int32 *e );
Bool   STD_CDECL stdpInt32Equal      ( Int32 *e1, Int32 *e2 );
Bool   STD_CDECL stdpInt32LessEq     ( Int32 *e1, Int32 *e2 );


uInt32 STD_CDECL stdpInt64Hash       (  Int64 *e );
Bool   STD_CDECL stdpInt64Equal      (  Int64 *e1,  Int64 *e2 );
Bool   STD_CDECL stdpInt64LessEq     (  Int64 *e1,  Int64 *e2 );

uInt32 STD_CDECL stdpuInt64Hash      ( uInt64 *e );
Bool   STD_CDECL stdpuInt64Equal     ( uInt64 *e1, uInt64 *e2 );
Bool   STD_CDECL stdpuInt64LessEq    ( uInt64 *e1, uInt64 *e2 );


uInt32 STD_CDECL stdNStringHash      ( cString e );
Bool   STD_CDECL stdNStringEqual     ( cString e1, cString e2 );

cString STD_CDECL stdStringIsPrefix   ( cString e1, cString e2 );
cString STD_CDECL stdCIStringIsPrefix ( cString e1, cString e2 );


uInt32 STD_CDECL stdCIStringHash     ( cString e );
Bool   STD_CDECL stdCIStringEqual    ( cString e1, cString e2 );
Bool   STD_CDECL stdCIStringLessEq   ( cString e1, cString e2 );


#define  stdSetHash       setHash
#define  stdSetEqual      setEqual

#define  stdAddressHash   stdPointerHash
#define  stdAddressEqual  stdPointerEqual
#define  stdAddressLessEq stdPointerLessEq

#define  stduIntHash      stdIntHash
#define  stduIntEqual     stdIntEqual
#define  stduIntsLessEq   stdIntLessEq




#define IRRELEVANT_POINTER_BITS     stdMAX(stdALIGN_SHIFT, 5)

#define  STD_POINTER_HASH(e)        ( (((Address)(e)) >>  (IRRELEVANT_POINTER_BITS    ) ) \
                                    ^ (((Address)(e)) >>  (IRRELEVANT_POINTER_BITS + 3) ) \
                                    ^ (((Address)(e)) >>  (IRRELEVANT_POINTER_BITS + 6) ) \
                                    )

#define  _stdIntHash(e)               (e)
#define  _stdInt64Hash(e)             (((uInt32)(e))^((uInt32)(((uInt64)(e))>>32)))
#define  _stdIntEqual(e1,e2)          ((e1) == (e2))
#define  _stdPointerEqual(e1,e2)      ((e1) == (e2))

#ifdef __cplusplus
}
#endif

#endif
