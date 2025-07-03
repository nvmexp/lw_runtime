 /****************************************************************************\
|*                                                                            *|
|*      Copyright 2016-2017 LWPU Corporation.  All rights reserved.         *|
|*                                                                            *|
|*  NOTICE TO USER:                                                           *|
|*                                                                            *|
|*  This source code is subject to LWPU ownership rights under U.S. and     *|
|*  international Copyright laws.                                             *|
|*                                                                            *|
|*  This software and the information contained herein is PROPRIETARY and     *|
|*  CONFIDENTIAL to LWPU and is being provided under the terms and          *|
|*  conditions of a Non-Disclosure Agreement. Any reproduction or             *|
|*  disclosure to any third party without the express written consent of      *|
|*  LWPU is prohibited.                                                     *|
|*                                                                            *|
|*  LWPU MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE       *|
|*  CODE FOR ANY PURPOSE. IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR           *|
|*  IMPLIED WARRANTY OF ANY KIND.  LWPU DISCLAIMS ALL WARRANTIES WITH       *|
|*  REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF           *|
|*  MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR            *|
|*  PURPOSE. IN NO EVENT SHALL LWPU BE LIABLE FOR ANY SPECIAL,              *|
|*  INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES            *|
|*  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN        *|
|*  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING       *|
|*  OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE        *|
|*  CODE.                                                                     *|
|*                                                                            *|
|*  U.S. Government End Users. This source code is a "commercial item"        *|
|*  as that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting         *|
|*  of "commercial computer software" and "commercial computer software       *|
|*  documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)     *|
|*  and is provided to the U.S. Government only as a commercial end item.     *|
|*  Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through          *|
|*  227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the         *|
|*  source code with only those rights set forth herein.                      *|
|*                                                                            *|
|*  Module: types.h                                                           *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _TYPES_H
#define _TYPES_H

//******************************************************************************
//
//  Fowards
//
//******************************************************************************
class CCommand;
class CKMutant;
class CDebugGlobal;

class CListObject;
class CList;

class CBinaryNode;
class CAvlNode;
class CAvlCacheNode;
class CAvlPhysicalNode;
class CPhysicalCacheNode;
class CVirtualCacheNode;
class CBinaryTree;
class CAvlTree;
class CAvlCacheTree;
class CAvlPhysicalTree;
class CCache;
class CPhysicalCache;
class CVirtualCache;
class CLinkedCache;

class CDmlState;
class CProgressState;
class CCacheState;
class CCommand;

#if 0
//******************************************************************************
//
//  Internal lWpu type definitions (For use with driver headers)
//
//******************************************************************************
typedef unsigned char   LwV8;   /* "void": enumerated or multiple fields   */
typedef unsigned short  LwV16;  /* "void": enumerated or multiple fields   */
typedef unsigned long   LwV32;  /* "void": enumerated or multiple fields   */
typedef unsigned char   LwU8;   /* 0 to 255                                */
typedef unsigned short  LwU16;  /* 0 to 65535                              */
typedef unsigned long   LwU32;  /* 0 to 4294967295                         */
typedef signed char     LwS8;   /* -128 to 127                             */
typedef signed short    LwS16;  /* -32768 to 32767                         */
typedef signed int      LwS32;  /* -2147483648 to 2147483647               */
typedef float           LwF32;  /* IEEE Single Precision (S1E8M23)         */
typedef double          LwF64;  /* IEEE Double Precision (S1E11M52)        */
typedef unsigned __int64 LwU64; /* 0 to 18446744073709551615               */
typedef          __int64 LwS64; /* 2^-63 to 2^63-1                         */
typedef LwU8            LwBool;
typedef LwU32           LwHandle;
typedef LwU64           LwP64;  /* 64 bit void pointer                     */
#endif
//******************************************************************************
//
//  Other miscellaneous type definitions
//
//******************************************************************************
typedef ULONG64         QWORD;

//******************************************************************************
//
//  Template type definitions (Mostly smart pointer definitions)
//
//******************************************************************************
typedef CRefPtr<CKMutant>                   CKMutantPtr;
typedef CRefPtr<CDebugGlobal>               CDebugGlobalPtr;

typedef CRefPtr<CListObject>                CListObjectPtr;
typedef CRefPtr<CList>                      CListPtr;

typedef CRefPtr<CBinaryNode>                CBinaryNodePtr;
typedef CRefPtr<CAvlNode>                   CAvlNodePtr;
typedef CDrvRefPtr<CAvlCacheNode,           CAvlNodePtr>            CAvlCacheNodePtr;
typedef CDrvRefPtr<CAvlPhysicalNode,        CAvlNodePtr>            CAvlPhysicalNodePtr;
typedef CDrvRefPtr<CPhysicalCacheNode,      CAvlCacheNodePtr>       CPhysicalCacheNodePtr;
typedef CDrvRefPtr<CVirtualCacheNode,       CAvlCacheNodePtr>       CVirtualCacheNodePtr;
typedef CRefPtr<CBinaryTree>                CBinaryTreePtr;
typedef CRefPtr<CAvlTree>                   CAvlTreePtr;
typedef CDrvRefPtr<CAvlCacheTree,           CAvlTreePtr>            CAvlCacheTreePtr;
typedef CDrvRefPtr<CAvlPhysicalTree,        CAvlTreePtr>            CAvlPhysicalTreePtr;
typedef CRefPtr<CCache>                     CCachePtr;
typedef CDrvRefPtr<CPhysicalCache,          CCachePtr>              CPhysicalCachePtr;
typedef CDrvRefPtr<CVirtualCache,           CCachePtr>              CVirtualCachePtr;

typedef CArrayPtr<bool>                     BoolArray;
typedef CArrayPtr<char>                     CharArray;
typedef CArrayPtr<ULONG>                    UlongArray;
typedef CArrayPtr<ULONG64>                  Ulong64Array;
typedef CArrayPtr<POINTER>                  ptrArray;
typedef CArrayPtr<BYTE>                     ByteArray;
typedef CArrayPtr<WORD>                     WordArray;
typedef CArrayPtr<DWORD>                    DwordArray;
typedef CArrayPtr<QWORD>                    QwordArray;
typedef CArrayPtr<BoolArray>                BoolArrays;
typedef CArrayPtr<CharArray>                CharArrays;
typedef CArrayPtr<UlongArray>               UlongArrays;
typedef CArrayPtr<Ulong64Array>             Ulong64Arrays;
typedef CArrayPtr<ptrArray>                 ptrArrays;
typedef CArrayPtr<ByteArray>                ByteArrays;
typedef CArrayPtr<WordArray>                WordArrays;
typedef CArrayPtr<DwordArray>               DwordArrays;
typedef CArrayPtr<QwordArray>               QwordArrays;
typedef CArrayPtr<CBinaryNodePtr>           CBinaryNodeArray;
typedef CArrayPtr<CAvlNodePtr>              CAvlNodeArray;

//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _TYPES_H
