/*
* Copyright 2009-2017 LWPU Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to LWPU ownership rights under U.S. and
* international Copyright laws.
*
* This software and the information contained herein is PROPRIETARY and
* CONFIDENTIAL to LWPU and is being provided under the terms and conditions
* of a form of LWPU software license agreement.
*
* LWPU MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  LWPU DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL LWPU BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.   This source code is a "commercial item" as
* that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer  software"  and "commercial computer software
* documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*
* Any use of this source code in individual and commercial software must
* include, in the user documentation and internal comments to the code,
* the above Disclaimer and U.S. Government End Users Notice.
*/

/** \mainpage
 * \section Introduction
 * The LWPU Tools Extension library is a set of functions that a
 * developer can use to provide additional information to tools.
 * The additional information is used by the tool to improve
 * analysis and visualization of data.
 *
 * The library introduces close to zero overhead if no tool is
 * attached to the application.  The overhead when a tool is
 * attached is specific to the tool.
 */

#ifndef LWTOOLSEXT_META_H_
#define LWTOOLSEXT_META_H_



#ifdef _MSC_VER
#  define LWTX_PACK( __Declaration__ ) __pragma( pack(push, 1) ) __Declaration__ __pragma( pack(pop) )
#elif (defined(__GNUC__) || defined(__clang__))
#  define LWTX_PACK( __Declaration__ ) __Declaration__ __attribute__((__packed__))
#endif


/* Structs defining parameters for LWTX API functions */

typedef struct LwtxMarkEx{ const lwtxEventAttributes_t* eventAttrib; } LwtxMarkEx;
LWTX_PACK(struct LwtxDomainMarkEx { lwtxDomainHandle_t domain; LwtxMarkEx core; });
typedef struct LwtxDomainMarkEx LwtxDomainMarkEx;

typedef struct LwtxMarkA        { const char* message; } LwtxMarkA;
typedef struct LwtxMarkW        { const wchar_t* message; } LwtxMarkW;
typedef struct LwtxRangeStartEx { const lwtxEventAttributes_t* eventAttrib; } LwtxRangeStartEx;

LWTX_PACK(struct LwtxDomainRangeStartEx { lwtxDomainHandle_t domain; LwtxRangeStartEx core; });
typedef struct LwtxDomainRangeStartEx LwtxDomainRangeStartEx;

typedef struct LwtxRangeStartA  { const char* message; } LwtxRangeStartA;
typedef struct LwtxRangeStartW  { const wchar_t* message; } LwtxRangeStartW;
typedef struct LwtxRangeEnd     { lwtxRangeId_t id; } LwtxRangeEnd;

LWTX_PACK(struct LwtxDomainRangeEnd { lwtxDomainHandle_t domain; LwtxRangeEnd core; });
typedef struct LwtxDomainRangeEnd LwtxDomainRangeEnd;

typedef struct LwtxRangePushEx  { const lwtxEventAttributes_t* eventAttrib; } LwtxRangePushEx;

LWTX_PACK(struct LwtxDomainRangePushEx { lwtxDomainHandle_t domain; LwtxRangePushEx core; });
typedef struct LwtxDomainRangePushEx LwtxDomainRangePushEx;

typedef struct LwtxRangePushA   { const char* message; } LwtxRangePushA;
typedef struct LwtxRangePushW   { const wchar_t* message; } LwtxRangePushW;
typedef struct LwtxDomainRangePop   { lwtxDomainHandle_t domain; } LwtxDomainRangePop;
/*     LwtxRangePop     - no parameters, params will be NULL. */
typedef struct LwtxDomainResourceCreate  { lwtxDomainHandle_t domain; const lwtxResourceAttributes_t* attribs; } LwtxDomainResourceCreate;
typedef struct LwtxDomainResourceDestroy  { lwtxResourceHandle_t handle; } LwtxDomainResourceDestroy;
typedef struct LwtxDomainRegisterString  { lwtxDomainHandle_t domain; const void* str; } LwtxDomainRegisterString;
typedef struct LwtxDomainCreate  { const void* name; } LwtxDomainCreate;
typedef struct LwtxDomainDestroy  { lwtxDomainHandle_t domain; } LwtxDomainDestroy;


#ifdef LWTOOLSEXT_SYNC_H_
typedef struct LwtxSynlwserCommon  { lwtxSynlwser_t handle; } LwtxSynlwserCommon;
typedef struct LwtxSynlwserCreate  { lwtxDomainHandle_t domain; const lwtxSynlwserAttributes_t* attribs; } LwtxSynlwserCreate;
#endif

/* All other LWTX API functions are for naming resources. 
 * A generic params struct is used for all such functions,
 * passing all resource handles as a uint64_t.
 */
typedef struct LwtxNameResourceA
{
    uint64_t resourceHandle;
    const char* name;
} LwtxNameResourceA;

typedef struct LwtxNameResourceW
{
    uint64_t resourceHandle;
    const wchar_t* name;
} LwtxNameResourceW;

LWTX_PACK(struct LwtxDomainNameResourceA { lwtxDomainHandle_t domain; LwtxNameResourceA core; });
typedef struct LwtxDomainNameResourceA LwtxDomainNameResourceA;
LWTX_PACK(struct LwtxDomainNameResourceW { lwtxDomainHandle_t domain; LwtxNameResourceW core; });
typedef struct LwtxDomainNameResourceW LwtxDomainNameResourceW;


#endif /* LWTOOLSEXT_META_H_ */
