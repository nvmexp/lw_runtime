/* This file was procedurally generated!  Do not modify this file by hand.  */

/*
* Copyright 2009-2016  LWPU Corporation.  All rights reserved.
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

#ifndef LWTX_IMPL_GUARD
#error Never include this file directly -- it is automatically included by lwToolsExt.h (except when LWTX_NO_IMPL is defined).
#endif

/* ---- Include required platform headers ---- */

#if defined(_WIN32) 

#include <Windows.h>

#else
#include <unistd.h>

#if defined(__ANDROID__)
#include <android/api-level.h> 
#endif

#if defined(__linux__) || defined(__CYGWIN__)
#include <sched.h>
#endif

#include <limits.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>

#include <string.h>
#include <sys/types.h>
#include <pthread.h>
#include <stdlib.h>
#include <wchar.h>

#endif

/* ---- Define macros used in this file ---- */

#define LWTX_INIT_STATE_FRESH 0
#define LWTX_INIT_STATE_STARTED 1
#define LWTX_INIT_STATE_COMPLETE 2

#ifdef LWTX_DEBUG_PRINT
#ifdef __ANDROID__
#include <android/log.h>
#define LWTX_ERR(...) __android_log_print(ANDROID_LOG_ERROR, "LWTOOLSEXT", __VA_ARGS__);
#define LWTX_INFO(...) __android_log_print(ANDROID_LOG_INFO, "LWTOOLSEXT", __VA_ARGS__);
#else
#include <stdio.h>
#define LWTX_ERR(...) fprintf(stderr, "LWTX_ERROR: " __VA_ARGS__)
#define LWTX_INFO(...) fprintf(stderr, "LWTX_INFO: " __VA_ARGS__)
#endif
#else /* !defined(LWTX_DEBUG_PRINT) */
#define LWTX_ERR(...)
#define LWTX_INFO(...)
#endif

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

/* ---- Forward declare all functions referenced in globals ---- */

LWTX_LINKONCE_FWDDECL_FUNCTION void LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)(void);
LWTX_LINKONCE_FWDDECL_FUNCTION int LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxEtiGetModuleFunctionTable)(
    LwtxCallbackModule module,
    LwtxFunctionTable* out_table,
    unsigned int* out_size);
LWTX_LINKONCE_FWDDECL_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxEtiSetInjectionLwtxVersion)(
    uint32_t version);
LWTX_LINKONCE_FWDDECL_FUNCTION const void* LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxGetExportTable)(
    uint32_t exportTableId);

#include "lwtxInitDecls.h"

/* ---- Define all globals ---- */

typedef struct lwtxGlobals_t
{
    volatile unsigned int initState;
    LwtxExportTableCallbacks etblCallbacks;
    LwtxExportTableVersionInfo etblVersionInfo;

    /* Implementation function pointers */
    lwtxMarkEx_impl_fntype lwtxMarkEx_impl_fnptr;
    lwtxMarkA_impl_fntype lwtxMarkA_impl_fnptr;
    lwtxMarkW_impl_fntype lwtxMarkW_impl_fnptr;
    lwtxRangeStartEx_impl_fntype lwtxRangeStartEx_impl_fnptr;
    lwtxRangeStartA_impl_fntype lwtxRangeStartA_impl_fnptr;
    lwtxRangeStartW_impl_fntype lwtxRangeStartW_impl_fnptr;
    lwtxRangeEnd_impl_fntype lwtxRangeEnd_impl_fnptr;
    lwtxRangePushEx_impl_fntype lwtxRangePushEx_impl_fnptr;
    lwtxRangePushA_impl_fntype lwtxRangePushA_impl_fnptr;
    lwtxRangePushW_impl_fntype lwtxRangePushW_impl_fnptr;
    lwtxRangePop_impl_fntype lwtxRangePop_impl_fnptr;
    lwtxNameCategoryA_impl_fntype lwtxNameCategoryA_impl_fnptr;
    lwtxNameCategoryW_impl_fntype lwtxNameCategoryW_impl_fnptr;
    lwtxNameOsThreadA_impl_fntype lwtxNameOsThreadA_impl_fnptr;
    lwtxNameOsThreadW_impl_fntype lwtxNameOsThreadW_impl_fnptr;

    lwtxNameLwDeviceA_fakeimpl_fntype lwtxNameLwDeviceA_impl_fnptr;
    lwtxNameLwDeviceW_fakeimpl_fntype lwtxNameLwDeviceW_impl_fnptr;
    lwtxNameLwContextA_fakeimpl_fntype lwtxNameLwContextA_impl_fnptr;
    lwtxNameLwContextW_fakeimpl_fntype lwtxNameLwContextW_impl_fnptr;
    lwtxNameLwStreamA_fakeimpl_fntype lwtxNameLwStreamA_impl_fnptr;
    lwtxNameLwStreamW_fakeimpl_fntype lwtxNameLwStreamW_impl_fnptr;
    lwtxNameLwEventA_fakeimpl_fntype lwtxNameLwEventA_impl_fnptr;
    lwtxNameLwEventW_fakeimpl_fntype lwtxNameLwEventW_impl_fnptr;

    lwtxNameClDeviceA_fakeimpl_fntype lwtxNameClDeviceA_impl_fnptr;
    lwtxNameClDeviceW_fakeimpl_fntype lwtxNameClDeviceW_impl_fnptr;
    lwtxNameClContextA_fakeimpl_fntype lwtxNameClContextA_impl_fnptr;
    lwtxNameClContextW_fakeimpl_fntype lwtxNameClContextW_impl_fnptr;
    lwtxNameClCommandQueueA_fakeimpl_fntype lwtxNameClCommandQueueA_impl_fnptr;
    lwtxNameClCommandQueueW_fakeimpl_fntype lwtxNameClCommandQueueW_impl_fnptr;
    lwtxNameClMemObjectA_fakeimpl_fntype lwtxNameClMemObjectA_impl_fnptr;
    lwtxNameClMemObjectW_fakeimpl_fntype lwtxNameClMemObjectW_impl_fnptr;
    lwtxNameClSamplerA_fakeimpl_fntype lwtxNameClSamplerA_impl_fnptr;
    lwtxNameClSamplerW_fakeimpl_fntype lwtxNameClSamplerW_impl_fnptr;
    lwtxNameClProgramA_fakeimpl_fntype lwtxNameClProgramA_impl_fnptr;
    lwtxNameClProgramW_fakeimpl_fntype lwtxNameClProgramW_impl_fnptr;
    lwtxNameClEventA_fakeimpl_fntype lwtxNameClEventA_impl_fnptr;
    lwtxNameClEventW_fakeimpl_fntype lwtxNameClEventW_impl_fnptr;

    lwtxNameLwdaDeviceA_impl_fntype lwtxNameLwdaDeviceA_impl_fnptr;
    lwtxNameLwdaDeviceW_impl_fntype lwtxNameLwdaDeviceW_impl_fnptr;
    lwtxNameLwdaStreamA_fakeimpl_fntype lwtxNameLwdaStreamA_impl_fnptr;
    lwtxNameLwdaStreamW_fakeimpl_fntype lwtxNameLwdaStreamW_impl_fnptr;
    lwtxNameLwdaEventA_fakeimpl_fntype lwtxNameLwdaEventA_impl_fnptr;
    lwtxNameLwdaEventW_fakeimpl_fntype lwtxNameLwdaEventW_impl_fnptr;

    lwtxDomainMarkEx_impl_fntype lwtxDomainMarkEx_impl_fnptr;
    lwtxDomainRangeStartEx_impl_fntype lwtxDomainRangeStartEx_impl_fnptr;
    lwtxDomainRangeEnd_impl_fntype lwtxDomainRangeEnd_impl_fnptr;
    lwtxDomainRangePushEx_impl_fntype lwtxDomainRangePushEx_impl_fnptr;
    lwtxDomainRangePop_impl_fntype lwtxDomainRangePop_impl_fnptr;
    lwtxDomainResourceCreate_impl_fntype lwtxDomainResourceCreate_impl_fnptr;
    lwtxDomainResourceDestroy_impl_fntype lwtxDomainResourceDestroy_impl_fnptr;
    lwtxDomainNameCategoryA_impl_fntype lwtxDomainNameCategoryA_impl_fnptr;
    lwtxDomainNameCategoryW_impl_fntype lwtxDomainNameCategoryW_impl_fnptr;
    lwtxDomainRegisterStringA_impl_fntype lwtxDomainRegisterStringA_impl_fnptr;
    lwtxDomainRegisterStringW_impl_fntype lwtxDomainRegisterStringW_impl_fnptr;
    lwtxDomainCreateA_impl_fntype lwtxDomainCreateA_impl_fnptr;
    lwtxDomainCreateW_impl_fntype lwtxDomainCreateW_impl_fnptr;
    lwtxDomainDestroy_impl_fntype lwtxDomainDestroy_impl_fnptr;
    lwtxInitialize_impl_fntype lwtxInitialize_impl_fnptr;

    lwtxDomainSynlwserCreate_impl_fntype lwtxDomainSynlwserCreate_impl_fnptr;
    lwtxDomainSynlwserDestroy_impl_fntype lwtxDomainSynlwserDestroy_impl_fnptr;
    lwtxDomainSynlwserAcquireStart_impl_fntype lwtxDomainSynlwserAcquireStart_impl_fnptr;
    lwtxDomainSynlwserAcquireFailed_impl_fntype lwtxDomainSynlwserAcquireFailed_impl_fnptr;
    lwtxDomainSynlwserAcquireSuccess_impl_fntype lwtxDomainSynlwserAcquireSuccess_impl_fnptr;
    lwtxDomainSynlwserReleasing_impl_fntype lwtxDomainSynlwserReleasing_impl_fnptr;

    /* Tables of function pointers -- Extra null added to the end to ensure
    *  a crash instead of silent corruption if a tool reads off the end. */
    LwtxFunctionPointer* functionTable_CORE  [LWTX_CBID_CORE_SIZE   + 1];
    LwtxFunctionPointer* functionTable_LWDA  [LWTX_CBID_LWDA_SIZE   + 1];
    LwtxFunctionPointer* functionTable_OPENCL[LWTX_CBID_OPENCL_SIZE + 1];
    LwtxFunctionPointer* functionTable_LWDART[LWTX_CBID_LWDART_SIZE + 1];
    LwtxFunctionPointer* functionTable_CORE2 [LWTX_CBID_CORE2_SIZE  + 1];
    LwtxFunctionPointer* functionTable_SYNC  [LWTX_CBID_SYNC_SIZE   + 1];
} lwtxGlobals_t;

LWTX_LINKONCE_DEFINE_GLOBAL lwtxGlobals_t LWTX_VERSIONED_IDENTIFIER(lwtxGlobals) =
{
    LWTX_INIT_STATE_FRESH,

    {
        sizeof(LwtxExportTableCallbacks),
        LWTX_VERSIONED_IDENTIFIER(lwtxEtiGetModuleFunctionTable)
    },
    {
        sizeof(LwtxExportTableVersionInfo),
        LWTX_VERSION,
        0,
        LWTX_VERSIONED_IDENTIFIER(lwtxEtiSetInjectionLwtxVersion)
    },

    /* Implementation function pointers */
    LWTX_VERSIONED_IDENTIFIER(lwtxMarkEx_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxMarkA_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxMarkW_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxRangeStartEx_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxRangeStartA_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxRangeStartW_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxRangeEnd_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxRangePushEx_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxRangePushA_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxRangePushW_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxRangePop_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxNameCategoryA_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxNameCategoryW_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxNameOsThreadA_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxNameOsThreadW_impl_init),

    LWTX_VERSIONED_IDENTIFIER(lwtxNameLwDeviceA_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxNameLwDeviceW_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxNameLwContextA_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxNameLwContextW_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxNameLwStreamA_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxNameLwStreamW_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxNameLwEventA_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxNameLwEventW_impl_init),

    LWTX_VERSIONED_IDENTIFIER(lwtxNameClDeviceA_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxNameClDeviceW_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxNameClContextA_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxNameClContextW_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxNameClCommandQueueA_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxNameClCommandQueueW_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxNameClMemObjectA_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxNameClMemObjectW_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxNameClSamplerA_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxNameClSamplerW_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxNameClProgramA_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxNameClProgramW_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxNameClEventA_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxNameClEventW_impl_init),

    LWTX_VERSIONED_IDENTIFIER(lwtxNameLwdaDeviceA_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxNameLwdaDeviceW_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxNameLwdaStreamA_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxNameLwdaStreamW_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxNameLwdaEventA_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxNameLwdaEventW_impl_init),

    LWTX_VERSIONED_IDENTIFIER(lwtxDomainMarkEx_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxDomainRangeStartEx_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxDomainRangeEnd_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxDomainRangePushEx_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxDomainRangePop_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxDomainResourceCreate_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxDomainResourceDestroy_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxDomainNameCategoryA_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxDomainNameCategoryW_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxDomainRegisterStringA_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxDomainRegisterStringW_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxDomainCreateA_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxDomainCreateW_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxDomainDestroy_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxInitialize_impl_init),

    LWTX_VERSIONED_IDENTIFIER(lwtxDomainSynlwserCreate_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxDomainSynlwserDestroy_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxDomainSynlwserAcquireStart_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxDomainSynlwserAcquireFailed_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxDomainSynlwserAcquireSuccess_impl_init),
    LWTX_VERSIONED_IDENTIFIER(lwtxDomainSynlwserReleasing_impl_init),

    /* Tables of function pointers */
    {
        0,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxMarkEx_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxMarkA_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxMarkW_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxRangeStartEx_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxRangeStartA_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxRangeStartW_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxRangeEnd_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxRangePushEx_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxRangePushA_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxRangePushW_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxRangePop_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameCategoryA_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameCategoryW_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameOsThreadA_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameOsThreadW_impl_fnptr,
        0
    },
    {
        0,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwDeviceA_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwDeviceW_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwContextA_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwContextW_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwStreamA_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwStreamW_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwEventA_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwEventW_impl_fnptr,
        0
    },
    {
        0,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClDeviceA_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClDeviceW_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClContextA_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClContextW_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClCommandQueueA_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClCommandQueueW_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClMemObjectA_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClMemObjectW_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClSamplerA_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClSamplerW_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClProgramA_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClProgramW_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClEventA_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClEventW_impl_fnptr,
        0
    },
    {
        0,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwdaDeviceA_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwdaDeviceW_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwdaStreamA_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwdaStreamW_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwdaEventA_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwdaEventW_impl_fnptr,
        0
    },
    {
        0,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainMarkEx_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainRangeStartEx_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainRangeEnd_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainRangePushEx_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainRangePop_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainResourceCreate_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainResourceDestroy_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainNameCategoryA_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainNameCategoryW_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainRegisterStringA_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainRegisterStringW_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainCreateA_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainCreateW_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainDestroy_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxInitialize_impl_fnptr,
        0
    },
    {
        0,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainSynlwserCreate_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainSynlwserDestroy_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainSynlwserAcquireStart_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainSynlwserAcquireFailed_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainSynlwserAcquireSuccess_impl_fnptr,
        (LwtxFunctionPointer*)&LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainSynlwserReleasing_impl_fnptr,
        0
    }
};

/* ---- Define static inline implementations of core API functions ---- */

#include "lwtxImplCore.h"

/* ---- Define implementations of export table functions ---- */

LWTX_LINKONCE_DEFINE_FUNCTION int LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxEtiGetModuleFunctionTable)(
    LwtxCallbackModule module,
    LwtxFunctionTable* out_table,
    unsigned int* out_size)
{
    unsigned int bytes = 0;
    LwtxFunctionTable table = (LwtxFunctionTable)0;

    switch (module)
    {
    case LWTX_CB_MODULE_CORE:
        table = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).functionTable_CORE;
        bytes = (unsigned int)sizeof(LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).functionTable_CORE);
        break;
    case LWTX_CB_MODULE_LWDA:
        table = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).functionTable_LWDA;
        bytes = (unsigned int)sizeof(LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).functionTable_LWDA);
        break;
    case LWTX_CB_MODULE_OPENCL:
        table = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).functionTable_OPENCL;
        bytes = (unsigned int)sizeof(LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).functionTable_OPENCL);
        break;
    case LWTX_CB_MODULE_LWDART:
        table = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).functionTable_LWDART;
        bytes = (unsigned int)sizeof(LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).functionTable_LWDART);
        break;
    case LWTX_CB_MODULE_CORE2:
        table = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).functionTable_CORE2;
        bytes = (unsigned int)sizeof(LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).functionTable_CORE2);
        break;
    case LWTX_CB_MODULE_SYNC:
        table = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).functionTable_SYNC;
        bytes = (unsigned int)sizeof(LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).functionTable_SYNC);
        break;
    default: return 0;
    }

    if (out_size)
        *out_size = (bytes / (unsigned int)sizeof(LwtxFunctionPointer*)) - 1;

    if (out_table)
        *out_table = table;

    return 1;
}

LWTX_LINKONCE_DEFINE_FUNCTION const void* LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxGetExportTable)(uint32_t exportTableId)
{
    switch (exportTableId)
    {
    case LWTX_ETID_CALLBACKS:       return &LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).etblCallbacks;
    case LWTX_ETID_VERSIONINFO:     return &LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).etblVersionInfo;
    default:                        return 0;
    }
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxEtiSetInjectionLwtxVersion)(uint32_t version)
{
    /* Reserved for custom implementations to resolve problems with tools */
    (void)version;
}

/* ---- Define implementations of init versions of all API functions ---- */

#include "lwtxInitDefs.h"

/* ---- Define implementations of initialization functions ---- */

#include "lwtxInit.h"

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */
