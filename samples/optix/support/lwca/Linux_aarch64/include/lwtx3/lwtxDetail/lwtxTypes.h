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

/* This header defines types which are used by the internal implementation
*  of LWTX and callback subscribers.  API clients do not use these types,
*  so they are defined here instead of in lwToolsExt.h to clarify they are
*  not part of the LWTX client API. */

#ifndef LWTX_IMPL_GUARD
#error Never include this file directly -- it is automatically included by lwToolsExt.h.
#endif

/* ------ Dependency-free types binary-compatible with real types ------- */

/* In order to avoid having the LWTX core API headers depend on non-LWTX
*  headers like lwca.h, LWTX defines binary-compatible types to use for
*  safely making the initialization versions of all LWTX functions without
*  needing to have definitions for the real types. */

typedef int   lwtx_LWdevice;
typedef void* lwtx_LWcontext;
typedef void* lwtx_LWstream;
typedef void* lwtx_LWevent;

typedef void* lwtx_lwdaStream_t;
typedef void* lwtx_lwdaEvent_t;

typedef void* lwtx_cl_platform_id;
typedef void* lwtx_cl_device_id;
typedef void* lwtx_cl_context;
typedef void* lwtx_cl_command_queue;
typedef void* lwtx_cl_mem;
typedef void* lwtx_cl_program;
typedef void* lwtx_cl_kernel;
typedef void* lwtx_cl_event;
typedef void* lwtx_cl_sampler;

typedef struct lwtxSynlwser* lwtxSynlwser_t;
struct lwtxSynlwserAttributes_v0;
typedef struct lwtxSynlwserAttributes_v0 lwtxSynlwserAttributes_t;

/* --------- Types for function pointers (with fake API types) ---------- */

typedef void (LWTX_API * lwtxMarkEx_impl_fntype)(const lwtxEventAttributes_t* eventAttrib);
typedef void (LWTX_API * lwtxMarkA_impl_fntype)(const char* message);
typedef void (LWTX_API * lwtxMarkW_impl_fntype)(const wchar_t* message);
typedef lwtxRangeId_t (LWTX_API * lwtxRangeStartEx_impl_fntype)(const lwtxEventAttributes_t* eventAttrib);
typedef lwtxRangeId_t (LWTX_API * lwtxRangeStartA_impl_fntype)(const char* message);
typedef lwtxRangeId_t (LWTX_API * lwtxRangeStartW_impl_fntype)(const wchar_t* message);
typedef void (LWTX_API * lwtxRangeEnd_impl_fntype)(lwtxRangeId_t id);
typedef int (LWTX_API * lwtxRangePushEx_impl_fntype)(const lwtxEventAttributes_t* eventAttrib);
typedef int (LWTX_API * lwtxRangePushA_impl_fntype)(const char* message);
typedef int (LWTX_API * lwtxRangePushW_impl_fntype)(const wchar_t* message);
typedef int (LWTX_API * lwtxRangePop_impl_fntype)(void);
typedef void (LWTX_API * lwtxNameCategoryA_impl_fntype)(uint32_t category, const char* name);
typedef void (LWTX_API * lwtxNameCategoryW_impl_fntype)(uint32_t category, const wchar_t* name);
typedef void (LWTX_API * lwtxNameOsThreadA_impl_fntype)(uint32_t threadId, const char* name);
typedef void (LWTX_API * lwtxNameOsThreadW_impl_fntype)(uint32_t threadId, const wchar_t* name);

/* Real impl types are defined in lwtxImplLwda_v3.h, where LWCA headers are included */
typedef void (LWTX_API * lwtxNameLwDeviceA_fakeimpl_fntype)(lwtx_LWdevice device, const char* name);
typedef void (LWTX_API * lwtxNameLwDeviceW_fakeimpl_fntype)(lwtx_LWdevice device, const wchar_t* name);
typedef void (LWTX_API * lwtxNameLwContextA_fakeimpl_fntype)(lwtx_LWcontext context, const char* name);
typedef void (LWTX_API * lwtxNameLwContextW_fakeimpl_fntype)(lwtx_LWcontext context, const wchar_t* name);
typedef void (LWTX_API * lwtxNameLwStreamA_fakeimpl_fntype)(lwtx_LWstream stream, const char* name);
typedef void (LWTX_API * lwtxNameLwStreamW_fakeimpl_fntype)(lwtx_LWstream stream, const wchar_t* name);
typedef void (LWTX_API * lwtxNameLwEventA_fakeimpl_fntype)(lwtx_LWevent event, const char* name);
typedef void (LWTX_API * lwtxNameLwEventW_fakeimpl_fntype)(lwtx_LWevent event, const wchar_t* name);

/* Real impl types are defined in lwtxImplOpenCL_v3.h, where OPENCL headers are included */
typedef void (LWTX_API * lwtxNameClDeviceA_fakeimpl_fntype)(lwtx_cl_device_id device, const char* name);
typedef void (LWTX_API * lwtxNameClDeviceW_fakeimpl_fntype)(lwtx_cl_device_id device, const wchar_t* name);
typedef void (LWTX_API * lwtxNameClContextA_fakeimpl_fntype)(lwtx_cl_context context, const char* name);
typedef void (LWTX_API * lwtxNameClContextW_fakeimpl_fntype)(lwtx_cl_context context, const wchar_t* name);
typedef void (LWTX_API * lwtxNameClCommandQueueA_fakeimpl_fntype)(lwtx_cl_command_queue command_queue, const char* name);
typedef void (LWTX_API * lwtxNameClCommandQueueW_fakeimpl_fntype)(lwtx_cl_command_queue command_queue, const wchar_t* name);
typedef void (LWTX_API * lwtxNameClMemObjectA_fakeimpl_fntype)(lwtx_cl_mem memobj, const char* name);
typedef void (LWTX_API * lwtxNameClMemObjectW_fakeimpl_fntype)(lwtx_cl_mem memobj, const wchar_t* name);
typedef void (LWTX_API * lwtxNameClSamplerA_fakeimpl_fntype)(lwtx_cl_sampler sampler, const char* name);
typedef void (LWTX_API * lwtxNameClSamplerW_fakeimpl_fntype)(lwtx_cl_sampler sampler, const wchar_t* name);
typedef void (LWTX_API * lwtxNameClProgramA_fakeimpl_fntype)(lwtx_cl_program program, const char* name);
typedef void (LWTX_API * lwtxNameClProgramW_fakeimpl_fntype)(lwtx_cl_program program, const wchar_t* name);
typedef void (LWTX_API * lwtxNameClEventA_fakeimpl_fntype)(lwtx_cl_event evnt, const char* name);
typedef void (LWTX_API * lwtxNameClEventW_fakeimpl_fntype)(lwtx_cl_event evnt, const wchar_t* name);

/* Real impl types are defined in lwtxImplLwdaRt_v3.h, where LWDART headers are included */
typedef void (LWTX_API * lwtxNameLwdaDeviceA_impl_fntype)(int device, const char* name);
typedef void (LWTX_API * lwtxNameLwdaDeviceW_impl_fntype)(int device, const wchar_t* name);
typedef void (LWTX_API * lwtxNameLwdaStreamA_fakeimpl_fntype)(lwtx_lwdaStream_t stream, const char* name);
typedef void (LWTX_API * lwtxNameLwdaStreamW_fakeimpl_fntype)(lwtx_lwdaStream_t stream, const wchar_t* name);
typedef void (LWTX_API * lwtxNameLwdaEventA_fakeimpl_fntype)(lwtx_lwdaEvent_t event, const char* name);
typedef void (LWTX_API * lwtxNameLwdaEventW_fakeimpl_fntype)(lwtx_lwdaEvent_t event, const wchar_t* name);

typedef void (LWTX_API * lwtxDomainMarkEx_impl_fntype)(lwtxDomainHandle_t domain, const lwtxEventAttributes_t* eventAttrib);
typedef lwtxRangeId_t (LWTX_API * lwtxDomainRangeStartEx_impl_fntype)(lwtxDomainHandle_t domain, const lwtxEventAttributes_t* eventAttrib);
typedef void (LWTX_API * lwtxDomainRangeEnd_impl_fntype)(lwtxDomainHandle_t domain, lwtxRangeId_t id);
typedef int (LWTX_API * lwtxDomainRangePushEx_impl_fntype)(lwtxDomainHandle_t domain, const lwtxEventAttributes_t* eventAttrib);
typedef int (LWTX_API * lwtxDomainRangePop_impl_fntype)(lwtxDomainHandle_t domain);
typedef lwtxResourceHandle_t (LWTX_API * lwtxDomainResourceCreate_impl_fntype)(lwtxDomainHandle_t domain, lwtxResourceAttributes_t* attribs);
typedef void (LWTX_API * lwtxDomainResourceDestroy_impl_fntype)(lwtxResourceHandle_t resource);
typedef void (LWTX_API * lwtxDomainNameCategoryA_impl_fntype)(lwtxDomainHandle_t domain, uint32_t category, const char* name);
typedef void (LWTX_API * lwtxDomainNameCategoryW_impl_fntype)(lwtxDomainHandle_t domain, uint32_t category, const wchar_t* name);
typedef lwtxStringHandle_t (LWTX_API * lwtxDomainRegisterStringA_impl_fntype)(lwtxDomainHandle_t domain, const char* string);
typedef lwtxStringHandle_t (LWTX_API * lwtxDomainRegisterStringW_impl_fntype)(lwtxDomainHandle_t domain, const wchar_t* string);
typedef lwtxDomainHandle_t (LWTX_API * lwtxDomainCreateA_impl_fntype)(const char* message);
typedef lwtxDomainHandle_t (LWTX_API * lwtxDomainCreateW_impl_fntype)(const wchar_t* message);
typedef void (LWTX_API * lwtxDomainDestroy_impl_fntype)(lwtxDomainHandle_t domain);
typedef void (LWTX_API * lwtxInitialize_impl_fntype)(const void* reserved);

typedef lwtxSynlwser_t (LWTX_API * lwtxDomainSynlwserCreate_impl_fntype)(lwtxDomainHandle_t domain, const lwtxSynlwserAttributes_t* attribs);
typedef void (LWTX_API * lwtxDomainSynlwserDestroy_impl_fntype)(lwtxSynlwser_t handle);
typedef void (LWTX_API * lwtxDomainSynlwserAcquireStart_impl_fntype)(lwtxSynlwser_t handle);
typedef void (LWTX_API * lwtxDomainSynlwserAcquireFailed_impl_fntype)(lwtxSynlwser_t handle);
typedef void (LWTX_API * lwtxDomainSynlwserAcquireSuccess_impl_fntype)(lwtxSynlwser_t handle);
typedef void (LWTX_API * lwtxDomainSynlwserReleasing_impl_fntype)(lwtxSynlwser_t handle);

/* ---------------- Types for callback subscription --------------------- */

typedef const void *(LWTX_API * LwtxGetExportTableFunc_t)(uint32_t exportTableId);
typedef int (LWTX_API * LwtxInitializeInjectionLwtxFunc_t)(LwtxGetExportTableFunc_t exportTable);

typedef enum LwtxCallbackModule
{
    LWTX_CB_MODULE_ILWALID                 = 0,
    LWTX_CB_MODULE_CORE                    = 1,
    LWTX_CB_MODULE_LWDA                    = 2,
    LWTX_CB_MODULE_OPENCL                  = 3,
    LWTX_CB_MODULE_LWDART                  = 4,
    LWTX_CB_MODULE_CORE2                   = 5,
    LWTX_CB_MODULE_SYNC                    = 6,
    /* --- New constants must only be added directly above this line --- */
    LWTX_CB_MODULE_SIZE,
    LWTX_CB_MODULE_FORCE_INT               = 0x7fffffff
} LwtxCallbackModule;

typedef enum LwtxCallbackIdCore
{
    LWTX_CBID_CORE_ILWALID                 =  0,
    LWTX_CBID_CORE_MarkEx                  =  1,
    LWTX_CBID_CORE_MarkA                   =  2,
    LWTX_CBID_CORE_MarkW                   =  3,
    LWTX_CBID_CORE_RangeStartEx            =  4,
    LWTX_CBID_CORE_RangeStartA             =  5,
    LWTX_CBID_CORE_RangeStartW             =  6,
    LWTX_CBID_CORE_RangeEnd                =  7,
    LWTX_CBID_CORE_RangePushEx             =  8,
    LWTX_CBID_CORE_RangePushA              =  9,
    LWTX_CBID_CORE_RangePushW              = 10,
    LWTX_CBID_CORE_RangePop                = 11,
    LWTX_CBID_CORE_NameCategoryA           = 12,
    LWTX_CBID_CORE_NameCategoryW           = 13,
    LWTX_CBID_CORE_NameOsThreadA           = 14,
    LWTX_CBID_CORE_NameOsThreadW           = 15,
    /* --- New constants must only be added directly above this line --- */
    LWTX_CBID_CORE_SIZE,
    LWTX_CBID_CORE_FORCE_INT = 0x7fffffff
} LwtxCallbackIdCore;

typedef enum LwtxCallbackIdCore2
{
    LWTX_CBID_CORE2_ILWALID                 = 0,
    LWTX_CBID_CORE2_DomainMarkEx            = 1,
    LWTX_CBID_CORE2_DomainRangeStartEx      = 2,
    LWTX_CBID_CORE2_DomainRangeEnd          = 3,
    LWTX_CBID_CORE2_DomainRangePushEx       = 4,
    LWTX_CBID_CORE2_DomainRangePop          = 5,
    LWTX_CBID_CORE2_DomainResourceCreate    = 6,
    LWTX_CBID_CORE2_DomainResourceDestroy   = 7,
    LWTX_CBID_CORE2_DomainNameCategoryA     = 8,
    LWTX_CBID_CORE2_DomainNameCategoryW     = 9,
    LWTX_CBID_CORE2_DomainRegisterStringA   = 10,
    LWTX_CBID_CORE2_DomainRegisterStringW   = 11,
    LWTX_CBID_CORE2_DomainCreateA           = 12,
    LWTX_CBID_CORE2_DomainCreateW           = 13,
    LWTX_CBID_CORE2_DomainDestroy           = 14,
    LWTX_CBID_CORE2_Initialize              = 15,
    /* --- New constants must only be added directly above this line --- */
    LWTX_CBID_CORE2_SIZE,
    LWTX_CBID_CORE2_FORCE_INT               = 0x7fffffff
} LwtxCallbackIdCore2;

typedef enum LwtxCallbackIdLwda
{
    LWTX_CBID_LWDA_ILWALID                 =  0,
    LWTX_CBID_LWDA_NameLwDeviceA           =  1,
    LWTX_CBID_LWDA_NameLwDeviceW           =  2,
    LWTX_CBID_LWDA_NameLwContextA          =  3,
    LWTX_CBID_LWDA_NameLwContextW          =  4,
    LWTX_CBID_LWDA_NameLwStreamA           =  5,
    LWTX_CBID_LWDA_NameLwStreamW           =  6,
    LWTX_CBID_LWDA_NameLwEventA            =  7,
    LWTX_CBID_LWDA_NameLwEventW            =  8,
    /* --- New constants must only be added directly above this line --- */
    LWTX_CBID_LWDA_SIZE,
    LWTX_CBID_LWDA_FORCE_INT               = 0x7fffffff
} LwtxCallbackIdLwda;

typedef enum LwtxCallbackIdLwdaRt
{
    LWTX_CBID_LWDART_ILWALID               =  0,
    LWTX_CBID_LWDART_NameLwdaDeviceA       =  1,
    LWTX_CBID_LWDART_NameLwdaDeviceW       =  2,
    LWTX_CBID_LWDART_NameLwdaStreamA       =  3,
    LWTX_CBID_LWDART_NameLwdaStreamW       =  4,
    LWTX_CBID_LWDART_NameLwdaEventA        =  5,
    LWTX_CBID_LWDART_NameLwdaEventW        =  6,
    /* --- New constants must only be added directly above this line --- */
    LWTX_CBID_LWDART_SIZE,
    LWTX_CBID_LWDART_FORCE_INT             = 0x7fffffff
} LwtxCallbackIdLwdaRt;

typedef enum LwtxCallbackIdOpenCL
{
    LWTX_CBID_OPENCL_ILWALID               =  0,
    LWTX_CBID_OPENCL_NameClDeviceA         =  1,
    LWTX_CBID_OPENCL_NameClDeviceW         =  2,
    LWTX_CBID_OPENCL_NameClContextA        =  3,
    LWTX_CBID_OPENCL_NameClContextW        =  4,
    LWTX_CBID_OPENCL_NameClCommandQueueA   =  5,
    LWTX_CBID_OPENCL_NameClCommandQueueW   =  6,
    LWTX_CBID_OPENCL_NameClMemObjectA      =  7,
    LWTX_CBID_OPENCL_NameClMemObjectW      =  8,
    LWTX_CBID_OPENCL_NameClSamplerA        =  9,
    LWTX_CBID_OPENCL_NameClSamplerW        = 10,
    LWTX_CBID_OPENCL_NameClProgramA        = 11,
    LWTX_CBID_OPENCL_NameClProgramW        = 12,
    LWTX_CBID_OPENCL_NameClEventA          = 13,
    LWTX_CBID_OPENCL_NameClEventW          = 14,
    /* --- New constants must only be added directly above this line --- */
    LWTX_CBID_OPENCL_SIZE,
    LWTX_CBID_OPENCL_FORCE_INT             = 0x7fffffff
} LwtxCallbackIdOpenCL;

typedef enum LwtxCallbackIdSync
{
    LWTX_CBID_SYNC_ILWALID                      = 0,
    LWTX_CBID_SYNC_DomainSynlwserCreate         = 1,
    LWTX_CBID_SYNC_DomainSynlwserDestroy        = 2,
    LWTX_CBID_SYNC_DomainSynlwserAcquireStart   = 3,
    LWTX_CBID_SYNC_DomainSynlwserAcquireFailed  = 4,
    LWTX_CBID_SYNC_DomainSynlwserAcquireSuccess = 5,
    LWTX_CBID_SYNC_DomainSynlwserReleasing      = 6,
    /* --- New constants must only be added directly above this line --- */
    LWTX_CBID_SYNC_SIZE,
    LWTX_CBID_SYNC_FORCE_INT                    = 0x7fffffff
} LwtxCallbackIdSync;

/* IDs for LWTX Export Tables */
typedef enum LwtxExportTableID
{
    LWTX_ETID_ILWALID                      = 0,
    LWTX_ETID_CALLBACKS                    = 1,
    LWTX_ETID_RESERVED0                    = 2,
    LWTX_ETID_VERSIONINFO                  = 3,
    /* --- New constants must only be added directly above this line --- */
    LWTX_ETID_SIZE,
    LWTX_ETID_FORCE_INT                    = 0x7fffffff
} LwtxExportTableID;

typedef void (* LwtxFunctionPointer)(void); /* generic uncallable function pointer, must be casted to appropriate function type */
typedef LwtxFunctionPointer** LwtxFunctionTable; /* double pointer because array(1) of pointers(2) to function pointers */

typedef struct LwtxExportTableCallbacks
{
    size_t struct_size;

    /* returns an array of pointer to function pointers*/
    int (LWTX_API *GetModuleFunctionTable)(
        LwtxCallbackModule module,
        LwtxFunctionTable* out_table,
        unsigned int* out_size);
} LwtxExportTableCallbacks;

typedef struct LwtxExportTableVersionInfo
{
    /* sizeof(LwtxExportTableVersionInfo) */
    size_t struct_size;

    /* The API version comes from the LWTX library linked to the app.  The
    * injection library is can use this info to make some assumptions */
    uint32_t version;

    /* Reserved for alignment, do not use */
    uint32_t reserved0;

    /* This must be set by tools when attaching to provide applications
    *  the ability to, in emergency situations, detect problematic tools
    *  versions and modify the LWTX source to prevent attaching anything
    *  that causes trouble in the app.  Lwrrently, this value is ignored. */
    void (LWTX_API *SetInjectionLwtxVersion)(
        uint32_t version);
} LwtxExportTableVersionInfo;







