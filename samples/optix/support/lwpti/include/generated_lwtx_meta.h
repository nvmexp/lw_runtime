/*
 * Copyright 2013-2018 LWPU Corporation.  All rights reserved.
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

#if defined(__GNUC__) && defined(LWPTI_LIB)
    #pragma GCC visibility push(default)
#endif

// *************************************************************************
//      Definitions of structs to hold parameters for each function
// *************************************************************************

typedef struct lwtxMarkEx_params_st {
  const lwtxEventAttributes_t* eventAttrib;
} lwtxMarkEx_params;

typedef struct lwtxMarkA_params_st {
  const char* message;
} lwtxMarkA_params;

typedef struct lwtxMarkW_params_st {
  const wchar_t* message;
} lwtxMarkW_params;

typedef struct lwtxRangeStartEx_params_st {
  const lwtxEventAttributes_t* eventAttrib;
} lwtxRangeStartEx_params;

typedef struct lwtxRangeStartA_params_st {
  const char* message;
} lwtxRangeStartA_params;

typedef struct lwtxRangeStartW_params_st {
  const wchar_t* message;
} lwtxRangeStartW_params;

typedef struct lwtxRangeEnd_params_st {
  lwtxRangeId_t id;
} lwtxRangeEnd_params;

typedef struct lwtxRangePushEx_params_st {
  const lwtxEventAttributes_t* eventAttrib;
} lwtxRangePushEx_params;

typedef struct lwtxRangePushA_params_st {
  const char* message;
} lwtxRangePushA_params;

typedef struct lwtxRangePushW_params_st {
  const wchar_t* message;
} lwtxRangePushW_params;

typedef struct lwtxRangePop_params_st {
  /* WAR: Windows compiler doesn't allow empty structs */
  /* This field shouldn't be used */
  void *dummy;
} lwtxRangePop_params;

typedef struct lwtxNameCategoryA_params_st {
  uint32_t category;
  const char* name;
} lwtxNameCategoryA_params;

typedef struct lwtxNameCategoryW_params_st {
  uint32_t category;
  const wchar_t* name;
} lwtxNameCategoryW_params;

typedef struct lwtxNameOsThreadA_params_st {
  uint32_t threadId;
  const char* name;
} lwtxNameOsThreadA_params;

typedef struct lwtxNameOsThreadW_params_st {
  uint32_t threadId;
  const wchar_t* name;
} lwtxNameOsThreadW_params;

typedef struct lwtxNameLwDeviceA_params_st {
  LWdevice device;
  const char* name;
} lwtxNameLwDeviceA_params;

typedef struct lwtxNameLwDeviceW_params_st {
  LWdevice device;
  const wchar_t* name;
} lwtxNameLwDeviceW_params;

typedef struct lwtxNameLwContextA_params_st {
  LWcontext context;
  const char* name;
} lwtxNameLwContextA_params;

typedef struct lwtxNameLwContextW_params_st {
  LWcontext context;
  const wchar_t* name;
} lwtxNameLwContextW_params;

typedef struct lwtxNameLwStreamA_params_st {
  LWstream stream;
  const char* name;
} lwtxNameLwStreamA_params;

typedef struct lwtxNameLwStreamW_params_st {
  LWstream stream;
  const wchar_t* name;
} lwtxNameLwStreamW_params;

typedef struct lwtxNameLwEventA_params_st {
  LWevent event;
  const char* name;
} lwtxNameLwEventA_params;

typedef struct lwtxNameLwEventW_params_st {
  LWevent event;
  const wchar_t* name;
} lwtxNameLwEventW_params;

typedef struct lwtxNameLwdaDeviceA_params_st {
  int device;
  const char* name;
} lwtxNameLwdaDeviceA_params;

typedef struct lwtxNameLwdaDeviceW_params_st {
  int device;
  const wchar_t* name;
} lwtxNameLwdaDeviceW_params;

typedef struct lwtxNameLwdaStreamA_params_st {
  lwdaStream_t stream;
  const char* name;
} lwtxNameLwdaStreamA_params;

typedef struct lwtxNameLwdaStreamW_params_st {
  lwdaStream_t stream;
  const wchar_t* name;
} lwtxNameLwdaStreamW_params;

typedef struct lwtxNameLwdaEventA_params_st {
  lwdaEvent_t event;
  const char* name;
} lwtxNameLwdaEventA_params;

typedef struct lwtxNameLwdaEventW_params_st {
  lwdaEvent_t event;
  const wchar_t* name;
} lwtxNameLwdaEventW_params;

typedef struct lwtxDomainCreateA_params_st {
  const char* name;
} lwtxDomainCreateA_params;

typedef struct lwtxDomainDestroy_params_st {
  lwtxDomainHandle_t domain;
} lwtxDomainDestroy_params;

typedef struct lwtxDomainMarkEx_params_st {
  lwtxDomainHandle_t domain;
  lwtxMarkEx_params core;
} lwtxDomainMarkEx_params;

typedef struct lwtxDomainRangeStartEx_params_st {
  lwtxDomainHandle_t domain;
  lwtxRangeStartEx_params core;
} lwtxDomainRangeStartEx_params;

typedef struct lwtxDomainRangeEnd_params_st {
  lwtxDomainHandle_t domain;
  lwtxRangeEnd_params core;
} lwtxDomainRangeEnd_params;

typedef struct lwtxDomainRangePushEx_params_st {
  lwtxDomainHandle_t domain;
  lwtxRangePushEx_params core;
} lwtxDomainRangePushEx_params;

typedef struct lwtxDomainRangePop_params_st {
  lwtxDomainHandle_t domain;
} lwtxDomainRangePop_params;

typedef struct lwtxSynlwserCreate_params_st {
  lwtxDomainHandle_t domain;
  const lwtxSynlwserAttributes_t* attribs;
} lwtxSynlwserCreate_params;

typedef struct lwtxSynlwserCommon_params_st {
  lwtxSynlwser_t handle;
} lwtxSynlwserCommon_params;

typedef struct lwtxDomainRegisterStringA_params_st {
    lwtxDomainHandle_t domain;
    lwtxStringHandle_t handle;
} lwtxDomainRegisterStringA_params;

typedef struct lwtxDomainRegisterStringW_params_st {
    lwtxDomainHandle_t domain;
    lwtxStringHandle_t handle;
} lwtxDomainRegisterStringW_params;

#if defined(__GNUC__) && defined(LWPTI_LIB)
    #pragma GCC visibility pop
#endif    
