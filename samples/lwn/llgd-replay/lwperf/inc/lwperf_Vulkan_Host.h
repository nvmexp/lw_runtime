#ifndef LWPERF_VULKAN_HOST_H
#define LWPERF_VULKAN_HOST_H

/*
 * Copyright 2014-2021  LWPU Corporation.  All rights reserved.
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

#include <stddef.h>
#include <stdint.h>
#include "lwperf_common.h"
#include "lwperf_host.h"

#if defined(__GNUC__) && defined(LWPA_SHARED_LIB)
    #pragma GCC visibility push(default)
    #if !defined(LWPW_LOCAL)
        #define LWPW_LOCAL __attribute__ ((visibility ("hidden")))
    #endif
#else
    #if !defined(LWPW_LOCAL)
        #define LWPW_LOCAL
    #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @file   lwperf_vulkan_host.h
 */

    typedef struct LWPA_MetricsContext LWPA_MetricsContext;

    typedef struct LWPW_VK_MetricsContext_Create_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        const char* pChipName;
        /// [out]
        struct LWPA_MetricsContext* pMetricsContext;
    } LWPW_VK_MetricsContext_Create_Params;
#define LWPW_VK_MetricsContext_Create_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_MetricsContext_Create_Params, pMetricsContext)

    LWPA_Status LWPW_VK_MetricsContext_Create(LWPW_VK_MetricsContext_Create_Params* pParams);

    typedef struct LWPW_VK_RawMetricsConfig_Create_Params
    {
        /// [in]
        size_t structSize;
        /// [in] assign to NULL
        void* pPriv;
        /// [in]
        LWPA_ActivityKind activityKind;
        /// [in]
        const char* pChipName;
        /// [out] new LWPA_RawMetricsConfig object
        struct LWPA_RawMetricsConfig* pRawMetricsConfig;
    } LWPW_VK_RawMetricsConfig_Create_Params;
#define LWPW_VK_RawMetricsConfig_Create_Params_STRUCT_SIZE LWPA_STRUCT_SIZE(LWPW_VK_RawMetricsConfig_Create_Params, pRawMetricsConfig)

    LWPA_Status LWPW_VK_RawMetricsConfig_Create(LWPW_VK_RawMetricsConfig_Create_Params* pParams);



#ifdef __cplusplus
} // extern "C"
#endif

#if defined(__GNUC__) && defined(LWPA_SHARED_LIB)
    #pragma GCC visibility pop
#endif

#endif // LWPERF_VULKAN_HOST_H
