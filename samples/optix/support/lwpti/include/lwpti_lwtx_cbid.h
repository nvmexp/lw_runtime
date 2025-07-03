/*
 * Copyright 2013-2017 LWPU Corporation.  All rights reserved.
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

typedef enum {
  LWPTI_CBID_LWTX_ILWALID                               = 0,
  LWPTI_CBID_LWTX_lwtxMarkA                             = 1,
  LWPTI_CBID_LWTX_lwtxMarkW                             = 2,
  LWPTI_CBID_LWTX_lwtxMarkEx                            = 3,
  LWPTI_CBID_LWTX_lwtxRangeStartA                       = 4,
  LWPTI_CBID_LWTX_lwtxRangeStartW                       = 5,
  LWPTI_CBID_LWTX_lwtxRangeStartEx                      = 6,
  LWPTI_CBID_LWTX_lwtxRangeEnd                          = 7,
  LWPTI_CBID_LWTX_lwtxRangePushA                        = 8,
  LWPTI_CBID_LWTX_lwtxRangePushW                        = 9,
  LWPTI_CBID_LWTX_lwtxRangePushEx                       = 10,
  LWPTI_CBID_LWTX_lwtxRangePop                          = 11,
  LWPTI_CBID_LWTX_lwtxNameCategoryA                     = 12,
  LWPTI_CBID_LWTX_lwtxNameCategoryW                     = 13,
  LWPTI_CBID_LWTX_lwtxNameOsThreadA                     = 14,
  LWPTI_CBID_LWTX_lwtxNameOsThreadW                     = 15,
  LWPTI_CBID_LWTX_lwtxNameLwDeviceA                     = 16,
  LWPTI_CBID_LWTX_lwtxNameLwDeviceW                     = 17,
  LWPTI_CBID_LWTX_lwtxNameLwContextA                    = 18,
  LWPTI_CBID_LWTX_lwtxNameLwContextW                    = 19,
  LWPTI_CBID_LWTX_lwtxNameLwStreamA                     = 20,
  LWPTI_CBID_LWTX_lwtxNameLwStreamW                     = 21,
  LWPTI_CBID_LWTX_lwtxNameLwEventA                      = 22,
  LWPTI_CBID_LWTX_lwtxNameLwEventW                      = 23,
  LWPTI_CBID_LWTX_lwtxNameLwdaDeviceA                   = 24,
  LWPTI_CBID_LWTX_lwtxNameLwdaDeviceW                   = 25,
  LWPTI_CBID_LWTX_lwtxNameLwdaStreamA                   = 26,
  LWPTI_CBID_LWTX_lwtxNameLwdaStreamW                   = 27,
  LWPTI_CBID_LWTX_lwtxNameLwdaEventA                    = 28,
  LWPTI_CBID_LWTX_lwtxNameLwdaEventW                    = 29,
  LWPTI_CBID_LWTX_lwtxDomainMarkEx                      = 30,
  LWPTI_CBID_LWTX_lwtxDomainRangeStartEx                = 31,
  LWPTI_CBID_LWTX_lwtxDomainRangeEnd                    = 32,
  LWPTI_CBID_LWTX_lwtxDomainRangePushEx                 = 33,
  LWPTI_CBID_LWTX_lwtxDomainRangePop                    = 34,
  LWPTI_CBID_LWTX_lwtxDomainResourceCreate              = 35,
  LWPTI_CBID_LWTX_lwtxDomainResourceDestroy             = 36,
  LWPTI_CBID_LWTX_lwtxDomainNameCategoryA               = 37,
  LWPTI_CBID_LWTX_lwtxDomainNameCategoryW               = 38,
  LWPTI_CBID_LWTX_lwtxDomainRegisterStringA             = 39,
  LWPTI_CBID_LWTX_lwtxDomainRegisterStringW             = 40,
  LWPTI_CBID_LWTX_lwtxDomainCreateA                     = 41,
  LWPTI_CBID_LWTX_lwtxDomainCreateW                     = 42,
  LWPTI_CBID_LWTX_lwtxDomainDestroy                     = 43,
  LWPTI_CBID_LWTX_lwtxDomainSynlwserCreate              = 44,
  LWPTI_CBID_LWTX_lwtxDomainSynlwserDestroy             = 45,
  LWPTI_CBID_LWTX_lwtxDomainSynlwserAcquireStart        = 46,
  LWPTI_CBID_LWTX_lwtxDomainSynlwserAcquireFailed       = 47,
  LWPTI_CBID_LWTX_lwtxDomainSynlwserAcquireSuccess      = 48,
  LWPTI_CBID_LWTX_lwtxDomainSynlwserReleasing           = 49,
  LWPTI_CBID_LWTX_SIZE,
  LWPTI_CBID_LWTX_FORCE_INT                             = 0x7fffffff
} LWpti_lwtx_api_trace_cbid;

#if defined(__GNUC__) && defined(LWPTI_LIB)
    #pragma GCC visibility pop
#endif    
