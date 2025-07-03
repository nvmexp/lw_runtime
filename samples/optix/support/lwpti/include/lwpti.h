/*
 * Copyright 2010-2017 LWPU Corporation.  All rights reserved.
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

#if !defined(_LWPTI_H_)
#define _LWPTI_H_

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifdef NOMINMAX
#include <windows.h>
#else
#define NOMINMAX
#include <windows.h>
#undef NOMINMAX
#endif
#endif

#include <lwca.h>
#include <lwpti_result.h>
#include <lwpti_version.h>

/* Activity, callback, event and metric APIs */
#include <lwpti_activity.h>
#include <lwpti_callbacks.h>
#include <lwpti_events.h>
#include <lwpti_metrics.h>

/* Runtime, driver, and lwtx function identifiers */
#include <lwpti_driver_cbid.h>
#include <lwpti_runtime_cbid.h>
#include <lwpti_lwtx_cbid.h>

/* To support function parameter structures for obsoleted API. See
   lwca.h for the actual definition of these structures. */
typedef unsigned int LWdeviceptr_v1;
typedef struct LWDA_MEMCPY2D_v1_st { int dummy; } LWDA_MEMCPY2D_v1;
typedef struct LWDA_MEMCPY3D_v1_st { int dummy; } LWDA_MEMCPY3D_v1;
typedef struct LWDA_ARRAY_DESCRIPTOR_v1_st { int dummy; } LWDA_ARRAY_DESCRIPTOR_v1;
typedef struct LWDA_ARRAY3D_DESCRIPTOR_v1_st { int dummy; } LWDA_ARRAY3D_DESCRIPTOR_v1;

/* Function parameter structures */
#include <generated_lwda_runtime_api_meta.h>
#include <generated_lwda_meta.h>

/* The following parameter structures cannot be included unless a
   header that defines GL_VERSION is included before including them.
   If these are needed then make sure such a header is included
   already. */
#ifdef GL_VERSION
#include <generated_lwda_gl_interop_meta.h>
#include <generated_lwdaGL_meta.h>
#endif

//#include <generated_lwtx_meta.h>

/* The following parameter structures cannot be included by default as
   they are not guaranteed to be available on all systems. Uncomment
   the includes that are available, or use the include explicitly. */
#if defined(__linux__)
//#include <generated_lwda_vdpau_interop_meta.h>
//#include <generated_lwdaVDPAU_meta.h>
#endif

#ifdef _WIN32
//#include <generated_lwda_d3d9_interop_meta.h>
//#include <generated_lwda_d3d10_interop_meta.h>
//#include <generated_lwda_d3d11_interop_meta.h>
//#include <generated_lwdaD3D9_meta.h>
//#include <generated_lwdaD3D10_meta.h>
//#include <generated_lwdaD3D11_meta.h>
#endif

#endif /*_LWPTI_H_*/


