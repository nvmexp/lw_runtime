/*
 * Copyright 2010-2018 LWPU Corporation.  All rights reserved.
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

#if !defined(_LWPTI_VERSION_H_)
#define _LWPTI_VERSION_H_

#include <lwda_stdint.h>
#include <lwpti_result.h>

#ifndef LWPTIAPI
#ifdef _WIN32
#define LWPTIAPI __stdcall
#else
#define LWPTIAPI
#endif
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__GNUC__) && defined(LWPTI_LIB)
    #pragma GCC visibility push(default)
#endif

/**
 * \defgroup LWPTI_VERSION_API LWPTI Version
 * Function and macro to determine the LWPTI version.
 * @{
 */

/**
 * \brief The API version for this implementation of LWPTI.
 *
 * The API version for this implementation of LWPTI. This define along
 * with \ref lwptiGetVersion can be used to dynamically detect if the
 * version of LWPTI compiled against matches the version of the loaded
 * LWPTI library.
 *
 * v1 : LWDAToolsSDK 4.0
 * v2 : LWDAToolsSDK 4.1
 * v3 : LWCA Toolkit 5.0
 * v4 : LWCA Toolkit 5.5
 * v5 : LWCA Toolkit 6.0
 * v6 : LWCA Toolkit 6.5
 * v7 : LWCA Toolkit 6.5(with sm_52 support)
 * v8 : LWCA Toolkit 7.0
 * v9 : LWCA Toolkit 8.0
 * v10 : LWCA Toolkit 9.0
 * v11 : LWCA Toolkit 9.1
 * v12 : LWCA Toolkit 10.0, 10.1 and 10.2
 * v13 : LWCA Toolkit 11.0
 */
#define LWPTI_API_VERSION 13

/**
 * \brief Get the LWPTI API version.
 *
 * Return the API version in \p *version.
 *
 * \param version Returns the version
 *
 * \retval LWPTI_SUCCESS on success
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p version is NULL
 * \sa LWPTI_API_VERSION
 */
LWptiResult LWPTIAPI lwptiGetVersion(uint32_t *version);

/** @} */ /* END LWPTI_VERSION_API */

#if defined(__GNUC__) && defined(LWPTI_LIB)
    #pragma GCC visibility pop
#endif

#if defined(__cplusplus)
}
#endif

#endif /*_LWPTI_VERSION_H_*/
