/*
 * Copyright 1993-2018 LWPU Corporation.  All rights reserved.
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

#ifndef lwda_profiler_H
#define lwda_profiler_H

#include <lwca.h>

#if defined(__LWDA_API_VERSION_INTERNAL) || defined(__DOXYGEN_ONLY__) || defined(LWDA_ENABLE_DEPRECATED)
#define __LWDA_DEPRECATED
#elif defined(_MSC_VER)
#define __LWDA_DEPRECATED __declspec(deprecated)
#elif defined(__GNUC__)
#define __LWDA_DEPRECATED __attribute__((deprecated))
#else
#define __LWDA_DEPRECATED
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Profiler Output Modes
 */
/*DEVICE_BUILTIN*/
typedef enum LWoutput_mode_enum
{
    LW_OUT_KEY_VALUE_PAIR  = 0x00, /**< Output mode Key-Value pair format. */
    LW_OUT_CSV             = 0x01  /**< Output mode Comma separated values format. */
}LWoutput_mode;


/**
 * \ingroup LWDA_DRIVER
 * \defgroup LWDA_PROFILER_DEPRECATED Profiler Control [DEPRECATED]
 *
 * ___MANBRIEF___ profiler control functions of the low-level LWCA driver API
 * (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the profiler control functions of the low-level LWCA
 * driver application programming interface.
 *
 * @{
 */

/**
 * \brief Initialize the profiling.
 *
 * \deprecated
 *
 * Using this API user can initialize the LWCA profiler by specifying
 * the configuration file, output file and output file format. This
 * API is generally used to profile different set of counters by
 * looping the kernel launch. The \p configFile parameter can be used
 * to select profiling options including profiler counters. Refer to
 * the "Compute Command Line Profiler User Guide" for supported
 * profiler options and counters.
 *
 * Limitation: The LWCA profiler cannot be initialized with this API
 * if another profiling tool is already active, as indicated by the
 * ::LWDA_ERROR_PROFILER_DISABLED return code.
 *
 * Typical usage of the profiling APIs is as follows: 
 *
 * for each set of counters/options\n
 * {\n
 *     lwProfilerInitialize(); //Initialize profiling, set the counters or options in the config file \n
 *     ...\n
 *     lwProfilerStart(); \n
 *     // code to be profiled \n
 *     lwProfilerStop(); \n
 *     ...\n
 *     lwProfilerStart(); \n
 *     // code to be profiled \n
 *     lwProfilerStop(); \n
 *     ...\n
 * }\n
 *
 * \param configFile - Name of the config file that lists the counters/options
 * for profiling.
 * \param outputFile - Name of the outputFile where the profiling results will
 * be stored.
 * \param outputMode - outputMode, can be ::LW_OUT_KEY_VALUE_PAIR or ::LW_OUT_CSV.
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_CONTEXT,
 * ::LWDA_ERROR_ILWALID_VALUE,
 * ::LWDA_ERROR_PROFILER_DISABLED
 * \notefnerr
 *
 * \sa
 * ::lwProfilerStart,
 * ::lwProfilerStop,
 * ::lwdaProfilerInitialize
 */
__LWDA_DEPRECATED LWresult LWDAAPI lwProfilerInitialize(const char *configFile, const char *outputFile, LWoutput_mode outputMode);
 
/** @} */ /* END LWDA_PROFILER_DEPRECATED */

/**
 * \ingroup LWDA_DRIVER
 * \defgroup LWDA_PROFILER Profiler Control 
 *
 * ___MANBRIEF___ profiler control functions of the low-level LWCA driver API
 * (___LWRRENT_FILE___) ___ENDMANBRIEF___
 *
 * This section describes the profiler control functions of the low-level LWCA
 * driver application programming interface.
 *
 * @{
 */

/**
 * \brief Enable profiling.
 *
 * Enables profile collection by the active profiling tool for the
 * current context. If profiling is already enabled, then
 * lwProfilerStart() has no effect.
 *
 * lwProfilerStart and lwProfilerStop APIs are used to
 * programmatically control the profiling granularity by allowing
 * profiling to be done only on selective pieces of code.
 * 
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_CONTEXT
 * \notefnerr
 *
 * \sa
 * ::lwProfilerInitialize,
 * ::lwProfilerStop,
 * ::lwdaProfilerStart
 */
LWresult LWDAAPI lwProfilerStart(void);

/**
 * \brief Disable profiling.
 *
 * Disables profile collection by the active profiling tool for the
 * current context. If profiling is already disabled, then
 * lwProfilerStop() has no effect.
 *
 * lwProfilerStart and lwProfilerStop APIs are used to
 * programmatically control the profiling granularity by allowing
 * profiling to be done only on selective pieces of code.
 *
 * \return
 * ::LWDA_SUCCESS,
 * ::LWDA_ERROR_ILWALID_CONTEXT
 * \notefnerr
 *
 * \sa
 * ::lwProfilerInitialize,
 * ::lwProfilerStart,
 * ::lwdaProfilerStop
 */
LWresult LWDAAPI lwProfilerStop(void);

/** @} */ /* END LWDA_PROFILER */

#ifdef __cplusplus
};
#endif

#undef __LWDA_DEPRECATED

#endif

