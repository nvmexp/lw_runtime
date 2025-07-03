/*
 * Copyright 2010-2020 LWPU Corporation.  All rights reserved.
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

#if !defined(_LWPTI_RESULT_H_)
#define _LWPTI_RESULT_H_

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
 * \defgroup LWPTI_RESULT_API LWPTI Result Codes
 * Error and result codes returned by LWPTI functions.
 * @{
 */

/**
 * \brief LWPTI result codes.
 *
 * Error and result codes returned by LWPTI functions.
 */
typedef enum {
    /**
     * No error.
     */
    LWPTI_SUCCESS                                       = 0,
    /**
     * One or more of the parameters is invalid.
     */
    LWPTI_ERROR_ILWALID_PARAMETER                       = 1,
    /**
     * The device does not correspond to a valid LWCA device.
     */
    LWPTI_ERROR_ILWALID_DEVICE                          = 2,
    /**
     * The context is NULL or not valid.
     */
    LWPTI_ERROR_ILWALID_CONTEXT                         = 3,
    /**
     * The event domain id is invalid.
     */
    LWPTI_ERROR_ILWALID_EVENT_DOMAIN_ID                 = 4,
    /**
     * The event id is invalid.
     */
    LWPTI_ERROR_ILWALID_EVENT_ID                        = 5,
    /**
     * The event name is invalid.
     */
    LWPTI_ERROR_ILWALID_EVENT_NAME                      = 6,
    /**
     * The current operation cannot be performed due to dependency on
     * other factors.
     */
    LWPTI_ERROR_ILWALID_OPERATION                       = 7,
    /**
     * Unable to allocate enough memory to perform the requested
     * operation.
     */
    LWPTI_ERROR_OUT_OF_MEMORY                           = 8,
    /**
     * An error oclwrred on the performance monitoring hardware.
     */
    LWPTI_ERROR_HARDWARE                                = 9,
    /**
     * The output buffer size is not sufficient to return all
     * requested data.
     */
    LWPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT           = 10,
    /**
     * API is not implemented.
     */
    LWPTI_ERROR_API_NOT_IMPLEMENTED                     = 11,
    /**
     * The maximum limit is reached.
     */
    LWPTI_ERROR_MAX_LIMIT_REACHED                       = 12,
    /**
     * The object is not yet ready to perform the requested operation.
     */
    LWPTI_ERROR_NOT_READY                               = 13,
    /**
     * The current operation is not compatible with the current state
     * of the object
     */
    LWPTI_ERROR_NOT_COMPATIBLE                          = 14,
    /**
     * LWPTI is unable to initialize its connection to the LWCA
     * driver.
     */
    LWPTI_ERROR_NOT_INITIALIZED                         = 15,
    /**
     * The metric id is invalid.
     */
    LWPTI_ERROR_ILWALID_METRIC_ID                        = 16,
    /**
     * The metric name is invalid.
     */
    LWPTI_ERROR_ILWALID_METRIC_NAME                      = 17,
    /**
     * The queue is empty.
     */
    LWPTI_ERROR_QUEUE_EMPTY                              = 18,
    /**
     * Invalid handle (internal?).
     */
    LWPTI_ERROR_ILWALID_HANDLE                           = 19,
    /**
     * Invalid stream.
     */
    LWPTI_ERROR_ILWALID_STREAM                           = 20,
    /**
     * Invalid kind.
     */
    LWPTI_ERROR_ILWALID_KIND                             = 21,
    /**
     * Invalid event value.
     */
    LWPTI_ERROR_ILWALID_EVENT_VALUE                      = 22,
    /**
     * LWPTI is disabled due to conflicts with other enabled profilers
     */
    LWPTI_ERROR_DISABLED                                 = 23,
    /**
     * Invalid module.
     */
    LWPTI_ERROR_ILWALID_MODULE                           = 24,
    /**
     * Invalid metric value.
     */
    LWPTI_ERROR_ILWALID_METRIC_VALUE                     = 25,
    /**
     * The performance monitoring hardware is in use by other client.
     */
    LWPTI_ERROR_HARDWARE_BUSY                            = 26,
    /**
     * The attempted operation is not supported on the current
     * system or device.
     */
    LWPTI_ERROR_NOT_SUPPORTED                            = 27,
    /**
     * Unified memory profiling is not supported on the system.
     * Potential reason could be unsupported OS or architecture.
     */
    LWPTI_ERROR_UM_PROFILING_NOT_SUPPORTED               = 28,
    /**
     * Unified memory profiling is not supported on the device
     */
    LWPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_DEVICE     = 29,
    /**
     * Unified memory profiling is not supported on a multi-GPU
     * configuration without P2P support between any pair of devices
     */
    LWPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_NON_P2P_DEVICES = 30,
    /**
     * Unified memory profiling is not supported under the
     * Multi-Process Service (MPS) environment. LWCA 7.5 removes this
     * restriction.
     */
    LWPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_WITH_MPS      = 31,
    /**
     * In LWCA 9.0, devices with compute capability 7.0 don't
     * support CDP tracing
     */
    LWPTI_ERROR_CDP_TRACING_NOT_SUPPORTED                = 32,
    /**
     * Profiling on virtualized GPU is not supported.
     */
    LWPTI_ERROR_VIRTUALIZED_DEVICE_NOT_SUPPORTED         = 33,
    /**
     * Profiling results might be incorrect for LWCA applications
     * compiled with lwcc version older than 9.0 for devices with
     * compute capability 6.0 and 6.1.
     * Profiling session will continue and LWPTI will notify it using this error code.
     * User is advised to recompile the application code with lwcc version 9.0 or later.
     * Ignore this warning if code is already compiled with the recommended lwcc version.
     */
    LWPTI_ERROR_LWDA_COMPILER_NOT_COMPATIBLE             = 34,
    /**
     * User doesn't have sufficient privileges which are required to
     * start the profiling session.
     * One possible reason for this may be that the LWPU driver or your system
     * administrator may have restricted access to the LWPU GPU performance counters.
     * To learn how to resolve this issue and find more information, please visit
     * https://developer.lwpu.com/LWPTI_ERROR_INSUFFICIENT_PRIVILEGES
     */
    LWPTI_ERROR_INSUFFICIENT_PRIVILEGES                  = 35,
    /**
     * Old profiling api's are not supported with new profiling api's
     */
    LWPTI_ERROR_OLD_PROFILER_API_INITIALIZED             = 36,
    /**
     * Missing definition of the OpenACC API routine in the linked OpenACC library.
     *
     * One possible reason is that OpenACC library is linked statically in the
     * user application, which might not have the definition of all the OpenACC
     * API routines needed for the OpenACC profiling, as compiler might ignore
     * definitions for the functions not used in the application. This issue
     * can be mitigated by linking the OpenACC library dynamically.
     */
    LWPTI_ERROR_OPENACC_UNDEFINED_ROUTINE                = 37,
    /**
     * Legacy LWPTI Profiling is not supported on devices with Compute Capability
     * 7.5 or higher (Turing+). Using this error to specify this case and differentiate
     * it from other errors.
     */
    LWPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED            = 38,
    /**
     * LWPTI doesn't allow multiple callback subscribers. Only a single subscriber
     * can be registered at a time.
     * Same error code is used when application is launched using LWPU tools
     * like lwperf, Visual Profiler, Nsight Systems, Nsight Compute, lwca-gdb and
     * lwca-memcheck.
     */
    LWPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED       = 39,
    /**
     * An unknown internal error has oclwrred.
     */
    LWPTI_ERROR_UNKNOWN                                  = 999,
    LWPTI_ERROR_FORCE_INT                                = 0x7fffffff
} LWptiResult;

/**
 * \brief Get the descriptive string for a LWptiResult.
 *
 * Return the descriptive string for a LWptiResult in \p *str.
 * \note \b Thread-safety: this function is thread safe.
 *
 * \param result The result to get the string for
 * \param str Returns the string
 *
 * \retval LWPTI_SUCCESS on success
 * \retval LWPTI_ERROR_ILWALID_PARAMETER if \p str is NULL or \p
 * result is not a valid LWptiResult
 */
LWptiResult LWPTIAPI lwptiGetResultString(LWptiResult result, const char **str);

/** @} */ /* END LWPTI_RESULT_API */

#if defined(__GNUC__) && defined(LWPTI_LIB)
    #pragma GCC visibility pop
#endif

#if defined(__cplusplus)
}
#endif

#endif /*_LWPTI_RESULT_H_*/


