 /* Copyright 2009-2016 LWPU Corporation.  All rights reserved. 
  * 
  * NOTICE TO LICENSEE: 
  * 
  * The source code and/or documentation ("Licensed Deliverables") are 
  * subject to LWPU intellectual property rights under U.S. and 
  * international Copyright laws. 
  * 
  * The Licensed Deliverables contained herein are PROPRIETARY and 
  * CONFIDENTIAL to LWPU and are being provided under the terms and 
  * conditions of a form of LWPU software license agreement by and 
  * between LWPU and Licensee ("License Agreement") or electronically 
  * accepted by Licensee.  Notwithstanding any terms or conditions to 
  * the contrary in the License Agreement, reproduction or disclosure 
  * of the Licensed Deliverables to any third party without the express 
  * written consent of LWPU is prohibited. 
  * 
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE 
  * LICENSE AGREEMENT, LWPU MAKES NO REPRESENTATION ABOUT THE 
  * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE 
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
  * C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government 
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
#ifndef LW_NPPCORE_H
#define LW_NPPCORE_H

#include <lwda_runtime_api.h>

/**
 * \file nppcore.h
 * Basic NPP functionality. 
 *  This file contains functions to query the NPP version as well as 
 *  info about the LWCA compute capabilities on a given computer.
 */
 
#include "nppdefs.h"

#ifdef __cplusplus
extern "C" {
#endif
 
/** \defgroup core_npp NPP Core
 * Basic functions for library management, in particular library version
 * and device property query functions.
 * @{
 */

/**
 * Get the NPP library version.
 *
 * \return A struct containing separate values for major and minor revision 
 *      and build number.
 */
const NppLibraryVersion * 
nppGetLibVersion(void);

/**
 * What LWCA compute model is supported by the active LWCA device?
 * 
 * Before trying to call any NPP functions, the user should make a call
 * this function to ensure that the current machine has a LWCA capable device.
 *
 * \return An enum value representing if a LWCA capable device was found and what
 *      level of compute capabilities it supports.
 */
NppGpuComputeCapability 
nppGetGpuComputeCapability(void);

/**
 * Get the number of Streaming Multiprocessors (SM) on the active LWCA device.
 *
 * \return Number of SMs of the default LWCA device.
 */
int 
nppGetGpuNumSMs(void);

/**
 * Get the maximum number of threads per block on the active LWCA device.
 *
 * \return Maximum number of threads per block on the active LWCA device.
 */
int 
nppGetMaxThreadsPerBlock(void);

/**
 * Get the maximum number of threads per SM for the active GPU
 *
 * \return Maximum number of threads per SM for the active GPU
 */
int 
nppGetMaxThreadsPerSM(void);

/**
 * Get the maximum number of threads per SM, maximum threads per block, and number of SMs for the active GPU
 *
 * \return lwdaSuccess for success, -1 for failure
 */
int 
nppGetGpuDeviceProperties(int * pMaxThreadsPerSM, int * pMaxThreadsPerBlock, int * pNumberOfSMs);

/** 
 * Get the name of the active LWCA device.
 *
 * \return Name string of the active graphics-card/compute device in a system.
 */
const char * 
nppGetGpuName(void);

/**
 * Get the NPP LWCA stream.
 * NPP enables conlwrrent device tasks via a global stream state varible.
 * The NPP stream by default is set to stream 0, i.e. non-conlwrrent mode.
 * A user can set the NPP stream to any valid LWCA stream. All LWCA commands
 * issued by NPP (e.g. kernels launched by the NPP library) are then
 * issed to that NPP stream.
 */
lwdaStream_t
nppGetStream(void);

/**
 * Get the number of SMs on the device associated with the current NPP LWCA stream.
 * NPP enables conlwrrent device tasks via a global stream state varible.
 * The NPP stream by default is set to stream 0, i.e. non-conlwrrent mode.
 * A user can set the NPP stream to any valid LWCA stream. All LWCA commands
 * issued by NPP (e.g. kernels launched by the NPP library) are then
 * issed to that NPP stream.  This call avoids a lwdaGetDeviceProperties() call.
 */
unsigned int
nppGetStreamNumSMs(void);

/**
 * Get the maximum number of threads per SM on the device associated with the current NPP LWCA stream.
 * NPP enables conlwrrent device tasks via a global stream state varible.
 * The NPP stream by default is set to stream 0, i.e. non-conlwrrent mode.
 * A user can set the NPP stream to any valid LWCA stream. All LWCA commands
 * issued by NPP (e.g. kernels launched by the NPP library) are then
 * issed to that NPP stream.  This call avoids a lwdaGetDeviceProperties() call.
 */
unsigned int
nppGetStreamMaxThreadsPerSM(void);

/**
 * Set the NPP LWCA stream.  This function now returns an error if a problem oclwrs with Lwca stream management. 
 *   This function should only be called if a call to nppGetStream() returns a stream number which is different from
 *   the desired stream since unnecessarily flushing the current stream can significantly affect performance.
 * \see nppGetStream()
 */
NppStatus
nppSetStream(lwdaStream_t hStream);


/** @} Module LabelCoreNPP */


#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LW_NPPCORE_H */
