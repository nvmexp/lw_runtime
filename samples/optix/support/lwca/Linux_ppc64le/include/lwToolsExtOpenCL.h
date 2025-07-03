/*
* Copyright 2009-2017 LWPU Corporation.  All rights reserved.
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

#ifndef LWTOOLSEXT_OPENCL_H_
#define LWTOOLSEXT_OPENCL_H_

#include <CL/cl.h>

#include "lwToolsExt.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* ========================================================================= */
/** \name Functions for OpenCL Resource Naming
 */
/** \addtogroup RESOURCE_NAMING
 * \section RESOURCE_NAMING_OPENCL OpenCL Resource Naming
 *
 * This section covers the API functions that allow to annotate OpenCL resources
 * with user-provided names.
 *
 * @{
 */

/*  ------------------------------------------------------------------------- */
/* \cond SHOW_HIDDEN 
* \brief Used to build a non-colliding value for resource types separated class
* \version \LWTX_VERSION_2
*/
#define LWTX_RESOURCE_CLASS_OPENCL 6 
/** \endcond */

/*  ------------------------------------------------------------------------- */
/** \brief Resource types for OpenCL
*/
typedef enum lwtxResourceOpenCLType_t
{
    LWTX_RESOURCE_TYPE_OPENCL_DEVICE = LWTX_RESOURCE_MAKE_TYPE(OPENCL, 1),
    LWTX_RESOURCE_TYPE_OPENCL_CONTEXT = LWTX_RESOURCE_MAKE_TYPE(OPENCL, 2),
    LWTX_RESOURCE_TYPE_OPENCL_COMMANDQUEUE = LWTX_RESOURCE_MAKE_TYPE(OPENCL, 3),
    LWTX_RESOURCE_TYPE_OPENCL_MEMOBJECT = LWTX_RESOURCE_MAKE_TYPE(OPENCL, 4),
    LWTX_RESOURCE_TYPE_OPENCL_SAMPLER = LWTX_RESOURCE_MAKE_TYPE(OPENCL, 5),
    LWTX_RESOURCE_TYPE_OPENCL_PROGRAM = LWTX_RESOURCE_MAKE_TYPE(OPENCL, 6),
    LWTX_RESOURCE_TYPE_OPENCL_EVENT = LWTX_RESOURCE_MAKE_TYPE(OPENCL, 7)
} lwtxResourceOpenCLType_t;


/* ------------------------------------------------------------------------- */
/** \brief Annotates an OpenCL device.
 *
 * Allows to associate an OpenCL device with a user-provided name.
 *
 * \param device - The handle of the OpenCL device to name.
 * \param name   - The name of the OpenCL device.
 *
 * \version \LWTX_VERSION_1
 * @{ */
LWTX_DECLSPEC void LWTX_API lwtxNameClDeviceA(cl_device_id device, const char* name);
LWTX_DECLSPEC void LWTX_API lwtxNameClDeviceW(cl_device_id device, const wchar_t* name);
/** @} */

/* ------------------------------------------------------------------------- */
/** \brief Annotates an OpenCL context.
 *
 * Allows to associate an OpenCL context with a user-provided name.
 *
 * \param context - The handle of the OpenCL context to name.
 * \param name    - The name of the OpenCL context.
 *
 * \version \LWTX_VERSION_1
 * @{ */
LWTX_DECLSPEC void LWTX_API lwtxNameClContextA(cl_context context, const char* name);
LWTX_DECLSPEC void LWTX_API lwtxNameClContextW(cl_context context, const wchar_t* name);
/** @} */

/* ------------------------------------------------------------------------- */
/** \brief Annotates an OpenCL command queue.
 *
 * Allows to associate an OpenCL command queue with a user-provided name.
 *
 * \param command_queue - The handle of the OpenCL command queue to name.
 * \param name          - The name of the OpenCL command queue.
 *
 * \version \LWTX_VERSION_1
 * @{ */
LWTX_DECLSPEC void LWTX_API lwtxNameClCommandQueueA(cl_command_queue command_queue, const char* name);
LWTX_DECLSPEC void LWTX_API lwtxNameClCommandQueueW(cl_command_queue command_queue, const wchar_t* name);
/** @} */

/* ------------------------------------------------------------------------- */
/** \brief Annotates an OpenCL memory object.
 *
 * Allows to associate an OpenCL memory object with a user-provided name.
 *
 * \param memobj - The handle of the OpenCL memory object to name.
 * \param name   - The name of the OpenCL memory object.
 *
 * \version \LWTX_VERSION_1
 * @{ */
LWTX_DECLSPEC void LWTX_API lwtxNameClMemObjectA(cl_mem memobj, const char* name);
LWTX_DECLSPEC void LWTX_API lwtxNameClMemObjectW(cl_mem memobj, const wchar_t* name);
/** @} */

/* ------------------------------------------------------------------------- */
/** \brief Annotates an OpenCL sampler.
 *
 * Allows to associate an OpenCL sampler with a user-provided name.
 *
 * \param sampler - The handle of the OpenCL sampler to name.
 * \param name    - The name of the OpenCL sampler.
 *
 * \version \LWTX_VERSION_1
 * @{ */
LWTX_DECLSPEC void LWTX_API lwtxNameClSamplerA(cl_sampler sampler, const char* name);
LWTX_DECLSPEC void LWTX_API lwtxNameClSamplerW(cl_sampler sampler, const wchar_t* name);
/** @} */

/* ------------------------------------------------------------------------- */
/** \brief Annotates an OpenCL program.
 *
 * Allows to associate an OpenCL program with a user-provided name.
 *
 * \param program - The handle of the OpenCL program to name.
 * \param name    - The name of the OpenCL program.
 *
 * \code
 * cpProgram = clCreateProgramWithSource(cxGPUContext, 1,
 *     (const char **) &cSourceCL, &program_length, &ciErrNum);
 * shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
 * lwtxNameClProgram(cpProgram, L"PROGRAM_NAME");
 * \endcode
 *
 * \version \LWTX_VERSION_1
 * @{ */
LWTX_DECLSPEC void LWTX_API lwtxNameClProgramA(cl_program program, const char* name);
LWTX_DECLSPEC void LWTX_API lwtxNameClProgramW(cl_program program, const wchar_t* name);
/** @} */

/* ------------------------------------------------------------------------- */
/** \brief Annotates an OpenCL event.
 *
 * Allows to associate an OpenCL event with a user-provided name.
 *
 * \param evnt - The handle of the OpenCL event to name.
 * \param name - The name of the OpenCL event.
 *
 * \version \LWTX_VERSION_1
 * @{ */
LWTX_DECLSPEC void LWTX_API lwtxNameClEventA(cl_event evnt, const char* name);
LWTX_DECLSPEC void LWTX_API lwtxNameClEventW(cl_event evnt, const wchar_t* name);
/** @} */

/** @} */ /* END RESOURCE_NAMING */

/* ========================================================================= */
#ifdef UNICODE
  #define lwtxNameClDevice        lwtxNameClDeviceW
  #define lwtxNameClContext       lwtxNameClContextW
  #define lwtxNameClCommandQueue  lwtxNameClCommandQueueW
  #define lwtxNameClMemObject     lwtxNameClMemObjectW
  #define lwtxNameClSampler       lwtxNameClSamplerW
  #define lwtxNameClProgram       lwtxNameClProgramW
  #define lwtxNameClEvent         lwtxNameClEventW
#else
  #define lwtxNameClDevice        lwtxNameClDeviceA
  #define lwtxNameClContext       lwtxNameClContextA
  #define lwtxNameClCommandQueue  lwtxNameClCommandQueueA
  #define lwtxNameClMemObject     lwtxNameClMemObjectA
  #define lwtxNameClSampler       lwtxNameClSamplerA
  #define lwtxNameClProgram       lwtxNameClProgramA
  #define lwtxNameClEvent         lwtxNameClEventA
#endif

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* LWTOOLSEXT_OPENCL_H_ */
