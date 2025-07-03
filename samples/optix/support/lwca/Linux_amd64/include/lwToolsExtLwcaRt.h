/*
* Copyright 2009-2017  LWPU Corporation.  All rights reserved.
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

#ifndef LWTOOLSEXT_LWDART_H_
#define LWTOOLSEXT_LWDART_H_

#include "lwca.h"
#include "driver_types.h"

#include "lwToolsExt.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* ========================================================================= */
/** \name Functions for LWCA Resource Naming
*/
/** \addtogroup RESOURCE_NAMING
 * \section RESOURCE_NAMING_LWDART LWCA Runtime Resource Naming
 *
 * This section covers the API functions that allow to annotate LWCA resources
 * with user-provided names.
 *
 * @{
 */

/*  ------------------------------------------------------------------------- */
/* \cond SHOW_HIDDEN 
* \brief Used to build a non-colliding value for resource types separated class
* \version \LWTX_VERSION_2
*/
#define LWTX_RESOURCE_CLASS_LWDART 5
/** \endcond */

/*  ------------------------------------------------------------------------- */
/** \brief Resource types for LWDART
*/
typedef enum lwtxResourceLWDARTType_t
{
    LWTX_RESOURCE_TYPE_LWDART_DEVICE = LWTX_RESOURCE_MAKE_TYPE(LWDART, 0), /* int device */
    LWTX_RESOURCE_TYPE_LWDART_STREAM = LWTX_RESOURCE_MAKE_TYPE(LWDART, 1), /* lwdaStream_t */
    LWTX_RESOURCE_TYPE_LWDART_EVENT = LWTX_RESOURCE_MAKE_TYPE(LWDART, 2)  /* lwdaEvent_t */
} lwtxResourceLWDARTType_t;


/* ------------------------------------------------------------------------- */
/** \brief Annotates a LWCA device.
 *
 * Allows the user to associate a LWCA device with a user-provided name.
 *
 * \param device - The id of the LWCA device to name.
 * \param name   - The name of the LWCA device.
 *
 * \version \LWTX_VERSION_1
 * @{ */
LWTX_DECLSPEC void LWTX_API lwtxNameLwdaDeviceA(int device, const char* name);
LWTX_DECLSPEC void LWTX_API lwtxNameLwdaDeviceW(int device, const wchar_t* name);
/** @} */

/* ------------------------------------------------------------------------- */
/** \brief Annotates a LWCA stream.
 *
 * Allows the user to associate a LWCA stream with a user-provided name.
 *
 * \param stream - The handle of the LWCA stream to name.
 * \param name   - The name of the LWCA stream.
 *
 * \version \LWTX_VERSION_1
 * @{ */
LWTX_DECLSPEC void LWTX_API lwtxNameLwdaStreamA(lwdaStream_t stream, const char* name);
LWTX_DECLSPEC void LWTX_API lwtxNameLwdaStreamW(lwdaStream_t stream, const wchar_t* name);
/** @} */

/* ------------------------------------------------------------------------- */
/** \brief Annotates a LWCA event.
 *
 * Allows the user to associate a LWCA event with a user-provided name.
 *
 * \param event - The handle of the LWCA event to name.
 * \param name  - The name of the LWCA event.
 *
 * \version \LWTX_VERSION_1
 * @{ */
LWTX_DECLSPEC void LWTX_API lwtxNameLwdaEventA(lwdaEvent_t event, const char* name);
LWTX_DECLSPEC void LWTX_API lwtxNameLwdaEventW(lwdaEvent_t event, const wchar_t* name);
/** @} */

/** @} */ /* END RESOURCE_NAMING */

/* ========================================================================= */
#ifdef UNICODE
  #define lwtxNameLwdaDevice lwtxNameLwdaDeviceW
  #define lwtxNameLwdaStream lwtxNameLwdaStreamW
  #define lwtxNameLwdaEvent  lwtxNameLwdaEventW
#else
  #define lwtxNameLwdaDevice lwtxNameLwdaDeviceA
  #define lwtxNameLwdaStream lwtxNameLwdaStreamA
  #define lwtxNameLwdaEvent  lwtxNameLwdaEventA
#endif

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* LWTOOLSEXT_LWDART_H_ */
