/*
 * Copyright (c) 2018-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCIBUF_INTERNAL_TEGRA_H
#define INCLUDED_LWSCIBUF_INTERNAL_TEGRA_H

/**
 * NOTE:
 * If an element including this header file is being built from perforce,
 * it needs to define 'LW_DEFINES += LW_SCI_DESKTOP_COMPATIBLE_HEADERS' in its
 * makefile.
 */
#ifdef LW_SCI_DESKTOP_COMPATIBLE_HEADERS
#include "mobile_common.h"
#else
#if (LW_IS_SAFETY == 0)
#include "lwrm_memmgr.h"
#else
#include "lwrm_memmgr_safe.h"
#endif //LW_IS_SAFETY
#endif //LW_SCI_DESKTOP_COMPATIBLE_HEADERS

#if defined(__cplusplus)
extern "C"
{
#endif

/**
 * @defgroup lwscibuf_obj_datatype_int LwSciBuf object internal datatypes
 * List of all LwSciBuf object internal data types
 * @{
 */

/**
 * @brief Structure to represent buffer handle referencing
 *        the allocated buffer.
 *
 * @implements{18840138}
 */
typedef struct {
   /**
    * Handle to buffer allocated through liblwrm_mem API.
    */
    LwRmMemHandle memHandle;
} LwSciBufRmHandle;

/**
 * @}
 */

#if defined(__cplusplus)
}
#endif

#endif /* INCLUDED_LWSCIBUF_INTERNAL_TEGRA_H */
