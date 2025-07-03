/*
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

/**
 * \file
 * \brief <b>LwSciSync core semaphore attribute definitions</b>
 *
 * @b Description: This file declares semaphore related core items
 */

#ifndef INCLUDED_LWSCISYNC_ATTRIBUTE_CORE_SEMAPHORE_H
#define INCLUDED_LWSCISYNC_ATTRIBUTE_CORE_SEMAPHORE_H

#include "lwscisync_attribute_core.h"

/**
 * \brief Copy all semaphore fields in the attr list
 *
 * \param[in] coreAttrList LwSciSync attr list to copy
 * \param[out] newCoreAttrList attr list to copy into
 *
 * \return LwSciError
 * - LwSciError_Success if successful
 * - error code in case of underlying LwSciBuf failure
 */
LwSciError CopySemaAttrList(
    const LwSciSyncCoreAttrList* coreAttrList,
    LwSciSyncCoreAttrList* newCoreAttrList);

#endif
