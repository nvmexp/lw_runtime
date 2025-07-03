/*
 * Copyright (c) 2019-2020, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

/**
 * \file
 * \brief <b>LwSciSync core semaphore attribute implementation</b>
 *
 * @b Description: This file implements core semaphore attribute logic
 */

#include "../inc/lwscisync_attribute_core_cluster.h"
#include "lwscisync_attribute_core_semaphore.h"

LwSciError CopySemaAttrList(
    const LwSciSyncCoreAttrList* coreAttrList,
    LwSciSyncCoreAttrList* newCoreAttrList)
{
    LwSciError error = LwSciError_Success;

    /** Copy semaAttrList */
    if (NULL != coreAttrList->semaAttrList) {
        error = LwSciBufAttrListClone(coreAttrList->semaAttrList,
                &newCoreAttrList->semaAttrList);
        if (LwSciError_Success != error) {
            goto fn_exit;
        }
    }

fn_exit:
    return error;
}

void LwSciSyncCoreAttrListGetSemaAttrList(
    LwSciSyncAttrList syncAttrList,
    LwSciBufAttrList* semaAttrList)
{
    LwSciSyncCoreAttrListObj* objAttrList = NULL;

    LwSciSyncCoreAttrListGetObjFromRef(syncAttrList, &objAttrList);

    *semaAttrList = objAttrList->coreAttrList->semaAttrList;
}
