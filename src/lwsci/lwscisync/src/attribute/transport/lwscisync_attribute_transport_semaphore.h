/*
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef INCLUDED_LWSCISYNC_ATTRIBUTE_TRANSPORT_SEMAPHORE_H
#define INCLUDED_LWSCISYNC_ATTRIBUTE_TRANSPORT_SEMAPHORE_H

#include "lwscisync_attribute_core.h"
#include "lwscisync_attribute_core_cluster.h"

/** Export the semaphore attr list */
LwSciError ExportSemaAttrList(
    const LwSciSyncCoreAttrList* coreAttrList,
    LwSciSyncCoreAttrListState state,
    LwSciIpcEndpoint ipcEndpoint,
    void** txbufPtr,
    size_t* txbufSize);

/** Import the semaphore attr list */
LwSciError ImportSemaAttrList(
    LwSciSyncCoreAttrList* coreAttrList,
    LwSciSyncModule module,
    LwSciIpcEndpoint ipcEndpoint,
    LwSciSyncAttrList inputAttrList,
    bool importReconciled,
    const void* inputValue,
    size_t length);

#endif
