/*
 * Copyright (c) 2019-2020 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */
#ifndef INCLUDED_IPC_WRAPPER_OLD_H
#define INCLUDED_IPC_WRAPPER_OLD_H

#include <lwsciipc_internal.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct IpcWrapperOldRec* IpcWrapperOld;

LwSciError ipcInit(const char* endpointName, IpcWrapperOld* ipcWrapper);

void ipcDeinit(IpcWrapperOld ipcWrapper);

LwSciError ipcSend(IpcWrapperOld ipcWrapper, void* buf, size_t size);

LwSciError ipcSendTimeout(IpcWrapperOld ipcWrapper, void* buf, size_t size,
                          int64_t timeoutNs);

LwSciError ipcRecvFill(IpcWrapperOld ipcWrapper, void* buf, size_t size);

LwSciError ipcRecvFillTimeout(IpcWrapperOld ipcWrapper, void* buf, size_t size,
                              int64_t timeoutNs);

LwSciIpcEndpoint ipcWrapperGetEndpoint(IpcWrapperOld ipcWrapper);

#ifdef __cplusplus
}
#endif

#endif
