/*
 * Copyright (c) 2019-2020 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */

#ifndef INCLUDED_IPC_WRAPPER_OLD_CORE_H
#define INCLUDED_IPC_WRAPPER_OLD_CORE_H

#include <lwscilog.h>

#define IPC_CHECK_API(func, expected)                                          \
    ({                                                                         \
        err = (func);                                                          \
        if (err != (expected)) {                                               \
            LWSCI_ERR_HEXUINT(#func " failed: \n", err);                       \
            goto fail;                                                         \
        }                                                                      \
    })

#ifdef __cplusplus
extern "C" {
#endif

struct IpcWrapperOldRec {
    LwSciIpcEndpoint endpoint;  /* LwIPC handle */
    struct LwSciIpcEndpointInfo info; /* channel info */

#ifdef QNX
    int32_t chId;   /* channel id to get event */
    int32_t connId;   /* connection id to send event in library */
#endif

#ifdef LINUX
    int32_t fd;   /* fd to get event */
    uint32_t event;
#endif
};

void sigHandler(int sigNum);
void setupTerminationHandlers(void);

int registerGlobalWrapper(IpcWrapperOld ipcWrapper);
void deregisterGlobalWrapper(IpcWrapperOld ipcWrapper);

#ifdef __cplusplus
}
#endif

#endif
