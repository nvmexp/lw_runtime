/*
 * Copyright (c) 2019-2021 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */

#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "ipc_wrapper_old.h"
#include "ipc_wrapper_old_core.h"

LwSciError ipcInit(const char* endpointName, IpcWrapperOld* _ipcWrapper)
{
    LwSciError err = LwSciError_Success;
    IpcWrapperOld ipcWrapper = NULL;

    setupTerminationHandlers();

    ipcWrapper = (IpcWrapperOld)calloc(1U, sizeof(struct IpcWrapperOldRec));
    if (ipcWrapper == NULL) {
        err = LwSciError_InsufficientMemory;
        goto fail;
    }

    IPC_CHECK_API(LwSciIpcOpenEndpoint(endpointName, &ipcWrapper->endpoint),
                  LwSciError_Success);

    IPC_CHECK_API(
        LwSciIpcGetLinuxEventFd(ipcWrapper->endpoint, &ipcWrapper->fd),
        LwSciError_Success);

    IPC_CHECK_API(
        LwSciIpcGetEndpointInfo(ipcWrapper->endpoint, &ipcWrapper->info),
        LwSciError_Success);

    LwSciIpcResetEndpoint(ipcWrapper->endpoint);

    *_ipcWrapper = ipcWrapper;
    if (0 != registerGlobalWrapper(ipcWrapper)) {
        LWSCI_ERR_STR("too many ipcWrappers\n");
        err = LwSciError_InsufficientResource;
        goto fail;
    }

fail:
    return err;
}

void ipcDeinit(IpcWrapperOld ipcWrapper)
{
    LwSciIpcCloseEndpoint(ipcWrapper->endpoint);
    deregisterGlobalWrapper(ipcWrapper);
    free(ipcWrapper);
}

LwSciError ipcSendTimeout(IpcWrapperOld ipcWrapper, void* buf, size_t size,
                          int64_t timeoutNs)
{
    LwSciError err = LwSciError_Success;
    int32_t bytes = 0;
    bool done = false;
    fd_set rfds;
    int retval = 0;
    struct timeval timeout = {0};

    if (timeoutNs > 0) {
        timeout.tv_sec = timeoutNs / 1000000000;
        timeout.tv_usec = (timeoutNs % 1000000000) / 1000;
    }

    while (!done) {
        IPC_CHECK_API(
            LwSciIpcGetEvent(ipcWrapper->endpoint, &ipcWrapper->event),
            LwSciError_Success);
        if (ipcWrapper->event & LW_SCI_IPC_EVENT_WRITE) {
            IPC_CHECK_API(
                LwSciIpcWrite(ipcWrapper->endpoint, buf, size, &bytes),
                LwSciError_Success);
            if (bytes != size) {
                LWSCI_ERR_INT("bytes ", bytes);
                LWSCI_ERR_ULONG(" != size \n", size);
                return LwSciError_LwSciIplwnknown;
            }
            done = true;
        } else {
            FD_ZERO(&rfds);
            FD_SET(ipcWrapper->fd, &rfds);

            retval = select(ipcWrapper->fd + 1, &rfds, NULL, NULL,
                            timeoutNs > 0 ? &timeout : NULL);
            if (retval == 0) {
                return LwSciError_Timeout;
            } else if ((retval < 0) & (errno != EINTR)) {
                return LwSciError_LwSciIplwnknown;
            }
        }
    }
    return LwSciError_Success;
fail:
    return err;
}

LwSciError ipcSend(IpcWrapperOld ipcWrapper, void* buf, size_t size)
{
    return ipcSendTimeout(ipcWrapper, buf, size, 0);
}

LwSciError ipcRecvFillTimeout(IpcWrapperOld ipcWrapper, void* buf, size_t size,
                              int64_t timeout_ns)
{
    LwSciError err = LwSciError_Success;
    int32_t bytes = 0;
    bool done = false;
    fd_set rfds;
    int retval = 0;
    struct timeval timeout = {0};

    if (timeout_ns > 0) {
        timeout.tv_sec = timeout_ns / 1000000000;
        timeout.tv_usec = (timeout_ns % 1000000000) / 1000;
    }

    while (!done) {
        IPC_CHECK_API(
            LwSciIpcGetEvent(ipcWrapper->endpoint, &ipcWrapper->event),
            LwSciError_Success);

        if (ipcWrapper->event & LW_SCI_IPC_EVENT_READ) {
            IPC_CHECK_API(LwSciIpcRead(ipcWrapper->endpoint, buf, size, &bytes),
                          LwSciError_Success);
            if (bytes != size) {
                LWSCI_ERR_INT("bytes ", bytes);
                LWSCI_ERR_ULONG(" != size \n", size);
                return LwSciError_LwSciIplwnknown;
            }
            done = true;
        } else {
            FD_ZERO(&rfds);
            FD_SET(ipcWrapper->fd, &rfds);

            retval = select(ipcWrapper->fd + 1, &rfds, NULL, NULL,
                            timeout_ns > 0 ? &timeout : NULL);
            if (retval == 0) {
                return LwSciError_Timeout;
            } else if ((retval < 0) & (errno != EINTR)) {
                return LwSciError_LwSciIplwnknown;
            }
        }
    }
    return LwSciError_Success;
fail:
    return err;
}

LwSciError ipcRecvFill(IpcWrapperOld ipcWrapper, void* buf, size_t size)
{
    return ipcRecvFillTimeout(ipcWrapper, buf, size, 0);
}
