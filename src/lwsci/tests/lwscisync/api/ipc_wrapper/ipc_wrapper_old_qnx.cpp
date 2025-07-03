/*
 * Copyright (c) 2019-2020 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/neutrino.h>

#include "ipc_wrapper_old.h"
#include "ipc_wrapper_old_core.h"

#define TEST_EVENT_CODE (0x30) /* 1 byte */

LwSciError ipcInit(const char* endpointName, IpcWrapperOld* _ipcWrapper)
{
    LwSciError err = LwSciError_Success;
    int32_t chid;
    int32_t coid;
    int32_t ret;
    IpcWrapperOld ipcWrapper = NULL;

    setupTerminationHandlers();

    ipcWrapper = (IpcWrapperOld)calloc(1U, sizeof(struct IpcWrapperOldRec));
    if (ipcWrapper == NULL) {
        err = LwSciError_InsufficientMemory;
        goto fail;
    }

    IPC_CHECK_API(LwSciIpcOpenEndpoint(endpointName, &ipcWrapper->endpoint),
                  LwSciError_Success);

    chid = ChannelCreate_r(_NTO_CHF_UNBLOCK);
    if (chid < 0) {
        err = LwSciIpcErrnoToLwSciErr(chid);
        LWSCI_ERR_INT("ChannelCreate_r: fail \n", err);
        goto fail;
    }
    ipcWrapper->chId = chid;

    coid = ConnectAttach_r(0, 0, chid, _NTO_SIDE_CHANNEL, 0);
    if (coid < 0) {
        err = LwSciIpcErrnoToLwSciErr(coid);
        LWSCI_ERR_INT("%ConnectAttach_r: fail \n", err);
        goto fail;
    }
    ipcWrapper->connId = coid;

    IPC_CHECK_API(LwSciIpcSetQnxPulseParam(
                      ipcWrapper->endpoint, ipcWrapper->connId,
                      SIGEV_PULSE_PRIO_INHERIT, TEST_EVENT_CODE, (void*)NULL),
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

    if (ipcWrapper->connId != 0) {
        (void)ConnectDetach_r(ipcWrapper->connId);
        ipcWrapper->connId = 0;
    }
    if (ipcWrapper->chId != 0) {
        (void)ChannelDestroy_r(ipcWrapper->chId);
        ipcWrapper->chId = 0;
    }

    deregisterGlobalWrapper(ipcWrapper);
    free(ipcWrapper);
}

static LwSciError waitEvent(IpcWrapperOld ipcWrapper, uint32_t value,
                            int64_t timeoutNs)
{
    LwSciError err = LwSciError_Success;
    struct _pulse pulse;
    uint32_t event = 0;
    int32_t ret = 0;

    while (true) {
        IPC_CHECK_API(LwSciIpcGetEvent(ipcWrapper->endpoint, &event),
                      LwSciError_Success);
        if (event & value) {
            break;
        }

        if (timeoutNs > 0) {
            TimerTimeout(CLOCK_MONOTONIC, _NTO_TIMEOUT_RECEIVE, NULL,
                         (uint64_t*)&timeoutNs, NULL);
        }

        ret = MsgReceivePulse_r(ipcWrapper->chId, &pulse, sizeof(pulse), NULL);
        if (ret < 0) {
            err = LwSciIpcErrnoToLwSciErr(ret);
            LWSCI_ERR_INT("%s: receive pulse error: \n", err);
            return err;
        }
        if (pulse.code != TEST_EVENT_CODE) {
            LWSCI_ERR_INT("invalid pulse: \n", pulse.code);
            return LwSciError_LwSciIplwnknown;
        }
    }

fail:
    return err;
}

LwSciError ipcSendTimeout(IpcWrapperOld ipcWrapper, void* buf, size_t size,
                          int64_t timeoutNs)
{
    LwSciError err = LwSciError_Success;
    bool done = false;
    int32_t bytes = 0;

    while (done == false) {
        IPC_CHECK_API(waitEvent(ipcWrapper, LW_SCI_IPC_EVENT_WRITE, timeoutNs),
                      LwSciError_Success);

        IPC_CHECK_API(LwSciIpcWrite(ipcWrapper->endpoint, buf, size, &bytes),
                      LwSciError_Success);

        if (bytes != size) {
            LWSCI_ERR_INT("bytes ", bytes);
            LWSCI_ERR_ULONG("!= size \n", size);
            err = LwSciError_LwSciIplwnknown;
            goto fail;
        }
        done = true;
    }

fail:
    return err;
}

LwSciError ipcSend(IpcWrapperOld ipcWrapper, void* buf, size_t size)
{
    return ipcSendTimeout(ipcWrapper, buf, size, 0);
}

LwSciError ipcRecvFillTimeout(IpcWrapperOld ipcWrapper, void* buf, size_t size,
                              int64_t timeoutNs)
{
    LwSciError err = LwSciError_Success;
    bool done = false;
    int32_t bytes = 0;

    while (done == false) {
        IPC_CHECK_API(waitEvent(ipcWrapper, LW_SCI_IPC_EVENT_READ, timeoutNs),
                      LwSciError_Success);

        IPC_CHECK_API(LwSciIpcRead(ipcWrapper->endpoint, buf, size, &bytes),
                      LwSciError_Success);

        if (bytes != size) {
            LWSCI_ERR_INT("bytes ", bytes);
            LWSCI_ERR_ULONG("!= size \n", size);
            err = LwSciError_LwSciIplwnknown;
            goto fail;
        }
        done = true;
    }

fail:
    return err;
}

LwSciError ipcRecvFill(IpcWrapperOld ipcWrapper, void* buf, size_t size)
{
    return ipcRecvFillTimeout(ipcWrapper, buf, size, 0);
}
