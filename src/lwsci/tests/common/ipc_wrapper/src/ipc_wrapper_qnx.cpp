/*
 * Copyright (c) 2019-2020 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */

#include <ipc_wrapper_qnx.h>
#include <lwsciipc_internal.h>
#include <lwscilog.h>
#include <sys/neutrino.h>

#define TEST_EVENT_CODE (0x30) /* 1 byte */

IpcWrapperQNX::~IpcWrapperQNX()
{
    close();
}

LwSciError IpcWrapperQNX::send(void* buf, size_t size, int64_t timeoutNs)
{
    LwSciError err = LwSciError_Success;
    bool done = false;
    int32_t bytes = 0;

    while (done == false) {
        IPC_CHECK_API(waitEvent(LW_SCI_IPC_EVENT_WRITE, timeoutNs),
                      LwSciError_Success);

        IPC_CHECK_API(LwSciIpcWrite(endpoint, buf, size, &bytes),
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

LwSciError IpcWrapperQNX::recvFill(void* buf, size_t size, int64_t timeoutNs)
{
    LwSciError err = LwSciError_Success;
    bool done = false;
    int32_t bytes = 0;

    while (done == false) {
        IPC_CHECK_API(waitEvent(LW_SCI_IPC_EVENT_READ, timeoutNs),
                      LwSciError_Success);

        IPC_CHECK_API(LwSciIpcRead(endpoint, buf, size, &bytes),
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

std::unique_ptr<IpcWrapper> IpcWrapperQNX::open(const char* endpointName)
{
    LwSciError err = LwSciError_Success;
    LwSciIpcEndpoint endpoint = 0;
    int32_t chid = 0;
    int32_t coid = 0;
    struct LwSciIpcEndpointInfo info = {0};
    IPC_CHECK_API(LwSciIpcOpenEndpoint(endpointName, &endpoint),
                  LwSciError_Success);

    chid = ChannelCreate_r(_NTO_CHF_UNBLOCK);
    if (chid < 0) {
        err = LwSciIpcErrnoToLwSciErr(chid);
        LWSCI_ERR_INT("ChannelCreate_r: fail \n", err);
        goto fail;
        ;
    }
    coid = ConnectAttach_r(0, 0, chid, _NTO_SIDE_CHANNEL, 0);
    if (coid < 0) {
        err = LwSciIpcErrnoToLwSciErr(coid);
        LWSCI_ERR_INT("ConnectAttach_r: fail \n", err);
        goto fail;
    }
    IPC_CHECK_API(LwSciIpcSetQnxPulseParam(endpoint, coid,
                                           SIGEV_PULSE_PRIO_INHERIT,
                                           TEST_EVENT_CODE, (void*)NULL),
                  LwSciError_Success);
    IPC_CHECK_API(LwSciIpcGetEndpointInfo(endpoint, &info), LwSciError_Success);
    LwSciIpcResetEndpoint(endpoint);
    return std::unique_ptr<IpcWrapper>(
        new IpcWrapperQNX(endpoint, info, endpointName, chid, coid));

fail:
    if (endpoint != 0) {
        LwSciIpcCloseEndpoint(endpoint);
    }
    if (coid != 0) {
        (void)ConnectDetach_r(coid);
    }
    if (chid != 0) {
        (void)ChannelDestroy_r(chid);
    }
    return std::unique_ptr<IpcWrapper>(nullptr);
}

LwSciError IpcWrapperQNX::waitEvent(uint32_t value, int64_t timeoutNs)
{
    LwSciError err = LwSciError_Success;
    struct _pulse pulse;
    uint32_t event = 0;
    int32_t ret = 0;

    while (true) {
        IPC_CHECK_API(LwSciIpcGetEvent(endpoint, &event), LwSciError_Success);
        if (event & value) {
            break;
        }

        if (timeoutNs > 0) {
            TimerTimeout(CLOCK_MONOTONIC, _NTO_TIMEOUT_RECEIVE, NULL,
                         (uint64_t*)&timeoutNs, NULL);
        }

        ret = MsgReceivePulse_r(chId, &pulse, sizeof(pulse), NULL);
        if (ret < 0) {
            err = LwSciIpcErrnoToLwSciErr(ret);
            printf("%s: receive pulse error: %d\n", __func__, err);
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

void IpcWrapperQNX::close()
{
    if (connId != 0) {
        (void)ConnectDetach_r(connId);
        connId = 0;
    }
    if (chId != 0) {
        (void)ChannelDestroy_r(chId);
        chId = 0;
    }
    IpcWrapper::close();
}
