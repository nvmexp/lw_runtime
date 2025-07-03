/*
 * Copyright (c) 2019-2021 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */

#include <ipc_wrapper_linux.h>
#include <lwscilog.h>
#include <errno.h>
#include <sys/time.h>

IpcWrapperLinux::~IpcWrapperLinux()
{
    close();
}

LwSciError IpcWrapperLinux::send(void* buf, size_t size, int64_t timeoutNs)
{
    LwSciError err = LwSciError_Success;
    int32_t bytes = 0;
    bool done = false;
    struct timeval timeout = {0};

    if (timeoutNs > 0) {
        timeout.tv_sec = timeoutNs / 1000000000;
        timeout.tv_usec = (timeoutNs % 1000000000) / 1000;
    }

    while (!done) {
        IPC_CHECK_API(LwSciIpcGetEvent(endpoint, &event), LwSciError_Success);
        if (event & LW_SCI_IPC_EVENT_WRITE) {
            IPC_CHECK_API(LwSciIpcWrite(endpoint, buf, size, &bytes),
                          LwSciError_Success);
            if (bytes != size) {
                LWSCI_ERR_INT("bytes ", bytes);
                LWSCI_ERR_ULONG(" != size \n", size);
                return LwSciError_LwSciIplwnknown;
            }
            done = true;
        } else {
            int retval = 0;
            fd_set rfds;
            FD_ZERO(&rfds);
            FD_SET(fd, &rfds);

            retval = select(fd + 1, &rfds, NULL, NULL,
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

LwSciError IpcWrapperLinux::recvFill(void* buf, size_t size, int64_t timeout_ns)
{
    LwSciError err = LwSciError_Success;
    int32_t bytes = 0;
    bool done = false;
    struct timeval timeout = {0};

    if (timeout_ns > 0) {
        timeout.tv_sec = timeout_ns / 1000000000;
        timeout.tv_usec = (timeout_ns % 1000000000) / 1000;
    }

    while (!done) {
        IPC_CHECK_API(LwSciIpcGetEvent(endpoint, &event), LwSciError_Success);

        if (event & LW_SCI_IPC_EVENT_READ) {
            IPC_CHECK_API(LwSciIpcRead(endpoint, buf, size, &bytes),
                          LwSciError_Success);
            if (bytes != size) {
                LWSCI_ERR_INT("bytes ", bytes);
                LWSCI_ERR_ULONG(" != size \n", size);
                return LwSciError_LwSciIplwnknown;
            }
            done = true;
        } else {
            int retval = 0;
            fd_set rfds;
            FD_ZERO(&rfds);
            FD_SET(fd, &rfds);

            retval = select(fd + 1, &rfds, NULL, NULL,
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

std::unique_ptr<IpcWrapper> IpcWrapperLinux::open(const char* endpointName)
{
    LwSciError err = LwSciError_Success;
    LwSciIpcEndpoint endpoint = 0;
    int32_t fd = 0; /* fd to get event */
    struct LwSciIpcEndpointInfo info = {0};

    IPC_CHECK_API(LwSciIpcOpenEndpoint(endpointName, &endpoint),
                  LwSciError_Success);
    IPC_CHECK_API(LwSciIpcGetLinuxEventFd(endpoint, &fd), LwSciError_Success);
    IPC_CHECK_API(LwSciIpcGetEndpointInfo(endpoint, &info), LwSciError_Success);
    LwSciIpcResetEndpoint(endpoint);

    return std::unique_ptr<IpcWrapper>(
        new IpcWrapperLinux(endpoint, info, endpointName, fd));

fail:
    if (endpoint != 0) {
        LwSciIpcCloseEndpoint(endpoint);
    }
    return std::unique_ptr<IpcWrapper>(nullptr);
}
