/*
 * Copyright (c) 2019-2021 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */

#include <ipc_wrapper.h>
#include <signal.h>

#ifdef LINUX
#include "ipc_wrapper_linux.h"
#endif
#ifdef QNX
#include "ipc_wrapper_qnx.h"
#endif

std::mutex IpcWrapper::initCountMutex;
int IpcWrapper::initCount = 0;

void IpcWrapper::close()
{
    if (endpoint) {
        LwSciIpcCloseEndpoint(endpoint);
        endpoint = 0;
    }

    const std::lock_guard<std::mutex> lock(initCountMutex);
    if (initCount > 0) {
        initCount--;
        if (initCount == 0) {
            IpcWrapper::deinit();
        }
    }
}

void IpcWrapper::init()
{
    LwSciError err = LwSciIpcInit();
    if (err != LwSciError_Success) {
        printf("Fail to init ipc\n");
    }
}

std::unique_ptr<IpcWrapper> IpcWrapper::open(const char* endpointName)
{
    const std::lock_guard<std::mutex> lock(initCountMutex);
    if (initCount == 0) {
        IpcWrapper::init();
    }
    initCount++;
#ifdef LINUX
    return IpcWrapperLinux::open(endpointName);
#endif
#ifdef QNX
    return IpcWrapperQNX::open(endpointName);
#endif
fail:
    return std::unique_ptr<IpcWrapper>(nullptr);
}

void IpcWrapper::deinit()
{
    LwSciIpcDeinit();
}
