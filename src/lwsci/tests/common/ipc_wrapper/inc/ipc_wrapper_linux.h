/*
 * Copyright (c) 2019-2022 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */
#ifndef INCLUDED_IPC_WRAPPER_LINUX_H
#define INCLUDED_IPC_WRAPPER_LINUX_H

#include "ipc_wrapper.h"

class IpcWrapperLinux : public IpcWrapper
{
public:
    IpcWrapperLinux(LwSciIpcEndpoint endpoint,
                    struct LwSciIpcEndpointInfo info,
                    std::string name,
                    uint32_t fd) :
        IpcWrapper(endpoint, info, name),
        fd(fd)
    {
    }

    ~IpcWrapperLinux() override;

    LwSciError send(void* buf, size_t size, int64_t timeoutNs) override;

    LwSciError recvFill(void* buf, size_t size, int64_t timeoutNs) override;

    static std::unique_ptr<IpcWrapper> open(const char* endpointName);

private:
    int32_t fd; /* fd to get event */
    uint32_t event;
};

#endif // INCLUDED_IPC_WRAPPER_LINUX_H
