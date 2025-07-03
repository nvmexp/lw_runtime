/*
 * Copyright (c) 2019-2020 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */
#ifndef INCLUDED_IPC_WRAPPER_H
#define INCLUDED_IPC_WRAPPER_H

#include <memory>
#include <lwscierror.h>
#include <lwsciipc.h>
#include <string>
#include <vector>
// for debugging
#include <stdio.h>
#include <mutex>

#define IPC_CHECK_API(func, expected)                                          \
    ({                                                                         \
        err = (func);                                                          \
        if (err != (expected)) {                                               \
            fprintf(stderr, #func " failed: %X\n", err);                       \
            goto fail;                                                         \
        }                                                                      \
    })

class IpcWrapper
{
    friend class IpcWrapperLinux;
    friend class IpcWrapperQNX;

public:
    virtual LwSciError send(void* buf, size_t size, int64_t timeoutNs = -1) = 0;

    virtual LwSciError
    recvFill(void* buf, size_t size, int64_t timeoutNs = -1) = 0;

    virtual void close();

    LwSciIpcEndpoint getEndpoint() const
    {
        return endpoint;
    }

    const std::string& getName() const
    {
        return name;
    }

    virtual ~IpcWrapper() = default;

    static void init();

    static std::unique_ptr<IpcWrapper> open(const char* endpointName);

    static void deinit();

private:
    IpcWrapper(LwSciIpcEndpoint endpoint, struct LwSciIpcEndpointInfo info, std::string name) :
        endpoint(endpoint), info(info), name(name)
    {
    }

    // Forbid copy
    IpcWrapper(const IpcWrapper&) = delete;

    IpcWrapper& operator=(const IpcWrapper&) = delete;

    static std::mutex initCountMutex;
    static int initCount;

    LwSciIpcEndpoint endpoint;
    struct LwSciIpcEndpointInfo info;
    std::string name;
};

#endif // INCLUDED_IPC_WRAPPER_H
