/*
 * Copyright (c) 2019-2020 LWPU Corporation.  All Rights Reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */
#ifndef INCLUDED_IPC_WRAPPER_QNX_H
#define INCLUDED_IPC_WRAPPER_QNX_H

#include "ipc_wrapper.h"

class IpcWrapperQNX : public IpcWrapper
{
public:
    IpcWrapperQNX(LwSciIpcEndpoint endpoint,
                  struct LwSciIpcEndpointInfo info,
                  std::string name,
                  int32_t chId,
                  int32_t connId) :
        IpcWrapper(endpoint, info, name),
        chId(chId), connId(connId)
    {
    }

    virtual ~IpcWrapperQNX() override;

    LwSciError send(void* buf, size_t size, int64_t timeoutNs) override;

    LwSciError recvFill(void* buf, size_t size, int64_t timeoutNs) override;

    virtual void close() override;

    static std::unique_ptr<IpcWrapper> open(const char* endpointName);

private:
    LwSciError waitEvent(uint32_t value, int64_t timeoutNs = -1);

    int32_t chId;   /* channel id to get event */
    int32_t connId; /* connection id to send event in library */
};

#endif // INCLUDED_IPC_WRAPPER_QNX_H
