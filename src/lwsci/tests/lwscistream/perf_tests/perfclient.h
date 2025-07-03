//! \file
//! \brief LwSciStream test client declaration.
//!
//! \copyright
//! Copyright (c) 2019-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef PERFCLIENT_H
#define PERFCLIENT_H

#include <array>
#include <vector>
#include "util.h"

struct Packet{
    LwSciStreamCookie cookie{ 0U };
    LwSciStreamPacket handle{ 0U };
    std::array<LwSciBufObj, NUM_ELEMENTS> buffers;
};

class PerfClient
{
public:
    PerfClient(LwSciBufModule buf,
               LwSciSyncModule sync);

    virtual ~PerfClient(void);

protected:
    void setupSync(void);
    void recvWaiterAttr(void);
    void recvSignalObj(void);
    void sendEndpointElements(void);
    void recvAllocatedElements(void);
    void recvPacket(void);
    void recvPacketComplete(void);
    void recvSetupComplete(void);
    virtual void streaming(void) = 0;

    virtual void setEndpointBufAttr(LwSciBufAttrList attrList) = 0;
    void setCpuSyncAttrList(LwSciSyncAccessPerm cpuPerm,
                            LwSciSyncAttrList attrList);

protected:
    LwSciBufModule                  bufModule{ nullptr };
    LwSciSyncModule                 syncModule{ nullptr };

    LwSciStreamBlock                endpointHandle{ 0U };

    LwSciSyncCpuWaitContext         waitContext{ nullptr };
    std::vector<LwSciSyncObj>       syncs;

    std::array<Packet, NUM_PACKETS> packets;
    uint32_t                        numRecvPackets{ 0U };
} ;

#endif // PERFCLIENT_H
