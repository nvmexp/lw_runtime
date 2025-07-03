//! \file
//! \brief LwSciStream test Producer client declaration.
//!
//! \copyright
//! Copyright (c) 2019-2022 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef PERFPRODUCER_H
#define PERFPRODUCER_H

#include "perfclient.h"

class PerfProducer: public PerfClient
{
public:
    PerfProducer(LwSciBufModule buf,
                 LwSciSyncModule sync);
    ~PerfProducer(void) override;

    LwSciStreamBlock createStream(
        std::vector<LwSciIpcEndpoint>* ipcEndpoint = nullptr);
    virtual void run(void);

protected:
    void recvConnectEvent(void);
    void createPacket(void);
    void finalizePacket(void);

    void streaming(void) override;

    void setEndpointBufAttr(LwSciBufAttrList attrList) override;

protected:
    LwSciStreamBlock                producer{ 0U };
    LwSciStreamBlock                pool{ 0U };
    LwSciStreamBlock                multicast{ 0U };
    std::vector<LwSciStreamBlock>   c2cQueue;
    std::vector<LwSciStreamBlock>   ipcSrc;

    std::array<Packet, NUM_PACKETS> allocatedPackets;
} ;

#endif // PERFPRODUCER_H
