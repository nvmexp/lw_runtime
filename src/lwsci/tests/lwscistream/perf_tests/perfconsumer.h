//! \file
//! \brief LwSciStream test class Consumer client declaration.
//!
//! \copyright
//! Copyright (c) 2019-2022 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef PERFCONSUMER_H
#define PERFCONSUMER_H

#include "perfclient.h"
#ifdef __QNX__
#include "perftimerQNX.h"
#else
#include "perftimer.h"
#endif

class PerfConsumer : public PerfClient
{
public:
    PerfConsumer(LwSciBufModule buf,
                 LwSciSyncModule sync);
    ~PerfConsumer(void) override;

    void createStream(LwSciIpcEndpoint const ipcEndpoint,
        LwSciStreamBlock const upstreamBlock);
    virtual void run(void);

protected:
    void recvConnectEvent(void);
    void createPacket(void);
    void finalizePacket(void);

    void streaming(void) override;

    void setEndpointBufAttr(LwSciBufAttrList attrList) override;

protected:
    LwSciStreamBlock                    ipcDst{ 0U };
    LwSciStreamBlock                    c2cPool{ 0U };
    LwSciStreamBlock                    queue{ 0U };
    LwSciStreamBlock                    consumer{ 0U };

    std::vector<Packet>                 allocatedPackets;

    std::array<PerfTimer, NUM_PACKETS>  releaseTimer;
};

#endif // PERFCONSUMER_H
