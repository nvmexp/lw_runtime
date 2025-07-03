/*
 *  Copyright 2018-2019 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */

#pragma once

#include <queue>
#include <unistd.h>
#include "FMTimer.h"
#include "lwos.h"
#include "GlobalFmFabricNode.h"
#include "GlobalFabricManager.h"


class FMFabricNode;
class FMGlobalHeartbeat
{
public:
    FMGlobalHeartbeat( FMFabricNode *pFabricNode, uint32_t nodeId );
    ~FMGlobalHeartbeat();

    void startHeartbeat();
    void stopHeartbeat();
    bool updateHeartbeatCounts();
    void sendHeartbeat();
    void handleHeartbeatAck();

    FMFabricNode            *mpFabricNode;

private:
    uint32_t             mvNodeId;
    FMTimer              *mvTimer;
    uint32_t             mvSentCount;        // Number of heartbeat sent
    uint32_t             mvTotalMissedCount; // Total number of heartbeat not acked
    uint32_t             mvContMissedCount;  // Continuous number of heartbeat not acked
    uint32_t             mvTotalAckedCount;  // Total number of heartbeat acked
    bool                 mvAcked;            // if the current heartbeat acked
    LWOSCriticalSection  mvLock;

    static void handleHeartbeatTimer( void *timerCtx );
};

