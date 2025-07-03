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
#include <iostream>
#include <stdexcept>
#include <stdlib.h>
#include "topology.pb.h"
#include "FMErrorCodesInternal.h"
#include "GlobalFmHeartbeat.h"
#include "FMAutoLock.h"
#include "fm_log.h"

/*****************************************************************************/
/*                                                                           */
/* Global Heartbeat establishes a client connection with a Local Heartbeat   */
/* object.  It periodically sends a heartbeat message and expects to get a   */
/* response. If it fails to get a response it saves an error event that can  */
/* be queried from Cache Management.                                         */
/*                                                                           */
/*****************************************************************************/
FMGlobalHeartbeat::FMGlobalHeartbeat( FMFabricNode *pFabricNode, uint32_t nodeId )
{
    mpFabricNode = pFabricNode;
    mvNodeId  = nodeId;

    mvTimer = new FMTimer( FMGlobalHeartbeat::handleHeartbeatTimer, this );
    mvSentCount        = 0;
    mvTotalMissedCount = 0;
    mvContMissedCount  = 0;
    mvTotalAckedCount  = 0;
    mvAcked            = false;

    lwosInitializeCriticalSection( &mvLock );
};

FMGlobalHeartbeat::~FMGlobalHeartbeat()
{
    if ( mvTimer ) delete mvTimer;
    lwosDeleteCriticalSection( &mvLock );
};

void
FMGlobalHeartbeat::startHeartbeat()
{
    FM_LOG_DEBUG("Start heartbeat to " NODE_ID_LOG_STR " %d.", mvNodeId);
    mvTimer->start(HEARTBEAT_INTERVAL);
}

void
FMGlobalHeartbeat::stopHeartbeat()
{
    FM_LOG_DEBUG("Stop heartbeat to " NODE_ID_LOG_STR " %d.", mvNodeId);
    mvTimer->stop();
}

void
FMGlobalHeartbeat::handleHeartbeatAck()
{
    FMAutoLock lock(mvLock);
    mvTotalAckedCount++;
    mvAcked = true;

    // reset the continuous miss count
    mvContMissedCount = 0;
}

void
FMGlobalHeartbeat::handleHeartbeatTimer( void *timerCtx )
{
    FMGlobalHeartbeat *pGlobalHeatbeat = (FMGlobalHeartbeat *)timerCtx;
    if( pGlobalHeatbeat->updateHeartbeatCounts() )
    {
        pGlobalHeatbeat->sendHeartbeat();
    }
}

bool
FMGlobalHeartbeat::updateHeartbeatCounts()
{
    FMAutoLock lock(mvLock);
    if ( (mvSentCount > 0) && ( mvAcked == false ) )
    {
        mvTotalMissedCount++;
        mvContMissedCount++;
        FM_LOG_DEBUG("Missed heartbeat to " NODE_ID_LOG_STR " %d total %d, continuous %d.",
                     mvNodeId, mvTotalMissedCount, mvContMissedCount);
    }

    // when mvContMissedCount is greater than some threshold
    if ( mvContMissedCount > HEARTBEAT_THRESHOLD )
    {
        FM_LOG_ERROR("missed %d continuous heartbeats to " NODE_ID_LOG_STR " %d, total heartbeats missed %d,"
                     " triggering fatal error", mvContMissedCount, mvNodeId, mvTotalMissedCount);
        mpFabricNode->processHeartBeatFailure();

        stopHeartbeat();
        return false;
    }
    return true;
}

void
FMGlobalHeartbeat::sendHeartbeat()
{
    FMAutoLock lock(mvLock);

    lwswitch::heartbeat *pHeartbeat = new lwswitch::heartbeat();
    pHeartbeat->set_nodeid( mvNodeId ); // Set your own node ID

    lwswitch::fmMessage * pFmMessage = new lwswitch::fmMessage();
    pFmMessage->set_type( lwswitch::FM_HEARTBEAT );
    pFmMessage->set_allocated_heartbeat( pHeartbeat );

    FMIntReturn_t rc = mpFabricNode->SendControlMessage(pFmMessage, false);
    if ( rc == FM_INT_ST_OK )
    {
        mvSentCount++;
        mvAcked = false;
    }
    else{
        FM_LOG_DEBUG("Failed to send heartbeat to " NODE_ID_LOG_STR " %d, rc %d.",
                    mvNodeId, rc);
    }

    delete pFmMessage;

    // restart the timer for the next heartbeat;
    mvTimer->restart();
}
