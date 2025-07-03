
#include <iostream>
#include <stdexcept>
#include <stdlib.h>
#include "DcgmModuleLwSwitch.h"
#include "dcgm_lwswitch_structs.h"
#include "topology.pb.h"
#include "dcgm_structs.h"
#include "DcgmGlobalHeartbeat.h"
#include "DcgmFMAutoLock.h"
#include "logging.h"

/*****************************************************************************/
/*                                                                           */
/* Global Heartbeat establishes a client connection with a Local Heartbeat   */
/* object.  It periodically sends a heartbeat message and expects to get a   */
/* response. If it fails to get a response it saves an error event that can  */
/* be queried from Cache Management.                                         */
/*                                                                           */
/*****************************************************************************/

/*****************************************************************************/
DcgmGlobalHeartbeat::DcgmGlobalHeartbeat( DcgmFabricNode *pFabricNode, uint32_t nodeId )
{
    mpFabricNode = pFabricNode;
    mvNodeId  = nodeId;

    mvTimer = new DcgmFMTimer( DcgmGlobalHeartbeat::handleHeartbeatTimer, this );
    mvSentCount        = 0;
    mvTotalMissedCount = 0;
    mvContMissedCount  = 0;
    mvTotalAckedCount  = 0;
    mvAcked            = false;

    lwosInitializeCriticalSection( &mvLock );
};

DcgmGlobalHeartbeat::~DcgmGlobalHeartbeat()
{
    if ( mvTimer ) delete mvTimer;
    lwosDeleteCriticalSection( &mvLock );
};

void
DcgmGlobalHeartbeat::startHeartbeat()
{
    PRINT_INFO("%d", "Start heartbeat to node %d.", mvNodeId);
    mvTimer->start(HEARTBEAT_INTERVAL);
}

void
DcgmGlobalHeartbeat::stopHeartbeat()
{
    PRINT_INFO("%d", "Stop heartbeat to node %d.", mvNodeId);
    mvTimer->stop();
}

void
DcgmGlobalHeartbeat::handleHeartbeatAck()
{
    DcgmFMAutoLock lock(mvLock);
    mvTotalAckedCount++;
    mvAcked = true;

    // reset the continuous miss count
    mvContMissedCount = 0;
}

void
DcgmGlobalHeartbeat::handleHeartbeatTimer( void *timerCtx )
{
    DcgmGlobalHeartbeat *pGlobalHeatbeat = (DcgmGlobalHeartbeat *)timerCtx;
    pGlobalHeatbeat->updateHeartbeatCounts();
    pGlobalHeatbeat->sendHeartbeat();
}

void
DcgmGlobalHeartbeat::updateHeartbeatCounts()
{
    DcgmFMAutoLock lock(mvLock);
    if ( (mvSentCount > 0) && ( mvAcked == false ) )
    {
        mvTotalMissedCount++;
        mvContMissedCount++;

        PRINT_ERROR("%d, %d, %d",
                    "Missed heartbeat to node %d total %d, cont %d.",
                    mvNodeId, mvTotalMissedCount, mvContMissedCount);
    }

    // TODO actions to take
    // when mvContMissedCount is greater than some threshold
}

void
DcgmGlobalHeartbeat::sendHeartbeat()
{
    DcgmFMAutoLock lock(mvLock);

    lwswitch::heartbeat *pHeartbeat = new lwswitch::heartbeat();

    lwswitch::fmMessage * pFmMessage = new lwswitch::fmMessage();
    pFmMessage->set_type( lwswitch::FM_HEARTBEAT );
    pFmMessage->set_allocated_heartbeat( pHeartbeat );

    dcgmReturn_t rc = mpFabricNode->SendControlMessage(pFmMessage, false);
    if ( rc == DCGM_ST_OK )
    {
        mvSentCount++;
        mvAcked = false;
    }
    else{
        PRINT_ERROR("%d, %d",
                    "Failed to send heartbeat to node %d, rc %d.",
                    mvNodeId, rc);
    }

    delete pFmMessage;

    // restart the timer for the next heartbeat;
    mvTimer->restart();
}
