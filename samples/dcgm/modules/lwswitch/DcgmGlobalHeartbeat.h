
#ifndef DCGM_GLOBAL_HEARTBEAT_H
#define DCGM_GLOBAL_HEARTBEAT_H

#include <queue>
#include <unistd.h>
#include "DcgmFMTimer.h"
#include "lwos.h"
#include "DcgmFabricNode.h"

#define HEARTBEAT_INTERVAL    10 // second

class DcgmFabricNode;
class DcgmGlobalHeartbeat
{
public:
    DcgmGlobalHeartbeat( DcgmFabricNode *pFabricNode, uint32_t nodeId );
    ~DcgmGlobalHeartbeat();

    void startHeartbeat();
    void stopHeartbeat();
    void updateHeartbeatCounts();
    void sendHeartbeat();
    void handleHeartbeatAck();

    DcgmFabricNode   *mpFabricNode;

private:
    uint32_t             mvNodeId;
    DcgmFMTimer         *mvTimer;
    uint32_t             mvSentCount;        // Number of heartbeat sent
    uint32_t             mvTotalMissedCount; // Total number of heartbeat not acked
    uint32_t             mvContMissedCount;  // Continuous number of heartbeat not acked
    uint32_t             mvTotalAckedCount;  // Total number of heartbeat acked
    bool                 mvAcked;            // if the current heartbeat acked
    LWOSCriticalSection  mvLock;

    static void handleHeartbeatTimer( void *timerCtx );
};

#endif
