#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>

#include "logging.h"
#include "DcgmGlobalControlMsgHndl.h"

#include "lwml.h"
#include "lwml_internal.h"

DcgmGlobalControlMsgHndl::DcgmGlobalControlMsgHndl(FMConnInterface *ctrlConnIntf)
{
    mCtrlConnIntf = ctrlConnIntf;
};

DcgmGlobalControlMsgHndl::~DcgmGlobalControlMsgHndl()
{
};

void
DcgmGlobalControlMsgHndl::handleEvent( FabricManagerCommEventType eventType, uint32 nodeId )
{
    switch (eventType) {
        case FM_EVENT_PEER_FM_CONNECT: {
            PRINT_INFO("%d", "nodeId %d FM_EVENT_PEER_FM_CONNECT", nodeId);
            break;
        }
        case FM_EVENT_PEER_FM_DISCONNECT: {
            PRINT_INFO("%d", "nodeId %d FM_EVENT_PEER_FM_DISCONNECT", nodeId);
            break;
        }
        case FM_EVENT_GLOBAL_FM_CONNECT: {
            PRINT_INFO("%d", "nodeId %d FM_EVENT_GLOBAL_FM_CONNECT", nodeId);
            break;
        }
        case FM_EVENT_GLOBAL_FM_DISCONNECT: {
            PRINT_INFO("%d", "nodeId %d FM_EVENT_GLOBAL_FM_DISCONNECT", nodeId);
            break;
        }
    }
}

void
DcgmGlobalControlMsgHndl::handleMessage( lwswitch::fmMessage  *pFmMessage )
{
    PRINT_DEBUG("%d", "message type %d", pFmMessage->type());

    switch ( pFmMessage->type() )
    {
    case lwswitch::FM_NODE_STATS_ACK:
        handleNodeStatsAckMsg( pFmMessage );
        break;

    default:
        PRINT_ERROR("%d", "unknown message type %d", pFmMessage->type());
        break;
    }
}

void
DcgmGlobalControlMsgHndl::dumpMessage( lwswitch::fmMessage *pFmMessage )
{
#ifdef DEBUG
    std::string msgText;

    google::protobuf::TextFormat::PrintToString(*pFmMessage, &msgText);
    PRINT_DEBUG("%s", "%s", msgText.c_str());
#endif
}

/**
 *  on GFM or LFM, handle FM_NODE_STATS_ACK message
 */
void
DcgmGlobalControlMsgHndl::handleNodeStatsAckMsg( lwswitch::fmMessage *pFmMessage )
{

}
