#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>

#include "logging.h"
#include "DcgmLocalStatsMsgHndlr.h"

DcgmLocalStatsMsgHndlr::DcgmLocalStatsMsgHndlr(FMConnInterface *ctrlConnIntf,
                                               DcgmLocalStatsReporter *pLocalStatsReporter)
{
    mCtrlConnIntf = ctrlConnIntf;
    mLocalStatsReporter = pLocalStatsReporter;
};

DcgmLocalStatsMsgHndlr::~DcgmLocalStatsMsgHndlr()
{
};

void
DcgmLocalStatsMsgHndlr::handleEvent(FabricManagerCommEventType eventType, uint32 nodeId)
{
    switch (eventType) {
        case FM_EVENT_PEER_FM_CONNECT: {
            // TODO 
            break;
        }
        case FM_EVENT_PEER_FM_DISCONNECT: {
            // TODO
            break;
        }
        case FM_EVENT_GLOBAL_FM_CONNECT: {
            // TODO
            break;
        }
        case FM_EVENT_GLOBAL_FM_DISCONNECT: {
            // TODO
            break;
        }
    }
}

void
DcgmLocalStatsMsgHndlr::handleMessage(lwswitch::fmMessage  *pFmMessage)
{
    PRINT_DEBUG("%d", "message type %d", pFmMessage->type());

    switch ( pFmMessage->type() )
    {
    case lwswitch::FM_LWSWITCH_ERROR_FATAL_ACK:
        dumpMessage(pFmMessage);
        handleFatalErrorReportAckMsg( pFmMessage );
        break;

    case lwswitch::FM_LWSWITCH_ERROR_NON_FATAL_ACK:
        handleNonFatalErrorReportAckMsg( pFmMessage );
        break;

    case lwswitch::FM_NODE_STATS_ACK:
        handleNodeStatsAckMsg( pFmMessage );
        break;

    case lwswitch::FM_GET_ERROR_REQ:
        handleGetErrorRequestMsg( pFmMessage );
        break;

    case lwswitch::FM_GET_NODE_STATS_REQ:
        handleGetStatsRequestMsg( pFmMessage );
        break;

    default:
        PRINT_ERROR("%d", "unknown message type %d", pFmMessage->type());
        break;
    }
}

void
DcgmLocalStatsMsgHndlr::dumpMessage(lwswitch::fmMessage *pFmMessage)
{
#ifdef DEBUG
    std::string msgText;

    google::protobuf::TextFormat::PrintToString(*pFmMessage, &msgText);
    PRINT_DEBUG("%s", "%s", msgText.c_str());
#endif
}

/**
 *  handle FM_FATAL_ERROR_REPORT_ACK message from Global FM
 */
void
DcgmLocalStatsMsgHndlr::handleFatalErrorReportAckMsg(lwswitch::fmMessage *pFmMessage)
{

}

/**
 *  handle FM_NON_FATAL_ERROR_REPORT message from Global FM
 */
void
DcgmLocalStatsMsgHndlr::handleNonFatalErrorReportAckMsg(lwswitch::fmMessage *pFmMessage)
{

}

/**
 *  handle FM_NODE_STATS_ACK message from Global FM
 */
void
DcgmLocalStatsMsgHndlr::handleNodeStatsAckMsg(lwswitch::fmMessage *pFmMessage)
{

}

/**
 *  handle FM_GET_ERROR_REQ message from Global FM
 */
void
DcgmLocalStatsMsgHndlr::handleGetErrorRequestMsg(lwswitch::fmMessage *pFmMessage)
{
    uint32_t errorMask;
    std::queue < SwitchError_struct * > errQ;
    lwswitch::fmMessage *pFmResponse;
    const lwswitch::getSwitchErrorRequest *pGetError = &pFmMessage->geterrorrequest();
    dcgmReturn_t retVal;

    errorMask = pGetError->has_errormask() ? pGetError->errormask() : 0xFFFFFFFF;

    pFmResponse = mLocalStatsReporter->buildSwitchErrorMsg( errorMask, &errQ );
    if ( !pFmResponse )
    {
        PRINT_ERROR(" ", "Failed to build stats response message.");
        return;
    }

    // send the response back to global fabric manager
    pFmResponse->set_type( lwswitch::FM_GET_ERROR_RSP );
    pFmResponse->set_requestid( pFmMessage->requestid() );

    retVal = mCtrlConnIntf->SendMessageToGfm( pFmResponse, false );
    if ( retVal != DCGM_ST_OK )
    {
         PRINT_DEBUG(" ", "Failed to send response to global fabric manager");
    }
    delete pFmResponse;
}

/**
 *  handle FM_GET_NODE_STATS_REQ message from Global FM
 */
void
DcgmLocalStatsMsgHndlr::handleGetStatsRequestMsg(lwswitch::fmMessage *pFmMessage)
{
    std::queue < SwitchLatency_struct * > latencyQ;
    std::queue < LwlinkCounter_struct * > counterQ;
    const lwswitch::targetSwitch *pTarget;
    const lwswitch::getSwitchLatencyHist *pGetLatency;
    const lwswitch::getSwitchLwlinkCounter *pGetCounter;
    int i, j;
    lwswitch::fmMessage *pFmResponse = NULL;
    dcgmReturn_t retVal;
    std::list<uint32_t> switchPhysicalIdList;
    std::list<uint32_t>::iterator it;

    const lwswitch::getNodeStatsRequest req = pFmMessage->getstatsrequest();

    // get a list of the local switches for latency
    for ( i = 0; i < req.latestlatency_size(); i++ )
    {
        pGetLatency = &req.latestlatency(i);
        for ( j = 0; j < pGetLatency->targetswitches_size(); j++ )
        {
            pTarget = &pGetLatency->targetswitches(j);
            if( pTarget->has_targetswitchphysicalid() )
            {
                switchPhysicalIdList.push_back( pTarget->targetswitchphysicalid() );
            }
        }
    }

    // get the latency from local switches
    for ( it = switchPhysicalIdList.begin(); it != switchPhysicalIdList.end(); it++ )
    {
        mLocalStatsReporter->getSwitchInternalLatency( *it, &latencyQ );
    }

    // get a list of the local switches for lwlink counter
    switchPhysicalIdList.clear();
    for ( i = 0; i < req.lwlinkcounter_size(); i++ )
    {
        pGetCounter = &req.lwlinkcounter(i);
        for ( j = 0; j < pGetCounter->targetswitches_size(); j++ )
        {
            pTarget = &pGetCounter->targetswitches(j);
            // TODO: add node id check here
            if ( pTarget->has_targetswitchphysicalid() )
            {
                switchPhysicalIdList.push_back( pTarget->targetswitchphysicalid() );
            }
        }
    }

    // get the lwlink counters from local switches
    for ( it = switchPhysicalIdList.begin(); it != switchPhysicalIdList.end(); it++ )
    {
        mLocalStatsReporter->getSwitchLwlinkCounter( *it, &counterQ );
    }

    // build the stats response message
    pFmResponse = mLocalStatsReporter->buildSwitchStatsMsg( &latencyQ, &counterQ );
    if ( !pFmResponse )
    {
        PRINT_ERROR(" ", "Failed to build stats response message.");
        return;
    }

    // send the response back to global fabric manager
    pFmResponse->set_type( lwswitch::FM_GET_NODE_STATS_RSP );
    pFmResponse->set_requestid( pFmMessage->requestid() );

    dumpMessage(pFmResponse);

    retVal = mCtrlConnIntf->SendMessageToGfm( pFmResponse, false );
    if ( retVal != DCGM_ST_OK )
    {
         PRINT_DEBUG(" ", "Failed to send response to global fabric manager");
    }
    delete pFmResponse;
}

