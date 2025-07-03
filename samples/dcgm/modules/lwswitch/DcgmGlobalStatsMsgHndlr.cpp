#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>

#include "logging.h"
#include "DcgmGlobalStatsMsgHndlr.h"
#include "DcgmLogging.h"

DcgmGlobalStatsMsgHndlr::DcgmGlobalStatsMsgHndlr(DcgmGlobalFabricManager *pGfm)
{
    mpGfm = pGfm;
}

DcgmGlobalStatsMsgHndlr::~DcgmGlobalStatsMsgHndlr()
{

}

void
DcgmGlobalStatsMsgHndlr::handleEvent(FabricManagerCommEventType eventType, uint32 nodeId)
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
DcgmGlobalStatsMsgHndlr::handleMessage(lwswitch::fmMessage  *pFmMessage)
{
    PRINT_DEBUG("%d", "GlobalStatsHandler: message type %d", pFmMessage->type());

    switch ( pFmMessage->type() )
    {
    case lwswitch::FM_LWSWITCH_ERROR_FATAL:
        dumpMessage(pFmMessage);
        handleFatalErrorReportMsg( pFmMessage );
        break;

    case lwswitch::FM_LWSWITCH_ERROR_NON_FATAL:
        handleNonFatalErrorReportMsg( pFmMessage );
        break;

    case lwswitch::FM_NODE_STATS_REPORT:
        handleStatsReportMsg( pFmMessage );
        break;

    case lwswitch::FM_NODE_STATS_ACK:
        handleNodeStatsAckMsg( pFmMessage );
        break;

    case lwswitch::FM_LWLINK_ERROR_LWSWITCH_RECOVERY:
    case lwswitch::FM_LWLINK_ERROR_GPU_RECOVERY:
        dumpMessage(pFmMessage);
        handleLWLinkErrorRecoveryMsg( pFmMessage );
        break;

    case lwswitch::FM_LWLINK_ERROR_GPU_FATAL:
        dumpMessage(pFmMessage);
        handleLWLinkErrorFatalMsg( pFmMessage );
        break;

    default:
        PRINT_ERROR("%d", "GlobalStatsHandler: unknown message type %d", pFmMessage->type());
        break;
    }
}

void
DcgmGlobalStatsMsgHndlr::dumpMessage(lwswitch::fmMessage *pFmMessage)
{
#ifdef DEBUG
    std::string msgText;

    google::protobuf::TextFormat::PrintToString(*pFmMessage, &msgText);
    PRINT_DEBUG("%s", "GlobalStatsHandler: %s", msgText.c_str());
#endif
}

/**
 *  on Global FM, handle FM_FATAL_ERROR_REPORT message from Local FM
 */
void
DcgmGlobalStatsMsgHndlr::handleFatalErrorReportMsg(lwswitch::fmMessage *pFmMessage)
{
    FM_ERROR_CODE rc;
    int i;
    lwswitch::switchErrorReportAck *pResponse;
    lwswitch::fmMessage *pFmResponse;
    uint32 nodeId = pFmMessage->nodeid();

    pFmResponse = new lwswitch::fmMessage();
    pFmResponse->set_requestid( pFmMessage->requestid() );

    pResponse = new lwswitch::switchErrorReportAck;
    pFmResponse->set_allocated_errorreportack( pResponse );
    pFmResponse->set_type( lwswitch::FM_LWSWITCH_ERROR_FATAL_ACK );

    parseErrorReportMsg( pFmMessage->errorreport() );

    // send the response back to Local FM
    rc = sendMessage( nodeId, pFmResponse );
    if ( rc != FM_SUCCESS )
    {
        PRINT_DEBUG(" ", "GlobalStatsHandler: failed to send fatal error ack message to LocalFM");
    }

    // do globalFM additional fatal error handling
    mpGfm->queueErrorWorkerRequest( nodeId, pFmMessage );

    delete pFmResponse;
}

/**
 *  on Global FM, handle FM_NON_FATAL_ERROR_REPORT message from Local FM
 */
void
DcgmGlobalStatsMsgHndlr::handleNonFatalErrorReportMsg(lwswitch::fmMessage *pFmMessage)
{
    FM_ERROR_CODE rc;
    int i;
    lwswitch::switchErrorReportAck *pResponse;
    lwswitch::fmMessage *pFmResponse;  //response to send back to Global Fabric Manager
    uint32 nodeId = pFmMessage->nodeid();

    pFmResponse = new lwswitch::fmMessage();
    pFmResponse->set_requestid( pFmMessage->requestid() );

    pResponse = new lwswitch::switchErrorReportAck;
    pFmResponse->set_allocated_errorreportack( pResponse );
    pFmResponse->set_type( lwswitch::FM_LWSWITCH_ERROR_NON_FATAL_ACK );

    parseErrorReportMsg( pFmMessage->errorreport() );

    // send the response back to Local FM
    rc = sendMessage( nodeId, pFmResponse );
    if ( rc != FM_SUCCESS )
    {
        PRINT_DEBUG(" ", "GlobalStatsHandler: failed to send non-fatal error ack message to LocalFM");
    }

    delete pFmResponse;
}

void
DcgmGlobalStatsMsgHndlr::appendLWswitchSamples(dcgm_field_eid_t entityId, unsigned short dcgmFieldId,
                                               dcgmcm_sample_p samples)
{
    dcgmReturn_t result;
    dcgmcm_watch_info_t watchInfo;

    DcgmCacheManager *pCacheManager = mpGfm->GetCacheManager();
    if ( !pCacheManager )
        return;

    result = pCacheManager->GetEntityWatchInfoSnapshot( DCGM_FE_SWITCH, entityId,
                                                        dcgmFieldId, &watchInfo );
    // check if the field is watched
    if ( ( result != DCGM_ST_OK ) || !watchInfo.isWatched )
        return;

    result = pCacheManager->AppendSamples( DCGM_FE_SWITCH, entityId, dcgmFieldId, samples, 1 );
    if ( result != DCGM_ST_OK )
    {
        PRINT_ERROR("%d, %d, %d",
                    "GlobalStatsHandler: append sample error %d, entityId %d, dcgmFieldId %d",
                    result, entityId, dcgmFieldId);
    }
}

void
DcgmGlobalStatsMsgHndlr::cacheLatency(PortLatencyHist_t &latencyHisto, uint32_t switchPhysicalId,
                                      int portNum, int vcNum)
{
    dcgmcm_sample_t sample;

    dcgm_field_eid_t fid = DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P00;
    fid += 4 * portNum;

    memset(&sample, 0, sizeof(sample));
    sample.timestamp = timelib_usecSince1970();

    sample.val.i64 = latencyHisto.low;
    appendLWswitchSamples( switchPhysicalId, fid, &sample );

    sample.val.i64 = latencyHisto.med;
    appendLWswitchSamples( switchPhysicalId, fid + 1, &sample );

    sample.val.i64 = latencyHisto.high;
    appendLWswitchSamples( switchPhysicalId, fid + 2, &sample );

    sample.val.i64 = latencyHisto.panic;
    appendLWswitchSamples( switchPhysicalId, fid + 3, &sample );
} 

void
DcgmGlobalStatsMsgHndlr::cacheCounters(LwlinkCounter_t &linkCounter, int switchPhysicalId, int portNum)
{
    dcgmcm_sample_t sample;
    DcgmCacheManager *pCacheManager = mpGfm->GetCacheManager();

    dcgm_field_eid_t fid = DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P00;
    fid += 2 * portNum;
    memset(&sample, 0, sizeof(sample));
    sample.timestamp = timelib_usecSince1970();

    sample.val.i64 = linkCounter.txCounter0;
    appendLWswitchSamples( switchPhysicalId, fid, &sample );

    sample.val.i64 = linkCounter.rxCounter0;
    appendLWswitchSamples( switchPhysicalId, fid + 1, &sample );

    fid = DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P00;
    fid += 2 * portNum;

    sample.val.i64 = linkCounter.txCounter1;
    appendLWswitchSamples( switchPhysicalId, fid, &sample );

    sample.val.i64 = linkCounter.rxCounter1;
    appendLWswitchSamples( switchPhysicalId, fid + 1, &sample );
}

void
DcgmGlobalStatsMsgHndlr::cacheDetailedSwitchError(uint32_t switchPhysicalId,
                                                  const lwswitch::switchErrorInfo &errorInfo)
{
    uint32_t portNum = errorInfo.instance();
    uint32_t errorCode = errorInfo.errortype();

    // create dcgm switch error sample
    dcgmcm_sample_t sample;
    dcgm_field_eid_t errorFid;

    if ( errorInfo.severity() == lwswitch::LWSWITCH_ERROR_SEVERITY_FATAL )
    {
        PRINT_ERROR("%d, %d, %d", "detected LWSwitch fatal error, switchId=%d switch port=%d error code=%d ",
                    switchPhysicalId, portNum, errorCode);
        errorFid = DCGM_FI_DEV_LWSWITCH_FATAL_ERRORS;
    }
    else
    {
        PRINT_ERROR("%d, %d, %d", "detected LWSwitch non fatal error, switchId=%d switch port=%d error code=%d ",
                    switchPhysicalId, portNum, errorCode);
        errorFid = DCGM_FI_DEV_LWSWITCH_NON_FATAL_ERRORS;
    }

    memset(&sample, 0, sizeof(sample));
    sample.timestamp = timelib_usecSince1970();
    sample.val.i64 = errorCode; //value field is the error code (SXid) from driver.
    DcgmCacheManager *pCacheManager = mpGfm->GetCacheManager();
    appendLWswitchSamples( switchPhysicalId, errorFid, &sample );
}

void
DcgmGlobalStatsMsgHndlr::parseStatsReportMsg(uint32 nodeId, const lwswitch::nodeStats &statsReportMsg)
{

    int i, j;
    uint32_t switchPhysicalId = 0, portNum = 0;

    for ( i = 0; i < statsReportMsg.latestlatency_size(); i++ )
    {
        const lwswitch::switchLatencyHist &switchlatency = statsReportMsg.latestlatency(i);

        switchPhysicalId = 0;
        if ( switchlatency.has_switchphysicalid() )
        {
            switchPhysicalId = switchlatency.switchphysicalid();
        }

        for ( j = 0; j < switchlatency.latencyhist_size(); j++ )
        {
            const lwswitch::portLatencyHist &portLatencyMsg = switchlatency.latencyhist(j);
            portNum = portLatencyMsg.has_portnum() ? portLatencyMsg.portnum() : j;

            if ( IS_NODE_VALID( nodeId ) && IS_WILLOW_VALID( switchPhysicalId ) &&
                 IS_PORT_VALID( portNum ) )
            {
                // update latency on the global fabric manager
                PortLatencyHist_t latencyHisto = {0};

                if ( portLatencyMsg.has_elapsedtimemsec() )
                    latencyHisto.elapsedTimeMsec = portLatencyMsg.elapsedtimemsec();

                if ( portLatencyMsg.has_low() )
                    latencyHisto.low = portLatencyMsg.low();

                if ( portLatencyMsg.has_med() )
                    latencyHisto.med = portLatencyMsg.med();

                if ( portLatencyMsg.has_high() )
                    latencyHisto.high = portLatencyMsg.high();

                if ( portLatencyMsg.has_panic() )
                    latencyHisto.panic = portLatencyMsg.panic();
                
                // got good latency report, log the same to dcgm cachemanager
                cacheLatency( latencyHisto, switchPhysicalId, portNum, 0 );
            }
            else
            {
                PRINT_ERROR("%d, %d, %d",
                            "GlobalStatsHandler: Invalid parameters in stats message, node %d, switch %d, port %d",
                            nodeId, switchPhysicalId, portNum);
            }
        }
    }
    
    for ( i = 0; i < statsReportMsg.lwlinkcounter_size(); i++ )
    {
        const lwswitch::switchLwlinkCounter &switchcounter = statsReportMsg.lwlinkcounter(i);

        switchPhysicalId = 0;
        if ( switchcounter.has_switchphysicalid() )
        {
            switchPhysicalId = switchcounter.switchphysicalid();
        }

        for ( j = 0; j < switchcounter.linkcounter_size(); j++ )
        {
            const lwswitch::lwlinkCounter &linkCounterMsg = switchcounter.linkcounter(j);
            portNum = linkCounterMsg.has_portnum() ? linkCounterMsg.portnum() : j;

            if ( IS_NODE_VALID( nodeId ) && IS_WILLOW_VALID( switchPhysicalId ) &&
                 IS_PORT_VALID( portNum ) )
            {
                LwlinkCounter_t linkCounter = {0};
                if ( linkCounterMsg.has_txcounter0() )
                    linkCounter.txCounter0 = linkCounterMsg.txcounter0();

                if ( linkCounterMsg.has_rxcounter0() )
                    linkCounter.rxCounter0 = linkCounterMsg.rxcounter0();

                if ( linkCounterMsg.has_txcounter1() )
                    linkCounter.txCounter1 = linkCounterMsg.txcounter1();

                if ( linkCounterMsg.has_rxcounter1() )
                    linkCounter.rxCounter1 = linkCounterMsg.rxcounter1();

                cacheCounters( linkCounter, switchPhysicalId, portNum );
            }
            else
            {
                PRINT_ERROR("%d, %d, %d",
                            "GlobalStatsHandler: Invalid parameters in counter stats message, node %d, switch %d, port %d",
                            nodeId, switchPhysicalId, portNum);
            }
        }
    }
}

void
DcgmGlobalStatsMsgHndlr::logLWSwitchError(uint32_t switchPhysicalId,
                                          const lwswitch::switchErrorInfo &errorInfo)
{
    uint32_t partitionId = ILWALID_FABRIC_PARTITION_ID;
    uint32_t portNum = errorInfo.instance();

    // mGfmPartitionMgr is NULL in non shared fabric mode,
    // and partitionId will be set to ILWALID_FABRIC_PARTITION_ID, which will not be logged.
    if ( mpGfm && mpGfm->mGfmPartitionMgr )
    {
        partitionId = mpGfm->mGfmPartitionMgr->getActivePartitionIdForLWSwitchPort(0,
                                                                                   switchPhysicalId,
                                                                                   portNum);
    }

    if ( errorInfo.severity() == lwswitch::LWSWITCH_ERROR_SEVERITY_FATAL )
    {
        if ( partitionId != ILWALID_FABRIC_PARTITION_ID ) {
            FM_SYSLOG_ERR("Detected LWSwitch fatal error on partition %d, LWSwitch 0x%x, port %d.",
                          partitionId, switchPhysicalId, portNum);
        } else {
            FM_SYSLOG_ERR("Detected LWSwitch fatal error on LWSwitch 0x%x, port %d.",
                          switchPhysicalId, portNum);
        }
    }
    else
    {
        if ( partitionId != ILWALID_FABRIC_PARTITION_ID ) {
            FM_SYSLOG_ERR("Detected LWSwitch non fatal error on partition %d, LWSwitch 0x%x, port %d.",
                          partitionId, switchPhysicalId, portNum);
        } else {
            FM_SYSLOG_ERR("Detected non fatal error on LWSwitch 0x%x, port %d.",
                          switchPhysicalId, portNum);
        }
    }
}

void
DcgmGlobalStatsMsgHndlr::parseErrorReportMsg(const lwswitch::switchErrorReport &errorReportMsg)
{
    int i,j;
    for ( i = 0; i < errorReportMsg.switcherror_size(); i++ )
    {
        uint32_t switchPhysicalId = 0;
        const lwswitch::switchError &errorMsg = errorReportMsg.switcherror(i);
        if ( errorMsg.has_switchphysicalid() )
        {
            switchPhysicalId = errorMsg.switchphysicalid();
        }
        for ( j = 0; j < errorMsg.errorinfo_size(); j++ )
        {
            const lwswitch::switchErrorInfo &errorInfo = errorMsg.errorinfo(j);
            // publish individual error details to cache manager
            cacheDetailedSwitchError( switchPhysicalId, errorInfo );

            // syslog the error
            logLWSwitchError( switchPhysicalId, errorInfo );
        }
    }
}

/**
 *  on Global FM, handle FM_NODE_STATS_REPORT message from Local FM
 */
void
DcgmGlobalStatsMsgHndlr::handleStatsReportMsg(lwswitch::fmMessage *pFmMessage)
{
    FM_ERROR_CODE rc;
    int i, j;
    PortLatencyHist_t *pLatency;
    LwlinkCounter_t   *pCounter;
    uint32 nodeId = pFmMessage->nodeid();

    lwswitch::nodeStatsAck *pResponse;
    lwswitch::fmMessage *pFmResponse;  //response to send back to Global Fabric Manager
    const lwswitch::nodeStats &statsreport = pFmMessage->statsreport();

    pFmResponse = new lwswitch::fmMessage();
    pFmResponse->set_requestid( pFmMessage->requestid() );

    pResponse = new lwswitch::nodeStatsAck;
    pFmResponse->set_allocated_nodestatsack( pResponse );
    pFmResponse->set_type( lwswitch::FM_NODE_STATS_ACK );

    parseStatsReportMsg( nodeId, statsreport );

    // send the response back to Local FM
    rc = sendMessage( nodeId, pFmResponse );
    if ( rc != FM_SUCCESS )
    {
        PRINT_DEBUG(" ", "GlobalStatsHandler: failed to send stats ack message to LocalFM");
    }

    delete pFmResponse;
}

/**
 *  on Global FM handle FM_NODE_STATS_ACK message
 */
void
DcgmGlobalStatsMsgHndlr::handleNodeStatsAckMsg(lwswitch::fmMessage *pFmMessage)
{

}

void
DcgmGlobalStatsMsgHndlr::handleLWLinkErrorRecoveryMsg(lwswitch::fmMessage *pFmMessage)
{
    uint32 nodeId = pFmMessage->nodeid();
    lwswitch::lwlinkErrorMsg errMsg = pFmMessage->lwlinkerrormsg();
    if ( errMsg.has_recoverymsg() )
    {
        mpGfm->queueErrorWorkerRequest( nodeId, pFmMessage );
    }
    else
    {
        PRINT_WARNING("", "GlobalStatsHandler: received LWLink error recovery message without required fields");
    }
}

void
DcgmGlobalStatsMsgHndlr::handleLWLinkErrorFatalMsg(lwswitch::fmMessage *pFmMessage)
{
    uint32 nodeId = pFmMessage->nodeid();
    lwswitch::lwlinkErrorMsg errMsg = pFmMessage->lwlinkerrormsg();
    if ( errMsg.has_gpufatalmsg() )
    {
        mpGfm->queueErrorWorkerRequest( nodeId, pFmMessage );
    }
    else
    {
        PRINT_WARNING("", "GlobalStatsHandler: received LWLink error fatal message without required fields");
    }
}

/*
 * Send the message to the specified connection
 */
FM_ERROR_CODE
DcgmGlobalStatsMsgHndlr::sendMessage(uint32 nodeId, lwswitch::fmMessage *pFmMessage)
{
    dcgmReturn_t ret = mpGfm->SendMessageToLfm( nodeId, pFmMessage, false );
    FM_ERROR_CODE rc = (ret == DCGM_ST_OK) ? FM_SUCCESS : FM_MSG_SEND_ERR;
    return rc;
}
