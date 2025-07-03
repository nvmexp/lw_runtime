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
#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>

#include "fm_log.h"
#include "GlobalFmErrorStatusMsgHndlr.h"

GlobalFmErrorStatusMsgHndlr::GlobalFmErrorStatusMsgHndlr(GlobalFabricManager *pGfm)
{
    mpGfm = pGfm;
}

GlobalFmErrorStatusMsgHndlr::~GlobalFmErrorStatusMsgHndlr()
{

}

void
GlobalFmErrorStatusMsgHndlr::handleEvent(FabricManagerCommEventType eventType, uint32 nodeId)
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
GlobalFmErrorStatusMsgHndlr::handleMessage(lwswitch::fmMessage  *pFmMessage)
{
    FM_LOG_DEBUG("GlobalFmErrorStatusMsgHndlr: message type %d", pFmMessage->type());

    switch ( pFmMessage->type() )
    {
    case lwswitch::FM_LWSWITCH_ERROR_FATAL:
        dumpMessage(pFmMessage);
        handleLWSwitchFatalErrorReportMsg( pFmMessage );
        break;

    case lwswitch::FM_LWSWITCH_ERROR_NON_FATAL:
        handleLWSwitchNonFatalErrorReportMsg( pFmMessage );
        break;

    case lwswitch::FM_LWSWITCH_FATAL_ERROR_SCOPE:
        dumpMessage(pFmMessage);
        handleLWSwitchFatalErrorScopeMsg( pFmMessage );
        break;

    case lwswitch::FM_LWLINK_ERROR_LWSWITCH_RECOVERY:
    case lwswitch::FM_LWLINK_ERROR_GPU_RECOVERY:
        dumpMessage(pFmMessage);
        handleLWLinkErrorRecoveryMsg( pFmMessage );
        break;

    case lwswitch::FM_LWLINK_ERROR_GPU_FATAL:
        dumpMessage(pFmMessage);
        handleGpuLWLinkFatalErrorMsg( pFmMessage );
        break;

    default:
        FM_LOG_ERROR("fabric manager error handler received unknown message type %d", pFmMessage->type());
        break;
    }
}

void
GlobalFmErrorStatusMsgHndlr::dumpMessage(lwswitch::fmMessage *pFmMessage)
{
#ifdef DEBUG
    std::string msgText;

    google::protobuf::TextFormat::PrintToString(*pFmMessage, &msgText);
    FM_LOG_DEBUG("GlobalFmErrorStatusMsgHndlr: %s", msgText.c_str());
#endif
}

/**
 *  on Global FM, handle FM_LWSWITCH_ERROR_FATAL message from Local FM
 */
void
GlobalFmErrorStatusMsgHndlr::handleLWSwitchFatalErrorReportMsg(lwswitch::fmMessage *pFmMessage)
{
    // Parse and log the fatal error
    // as error handling is done in FM_LWSWITCH_FATAL_ERROR_SCOPE processing
    parseLWSwitchErrorReportMsg( pFmMessage->errorreport(),
                                 pFmMessage->has_nodeid() ?  pFmMessage->nodeid() : 0);
}

/**
 *  on Global FM, handle FM_LWSWITCH_FATAL_ERROR_SCOPE message from Local FM
 */
void
GlobalFmErrorStatusMsgHndlr::handleLWSwitchFatalErrorScopeMsg(lwswitch::fmMessage *pFmMessage)
{
    // do globalFM fatal error handling
    mpGfm->queueErrorWorkerRequest( pFmMessage->nodeid(), pFmMessage );
}

/**
 *  on Global FM, handle FM_LWSWITCH_ERROR_NON_FATAL message from Local FM
 */
void
GlobalFmErrorStatusMsgHndlr::handleLWSwitchNonFatalErrorReportMsg(lwswitch::fmMessage *pFmMessage)
{
    // Parse and log the non fatal error
    parseLWSwitchErrorReportMsg( pFmMessage->errorreport(),
                                 pFmMessage->has_nodeid() ?  pFmMessage->nodeid() : 0);
}

void
GlobalFmErrorStatusMsgHndlr::handleLWLinkErrorRecoveryMsg(lwswitch::fmMessage *pFmMessage)
{
    uint32 nodeId = pFmMessage->nodeid();
    lwswitch::lwlinkErrorMsg errMsg = pFmMessage->lwlinkerrormsg();
    if ( errMsg.has_recoverymsg() )
    {
        mpGfm->queueErrorWorkerRequest( nodeId, pFmMessage );
    }
    else
    {
        FM_LOG_WARNING("received LWLink recovery error message without required fields");
    }
}

void
GlobalFmErrorStatusMsgHndlr::handleGpuLWLinkFatalErrorMsg(lwswitch::fmMessage *pFmMessage)
{
    uint32 nodeId = pFmMessage->nodeid();
    lwswitch::lwlinkErrorMsg errMsg = pFmMessage->lwlinkerrormsg();
    if ( errMsg.has_gpufatalmsg() )
    {
        mpGfm->queueErrorWorkerRequest( nodeId, pFmMessage );
    }
    else
    {
        FM_LOG_WARNING("received GPU LWLink fatal error message without required fields");
    }
}

void
GlobalFmErrorStatusMsgHndlr::parseLWSwitchErrorReportMsg(const lwswitch::switchErrorReport &errorReportMsg,
                                                         uint32_t nodeId)
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
            // syslog the error
            logLWSwitchError( switchPhysicalId, errorInfo, nodeId );
        }
    }
}

void
GlobalFmErrorStatusMsgHndlr::logLWSwitchError(uint32_t switchPhysicalId,
                                              const lwswitch::switchErrorInfo &errorInfo,
                                              uint32 nodeId)
{
    uint32_t partitionId = ILWALID_FABRIC_PARTITION_ID;
    uint32_t portNum = errorInfo.instance();
    uint32_t errorValue = errorInfo.errorvalue();

    // mGfmPartitionMgr is NULL in non shared fabric mode,
    // and partitionId will be set to ILWALID_FABRIC_PARTITION_ID, which will not be logged.
    if ( mpGfm && mpGfm->mGfmPartitionMgr )
    {
        partitionId = mpGfm->mGfmPartitionMgr->getActivePartitionIdForLWSwitchPort(nodeId,
                                                                                   switchPhysicalId,
                                                                                   portNum);
    }

    FMPciInfo_t pciInfo = {0};
    mpGfm->getLWSwitchPciBdf(nodeId, switchPhysicalId, pciInfo);

    if ( errorInfo.severity() == lwswitch::LWSWITCH_ERROR_SEVERITY_FATAL )
    {
        if ( partitionId != ILWALID_FABRIC_PARTITION_ID ) {
            std::ostringstream ss;
            ss << "detected LWSwitch fatal error " << errorValue << " on partition " << partitionId << " LWSwitch pci bus id " << pciInfo.busId
               << " physical id " << switchPhysicalId << " port " << portNum;
            FM_LOG_ERROR("%s", ss.str().c_str());
            FM_SYSLOG_ERR("%s", ss.str().c_str());
        } else {
            std::ostringstream ss;
            ss << "detected LWSwitch fatal error " << errorValue << " on " NODE_ID_LOG_STR " " << nodeId << " on LWSwitch pci bus id " << pciInfo.busId
               << " physical id " << switchPhysicalId << " port " << portNum;
            FM_LOG_ERROR("%s", ss.str().c_str());
            FM_SYSLOG_ERR("%s", ss.str().c_str());
        }
    }
    else
    {
        if ( partitionId != ILWALID_FABRIC_PARTITION_ID ) {
            std::ostringstream ss;
            ss << "detected LWSwitch non-fatal error "<< errorValue << " on partition " << partitionId << " LWSwitch pci bus id " << pciInfo.busId
               << " physical id " << switchPhysicalId << " port " << portNum;
            FM_LOG_ERROR("%s", ss.str().c_str());
            FM_SYSLOG_ERR("%s", ss.str().c_str());
        } else {
            std::ostringstream ss;
            ss << "detected LWSwitch non-fatal error " << errorValue << " on " NODE_ID_LOG_STR " " << nodeId << " on LWSwitch pci bus id "  << pciInfo.busId
               << " physical id " << switchPhysicalId << " port " << portNum;
            FM_LOG_ERROR("%s", ss.str().c_str());
            FM_SYSLOG_ERR("%s", ss.str().c_str());
        }
    }
}

/*
 * Send the message to the specified connection
 */
FMIntReturn_t
GlobalFmErrorStatusMsgHndlr::sendMessage(uint32 nodeId, lwswitch::fmMessage *pFmMessage)
{
    FMIntReturn_t ret = mpGfm->SendMessageToLfm( nodeId, pFmMessage, false );
    FMIntReturn_t rc = (ret == FM_INT_ST_OK) ? FM_INT_ST_OK : FM_INT_ST_MSG_SEND_ERR;
    return rc;
}
