#include <iostream>
#include <sstream>

#include "logging.h"
#include "DcgmFMCommon.h"
#include "DcgmGlobalFabricManager.h"
#include "DcgmGFMHelper.h"
#include "DcgmGlobalFMErrorHndlr.h"
#include "DcgmLogging.h"
#include "DcgmFMAutoLock.h"

DcgmGlobalFMErrorHndlr::DcgmGlobalFMErrorHndlr(DcgmGlobalFabricManager *pGfm,
                                               uint32_t nodeId,
                                               uint32_t partitionId,
                                               GlobalFMErrorSource errSource,
                                               GlobalFMErrorTypes  errType,
                                               lwswitch::fmMessage &errMsg)
{
    mGfm = pGfm;
    mErrorInfo.nodeId = nodeId;
    mErrorInfo.partitionId = partitionId;
    mErrorInfo.errSource = errSource;
    mErrorInfo.errType = errType;
    mErrorInfo.errMsg = errMsg;
}

DcgmGlobalFMErrorHndlr::~DcgmGlobalFMErrorHndlr()
{
    // nothing as of now
}

void
DcgmGlobalFMErrorHndlr::processErrorMsg(void)
{
    switch(mErrorInfo.errType) {
        case ERROR_TYPE_LWLINK_RECOVERY: {
            handleErrorLWLinkRecovery();
            break;
        }
        case ERROR_TYPE_LWLINK_FATAL: {
            handleErrorLWLinkFatal();
            break;
        }
        case ERROR_TYPE_FATAL: {
            handleErrorFatal();
            break;
        }
        case ERROR_TYPE_CONFIG_NODE_FAILED:
        case ERROR_TYPE_CONFIG_SWITCH_FAILED:
        case ERROR_TYPE_CONFIG_SWITCH_PORT_FAILED:
        case ERROR_TYPE_CONFIG_GPU_FAILED: {
        case ERROR_TYPE_CONFIG_TIMEOUT:
            handleErrorConfigFailed();
            break;
        }
        case ERROR_TYPE_SOCKET_DISCONNECTED: {
            handleErrorSocketDisconnected();
            break;
        }
        case ERROR_TYPE_HEARTBEAT_FAILED: {
            handleErrorHeartbeatFailed();            
            break;
        }
        case ERROR_TYPE_SHARED_PARTITION_CONFIG_FAILED: {
            handleErrorSharedPartitionConfigFailed();
            break;
        }
        default: {
            PRINT_ERROR( "", "GlobalFMErrorHandler: unknown error type information" );
            break;
        }
    }
}

/*************************************************************************************
 * Handle an LWLink Long Recovery error message. When H/W trigger this error, 
 * the associated LWLink connection will transition to SAFE mode. As part of the
 * error handling, GFM will identify the associated connection and initiate a
 * re-train request to restore the connection to Active (HS) mode.
 *
 * Note:
 *  For LWSwitch, the recovery error is hooked up with non-fatal error handling.
 *  Due to this, the error will be reported to DCGM CacheManager directly by the
 *  LWSwitch LocalStats/GlobalStats handlers.
 *
 *  For GPU, on a single node system, the error is already reported to CacheManager
 *  when DCGM read the corresponding LWML events. For multi-node systems, the goal
 *  for DCGM is to mirror/transport each node's CacheManager memory to GlobalFM or
 *  Master DCGM entity.
 * 
 **************************************************************************************/
void
DcgmGlobalFMErrorHndlr::handleErrorLWLinkRecovery(void)
{
    int ret;
    uint32 nodeId = mErrorInfo.errMsg.nodeid();
    const lwswitch::lwlinkErrorMsg &errorMsg = mErrorInfo.errMsg.lwlinkerrormsg();
    const lwswitch::lwlinkErrorRecoveryMsg &recoveryMsg = errorMsg.recoverymsg();

    // first get the LWLink device which generated this recovery error
    DcgmFMLWLinkDevInfo lwLinkDevInfo;
    ret = mGfm->mLWLinkDevRepo.getDeviceInfo( nodeId, recoveryMsg.gpuorswitchid(), lwLinkDevInfo );
    if ( !ret ) {
        PRINT_WARNING( "", "GlobalFMErrorHandler: failed to find LWLink device which generated recovery error" );
        return;
    }

    // find the reported connections and retrain.
    for ( int i=0; i< recoveryMsg.linkindex_size(); i++ ) {
        DcgmLWLinkEndPointInfo endPoint;
        endPoint.nodeId = nodeId;
        endPoint.gpuOrSwitchId = recoveryMsg.gpuorswitchid();
        endPoint.linkIndex = recoveryMsg.linkindex( i );
        DcgmFMLWLinkDetailedConnInfo *connInfo = mGfm->mLWLinkConnRepo.getConnectionInfo( endPoint );
        if ( connInfo != NULL ) {
            // found the connection. re-train the connection
            std::stringstream outStr;
            connInfo->dumpConnInfo( &outStr, mGfm->mLWLinkDevRepo);

            DcgmLWLinkEndPointInfo masterEndpoint = connInfo->getMasterEndPointInfo();
            DcgmLWLinkEndPointInfo slaveEndpoint = connInfo->getSlaveEndPointInfo();
            DcgmFMLWLinkDevInfo masterDevInfo;
            DcgmFMLWLinkDevInfo slaveDevInfo;
            mGfm->mLWLinkDevRepo.getDeviceInfo( masterEndpoint.nodeId, masterEndpoint.gpuOrSwitchId, masterDevInfo );
            mGfm->mLWLinkDevRepo.getDeviceInfo( slaveEndpoint.nodeId, slaveEndpoint.gpuOrSwitchId, slaveDevInfo );

            if (mGfm->isSharedFabricMode() &&
                (masterDevInfo.getDeviceType() != slaveDevInfo.getDeviceType()))
            {
                // Do not re-train access link in shared fabric mode
                // as GPUs are not with the service VM for retraining
                PRINT_INFO("%s", "GlobalFMErrorHandler: Skip re-training following access connection due to LWLink recovery error %s\n",
                           outStr.str().c_str());
                outStr.str(""); // clear previous string
                continue;
            }

            PRINT_INFO("%s", "GlobalFMErrorHandler: re-training following connection due to LWLink recovery error %s\n",
                       outStr.str().c_str());
            DcgmGFMHelper::trainLWLinkConnection( mGfm->mLinkTrainIntf, mGfm->mLWLinkConnRepo, mGfm->mLWLinkDevRepo,
                                                  connInfo, LWLINK_TRAIN_SAFE_TO_HIGH );
            outStr.str(""); // clear previous string
            connInfo->dumpConnAndStateInfo( &outStr, mGfm->mLWLinkDevRepo);
            PRINT_INFO("%s", "GlobalFMErrorHandler: re-training completed for connection %s\n", outStr.str().c_str());
            // publish link state to dcgm cache manager after the retraining attempt
            masterDevInfo.publishLinkStateToCacheManager( mGfm );
            slaveDevInfo.publishLinkStateToCacheManager( mGfm );
        }
    }
}

/*************************************************************************************
 * Handle an LWLink Fatal error message. 
 *  To recover from an LWLink Fatal error, we need a GPU/LWSwtich Link level reset.
 *  The current version will just clear the FM session state so that no further LWCA
 *  clients are allowed by RM. Later the administrator will either reboot the machine
 *  or reset the devices out-of-band and restart FabricManager.
 *
 * Note:
 *  For LWSwitch, the fatal error is hooked up with fatal error handling.
 *  Due to this, the error will be reported to DCGM CacheManager directly by the
 *  LWSwitch LocalStats/GlobalStats handlers.
 *
 *  For GPU, on a single node system, the error is already reported to CacheManager
 *  when DCGM read the corresponding LWML events. For multi-node systems, the goal
 *  for DCGM is to mirror/transport each node's CacheManager memory to GlobalFM or
 *  Master DCGM entity.
 * 
 **************************************************************************************/
void
DcgmGlobalFMErrorHndlr::handleErrorLWLinkFatal(void)
{
    if (mErrorInfo.errSource == ERROR_SOURCE_GPU) {
        PRINT_ERROR( "", "GlobalFMErrorHandler: received GPU LWLink fatal error ");
        FM_SYSLOG_ERR("Found GPU LWLink fatal error");
    } else {
        PRINT_ERROR( "", "GlobalFMErrorHandler: received LWSwitch LWLink fatal error ");
        FM_SYSLOG_ERR("Found LWSwitch LWLink fatal error");
    }

    if (mGfm->isSharedFabricMode())
    {
        // Do not clear FM session on the service VM in shared fabric mode
        // Fatal error in one partition should not affect activation of other partitions
        // FM session needs to be set, as GPU health check depends on it.
        return;
    }

    // send clear FM session
    mGfm->mpConfig->sendConfigDeInitReqMsg(mErrorInfo.nodeId);
}

/*************************************************************************************
 * Handle a Fatal error reported by GPU/Switch
 *  To recover from a Fatal error, we need a GPU/LWSwtich Link level reset.
 *  The current version will just clear the FM session state so that no further LWCA
 *  clients are allowed by RM. Later the administrator will either reboot the machine
 *  or reset the devices out-of-band and restart FabricManager.
 *
 * Note:
 *  For LWSwitch, the fatal error is hooked up with fatal error handling.
 *  Due to this, the error will be reported to DCGM CacheManager directly by the
 *  LWSwitch LocalStats/GlobalStats handlers.
 *
 *  For GPU - No fatal error is reported to GlobalFM.
 **************************************************************************************/
void
DcgmGlobalFMErrorHndlr::handleErrorFatal(void)
{
    if (mErrorInfo.errSource == ERROR_SOURCE_GPU) {
        PRINT_ERROR( "", "GlobalFMErrorHandler: received GPU fatal error");
        FM_SYSLOG_ERR("Found GPU fatal error");

    } else {
        PRINT_ERROR( "", "GlobalFMErrorHandler: received LWSwitch fatal error");    
        FM_SYSLOG_ERR("Found LWSwitch fatal error");
    }

    if (mErrorInfo.errSource == ERROR_SOURCE_GPU) {
        // TODO - log the information to CacheManager.
        // No GPU fatal error is reported to GlobalFM as of now.
    }

    if (mGfm->isSharedFabricMode())
    {
        // Do not clear FM session on the service VM in shared fabric mode
        // Fatal error in one partition should not affect activation of other partitions
        // FM session needs to be set, as GPU health check depends on it.
        return;
    }

    // send clear FM session
    mGfm->mpConfig->sendConfigDeInitReqMsg(mErrorInfo.nodeId);
}

void
DcgmGlobalFMErrorHndlr::handleErrorConfigFailed(void)
{
    switch(mErrorInfo.errType) {
        case ERROR_TYPE_CONFIG_NODE_FAILED: {
            // a node global configuration failed
            FM_SYSLOG_ERR("failed to configure LWSwitch/GPU global settings");
            break;
        }
        case ERROR_TYPE_CONFIG_SWITCH_FAILED: {
            FM_SYSLOG_ERR("Failed to configure LWSwitch settings ");
            break;
        }
        case ERROR_TYPE_CONFIG_SWITCH_PORT_FAILED: {
            FM_SYSLOG_ERR("Failed to configure LWSwitch port");
            break;
        }
        case ERROR_TYPE_CONFIG_GPU_FAILED: {
            FM_SYSLOG_ERR("Failed to configure GPU settings");
            break;
        }
        default: {
            // shouldn't land here
            PRINT_ERROR("%d", "DcgmGlobalFMErrorHndlr: unhandled config error type %d", mErrorInfo.errType);
            break;
       }
    }

    // set error state
    mGfm->setNodeConfigError(mErrorInfo.nodeId);

    // send clear FM session
    mGfm->mpConfig->sendConfigDeInitReqMsg(mErrorInfo.nodeId);
}

void
DcgmGlobalFMErrorHndlr::handleErrorSocketDisconnected(void)
{
    FM_SYSLOG_ERR("lost socket connection between global and local fabric manager instance");

    // send clear FM session
    mGfm->mpConfig->sendConfigDeInitReqMsg(mErrorInfo.nodeId);
}

void
DcgmGlobalFMErrorHndlr::handleErrorHeartbeatFailed(void)
{
    FM_SYSLOG_ERR("failed heartbeat between global and local fabric manager instance");
}

//
// handle partition config failures in shared fabric mode
// error could happen in different context and thread
//   1. when a config request is sent to local fabric manager
//   2. when an asyc error response comes back from local fabric manager
//   3. when an expected response timed out
//
// multiple errors oclwrred in different places for the same partition
// are only handled once.
//
void
DcgmGlobalFMErrorHndlr::handleErrorSharedPartitionConfigFailed(void)
{
    DcgmGFMFabricPartitionMgr *pPartitionMgr = mGfm->mGfmPartitionMgr;
    PartitionInfo partInfo;
    if ( pPartitionMgr->getSharedLWSwitchPartitionInfo(mErrorInfo.nodeId,
                                                       mErrorInfo.partitionId,
                                                       partInfo) == false )
    {
        return;
    }

    PRINT_ERROR("%s %d %d %d", "%s: nodeId %d, partitionId %d, errorHandled %d.",
                __FUNCTION__, mErrorInfo.nodeId, mErrorInfo.partitionId, partInfo.errorHandled);

    // do not wait for all responses to come back
    // as the error could be caused by response timeout
    bool inErrHdlr = true;

    if ( !pPartitionMgr->isPartitionConfigFailed(mErrorInfo.nodeId, mErrorInfo.partitionId) )
    {
        // errors have not been already handled

        // multiple errors are only handled once
        pPartitionMgr->setPartitionConfigFailure(mErrorInfo.nodeId, mErrorInfo.partitionId);

        // Disable routing
        mGfm->mpConfig->configDeactivateSharedLWSwitchPartition(mErrorInfo.nodeId,
                                                                partInfo,
                                                                inErrHdlr);
        // reset access lwlinks
        pPartitionMgr->resetPartitionLWSwitchLinks(mErrorInfo.nodeId,
                                                   mErrorInfo.partitionId,
                                                   inErrHdlr);

        // reset trunk lwlinks
        pPartitionMgr->partitionResetFilteredTrunkLWLinks(mErrorInfo.nodeId,
                                                          mErrorInfo.partitionId,
                                                          inErrHdlr);
    }

    // Detach GPUs
    mGfm->mpConfig->configSharedLWSwitchPartitionDetachGPUs(mErrorInfo.nodeId,
                                                            mErrorInfo.partitionId,
                                                            inErrHdlr);

    // Clear pending request before starting config this partition
    mGfm->mpConfig->clearPartitionPendingConfigRequest( mErrorInfo.nodeId, mErrorInfo.
                                                        partitionId );
}



