#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/poll.h>
#include <unistd.h>

#include "FMErrorCodesInternal.h"
#include "LocalFabricManager.h"
#include "LocalFabricManagerCoOp.h"
#include "FmServerConnection.h"
#include "fabricmanager.pb.h"
#include "LocalFmCommandServer.h"
#include "LocalFMMemMgr.h"
#include "LocalFMMemMgrExporter.h"
#include "fm_config_options.h"
#include "LocalFMGpuMgr.h"
#include "FMCommonTypes.h"
#include "FMErrorCodesInternal.h"
#include "FMGpuDriverVersionCheck.h"
#include "LocalFmSwitchHeartbeatReporter.h"
#include "LocalFMMemMgrImporter.h"
#include "LocalFMSwitchEventReader.h"

#include "fm_log.h"
#include <g_lwconfig.h>


extern LWOSCriticalSection gLockRefernceCount;

/*****************************************************************************/
/*                                                                           */
/*                            Control                                        */
/*                                                                           */
/*****************************************************************************/
LocalFabricManagerControl::LocalFabricManagerControl(LocalFmArgs_t *lfm)
{
    // set all the member pointers which requires explicit clean-up to null.
    mFMGpuMgr = NULL;
    mFmSessionState = STATE_NOT_ALLOCATED;

    // init our localFM command server interface
    mLocalCmdServer = NULL;

    mLocalFmSwitchHeartbeatReporter = NULL;
    mpControlMsgHndl = NULL;
    mLocalFMCoOpMgr = NULL;
    mLinkTrainHndlr = NULL;
    mLWLinkDevRepo = NULL;
    mLWLinkDrvIntf = NULL;
    mDevInfoMsgHndlr = NULL;
    mLocalFMSwitchEvtReader = NULL;
    mpControlServer = NULL;
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    mLocalFMMemMgr = NULL;
    mLocalFMMemMgrExporter = NULL;
    mLocalFMMemMgrImporter = NULL;
#endif
    mvSwitchInterface.clear();
    mSimMode = lfm->simMode;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    mMulticastHndlr = NULL;
#endif

    //
    // create and initialize all the localFM class objects. this initialization
    // can throw exception. catch the same and do required clean-up as
    // destructor will not be called in this.
    //
    try {
        doLocalFMInitialization(lfm);
    } catch (const std::runtime_error &e) {
        // initialziation failed, do whatever clean-up we can do.
        cleanup();

        // finally pass the exception to caller
        throw std::runtime_error(e.what());
    }
}

void LocalFabricManagerControl::doLocalFMInitialization(LocalFmArgs_t *lfm)
{
    #ifndef LW_MODS_GDM
    FMIntReturn_t fmResult;

    mFabricMode = lfm->fabricMode;
    mMyNodeId = 0xFFFF; // indicate node id as uninitialized
    mContinueWithFailure = lfm->continueWithFailures;
    mAbortLwdaJobsOnFmExit = lfm->abortLwdaJobsOnFmExit;
    mDegradedSwitchInfo.clear();
    mExcludedLwswitchInfoList.clear();
    mSwitchHeartbeatTimeout = lfm->switchHeartbeatTimeout;
    mImexReqTimeout = lfm->imexReqTimeout;

    lwosInitializeCriticalSection(&gLockRefernceCount);

    //
    // LocalFM communicate with three different drivers, namely RM(GPU), LWSwitch and LWLinkCoreLib.
    // RM driver communication is abstracted into LocalFMGpuMgr class.
    // LWSwitch driver communication is abstracted into LocalFMSwitchInterface class.
    // LWLinkCoreLib driver communication is abstracted into LocalFMLWLinkDrvIntf class.
    // Lwrrently all of these drivers are combined into a single driver (namely lwpu.ko).
    // Fabric Manager use direct IOCTL to communicate with these drivers, so it must maintain
    // the application binary interface (ABI) compatibility with these drivers. Usually Fabric Manager
    // is tied-up/built along with driver builds. However, for small bug fixes/features which don't break
    // this driver ABI compatibility, lwstomers would like to upgrade FM only. In those cases,
    // customer driver version will be whitelisted for compatibility.

    // Lwrrently all the other drivers (LWSwitch and LWLinkCoreLib) uses the RM version itself. So the
    // version is validated explicitly using the FMGpuDriverVersionCheck interface class and other
    // driver interfaces will skip the version check. In future, if these
    // drivers are broken into individual drivers, then their respective abstraction class should
    // verify each driver versions.

    //
    // Always create the FMGpuDriverVersionCheck instance and validate version. This will throw
    // exception if the version doesn't match or whitelisted
    //

    FMGpuDriverVersionCheck gpuDrvVersionChk;
    gpuDrvVersionChk.checkGpuDriverVersionCompatibility("fabric manager");

    // rest of the code assumes the driver versions are validated.
    mFMGpuMgr = new LocalFMGpuMgr();


    // create and open lwswitch interfaces.
    createLwswitchInterface();

    //
    // RM capability management is hooked-up and validated when FmSession object is allocated.
    // So, there is no explicit API to acquire RM fabric management/privileged IOCTL capability.
    // If Fabric Manager don't have permission, then this FmSession allocation will fail
    //
    // Note: Lwrrently this RM capability management is applied/applicable to only Linux based clients.
    //

    if (mvSwitchInterface.size() > 0) {
        fmResult = mFMGpuMgr->allocFmSession(mAbortLwdaJobsOnFmExit);
        if (fmResult != FM_INT_ST_OK) {
            std::ostringstream ss;
            ss << "failed to allocate fabric manager session for LWPU GPU driver";
            FM_LOG_ERROR("%s", ss.str().c_str());
            throw std::runtime_error(ss.str());
        }
        mFmSessionState = STATE_ALLOCATED;
    }

    fmResult = mFMGpuMgr->initializeAllGpus();
    if (fmResult != FM_INT_ST_OK) {
        std::ostringstream ss;
        ss << "local fabric manager: failed to initialize and open handles to all the GPUs";
        FM_LOG_ERROR("%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }

    uint32_t rspTimeIntrvl = lfm->simMode ? FM_REQ_RESP_TIME_INTRVL_SIM : FM_REQ_RESP_TIME_INTRVL;
    uint32_t rspTimeThreshold = lfm->simMode ? FM_REQ_RESP_TIME_THRESHOLD_SIM : FM_REQ_RESP_TIME_THRESHOLD;

    // start the control server that accepts controls from gfm
    if (lfm->domainSocketPath == NULL || !(strnlen(lfm->domainSocketPath, FM_CONFIG_MAX_STRING_ITEM_LEN))) {
        // using TCP I/P sockets and default listening interface is localhost
        mpControlServer = new LocalFMLwcmServer( this, lfm->fmStartingTcpPort, lfm->bindInterfaceIp, true,
                                                 rspTimeIntrvl, rspTimeThreshold );
    } else {
        // use Unix Domain Socket interface. This is applicable for single node systems only
        mpControlServer = new LocalFMLwcmServer( this, lfm->fmStartingTcpPort, lfm->domainSocketPath, false,
                                                 rspTimeIntrvl, rspTimeThreshold );
    }

    mpControlMsgHndl  = new LocalFMControlMsgHndl( this );

    mLocalFMCoOpMgr = new LocalFMCoOpMgr(this, lfm->bindInterfaceIp, (lfm->fmStartingTcpPort + PEER_LFM_COORDINATION_PORT_OFFSET));
    mLWLinkDrvIntf = new LocalFMLWLinkDrvIntf();
    mLWLinkDevRepo = new LocalFMLWLinkDevRepo(mLWLinkDrvIntf);
    mLinkTrainHndlr = new LocalFMLWLinkMsgHndlr(this, mLWLinkDrvIntf, mLWLinkDevRepo);
    mDevInfoMsgHndlr = new LocalFMDevInfoMsgHdlr(this);

    // Start control server
    if (0 != mpControlServer->Start()) {
        std::ostringstream ss;
        ss << "local fabric manager: failed to start fabric manager control connection socket";
        FM_LOG_ERROR("%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }

    /* Wait for the notification that the server has started running.
       Return error if the server can't be started */
    if (0 != mpControlServer->WaitToStart()) {
        std::ostringstream ss;
        ss << "local fabric manager: timeout oclwrred while waiting for fabric manager control connection socket to start";
        FM_LOG_ERROR("%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }

    mLocalFmSwitchHeartbeatReporter = new LocalFmSwitchHeartbeatReporter(this, mSwitchHeartbeatTimeout);

#ifdef DEBUG
#ifndef LW_MODS
    // start our localFM command server interface
    mLocalCmdServer = new LocalFMCommandServer(this);
#endif
#endif
#endif

};

/*****************************************************************************/

LocalFabricManagerControl::~LocalFabricManagerControl()
{
    cleanup();
}

/*****************************************************************************/

void LocalFabricManagerControl::cleanup()
{
    if (mLocalCmdServer) {
        delete mLocalCmdServer;
    }

    // stop FM LWSwitch heartbeat reporting
    if (mLocalFmSwitchHeartbeatReporter) {
        mLocalFmSwitchHeartbeatReporter->stopHeartbeatReporting();
        delete mLocalFmSwitchHeartbeatReporter;
        mLocalFmSwitchHeartbeatReporter = NULL;
    }

    // stop GPU event watching thread before any event unregistration by destructors
    if (mFMGpuMgr) {
        mFMGpuMgr->stopGpuEventWatch();
    }

    delete mpControlMsgHndl;
    delete mLocalFMCoOpMgr;
    delete mLinkTrainHndlr;
    delete mLWLinkDevRepo;
    delete mLWLinkDrvIntf;
    delete mDevInfoMsgHndlr;
    if (mLocalFMSwitchEvtReader) {
        delete mLocalFMSwitchEvtReader;
        mLocalFMSwitchEvtReader = NULL;
    }
    delete mpControlServer;

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    // LocalFMMemMgr depends on LocalFMGpuMgr and an FM session and so LocalFMMemMgr should be
    // call before freeFmSession and delete LocalFMGpuMgr
    if (mLocalFMMemMgr) {
        delete mLocalFMMemMgr;
        mLocalFMMemMgr = NULL;
    }

    if (mLocalFMMemMgrExporter) {
        delete mLocalFMMemMgrExporter;
        mLocalFMMemMgrExporter = NULL;
	}

	if (mLocalFMMemMgrImporter) {
        delete mLocalFMMemMgrImporter;
        mLocalFMMemMgrImporter = NULL;
    }
#endif

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    if (mMulticastHndlr) {
        delete mMulticastHndlr;
        mMulticastHndlr = NULL;
    }
#endif
    // freeFmSession calls a method from LocalFMGpuMgr and so should be called before delete mFMGpuMgr
    //
    // if we successfully allocated FMSession object, free the same here. Otherwise if FM
    // stay running due to stayResident flag, LWCA initialization will wait indefinitely.
    // In this case, Fabric Manager main can't free the session as here localFM instance
    // creation itself is failed.
    //
    if ((mFMGpuMgr != NULL) && (mFmSessionState == STATE_ALLOCATED)) {
        mFMGpuMgr->freeFmSession();
    }

    std::vector <LocalFMSwitchInterface *>::iterator it;
    for ( it = mvSwitchInterface.begin(); it != mvSwitchInterface.end(); it++ ) {
        LocalFMSwitchInterface* switchIntf = *it;
        delete switchIntf;
    }

    delete mFMGpuMgr;

    lwosDeleteCriticalSection(&gLockRefernceCount);
};

/*****************************************************************************/

FMIntReturn_t LocalFabricManagerControl::SendMessageToLfm(uint32 fabricNodeId,
                                                             lwswitch::fmMessage *pFmMessage,
                                                             bool trackReq)
{
    if (mMyNodeId == 0xFFFF) {
        // GFM supposed to set the node's ID before LFM send any message to peer LFM
        FM_LOG_ERROR("request to send message to local fabric manager without initializing node's attributes");
        return FM_INT_ST_UNINITIALIZED;
    }

    pFmMessage->set_version( FABRIC_MANAGER_VERSION );
    pFmMessage->set_nodeid( mMyNodeId );

    return mLocalFMCoOpMgr->SendMessageToPeerLFM( fabricNodeId, pFmMessage, trackReq );
}

void LocalFabricManagerControl::setLocalNodeId(uint32 nodeId)
{
    // TODO: Review the placement of this function
    mMyNodeId = nodeId;

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    // FM session will be allocated only when we have an LWSwitch in localFM. So
    // check for the same before setting the session node id.
    if ( mFmSessionState == STATE_ALLOCATED ) {
        FMIntReturn_t fmResult;
        fmResult = mFMGpuMgr->fmSessionSetNodeId(mMyNodeId);
        if ( fmResult != FM_INT_ST_OK ) {
            FM_LOG_ERROR("request to set fabric manager session node id failed with %s", lwstatusToString(fmResult));
        }
    }
#endif

    // update FM local CoOp with own node id
    mLocalFMCoOpMgr->setLocalNodeId( mMyNodeId );

    mLWLinkDevRepo->setLocalNodeId( mMyNodeId );
}

FMIntReturn_t LocalFabricManagerControl::SendMessageToGfm(lwswitch::fmMessage *pFmMessage, bool trackReq )
{
    if (pFmMessage->type() == lwswitch::FM_HEARTBEAT_ACK) {
        // During FM initialization it is possible that LFM doesn't know it's node ID until configs are done.
        // GFM send LFM it node ID and LFM replies with the same
        const lwswitch::heartbeatAck *pResponse = &(pFmMessage->heartbeatack());
        pFmMessage->set_nodeid( pResponse->nodeid() );
    } else if (mMyNodeId == 0xFFFF) {
        // GFM supposed to set the node's ID before LFM send any message to GFM
        FM_LOG_ERROR("request to send message to global fabric manager without initializing node's attributes");
        return FM_INT_ST_UNINITIALIZED;
    } else {
        pFmMessage->set_nodeid( mMyNodeId );
    }

    pFmMessage->set_version( FABRIC_MANAGER_VERSION );

    return mpControlServer->sendFMMessage(pFmMessage, trackReq);
}

void
LocalFabricManagerControl::createLwswitchInterface( void )
{
    // get switch instances and start the corresponding switch interface
    LWSWITCH_GET_DEVICES_V2_PARAMS params;
    LW_STATUS status;
    uint32_t i;

    mvSwitchInterface.clear();

    status = lwswitch_api_get_devices(&params);

    if (status != LW_OK)
    {
        if (status == LW_ERR_LIB_RM_VERSION_MISMATCH)
        {
            std::ostringstream ss;
            ss << "fabric manager version is incompatible with LWSwitch driver. Please update with matching LWPU driver package";
            FM_LOG_ERROR("%s", ss.str().c_str());
            throw std::runtime_error(ss.str());
        }
        // all other errors, log the error code and bail out
        std::ostringstream ss;
        ss << "request to query LWSwitch device information from LWSwitch driver failed with error:" << lwstatusToString(status);
        FM_LOG_ERROR("%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }

    for (i = 0; i < params.deviceCount; i++)
    {
        LocalFMSwitchInterface *pSwitchIntf;
        // we add a switch with any excluded reason here, as switches that are degraded
        // by FM during it's first run will be excluded by the driver with appropriate reason
        // that means that subsequent runs of FM would fail to open the degraded switch. Hence
        // for those conditions, we add previously degraded switches to the list of excluded Switches
        // in addition to the manually excluded switches
        if (params.info[i].deviceReason != LWSWITCH_DEVICE_BLACKLIST_REASON_NONE) {
	        addExcludedLwSwitchInfo(params.info[i]);
            continue;
        }
        pSwitchIntf = new LocalFMSwitchInterface(params.info[i], mSwitchHeartbeatTimeout);
        mvSwitchInterface.push_back( pSwitchIntf );
    }
}

LocalFMSwitchInterface *
LocalFabricManagerControl::switchInterfaceAt( uint32_t physicalId )
{
    // NOTE: To maintain support for systems which don't have GPIO based
    // physical Ids for the switches, use the index as the physical Id.
    // This is available only in DEBUG build. Release builds only support
    // GPIO based physical ids.

    // first preference for physicalId
    std::vector <LocalFMSwitchInterface *>::iterator it;
    for ( it = mvSwitchInterface.begin(); it != mvSwitchInterface.end(); it++ ) {
        LocalFMSwitchInterface* switchIntf = *it;
        if ( switchIntf->getSwitchPhysicalId() == physicalId ) {
            // found the desired switch interface, return it
            return switchIntf;
        }
    }
#if defined(_DEBUG) || defined(LW_MODS)
    // for no GPIO based physical Ids, treat physicalId as enumeration index
    if ( (int)physicalId < (int)mvSwitchInterface.size() )
    {
        return mvSwitchInterface.at( physicalId );
    }
#endif
    return NULL;

}

LocalFMSwitchInterface *
LocalFabricManagerControl::switchInterfaceAt( FMUuid_t &uuid )
{
    std::vector <LocalFMSwitchInterface *>::iterator it;
    for ( it = mvSwitchInterface.begin(); it != mvSwitchInterface.end(); it++ ) {
        LocalFMSwitchInterface* switchIntf = *it;
        FMUuid_t tempUuid = switchIntf->getUuid();
        if ( tempUuid == uuid ) {
            // found the desired switch interface, return it
            return switchIntf;
        }
    }

    // not found
    return NULL;
}

LocalFMSwitchInterface *
LocalFabricManagerControl::switchInterfaceAtIndex( int index )
{
    if (mvSwitchInterface.size() == 0) {
        return NULL;
    }

    return mvSwitchInterface.at(index);
}

uint32_t
LocalFabricManagerControl::getSwitchDevIndex( uint32_t physicalId )
{
    // NOTE: To maintain support for systems which don't have GPIO based
    // physical Ids for the switches, use the index as the physical Id.
    // This is available only in DEBUG build. Release builds only support
    // GPIO based physical ids.

    // first preference for physicalId
    std::vector <LocalFMSwitchInterface *>::iterator it;
    for ( it = mvSwitchInterface.begin(); it != mvSwitchInterface.end(); it++ ) {
        LocalFMSwitchInterface* switchIntf = *it;
        if ( switchIntf->getSwitchPhysicalId() == physicalId ) {
            // found the desired switch interface, return its index
            return switchIntf->getSwitchDevIndex();
        }
    }
#if defined(_DEBUG) || defined(LW_MODS)
    // for no GPIO based physical Ids, treat physicalId as enumeration index
    if ( (int)physicalId < (int)mvSwitchInterface.size() )
    {
        LocalFMSwitchInterface* switchIntf = mvSwitchInterface.at( physicalId );
        if (switchIntf == NULL) {
            return -1;
        }
        return switchIntf->getSwitchDevIndex();
    }
#endif
    return -1;
}

/*****************************************************************************/
/*                                                                           */
/* Second level handling of request. Colwert command to correspoonding IOCTL */
/* and issue to the appropriate switch thread                                */
/*                                                                           */
/*****************************************************************************/
int LocalFabricManagerControl::ProcessMessage(lwswitch::fmMessage * pFmMessage, bool &isResponse)
{
    int ret = 0;
    int i, j;
    switchIoctl_t *ioctlStruct;

    switch (pFmMessage->type())
    {
        case lwswitch::FM_NODE_GLOBAL_CONFIG_REQ:
        case lwswitch::FM_SWITCH_PORT_CONFIG_REQ:
        case lwswitch::FM_INGRESS_REQUEST_TABLE_REQ:
        case lwswitch::FM_INGRESS_RESPONSE_TABLE_REQ:
        case lwswitch::FM_GPU_CONFIG_REQ:
        case lwswitch::FM_GANGED_LINK_TABLE_REQ:
        case lwswitch::FM_HEARTBEAT:
        case lwswitch::FM_CONFIG_INIT_DONE_REQ:
        case lwswitch::FM_CONFIG_DEINIT_REQ:
        case lwswitch::FM_GPU_ATTACH_REQ:
        case lwswitch::FM_GPU_DETACH_REQ:
        case lwswitch::FM_SWITCH_DISABLE_LINK_REQ:
        case lwswitch::FM_GPU_SET_DISABLED_LINK_MASK_REQ:
        case lwswitch::FM_GPU_GET_GFID_REQ:
        case lwswitch::FM_GPU_CFG_GFID_REQ:
        case lwswitch::FM_RMAP_TABLE_REQ:
        case lwswitch::FM_RID_TABLE_REQ:
        case lwswitch::FM_RLAN_TABLE_REQ:
        case lwswitch::FM_DEGRADED_GPU_INFO:
        case lwswitch::FM_DEGRADED_LWSWITCH_INFO:
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
        case lwswitch::FM_MCID_TABLE_SET_REQ:
#endif
            mpControlMsgHndl->handleMessage(pFmMessage);
            isResponse = false;
            break;

        case lwswitch::FM_NODE_INFO_MSG:
        {
            mLocalFMCoOpMgr->handleMessage(pFmMessage);
            isResponse = false;
            break;
        }
        case lwswitch::FM_NODE_GET_LWLINK_DEVICE_INFO_REQ:
        case lwswitch::FM_NODE_GET_LWSWITCH_DEVICE_INFO_REQ:
        case lwswitch::FM_NODE_GET_GPU_DEVICE_INFO_REQ:
        case lwswitch::FM_NODE_GET_GPU_LWLINK_SPEED_INFO_REQ:
        case lwswitch::FM_NODE_GET_VERSION_INFO_REQ:
        {
            mDevInfoMsgHndlr->handleMessage(pFmMessage);
            isResponse = false;
            break;
        }
        case lwswitch::FM_MASTER_LWLINK_CONN_PARALLEL_SWITCH_OFF:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_TO_SAFE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_TO_HIGH:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_HIGH_TO_SAFE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_SAFE_TO_OFF:
        case lwswitch::FM_MASTER_LWLINK_CONN_SWITCH_OFF:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_SAFE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_HIGH_TO_SAFE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_SAFE_TO_OFF:
        case lwswitch::FM_SLAVE_LWLINK_CONN_SWITCH_OFF:
        case lwswitch::FM_SLAVE_LWLINK_CONN_TRAIN_TO_SAFE:
        case lwswitch::FM_SLAVE_LWLINK_CONN_TRAIN_TO_HIGH:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH_SUBLINK:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH_MAINLINK:
        case lwswitch::FM_SLAVE_LWLINK_CONN_TRAIN_HIGH_TO_SAFE:
        case lwswitch::FM_SLAVE_LWLINK_CONN_TRAIN_SAFE_TO_OFF:
        case lwswitch::FM_LWLINK_TRAIN_RSP_MASTER_SYNC:
        case lwswitch::FM_LWLINK_TRAIN_RSP_SLAVE_SYNC:
        case lwswitch::FM_LWLINK_TRAIN_RSP_SLAVE_CONFIRM:
        case lwswitch::FM_LWLINK_TRAIN_RSP_SLAVE_COMPLETE:
        case lwswitch::FM_LWLINK_ENABLE_TX_COMMON_MODE:
        case lwswitch::FM_LWLINK_DISABLE_TX_COMMON_MODE:
        case lwswitch::FM_LWLINK_CALIBRATE:
        case lwswitch::FM_LWLINK_ENABLE_DATA:
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        case lwswitch::FM_LWLINK_OPTICAL_INIT_LINKS:
        case lwswitch::FM_LWLINK_OPTICAL_ENABLE_IOBIST:
        case lwswitch::FM_LWLINK_OPTICAL_START_PRETRAIN_TX:
        case lwswitch::FM_LWLINK_OPTICAL_CHECK_PRETRAIN_TX:
        case lwswitch::FM_LWLINK_OPTICAL_START_PRETRAIN_RX:
        case lwswitch::FM_LWLINK_OPTICAL_CHECK_PRETRAIN_RX:
        case lwswitch::FM_LWLINK_OPTICAL_STOP_PRETRAIN:
        case lwswitch::FM_LWLINK_OPTICAL_DISABLE_IOBIST:
#endif
        case lwswitch::FM_LWLINK_INITPHASE1:
        case lwswitch::FM_LWLINK_INITPHASE5:
        case lwswitch::FM_LWLINK_RX_INIT_TERM:
        case lwswitch::FM_LWLINK_SET_RX_DETECT:
        case lwswitch::FM_LWLINK_GET_RX_DETECT:
        case lwswitch::FM_LWLINK_INITNEGOTIATE:
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_INITOPTIMIZE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_POST_INITOPTIMIZE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_INF_MODE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_INITOPTIMIZE_TO_HIGH:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_TO_OFF:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_MAINTENANCE_TX:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_MAINTENANCE_RX:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_DISABLE_INF_MODE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_FORCE_EQ:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_DISABLE_FORCE_EQ:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_CHECK_EOM_STATUS:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_GET_LINK_STATE:
        case lwswitch::FM_LWLINK_CONN_TRAIN_INTERNODE_GET_GRADING_AND_FOM_VALUES:
#endif
        case lwswitch::FM_LWLINK_INIT:
        case lwswitch::FM_LWLINK_INIT_STATUS:
        case lwswitch::FM_LWLINK_DISCOVER_INTRANODE_CONNS:
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        case lwswitch::FM_LWLINK_ADD_INTERNODE_CONN:
#endif
        case lwswitch::FM_LWLINK_GET_INTRANODE_CONNS:
        case lwswitch::FM_LWLINK_WRITE_DISCOVERY_TOKENS:
        case lwswitch::FM_LWLINK_READ_DISCOVERY_TOKENS:
        case lwswitch::FM_LWLINK_READ_SIDS:
        case lwswitch::FM_LWLINK_RESET_SWITCH_LINKS:
        case lwswitch::FM_LWLINK_RESET_ALL_SWITCH_LINKS:
        case lwswitch::FM_LWLINK_SWITCH_TRAINING_FAILED_LINK_INFO:
        case lwswitch::FM_LWLINK_GET_DEVICE_LWLINK_STATE:
        {
            mLinkTrainHndlr->handleMessage(pFmMessage);
            isResponse = false;
            break;
        }
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
        case lwswitch::FM_MULTICAST_GROUP_CREATE_RSP:
        case lwswitch::FM_MULTICAST_GROUP_BIND_RSP:
        case lwswitch::FM_MULTICAST_GROUP_RELEASE_RSP:
        case lwswitch::FM_MULTICAST_GROUP_SETUP_COMPLETE_REQ:
        case lwswitch::FM_MULTICAST_GROUP_RELEASE_COMPLETE_REQ:
        {
            mMulticastHndlr->handleMessage(pFmMessage);
            isResponse = false;
            break;
        }

        case lwswitch::FM_LWLINK_INBAND_MSG:
        {
            // Inband TODO
            isResponse = false;
            break;
        }
#endif
        default:
        {
            FM_LOG_WARNING("detected unsupported fabric manager request type %d in message request handler",
                            pFmMessage->type());
            break;
        }
    }

    // end of switch on command type
    return ret;
}


int LocalFabricManagerControl::ProcessPeerLFMMessage(uint32 nodeId, lwswitch::fmMessage* pFmMessage, bool &isResponse)
{
    // TODO merge this function and ProcessRequest() as both of them are handling messages
    // TODO handle/route messages to appropriate message handlers

    switch (pFmMessage->type())
    {
        case lwswitch::FM_MASTER_LWLINK_CONN_SWITCH_OFF:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_SAFE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_HIGH_TO_SAFE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_SAFE_TO_OFF:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH_SUBLINK:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH_MAINLINK:
        case lwswitch::FM_SLAVE_LWLINK_CONN_SWITCH_OFF:
        case lwswitch::FM_SLAVE_LWLINK_CONN_TRAIN_TO_SAFE:
        case lwswitch::FM_SLAVE_LWLINK_CONN_TRAIN_TO_HIGH:
        case lwswitch::FM_SLAVE_LWLINK_CONN_TRAIN_HIGH_TO_SAFE:
        case lwswitch::FM_SLAVE_LWLINK_CONN_TRAIN_SAFE_TO_OFF:
        case lwswitch::FM_LWLINK_TRAIN_RSP_MASTER_SYNC:
        case lwswitch::FM_LWLINK_TRAIN_RSP_SLAVE_SYNC:
        case lwswitch::FM_LWLINK_TRAIN_RSP_SLAVE_CONFIRM:
        case lwswitch::FM_LWLINK_TRAIN_RSP_SLAVE_COMPLETE:
        case lwswitch::FM_LWLINK_ENABLE_TX_COMMON_MODE:
        case lwswitch::FM_LWLINK_DISABLE_TX_COMMON_MODE:
        case lwswitch::FM_LWLINK_CALIBRATE:
        case lwswitch::FM_LWLINK_ENABLE_DATA:
        case lwswitch::FM_LWLINK_INIT:
        case lwswitch::FM_LWLINK_INIT_STATUS:
        case lwswitch::FM_LWLINK_DISCOVER_INTRANODE_CONNS:
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        case lwswitch::FM_LWLINK_ADD_INTERNODE_CONN:
#endif
        case lwswitch::FM_LWLINK_GET_INTRANODE_CONNS:
        case lwswitch::FM_LWLINK_WRITE_DISCOVERY_TOKENS:
        case lwswitch::FM_LWLINK_READ_DISCOVERY_TOKENS:
        case lwswitch::FM_LWLINK_READ_SIDS:
        case lwswitch::FM_LWLINK_GET_DEVICE_LWLINK_STATE:
        {
            mLinkTrainHndlr->handleMessage(pFmMessage);
            isResponse = false;
            break;
        }
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        case lwswitch::FM_MEMORY_FLA_IMPORT_REQ:
        case lwswitch::FM_MEMORY_FLA_UNIMPORT_REQ:
        case lwswitch::FM_MEMORY_FLA_FATAL_ERROR_MSG:
        {
            mLocalFMMemMgrExporter->handleMessage(pFmMessage);
            isResponse = false;
            break;
        }
        case lwswitch::FM_MEMORY_IMPORT_REQ:
        case lwswitch::FM_MEMORY_UNIMPORT_REQ:
        {
            mLocalFMMemMgr->handleMessage(pFmMessage);
            isResponse = false;
            break;
        }
        case lwswitch::FM_MEMORY_FLA_IMPORT_RSP:
        case lwswitch::FM_MEMORY_FLA_UNIMPORT_RSP:
        {
            mLocalFMMemMgrImporter->handleMessage(pFmMessage);
            isResponse = false;
            break;
        }
        case lwswitch::FM_MEMORY_IMPORT_RSP:
        case lwswitch::FM_KT_MEMORY_IMPORT_ERR:
        case lwswitch::FM_MEMORY_UNIMPORT_RSP:
        {
            mLocalFMMemMgr->handleMessage(pFmMessage);
            // TODO start tracking responses for Memory import/unimport messages
            isResponse = false;
            break;
        }
#endif

        default: {
            FM_LOG_WARNING("detected unsupported peer fabric manager request type %d in message request handler",
                            pFmMessage->type());
            break;
        }
    }
    return 1;
}

void LocalFabricManagerControl::ProcessUnSolicitedMessage(lwswitch::fmMessage * pFmMessage)
{

}

void LocalFabricManagerControl::ProcessConnect(void)
{

}

void LocalFabricManagerControl::ProcessDisconnect(void)
{

}

void LocalFabricManagerControl::getAllLWLinkDevInfo(FMLWLinkDevInfoList &lwlinkDevList)
{
    //
    // GFM is requesting LWLink device information. Fetch/update the information in repo context
    // with latest information from LWSwitchCoreLib driver.
    //
    mLWLinkDevRepo->populateLWLinkDeviceList();
    lwlinkDevList = mLWLinkDevRepo->getDeviceList();
}

void LocalFabricManagerControl::getAllLwswitchInfo(FMLWSwitchInfoList &switchInfoList)
{
    std::vector<LocalFMSwitchInterface*>::iterator it;

    for (it = mvSwitchInterface.begin() ; it != mvSwitchInterface.end(); ++it) {
        LocalFMSwitchInterface *switchInterface = *it;
        FMLWSwitchInfo switchInfo = {0};
        switchInfo.switchIndex = switchInterface->getSwitchDevIndex();
        switchInfo.physicalId = switchInterface->getSwitchPhysicalId();
        switchInfo.pciInfo = switchInterface->getSwtichPciInfo();
        switchInfo.enabledLinkMask = switchInterface->getEnabledPortMask();
        switchInfo.archType = switchInterface->getSwitchArchType();
        switchInfo.uuid = switchInterface->getUuid();
        switchInfoList.push_back(switchInfo);
    }
}

void LocalFabricManagerControl::addExcludedLwSwitchInfo(LWSWITCH_DEVICE_INSTANCE_INFO_V2 switchInfo)
{
    FMExcludedLWSwitchInfo_t excludedSwitchInfo = {0};
    excludedSwitchInfo.physicalId = switchInfo.physId;
    FMPciInfo_t pciInfo;
    pciInfo.domain = switchInfo.pciDomain;
    pciInfo.bus = switchInfo.pciBus;
    pciInfo.device = switchInfo.pciDevice;
    pciInfo.function = switchInfo.pciFunction;
    snprintf(pciInfo.busId, FM_DEVICE_PCI_BUS_ID_BUFFER_SIZE,
             FM_DEVICE_PCI_BUS_ID_FMT, FM_DEVICE_PCI_BUS_ID_FMT_ARGS(&pciInfo));

    FMUuid_t uuid;
    memset(uuid.bytes, 0, FM_UUID_BUFFER_SIZE);
    lwswitch_uuid_to_string(&switchInfo.uuid, uuid.bytes, FM_UUID_BUFFER_SIZE);
    excludedSwitchInfo.pciInfo = pciInfo;
    strncpy(excludedSwitchInfo.uuid.bytes, uuid.bytes, FM_UUID_BUFFER_SIZE - 1);
    lwswitch::SwitchDegradedReason reason;
    mpControlMsgHndl->colwertDriverToFmDegradedReason(reason, switchInfo.deviceReason);
    excludedSwitchInfo.excludedReason = reason;
    mExcludedLwswitchInfoList.push_back(excludedSwitchInfo);
}

void LocalFabricManagerControl::getExcludedLwswitchInfo(FMExcludedLWSwitchInfoList &excludedLwswitchInfoList)
{
    excludedLwswitchInfoList.assign(mExcludedLwswitchInfoList.begin(), mExcludedLwswitchInfoList.end());
}

void LocalFabricManagerControl::addDegradedSwitchInfo(uint32_t physicalId, lwswitch::SwitchDegradedReason reason)
{
    mDegradedSwitchInfo.insert(make_pair(physicalId, reason));
    closeAndDeleteLWSwitchInterface(physicalId);
}

void LocalFabricManagerControl::closeAndDeleteLWSwitchInterface(uint32_t physicalId)
{
    // remove switch from vector of switches
    LocalFMSwitchInterface* switchIntf = NULL;
    bool bFound = false;
    std::vector <LocalFMSwitchInterface *>::iterator it;
    for ( it = mvSwitchInterface.begin(); it != mvSwitchInterface.end(); it++ ) {
        switchIntf = *it;
        if ( switchIntf->getSwitchPhysicalId() == physicalId ) {
            // found the desired switch interface, remove it from vector
            bFound = true;
            mvSwitchInterface.erase(it);
            break;
        }
    }

    if (bFound) {
        delete switchIntf;
        switchIntf = NULL;
    }

    // no switch found with given physical id
}

bool LocalFabricManagerControl::isSwitchDegraded(uint32_t physicalId)
{
    if (mDegradedSwitchInfo.find(physicalId) != mDegradedSwitchInfo.end()) {
        return true;
    }

    return false;
}

void LocalFabricManagerControl::getGpuPciInfo(FMUuid_t &uuid, FMPciInfo_t &pciInfo)
{
    FMGpuInfoList gpuInfoList;
    FMGpuInfoList::iterator it;
    FMIntReturn_t fmResult = FM_INT_ST_OK;
    fmResult = mFMGpuMgr->getInitializedGpuInfo(gpuInfoList);

    for ( it = gpuInfoList.begin(); it != gpuInfoList.end(); it++ ) {
        FMGpuInfo_t tempGpuInfo = *it;
        if ( tempGpuInfo.uuid == uuid ) {
            // found the GPU we are looking for.
            pciInfo = tempGpuInfo.pciInfo;
            return;
        }
    }
}
void LocalFabricManagerControl::getAllGpuInfo(FMGpuInfoList &gpuInfoList)
{
    mFMGpuMgr->getInitializedGpuInfo(gpuInfoList);
}

void LocalFabricManagerControl::getExcludedGpuInfo(FMExcludedGpuInfoList &excludedGpuInfoList)
{
    FMIntReturn_t fmResult;
    fmResult = mFMGpuMgr->getExcludedGpuInfo(excludedGpuInfoList);
    if ( fmResult != FM_INT_ST_OK ) {
        FM_LOG_ERROR("request to get excluded GPU information failed with error %d", fmResult);
    }
}

FMIntReturn_t LocalFabricManagerControl::refreshRmLibProbedGpuInfo( void )
{
    return mFMGpuMgr->refreshRmLibProbedGpuInfo();
}

FMIntReturn_t LocalFabricManagerControl::attachGpu( FMUuid_t uuid, bool registerEvent )
{
    // Attach Gpu with given uuid
    return mFMGpuMgr->initializeGpu(uuid, registerEvent);
}

FMIntReturn_t LocalFabricManagerControl::detachAllGpus( void )
{
    // Detach all GPUs
    return mFMGpuMgr->deInitializeAllGpus();
}

FMIntReturn_t LocalFabricManagerControl::detachGpu( FMUuid_t uuid, bool unRegisterEvent )
{
    // Detach Gpu with given uuid
    return mFMGpuMgr->deInitializeGpu(uuid, unRegisterEvent);
}

FMIntReturn_t LocalFabricManagerControl::getGpuGfid(FMUuid_t uuid, FMPciInfo_t &vf, uint32_t &gfid, uint32_t &gfidMask)
{
    // Get GFID for a given GPU
    return mFMGpuMgr->getGfid(uuid, vf, gfid, gfidMask);
}

FMIntReturn_t LocalFabricManagerControl::configGpuGfid(FMUuid_t uuid, uint32_t gfid, bool activate)
{
    // Config GFID for a given GPU
    return mFMGpuMgr->configGfid(uuid, gfid, activate);
}

bool LocalFabricManagerControl::onConfigInitDoneReqRcvd(void)
{
    FMIntReturn_t fmResult;

    if ( mvSwitchInterface.size() == 0 )
    {
        // there is no lwswitch on the node, there is no need to set session
        return true;
    }

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    // create Mem Manager even for single node setups to support IMEX on same node
    mLocalFMMemMgr = new LocalFMMemMgr(mLocalFMCoOpMgr, this);

	// create Mem Manager even for single node setups to support IMEX on same node
    mLocalFMMemMgrExporter = new LocalFMMemMgrExporter(mLocalFMCoOpMgr, this);
    mLocalFMMemMgrImporter = new LocalFMMemMgrImporter(mLocalFMCoOpMgr, mImexReqTimeout, this);

    // set exporter on import side and vice versa for single node import/export cases
    mLocalFMMemMgrImporter->setExporter(mLocalFMMemMgrExporter);
    mLocalFMMemMgrExporter->setImporter(mLocalFMMemMgrImporter);
#endif

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    //
    // Multicast should only be initialized after LFM finishes initialization successfully,
    // because multicast cannot accept dynamic commands before all switches are fully
    // initialized.
    //
    if (!mMulticastHndlr) {
        //
        // onConfigInitDoneReqRcvd() might be called multiple times,
        // only need to create LocalFmMulticastHndlr once
        //
        mMulticastHndlr = new LocalFmMulticastHndlr(this);
    }
#endif

    /* creating error reporter here so that we need not create event handler for degraded switches.
       Since the LocalFMErrorReporter is only needed after configInitDone request is received,
       we can create it here. Creating it above in the LFM constructor would require the extra
       work of freeing event handler if and when switch is degraded
    */
    try {
        mLocalFMSwitchEvtReader = new LocalFMSwitchEventReader( this, mLWLinkDevRepo );
    }
    catch (const std::runtime_error &e) {
        FM_LOG_ERROR("%s", e.what());
        FM_SYSLOG_ERR("%s", e.what());
        return false;
    }

    // start our error monitoring service
    mLocalFMSwitchEvtReader->Start();

    // start GPU event monitoring thread
    fmResult = mFMGpuMgr->startGpuEventWatch();
    if ( fmResult != FM_INT_ST_OK ) {
        FM_LOG_ERROR("request to start GPU event watching thread failed with error:%d", fmResult);
        return false;
    }

    // config init done will be received once. So only checking whether FM session was allocated or not
    if ( mFmSessionState == STATE_ALLOCATED ) {
        fmResult = mFMGpuMgr->fmSessionSetState();
        if ( fmResult != FM_INT_ST_OK ) {
            FM_LOG_ERROR("request to set fabric manager session in GPU Driver failed with error:%d", fmResult);
            return false;
        }
    }

    // do our FM LWSwitch heartbeat reporting when all the steps are done.
    mLocalFmSwitchHeartbeatReporter->startHeartbeatReporting();

    return true;
}

bool LocalFabricManagerControl::onConfigDeInitReqRcvd(void)
{
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    if (mLocalFMMemMgrExporter) {
        mLocalFMMemMgrExporter->disableMessageProcessing();
    }

    if (mLocalFMMemMgrImporter) {
        mLocalFMMemMgrImporter->disableProcessing();
    }
#endif
    return freeFmSession();
}


bool LocalFabricManagerControl::freeFmSession(void)
{
    FMIntReturn_t fmResult;

    if ( mvSwitchInterface.size() == 0 )
    {
        // there is no lwswitch on the node, there is no need to close session
        return true;
    }

    if ( mFmSessionState == STATE_NOT_ALLOCATED )
    {
        // already closed, probably due to previous errors
        return true;
    }

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    FM_LOG_ERROR("Aborting all the LWCA jobs and leaving the system uninitialized.");
    FM_LOG_ERROR("New LWCA job launch will fail on this system. "
                 "Please refer to your system user guide for recovery procedure.");
#endif
    // free the FMSession state
    fmResult = mFMGpuMgr->freeFmSession();
    if (fmResult != FM_INT_ST_OK) {
        FM_LOG_ERROR("request to free fabric manager session in GPU Driver failed with %s",
                      lwstatusToString(fmResult));
        return false;
    }

    mFmSessionState = STATE_NOT_ALLOCATED;
    return true;
}

void LocalFabricManagerControl::handleStayResidentCleanup(void)
{
    //
    // This will be called by main() to do some essential clean-up as part of stayResident
    // handling. This can be removed once we have a proper destructor for LocalFabricManagerControl
    //
    // Free our allocated FMSession for now, so that LWCA initialization won't wait indefinitely.
    //

    freeFmSession();
}

void LocalFabricManagerControl::setFmDriverStateToStandby(void)
{
    //
    // On Fabric Manager graceful exit, make a best effort to set the LWSwitch device's
    // Driver state to STATE_STANDBY. Otherwise once FM successfully configured,
    // the switch state will be in STATE_CONFIGURED. In configured state, Driver will be
    // expecting heartbeat and will set the state to MANAGER_TIMEOUT once the heartbeat timer
    // expires.
    //

    std::vector <LocalFMSwitchInterface *>::iterator it;
    for (it = mvSwitchInterface.begin(); it != mvSwitchInterface.end(); it++) {
        LocalFMSwitchInterface *pSwitchIntf = *(it);
        if (pSwitchIntf == NULL) {
            FM_LOG_ERROR("failed to get LWSwitch driver interface object for setting fabric state to standby");
            continue;
        }

        switchIoctl_t ioctlStruct;
        LWSWITCH_SET_FM_DRIVER_STATE_PARAMS ioctlParams;
        memset(&ioctlParams, 0, sizeof(ioctlParams));
        ioctlParams.driverState = LWSWITCH_DRIVER_FABRIC_STATE_STANDBY;
        ioctlStruct.type = IOCTL_LWSWITCH_SET_FM_DRIVER_STATE;
        ioctlStruct.ioctlParams = &ioctlParams;
        ioctlStruct.paramSize = sizeof(ioctlParams);
        FMIntReturn_t rc = pSwitchIntf->doIoctl(&ioctlStruct);
        if (rc != FM_INT_ST_OK) {
            // request failed, log error and continue for rest of the switches
            std::ostringstream ss;
            FMPciInfo_t pciInfo = pSwitchIntf->getSwtichPciInfo();
            ss << "request to set fabric state to standby during fabric manager shutdown failed for LWSwitch index:" << pSwitchIntf->getSwitchDevIndex()
               << " pci bus id:" << pciInfo.busId << " with error:" << rc;
            FM_LOG_ERROR("%s", ss.str().c_str());
        }
    }
}

