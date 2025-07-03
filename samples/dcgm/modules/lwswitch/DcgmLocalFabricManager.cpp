
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/poll.h>
#include <unistd.h>

#include "lwml.h"
#include "lwml_internal.h"
#include "LwcmHostEngineHandler.h"
#include "DcgmModuleLwSwitch.h"
#include "dcgm_lwswitch_structs.h"
#include "LwcmProtobuf.h"
#include "dcgm_structs.h"
#include "DcgmLocalStatsReporter.h"
#include "DcgmLocalStatsMsgHndlr.h"
#include "DcgmLocalFabricManager.h"
#include "DcgmLocalFabricManagerCoOp.h"
#include "LwcmServerConnection.h"
#include "fabricmanager.pb.h"
#include "LwcmSettings.h"
#include "DcgmLocalCommandServer.h"
#include "DcgmLocalMemMgr.h"

#include "logging.h"
#include <g_lwconfig.h>


/*****************************************************************************/
/*                                                                           */
/*                            Control                                        */
/*                                                                           */
/*****************************************************************************/
DcgmLocalFabricManagerControl::DcgmLocalFabricManagerControl(bool sharedFabric, 
                                                             char *bindInterfaceIp,
                                                             unsigned short startingPort,
                                                             char *domainSocketPath)
{
    mInitComplete = false;
    lwmlReturn_t lwmlResult;

    mSharedFabric = sharedFabric;

    // Get the host engine and cache manager pointers so we can add our lwswitches to the cache manager
    mpHostEngineHandler = LwcmHostEngineHandler::Instance();
    if (NULL == mpHostEngineHandler) {
        PRINT_ERROR("", "failed to get DCGM host engine handler instance");
        throw std::runtime_error("failed to get DCGM host engine handler instance");
    }

    mpCacheManager = mpHostEngineHandler->GetCacheManager();
    if (NULL == mpCacheManager) {
        PRINT_ERROR("", "failed to get DCGM cache manager instance");
        throw std::runtime_error("failed to get DCGM cache manager instance");
    }

    mMyNodeId = 0xFFFF; // indicate node id as uninitialized

    // create and open lwswitch interfaces.
    createLwswitchInterface();

    lwmlResult = lwmlInternalGetExportTable((const void**)&metblLwmlCommonInternal, &ETID_LWMLCommonInternal);

    if (LWML_SUCCESS != lwmlResult) {
        std::ostringstream ss;
        ss << "LocalFabricManager: failed to get LWML exported function table with error: " << lwmlErrorString(lwmlResult);
        PRINT_ERROR("%s", "%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }

    // create FMSession object
    if ( mvSwitchInterface.size() > 0 ) {
        unsigned int notUsed = 0;
        lwmlResult = LWML_CALL_ETBL( metblLwmlCommonInternal, AllocFmSession, (notUsed));
        if(LWML_SUCCESS != lwmlResult) {
            std::ostringstream ss;
            ss << "LocalFabricManager: FMSession allocation failed with: " << lwmlErrorString(lwmlResult);
            PRINT_ERROR("%s", "%s", ss.str().c_str());
            throw std::runtime_error(ss.str());
        }
        mFmSessionState = STATE_ALLOCATED;
    } else {
        // there is no switch on this node, there is no need to open session
        mFmSessionState = STATE_CLOSED;
    }

    // start the control server that accepts controls from gfm
    if (domainSocketPath == NULL) {
        // using TCP I/P sockets and default listening interface is localhost
        mpControlServer = new DcgmFMLwcmServer( this, startingPort, bindInterfaceIp, true );
    } else {
        // use Unix Domain Socket interface. This is applicable for single node systems only
        mpControlServer = new DcgmFMLwcmServer( this, startingPort, domainSocketPath, false );
    }

    mpControlMsgHndl  = new DcgmLocalControlMsgHndl( this, metblLwmlCommonInternal);

    mFMLocalCoOpMgr = new DcgmFMLocalCoOpMgr(this, bindInterfaceIp, (startingPort + PEER_LFM_COORDINATION_PORT_OFFSET));

    mLWLinkDrvIntf = new DcgmFMLWLinkDrvIntf();
    mLWLinkDevRepo = new DcgmLFMLWLinkDevRepo(mLWLinkDrvIntf);
    mLinkTrainHndlr = new DcgmFMLWLinkMsgHndlr(this, mLWLinkDrvIntf, mLWLinkDevRepo);
    mDevInfoMsgHndlr = new LFMDevInfoMsgHdlr(this);
    mLocalStatsMsgHndlr = new DcgmLocalStatsMsgHndlr(this, mLocalStatsReporter);

    // create our stats collector
    mLocalStatsReporter = new DcgmLocalStatsReporter( this, mSwitchIdToFdInfoMap, mLWLinkDevRepo);
    
    
    //Create mem manager later only if this is a multi-node setup
    mLocalMemMgr = NULL;
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    mIsMultiNode = false;
#endif

    // Start control server
    if (0 != mpControlServer->Start()) {
        PRINT_ERROR("", "LocalFabricManager: failed to create control connection listening thread");
        throw std::runtime_error("LocalFabricManager: failed to create control connection listening thread");
    }

    /* Wait for the notification that the server has started running. 
       Return error if the server can't be started */
    if (0 != mpControlServer->WaitToStart()) {
        PRINT_ERROR("", "LocalFabricManager: failed to get control connection socket to ready state");
        throw std::runtime_error("LocalFabricManager: failed to get control connection socket to ready state");
    }

#ifdef DEBUG
    // start our localFM command server interface
    mLocalCmdServer = new DcgmLocalCommandServer(this);
#endif

};

/*****************************************************************************/

DcgmLocalFabricManagerControl::~DcgmLocalFabricManagerControl()
{

#ifdef DEBUG
    delete mLocalCmdServer;
#endif

    delete mFMLocalCoOpMgr;
    delete mLinkTrainHndlr;
    delete mLWLinkDevRepo;
    delete mLWLinkDrvIntf;
    delete mDevInfoMsgHndlr;
    delete mLocalStatsMsgHndlr; // delete this before mLocalStatsReporter
    delete mLocalStatsReporter;
    delete mpControlServer;

    if(mLocalMemMgr) 
    {
        delete mLocalMemMgr;
        mLocalMemMgr = NULL;
        return;
    }

    // free FMSession object
    freeFmSession();
};

/*****************************************************************************/

bool DcgmLocalFabricManagerControl::QueryInitComplete() 
{
    return mInitComplete;
}
/*****************************************************************************/

dcgmReturn_t DcgmLocalFabricManagerControl::SendMessageToLfm(uint32 fabricNodeId,
                                                             lwswitch::fmMessage *pFmMessage,
                                                             bool trackReq)
{
    if (mMyNodeId == 0xFFFF) {
        // GFM supposed to set the node's ID before LFM send any message to peer LFM
        PRINT_ERROR("", "SendMessageToLfm sending message with uninitialzed Node ID\n");        
    }

    pFmMessage->set_version( FABRIC_MANAGER_VERSION );
    pFmMessage->set_nodeid( mMyNodeId );

    return mFMLocalCoOpMgr->SendMessageToPeerLFM( fabricNodeId, pFmMessage, trackReq );
}

void DcgmLocalFabricManagerControl::setLocalNodeId(uint32 nodeId)
{
    // TODO: Review the placement of this function
    mMyNodeId = nodeId;

    // update FM local CoOp with own node id 
    mFMLocalCoOpMgr->setLocalNodeId( mMyNodeId );

    mLWLinkDevRepo->setLocalNodeId( mMyNodeId );
}

dcgmReturn_t DcgmLocalFabricManagerControl::SendMessageToGfm(lwswitch::fmMessage *pFmMessage, bool trackReq )
{
    if (mMyNodeId == 0xFFFF) {
        // GFM supposed to set the node's ID before LFM send any message to GFM
        PRINT_ERROR("", "SendMessageToGfm sending message with uninitialzed Node ID\n");        
    }

    pFmMessage->set_version( FABRIC_MANAGER_VERSION );
    pFmMessage->set_nodeid( mMyNodeId );

    return mpControlServer->sendFMMessage(pFmMessage, trackReq);
}

void
DcgmLocalFabricManagerControl::createLwswitchInterface( void )
{
    // get switch instances and start the corresponding switch interface
    LWSWITCH_GET_DEVICES_PARAMS params;
    LW_STATUS status;
    uint32_t i;

    mvSwitchInterface.clear();
    mSwitchIdToFdInfoMap.clear();

    status = lwswitch_api_get_devices(&params);
    if (status != LW_OK)
    {
        if (status == LW_ERR_LIB_RM_VERSION_MISMATCH)
        {
            PRINT_ERROR("", "Fabric Manager version is incompatible with LWSwitch driver. Please update with matching LWPU driver package");
            throw std::runtime_error("Fabric Manager version is incompatible with LWSwitch driver. Please update with matching LWPU driver package");
        }
        // all other errors, log the error code and bail out
        PRINT_ERROR("%d","failed to query device information from LWSwitch driver, return status: %d.", status);
        throw std::runtime_error("LocalFabricManager: failed to query device information from LWSwitch driver");
    }

    for (i = 0; i < params.deviceCount; i++)
    {
        DcgmSwitchInterface *pSwitchIntf;
        pSwitchIntf = new DcgmSwitchInterface( params.info[i].deviceInstance );
        mvSwitchInterface.push_back( pSwitchIntf );
    }

    // build a list of switch physicalId to fd mapping for stats reporting
    for (i = 0; i < mvSwitchInterface.size(); i++ )
    {
        DcgmSwitchInterface *pInterface = mvSwitchInterface.at(i);
        int fd = pInterface->getFd();
        uint32_t physicalId = pInterface->getSwitchPhysicalId();

        // Note/Todo - Remove this for Explorer production
        // For non-gpio based board, the physicalId will be 0
        // Use the switchIndex (enumeration order) as physicalId in that case
        if ( physicalId == 0 )
        {
            physicalId = pInterface->getSwitchDevIndex();
        }
        // build the mapping
        mSwitchIdToFdInfoMap.insert( std::make_pair(physicalId, fd) );
    }

}

DcgmSwitchInterface *
DcgmLocalFabricManagerControl::switchInterfaceAt( uint32_t physicalId )
{
    // Note/Todo: To maintain support for systems which don't have GPIO based
    // physical Ids for the switches, use the index as the physical Id.

    // TODO: For Explorer release branch, change this code to strictly check
    // for physical Ids.

    // first preference for physicalId
    std::vector <DcgmSwitchInterface *>::iterator it;
    for ( it = mvSwitchInterface.begin(); it != mvSwitchInterface.end(); it++ ) {
        DcgmSwitchInterface* switchIntf = *it;
        if ( switchIntf->getSwitchPhysicalId() == physicalId ) {
            // found the desired switch interface, return it
            return switchIntf;
        }
    }

    // for no GPIO based physical Ids, treat physicalId as enumeration index
    if ( (int)physicalId < (int)mvSwitchInterface.size() )
    {
        return mvSwitchInterface.at( physicalId );
    }

    return NULL;

}

uint32_t
DcgmLocalFabricManagerControl::getSwitchDevIndex( uint32_t physicalId )
{
    // Note/Todo: To maintain support for systems which don't have GPIO based
    // physical Ids for the switches, use the index as the physical Id.

    // TODO: For Explorer release branch, change this code to strictly check
    // for physical Ids.

    // first preference for physicalId
    std::vector <DcgmSwitchInterface *>::iterator it;
    for ( it = mvSwitchInterface.begin(); it != mvSwitchInterface.end(); it++ ) {
        DcgmSwitchInterface* switchIntf = *it;
        if ( switchIntf->getSwitchPhysicalId() == physicalId ) {
            // found the desired switch interface, return its index
            return switchIntf->getSwitchDevIndex();
        }
    }

    // for no GPIO based physical Ids, treat physicalId as enumeration index
    if ( (int)physicalId < (int)mvSwitchInterface.size() )
    {
        DcgmSwitchInterface* switchIntf = mvSwitchInterface.at( physicalId );
        return switchIntf->getSwitchDevIndex();
    }

    return 0;

}

/*****************************************************************************/
/*                                                                           */
/* Second level handling of request. Colwert command to correspoonding IOCTL */
/* and issue to the appropriate switch thread                                */
/*                                                                           */
/*****************************************************************************/
int DcgmLocalFabricManagerControl::ProcessMessage(lwswitch::fmMessage * pFmMessage, bool &isResponse)
{
    int ret = 0;
    dcgmReturn_t dcgmReturn;        
    int i, j;
    switchIoctl_t *ioctlStruct;

    PRINT_DEBUG("%d","Local FM got Global Message %d",pFmMessage->type());

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
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
        case lwswitch::FM_RMAP_TABLE_REQ:
        case lwswitch::FM_RID_TABLE_REQ:
        case lwswitch::FM_RLAN_TABLE_REQ:
#endif
            mpControlMsgHndl->handleMessage(pFmMessage);
            isResponse = false;
            break;

        case lwswitch::FM_LWSWITCH_ERROR_FATAL_ACK:
        case lwswitch::FM_LWSWITCH_ERROR_NON_FATAL_ACK:
        case lwswitch::FM_NODE_STATS_ACK:
            mLocalStatsMsgHndlr->handleMessage(pFmMessage);
            isResponse = true;
            break;

        case lwswitch::FM_GET_ERROR_REQ:
        case lwswitch::FM_GET_NODE_STATS_REQ:
            mLocalStatsMsgHndlr->handleMessage(pFmMessage);
            isResponse = false;
            break;

        case lwswitch::FM_NODE_INFO_MSG:
        {   
            mFMLocalCoOpMgr->handleMessage(pFmMessage);
            isResponse = false;
            break;
        }
        case lwswitch::FM_NODE_GET_LWLINK_DEVICE_INFO_REQ:
        case lwswitch::FM_NODE_GET_LWSWITCH_DEVICE_INFO_REQ:
        case lwswitch::FM_NODE_GET_GPU_DEVICE_INFO_REQ:            
        {   
            mDevInfoMsgHndlr->handleMessage(pFmMessage);
            isResponse = false;
            break;
        }
        case lwswitch::FM_MASTER_LWLINK_CONN_SWITCH_OFF:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_SAFE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_HIGH_TO_SAFE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_SAFE_TO_OFF:
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
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
        case lwswitch::FM_LWLINK_INITPHASE1:
        case lwswitch::FM_LWLINK_RX_INIT_TERM:
        case lwswitch::FM_LWLINK_SET_RX_DETECT:
        case lwswitch::FM_LWLINK_GET_RX_DETECT:
        case lwswitch::FM_LWLINK_INITNEGOTIATE:
#endif
        case lwswitch::FM_LWLINK_INIT:
        case lwswitch::FM_LWLINK_INIT_STATUS:    
        case lwswitch::FM_LWLINK_DISCOVER_INTRANODE_CONNS:
        case lwswitch::FM_LWLINK_ADD_INTERNODE_CONN:
        case lwswitch::FM_LWLINK_GET_INTRANODE_CONNS:            
        case lwswitch::FM_LWLINK_WRITE_DISCOVERY_TOKENS:
        case lwswitch::FM_LWLINK_READ_DISCOVERY_TOKENS:
        case lwswitch::FM_LWLINK_RESET_SWITCH_LINKS:
        case lwswitch::FM_LWLINK_RESET_ALL_SWITCH_LINKS:
        {
            mLinkTrainHndlr->handleMessage(pFmMessage);
            isResponse = false;
            break;
        }
        default:
        {
            PRINT_WARNING("%d","Unrecognized request type %d",pFmMessage->type());
            break;
        }
    }
        
    //end of switch on command type
    return ret;
}


int DcgmLocalFabricManagerControl::ProcessPeerLFMMessage(uint32 nodeId, lwswitch::fmMessage* pFmMessage, bool &isResponse)
{
    // TODO merge this function and ProcessRequest() as both of them are handling messages
    // TODO handle/route messages to appropriate message handlers    

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    //TODO: KT specific. remove when we have a better way of handling topology
    mIsMultiNode = true;
#endif
    switch (pFmMessage->type())
    {
        case lwswitch::FM_MASTER_LWLINK_CONN_SWITCH_OFF:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_SAFE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_HIGH_TO_SAFE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_SAFE_TO_OFF:
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
        case lwswitch::FM_LWLINK_ADD_INTERNODE_CONN:
        case lwswitch::FM_LWLINK_GET_INTRANODE_CONNS:            
        case lwswitch::FM_LWLINK_WRITE_DISCOVERY_TOKENS:
        case lwswitch::FM_LWLINK_READ_DISCOVERY_TOKENS:
        {
            mLinkTrainHndlr->handleMessage(pFmMessage);
            isResponse = false;
            break;
        }
        case lwswitch::FM_MEMORY_IMPORT_REQ:
        case lwswitch::FM_MEMORY_UNIMPORT_REQ:
        {
            mLocalMemMgr->handleMessage(pFmMessage);    
            isResponse = false;
            break;
        }
        case lwswitch::FM_MEMORY_IMPORT_RSP:
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        case lwswitch::FM_KT_MEMORY_IMPORT_ERR:
#endif
        case lwswitch::FM_MEMORY_UNIMPORT_RSP:
        {
            mLocalMemMgr->handleMessage(pFmMessage);    
            //TODO start tracking responses for Memory import/unimport messages
            isResponse = false;
            break;
        }

        default: {
            PRINT_WARNING("%d","Unrecognized request type %d",pFmMessage->type());
            break;
        }
    }
    return 1;
}

void DcgmLocalFabricManagerControl::ProcessUnSolicitedMessage(lwswitch::fmMessage * pFmMessage)
{

}

void DcgmLocalFabricManagerControl::ProcessConnect(void)
{

}

void DcgmLocalFabricManagerControl::ProcessDisconnect(void)
{

}

void DcgmLocalFabricManagerControl::getAllLWLinkDevInfo(DcgmFMLWLinkDevInfoList &lwlinkDevList)
{
    //
    // GFM is requesting LWLink device information. Fetch/update the information in repo context
    // with latest information from LWSwitchCoreLib driver.
    //
    mLWLinkDevRepo->populateLWLinkDeviceList();
    lwlinkDevList = mLWLinkDevRepo->getDeviceList();
}

void DcgmLocalFabricManagerControl::getAllLwswitchInfo(DcgmFMLWSwitchInfoList &switchInfoList)
{
    std::vector<DcgmSwitchInterface*>::iterator it;

    for (it = mvSwitchInterface.begin() ; it != mvSwitchInterface.end(); ++it) {
        DcgmSwitchInterface *switchInterface = *it;
        DcgmFMLWSwitchInfo switchInfo = {0};
        switchInfo.switchIndex = switchInterface->getSwitchDevIndex();
        switchInfo.physicalId = switchInterface->getSwitchPhysicalId();
        switchInfo.pciInfo = switchInterface->getSwtichPciInfo();
        switchInfo.enabledLinkMask = switchInterface->getEnabledPortMask();
        // TODO - fill other switch information like eccid when driver expose it
        switchInfoList.push_back(switchInfo);
    }
}

void DcgmLocalFabricManagerControl::getAllGpuInfo(DcgmFMGpuInfoList &gpuInfoList)
{
    std::vector<unsigned int> dcgmGpuIds;
    std::vector<unsigned int>::iterator it;
    dcgmReturn_t retVal;
    lwmlReturn_t lwmlResult;
    lwmlDevice_t lwmlDevice;

    retVal = mpHostEngineHandler->GetLwcmGpuIds( dcgmGpuIds, 1 );
    if ( DCGM_ST_OK != retVal )
    {
        PRINT_ERROR( "%d", "GetLwcmGpuIds failed with %d", retVal );
        return;
    }

    for ( it = dcgmGpuIds.begin() ; it != dcgmGpuIds.end(); it++ ) {
        unsigned int gpuIndex = (*it);
        DcgmFMGpuInfo gpuInfo = {0};

        int lwmlIndex = mpCacheManager->GpuIdToLwmlIndex( gpuIndex );
        if ( lwmlIndex < 0 )
        {
            PRINT_ERROR( "%d", "GpuIdToLwmlIndex failed for index %d.",
                        gpuIndex );
            continue;
        }

        // FM is interested in volta based GPUs only for now.
        lwmlChipArchitecture_t arch;
        mpHostEngineHandler->GetLwcmGpuArch( gpuIndex, arch );

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
        if ( ( arch != LWML_CHIP_ARCH_VOLTA ) &&
             ( arch != LWML_CHIP_ARCH_AMPERE ) )
            continue;
#else
        if ( arch != LWML_CHIP_ARCH_VOLTA )
            continue;
#endif

        lwmlResult = lwmlDeviceGetHandleByIndex( lwmlIndex, &lwmlDevice );
        if ( LWML_SUCCESS != lwmlResult )
        {
             PRINT_ERROR( "%d %d", "lwmlDeviceGetHandleByIndex failed for index %d with %d", 
                         gpuIndex, lwmlResult );
             return;
        } 

        lwmlPciInfo_t lwmlGpuPciInfo;
        lwmlResult = lwmlDeviceGetPciInfo( lwmlDevice, &lwmlGpuPciInfo );
        if( lwmlResult != LWML_SUCCESS )
        {
            PRINT_ERROR("%d %u", "lwmlDeviceGetPciInfo returned %d for lwmlIndex %u",
                        (int)lwmlResult, gpuIndex);
            return;
        }

        char uuidBuf[DCGM_DEVICE_UUID_BUFFER_SIZE] = {0};
        lwmlResult = lwmlDeviceGetUUID(lwmlDevice, uuidBuf, sizeof(uuidBuf));
        if( lwmlResult != LWML_SUCCESS )
        {
            PRINT_ERROR("%d %u", "lwmlDeviceGetUUID returned %d for lwmlIndex %u",
                        (int)lwmlResult, gpuIndex);
            return;
        }

        gpuInfo.pciInfo.domain = lwmlGpuPciInfo.domain;
        gpuInfo.pciInfo.bus = lwmlGpuPciInfo.bus;
        gpuInfo.pciInfo.device = lwmlGpuPciInfo.device;
        //TODO: fill PCI function information. DCGM/LWML don't have this info available straightforward
        gpuInfo.pciInfo.function = 0;
        strncpy(gpuInfo.uuid, uuidBuf, sizeof(gpuInfo.uuid));
        gpuInfo.gpuIndex = gpuIndex;
        gpuInfoList.push_back(gpuInfo);
    }
}

void DcgmLocalFabricManagerControl::getBlacklistGpuInfo(DcgmFMGpuInfoList &blacklistGpuInfoList)
{
    std::vector<lwmlBlacklistDeviceInfo_t> blacklist;
    std::vector<lwmlBlacklistDeviceInfo_t>::iterator it;
    int gpuIndex = 0;

    blacklistGpuInfoList.clear();

    // query the blacklisted gpu information from DCGM
    mpCacheManager->GetGpuBlacklist( blacklist );
    for ( it = blacklist.begin(); it != blacklist.end(); it++ ) {
        lwmlBlacklistDeviceInfo_t blacklistGpuInfo = (*it);
        DcgmFMGpuInfo fmGpuInfo = {0};
        fmGpuInfo.pciInfo.domain = blacklistGpuInfo.pciInfo.domain;
        fmGpuInfo.pciInfo.bus = blacklistGpuInfo.pciInfo.bus;
        fmGpuInfo.pciInfo.device = blacklistGpuInfo.pciInfo.device;
        //TODO: fill PCI function information. DCGM/LWM don't have this info available straightforward
        fmGpuInfo.pciInfo.function = 0;
        fmGpuInfo.gpuIndex = gpuIndex; // index is the order in which the blacklist vector is populated.
        gpuIndex++;
        strncpy( fmGpuInfo.uuid, blacklistGpuInfo.uuid, sizeof(fmGpuInfo.uuid) );
        blacklistGpuInfoList.push_back(fmGpuInfo);
    }
}

dcgmReturn_t DcgmLocalFabricManagerControl::attachAllGpus( void )
{
    // Attach all GPUs
    return mpCacheManager->AttachGpus();
}

dcgmReturn_t DcgmLocalFabricManagerControl::detachAllGpus( void )
{
    // Detach all GPUs
    return mpCacheManager->DetachGpus();
}

bool DcgmLocalFabricManagerControl::onConfigInitDoneReqRcvd(void)
{
    lwmlReturn_t lwmlResult;
    unsigned int notUsed = 0;

    if ( mvSwitchInterface.size() == 0 )
    {
        // there is no lwswitch on the node, there is no need to set session
        return true;
    }

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    //TODO: KT specific, fix once we have a better way to detect multi-node
    if(mIsMultiNode == true)
    {
        freeFmSession();
        //create Mem Manager
        mLocalMemMgr = new DcgmLocalMemMgr(mFMLocalCoOpMgr, this);
        //Start mem manager
        if (0 != mLocalMemMgr->Start()) {
            PRINT_ERROR("", "LocalFabricManager: failed to create memory manager thread");
            throw std::runtime_error("LocalFabricManager: failed to create memory manager thread");
        }
        return true;
    }
#endif

    // set the FMSession state only in shared fabric mode
    if( !isSharedFabricMode() )
    {
        lwmlResult = LWML_CALL_ETBL( metblLwmlCommonInternal, FmSessionSetState, (notUsed));
        if(LWML_SUCCESS != lwmlResult)
        {
            PRINT_ERROR("%d", "FmSessionSetState failed with %d", lwmlResult);
            return false;
        }

        mFmSessionState = STATE_SET;
    }

    // start our stats/error monitoring service
    mLocalStatsReporter->Start();

    return true;
}

bool DcgmLocalFabricManagerControl::onConfigDeInitReqRcvd(void)
{
    return freeFmSession();
}

bool DcgmLocalFabricManagerControl::freeFmSession(void)
{
    lwmlReturn_t lwmlResult;
    unsigned int notUsed = 0;

    if ( mvSwitchInterface.size() == 0 )
    {
        // there is no lwswitch on the node, there is no need to close session
        return true;
    }

    if ( mFmSessionState == STATE_CLOSED )
    {
        // already closed, probably due to previous errors
        return true;
    }

    // free the FMSession state
    lwmlResult = LWML_CALL_ETBL( metblLwmlCommonInternal, FreeFmSession, (notUsed));
    if(LWML_SUCCESS != lwmlResult)
    {
        PRINT_ERROR("%d", "FreeFmSession failed with %d", lwmlResult);
        return false;
    }

    mFmSessionState = STATE_CLOSED;
    return true;
}

/*****************************************************************************/
/*                                                                           */
/*                      Local Fabric Manager                                 */
/*                                                                           */
/*****************************************************************************/
DcgmLocalFabricManager::DcgmLocalFabricManager(bool sharedFabric,
                                               char *bindInterfaceIp,
                                               unsigned short startingPort,
                                               char *domainSocketPath)
{
    mpGlobalControl = new DcgmLocalFabricManagerControl(sharedFabric, bindInterfaceIp, startingPort, domainSocketPath);
}

DcgmLocalFabricManager::~DcgmLocalFabricManager()
{
    if (mpGlobalControl)
    {
        delete mpGlobalControl;
        mpGlobalControl = NULL;
    }
}

