
#include <sstream>

#include "logging.h"
#include "DcgmGFMHelper.h"
#include "DcgmLogging.h"
#include <g_lwconfig.h>


int
DcgmGFMHelper::lwLinkInitializeAllNodes(DcgmFMLWLinkIntf *linkTrainIntf, 
                                        DcgmFabricParser* pConfig,
                                        DcgmGFMLWLinkDevRepo &linkDevRepo)
{
    NodeIDList nodeIdList;
    std::map <NodeKeyType, NodeConfig *>::iterator nodeit;

    // create a list of node IDs from the parsed fabric config
    for ( nodeit = pConfig->NodeCfg.begin(); nodeit!= pConfig->NodeCfg.end(); nodeit++ ) {
        NodeConfig *pNode = nodeit->second;
        nodeIdList.push_back( pNode->nodeId );
    }

    // call the helper to initialize the nodes
    return lwLinkInitializeNodes(nodeIdList, linkTrainIntf, linkDevRepo);
}

int
DcgmGFMHelper::lwLinkInitializeNode(uint32 nodeId,
                                    DcgmFMLWLinkIntf *linkTrainIntf, 
                                    DcgmGFMLWLinkDevRepo &linkDevRepo)
{
    // only one node to initialize
    NodeIDList nodeIdList;
    nodeIdList.push_back( nodeId );
    // call the helper to initialize the node
    return lwLinkInitializeNodes(nodeIdList, linkTrainIntf, linkDevRepo);
}    

int
DcgmGFMHelper::lwLinkDiscoverIntraNodeConnOnNodes(DcgmFabricParser *pConfig,
                                                  DcgmFMLWLinkIntf *linkTrainIntf,
                                                  DcgmFMLWLinkConnRepo &linkConnRepo)
{
    NodeIDList nodeIdList;
    std::map <NodeKeyType, NodeConfig *>::iterator nodeit;

    // this routine will initiate LWLink connection discovery and report
    // all the connections local to each node. It is assumed that the links 
    // are in SAFE/HS mode prior to initiating discovery.

    // create a list of node IDs from the parsed fabric config
    for ( nodeit = pConfig->NodeCfg.begin(); nodeit!= pConfig->NodeCfg.end(); nodeit++ ) {
        NodeConfig *pNode = nodeit->second;
        nodeIdList.push_back( pNode->nodeId );
    }

    if (lwLinkSendDiscoverConnOnNodes(nodeIdList, linkTrainIntf)) {
        PRINT_ERROR("", "Failed to initiate device discovery on all nodes");
        return 1;
    }

    // call the helper to query all the discovered connections
    return lwLinkGetIntraNodeConns(nodeIdList, linkTrainIntf, linkConnRepo);

}

int
DcgmGFMHelper::lwLinkDiscoverIntraNodeConnOnNode(uint32 nodeId,
                                                 DcgmFMLWLinkIntf *linkTrainIntf,
                                                 DcgmFMLWLinkConnRepo &linkConnRepo)
{
    // this routine will initiate LWLink connection discovery and report
    // all the connections local to the node. It is assumed that the links 
    // are in SAFE/HS mode prior to initiating discovery.

    // only one node to initialize
    NodeIDList nodeIdList;
    nodeIdList.push_back( nodeId );

    // first send device discovery request to the node
    if (lwLinkSendDiscoverConnOnNodes(nodeIdList, linkTrainIntf)) {
        PRINT_ERROR("", "Failed to initiate device discovery on all nodes");
        return 1;
    }

    // call the helper to query all the discovered connections
    return lwLinkGetIntraNodeConns(nodeIdList, linkTrainIntf, linkConnRepo);
}

int
DcgmGFMHelper::lwLinkGetIntraNodeConnOnNodes(uint32 nodeId,
                                             DcgmFMLWLinkIntf *linkTrainIntf,
                                             DcgmFMLWLinkConnRepo &linkConnRepo)
{
    // this routine will NOT initiate LWLink connection discovery. But query
    // and populate lready discovered connections from the specified node.

    // only one node to query
    NodeIDList nodeIdList;
    nodeIdList.push_back( nodeId );

    // call the helper to query all the discovered connections
    return lwLinkGetIntraNodeConns(nodeIdList, linkTrainIntf, linkConnRepo);
}

int
DcgmGFMHelper::lwLinkDiscoverInterNodeConnections(DcgmFMLWLinkIntf *linkTrainIntf,
                                                  DcgmFabricParser *pConfig,
                                                  DcgmFMLWLinkConnRepo &linkConnRepo)
{

    std::map <NodeKeyType, NodeConfig *>::iterator nodeit;
    int retVal = 0;

    if (pConfig->NodeCfg.size() == 1) {
        // we have only one node. there is no inter node connections
        // to be discovered.
        return 0;
    }

    // this routine will initiate LWLink connection discovery and report
    // all the inter node connections. It is assumed that the links are in SAFE/HS
    // mode prior to initiating discovery.

    // first initiate write discovery on one node and read back discovery token
    // from all other available nodes and correlate connections. Repeat the same
    // process for every available nodes.
    for ( nodeit = pConfig->NodeCfg.begin(); nodeit!= pConfig->NodeCfg.end(); nodeit++ ) {
        NodeConfig *pNode = nodeit->second;
        DcgmLWLinkWriteDiscoveryTokenResp writeTokenResp;
        std::map<uint32, DcgmLWLinkReadDiscoveryTokenResp> readTokenResps;

        retVal = lwLinkWriteDiscoveryToken( pNode->nodeId, linkTrainIntf, writeTokenResp );
        if (retVal) {
            // write discovery token failed for the node, bail out
            return retVal;
        }

        // read discovery token from all node except the current node which wrote the tokens
        lwLinkReadDiscoveryToken( pNode->nodeId, linkTrainIntf, pConfig, readTokenResps );

        // identify the connections by matching the tokens
        lwLinkCorrelateConnections(writeTokenResp, readTokenResps, linkConnRepo);

        // for two nodes, we only need to write from one node and read from another node.
        // no need to repeat write/read discovery token for both nodes.
        if (pConfig->NodeCfg.size() == 2) {
            break;
        }
    }
    return retVal;
}

int
DcgmGFMHelper::lwLinkTrainIntraNodeConnections(DcgmFMLWLinkIntf *linkTrainIntf,
                                               DcgmFMLWLinkConnRepo &linkConnRepo,
                                               DcgmGFMLWLinkDevRepo &linkDevRepo,
                                               DcgmLWLinkTrainType trainTo,
                                               bool inErrHdlr)
{
    std::map<uint64, DcgmFMLWLinkDetailedConnInfo*> requestIds;
    std::map<uint64, DcgmFMLWLinkDetailedConnInfo*>::iterator reqIt;
    LWLinkIntraConnMap::iterator it;
    int retVal = 0;
    LWLinkIntraConnMap intraConnMap = linkConnRepo.getIntraConnections();
    const char* strTrainTo = getLWLinkTrainTypeString( trainTo );

    PRINT_INFO( "%s", "Training all intranode connections to %s", strTrainTo );

    // first send train request for all the connections
    for ( it = intraConnMap.begin(); it != intraConnMap.end(); it++ ) {
        DcgmLWLinkDetailedConnList &connList = it->second;
        DcgmLWLinkDetailedConnList::iterator jit;
        for ( jit = connList.begin(); jit != connList.end(); jit++ ) {
            uint64 requestId = 0;
            DcgmLWLinkReq linkReq = {{0}};
            DcgmFMLWLinkDetailedConnInfo *conn = *jit;
            // fill master end information of the connection
            DcgmLWLinkEndPointInfo masterEnd = conn->getMasterEndPointInfo();
            linkReq.connTrainReq.masterNodeId = masterEnd.nodeId;
            linkReq.connTrainReq.masterGpuOrSwitchId = masterEnd.gpuOrSwitchId;
            linkReq.connTrainReq.masterLinkIndex = masterEnd.linkIndex;
            // fill slave end information of the connection
            DcgmLWLinkEndPointInfo slaveEnd = conn->getSlaveEndPointInfo();            
            linkReq.connTrainReq.slaveNodeId = slaveEnd.nodeId;
            linkReq.connTrainReq.slaveGpuOrSwitchId = slaveEnd.gpuOrSwitchId;
            linkReq.connTrainReq.slaveLinkIndex = slaveEnd.linkIndex;
            // fill the train to state, High speed state
            linkReq.connTrainReq.trainTo = trainTo;
            linkTrainIntf->sendConnTrainReq( linkReq, requestId );
            if (requestId) {
                requestIds.insert( std::make_pair(requestId, conn) );                
            }
        }
    }

    if (inErrHdlr) {
        // no need to wait, as there are errors oclwrred already
        return retVal;
    }

    // wait for all the request to finish. Since requests has timeout, all the requests
    // should eventually finish.
    while (true) {
        for ( reqIt = requestIds.begin(); reqIt != requestIds.end(); ) {
            DcgmLWLinkReqResult reqResult = {0};
            if (linkTrainIntf->isLinkReqComplete( reqIt->first, reqResult )) {
                // update the training/link status information to the connection
                DcgmFMLWLinkDetailedConnInfo *connInfo = reqIt->second;
                if (reqResult.status != FM_LWLINK_ST_SUCCESS) {
                    // mark the final status as failed, but continue with rest of the requests
                    std::stringstream outStr;
                    connInfo->dumpConnInfo( &outStr, linkDevRepo );
                    retVal = reqResult.status;
                    PRINT_ERROR("%s %s %s", "Following intranode connection training to %s failed with error: %s\n %s",
                                strTrainTo, DcgmFMLWLinkError::getLinkErrorString(reqResult.status), outStr.str().c_str());
                    // log connection training failures to syslog
                    FM_SYSLOG_ERR("failed to train LWLink connection: %s", outStr.str().c_str());
                }
                // update the training/link status information to the connection/device repo
                updateDeviceAndConnLinkState( linkDevRepo, connInfo, reqResult.connTrainResp );
                requestIds.erase( reqIt++ );
            } else {
                ++reqIt;
            }
        }
        // one iteration is passed. exit if don't have any more request pending
        if ( requestIds.empty() ) {
            break;
        } else {
            // yield the CPU and try again for the request completion
            sched_yield();
        }
    }

    return retVal;
}

int
DcgmGFMHelper::lwlinkAddInterNodeConnections(DcgmFMLWLinkIntf *linkTrainIntf,
                                             DcgmFMLWLinkConnRepo &linkConnRepo,
                                             DcgmGFMLWLinkDevRepo &linkDevRepo)
{
    std::list<uint64> requestIds;
    int retVal = 0;
    LWLinkInterNodeConns::iterator it;
    LWLinkInterNodeConns interConnList = linkConnRepo.getInterConnections();

    PRINT_INFO("", "Adding internode connections to all available nodes");

    for ( it = interConnList.begin(); it != interConnList.end(); it++ ) {
        uint64 requestId = 0;
        DcgmLWLinkReq linkReq = {{0}};
        DcgmFMLWLinkDetailedConnInfo *tempInterConn = *it;
        DcgmLWLinkEndPointInfo localEndInfo = tempInterConn->getMasterEndPointInfo();
        DcgmLWLinkEndPointInfo remoteEndInfo = tempInterConn->getSlaveEndPointInfo();

        // add to master node first using master node as the local endpoint
        genAddInterNodeConnLinkReqMsg(linkDevRepo, linkReq, localEndInfo, remoteEndInfo);
        linkTrainIntf->sendAddInterNodeConnReq( linkReq, requestId );
        if (requestId) {
            requestIds.push_back( requestId );
        }

        // add to slave node using slave node as the local endpoint
        requestId = 0;
        genAddInterNodeConnLinkReqMsg(linkDevRepo, linkReq, remoteEndInfo, localEndInfo);
        linkTrainIntf->sendAddInterNodeConnReq( linkReq, requestId );
        if (requestId) {
            requestIds.push_back( requestId );
        }
    }

    // wait for ack/all the request to finish
    retVal = waitForLinkRequestToComplete( linkTrainIntf, requestIds, "LWLink add internode connection" );

    return retVal;
}

int
DcgmGFMHelper::lwLinkTrainInterNodeConnections(DcgmFMLWLinkIntf *linkTrainIntf,
                                               DcgmFMLWLinkConnRepo &linkConnRepo,
                                               DcgmGFMLWLinkDevRepo &linkDevRepo,
                                               DcgmLWLinkTrainType trainTo)
{
    std::map<uint64, DcgmFMLWLinkDetailedConnInfo*> requestIds;
    std::map<uint64, DcgmFMLWLinkDetailedConnInfo*>::iterator reqIt;
    LWLinkInterNodeConns::iterator it;
    int retVal = 0;
    LWLinkInterNodeConns interConnList = linkConnRepo.getInterConnections();
    const char* strTrainTo = getLWLinkTrainTypeString( trainTo );

    PRINT_INFO( "%s", "Training all internode connections to %s", strTrainTo );

    // first send train request for all the connections
    for ( it = interConnList.begin(); it != interConnList.end(); it++ ) {
        uint64 requestId = 0;
        DcgmLWLinkReq linkReq = {{0}};
        DcgmFMLWLinkDetailedConnInfo *conn = *it;
        // fill master end information of the connection
        DcgmLWLinkEndPointInfo masterEnd = conn->getMasterEndPointInfo();
        linkReq.connTrainReq.masterNodeId = masterEnd.nodeId;
        linkReq.connTrainReq.masterGpuOrSwitchId = masterEnd.gpuOrSwitchId;
        linkReq.connTrainReq.masterLinkIndex = masterEnd.linkIndex;
        // fill slave end information of the connection
        DcgmLWLinkEndPointInfo slaveEnd = conn->getSlaveEndPointInfo();            
        linkReq.connTrainReq.slaveNodeId = slaveEnd.nodeId;
        linkReq.connTrainReq.slaveGpuOrSwitchId = slaveEnd.gpuOrSwitchId;
        linkReq.connTrainReq.slaveLinkIndex = slaveEnd.linkIndex;
        // fill the train to state, High speed state
        linkReq.connTrainReq.trainTo = trainTo;
        linkTrainIntf->sendConnTrainReq( linkReq, requestId );
        if (requestId) {
            requestIds.insert( std::make_pair(requestId, conn) );                
        }
    }

    // wait for all the request to finish. Since requests has timeout, all the requests
    // should eventually finish.
    while (true) {
        for ( reqIt = requestIds.begin(); reqIt != requestIds.end(); ) {
            DcgmLWLinkReqResult reqResult = {0};
            if (linkTrainIntf->isLinkReqComplete( reqIt->first, reqResult )) {
                DcgmFMLWLinkDetailedConnInfo *connInfo = reqIt->second;
                if (reqResult.status != FM_LWLINK_ST_SUCCESS) {
                    // mark the final status as failed, but continue with rest of the requests
                    std::stringstream outStr;
                    connInfo->dumpConnInfo( &outStr, linkDevRepo );
                    retVal = reqResult.status;
                    PRINT_ERROR("%s %s %s", "Following internode connection training to %s failed with error: %s\n %s",
                                strTrainTo, DcgmFMLWLinkError::getLinkErrorString(reqResult.status), outStr.str().c_str());
                    // log connection training failures to syslog
                    FM_SYSLOG_ERR("failed to train LWLink connection: %s", outStr.str().c_str());
                }
                // update the training/link status information to the connection/device repo
                updateDeviceAndConnLinkState( linkDevRepo, connInfo, reqResult.connTrainResp );
                requestIds.erase( reqIt++ );
            } else {
                ++reqIt;
            }
        }
        // one iteration is passed. exit if don't have any more request pending
        if ( requestIds.empty() ) {
            break;
        } else {
            // yield the CPU and try again for the request completion
            sched_yield();
        }
    }

    return retVal;
}

int
DcgmGFMHelper::trainLWLinkConnection(DcgmFMLWLinkIntf *linkTrainIntf,
                                     DcgmFMLWLinkConnRepo &linkConnRepo,
                                     DcgmGFMLWLinkDevRepo &linkDevRepo,
                                     DcgmFMLWLinkDetailedConnInfo *connInfo,
                                     DcgmLWLinkTrainType trainTo)
{
    uint64 requestId = 0;
    int retVal = 0;
    DcgmLWLinkReq linkReq = {{0}};

    // first send train request for the connection

    // fill master end information of the connection
    DcgmLWLinkEndPointInfo masterEnd = connInfo->getMasterEndPointInfo();
    linkReq.connTrainReq.masterNodeId = masterEnd.nodeId;
    linkReq.connTrainReq.masterGpuOrSwitchId = masterEnd.gpuOrSwitchId;
    linkReq.connTrainReq.masterLinkIndex = masterEnd.linkIndex;
    // fill slave end information of the connection
    DcgmLWLinkEndPointInfo slaveEnd = connInfo->getSlaveEndPointInfo();            
    linkReq.connTrainReq.slaveNodeId = slaveEnd.nodeId;
    linkReq.connTrainReq.slaveGpuOrSwitchId = slaveEnd.gpuOrSwitchId;
    linkReq.connTrainReq.slaveLinkIndex = slaveEnd.linkIndex;
    // fill the train to state, High speed state
    linkReq.connTrainReq.trainTo = trainTo;
    linkTrainIntf->sendConnTrainReq( linkReq, requestId );
    if (requestId == 0) {
        // failed to send the request
        return 1;
    }

    // wait for the request to finish. Since requests has timeout, the request
    // should eventually finish.
    while (true) {
        DcgmLWLinkReqResult reqResult = {0};
        if (linkTrainIntf->isLinkReqComplete( requestId, reqResult )) {
            if (reqResult.status != FM_LWLINK_ST_SUCCESS) {
                // mark the final status as failed.
                std::stringstream outStr;
                connInfo->dumpConnInfo( &outStr, linkDevRepo );
                retVal = reqResult.status;
                PRINT_ERROR("%s %s", "Following LWLink connection training failed with error: %s\n %s",
                            DcgmFMLWLinkError::getLinkErrorString(reqResult.status), outStr.str().c_str()) ;
                // log connection training failures to syslog
                FM_SYSLOG_ERR("failed to train LWLink connection: %s", outStr.str().c_str());
            }
            // update the training/link status information to the connection/device repo
            updateDeviceAndConnLinkState( linkDevRepo, connInfo, reqResult.connTrainResp );
            break;
        }
        // one iteration is passed. yield the CPU and try again for the request completion
        sched_yield();
    }

    return 0;
}

int
DcgmGFMHelper::lwLinkSendResetSwitchLinks(uint32 nodeId,
                                          uint64 switchPhysicalId,
                                          uint64 linkMask,
                                          DcgmFMLWLinkIntf *linkTrainIntf,
                                          bool inErrHdlr)
{
    DcgmLWLinkReq linkReq = {{0}};
    DcgmLWLinkReqResult reqResult = {0};
    uint64 requestId = 0;
    int retVal;

    linkReq.nodeInitResetSwitchLinksReq.nodeId = nodeId;
    linkReq.nodeInitResetSwitchLinksReq.switchId = switchPhysicalId;
    linkReq.nodeInitResetSwitchLinksReq.linkMask = linkMask;
    retVal = linkTrainIntf->sendResetSwitchLinksReq( linkReq, requestId );
    if ( !requestId ) {
        // unable to send reset link request
        return retVal;
    }

    if (inErrHdlr) {
        // no need to wait in the context of error handler
        return retVal;
    }

    // wait for the request to finish
    while (true) {
        if (linkTrainIntf->isLinkReqComplete( requestId, reqResult )) {
            if (reqResult.status != FM_LWLINK_ST_SUCCESS) {
                PRINT_ERROR("%d %s", "LWLink reset switch link request failed for node %d with error: %s",
                            nodeId, DcgmFMLWLinkError::getLinkErrorString(reqResult.status));
                return reqResult.status;
            }
            break;
        }
        // yield the CPU and try again for the request completion
        sched_yield();
    }

    return 0;
}

int
DcgmGFMHelper::lwLinkResetAllSwitchLinks(uint32 nodeId, DcgmFMLWLinkIntf *linkTrainIntf)
{
    DcgmLWLinkReq linkReq = {{0}};
    DcgmLWLinkReqResult reqResult = {0};
    uint64 requestId = 0;
    int retVal;

    linkReq.nodeInitResetAllSwitchLinksReq.nodeId = nodeId;
    retVal = linkTrainIntf->sendResetAllSwitchLinksReq( linkReq, requestId );
    if ( !requestId ) {
        // unable to send reset link request
        return retVal;
    }

    // wait for the request to finish
    while (true) {
        if (linkTrainIntf->isLinkReqComplete( requestId, reqResult )) {
            if (reqResult.status != FM_LWLINK_ST_SUCCESS) {
                PRINT_ERROR("%d %s", "Reset all LWSwitch links failed for node %d with error: %s",
                            nodeId, DcgmFMLWLinkError::getLinkErrorString(reqResult.status));
                return reqResult.status;
            }
            break;
        }
        // yield the CPU and try again for the request completion
        sched_yield();
    }

    return 0;
}

int
DcgmGFMHelper::getLWLinkDeviceInfoFromNode(uint32 nodeId,
                                           GFMDevInfoMsgHdlr *devInfoMsgHdlr,
                                           DcgmGFMLWLinkDevRepo &linkDevRepo)
{
    uint64 requestId = 0;
    int retVal = 0;

    retVal = devInfoMsgHdlr->sendLWLinkDevInfoReq( nodeId, requestId );
    if ( !requestId ) {
        // unable to send gpu info request
        return retVal;
    }

    // wait for the request to finish
    while (true) {
        DevInfoReqResult reqResult = {0};
        if (devInfoMsgHdlr->isDevInfoReqComplete( requestId, reqResult )) {
            if (reqResult.status != FM_LWLINK_ST_SUCCESS) {
                // mark the final status as failed, but continue with rest of the requests
                retVal = reqResult.status;
                PRINT_ERROR("%d", "Failed to get LWLink Device info from node %d", nodeId);
                return retVal;
            } else {
                // request finished. copy the device information.
                lwswitch::deviceInfoResponseMsg devInfoRsp = reqResult.devInfoRspMsg.devinforsp();
                lwswitch::lwlinkDeviceInfoRsp lwlinkDevInfoRsp = devInfoRsp.lwlinkdevrsp();
                linkDevRepo.addDeviceInfo( nodeId, lwlinkDevInfoRsp );
                break;
            }
        }
        // yield the CPU and try again for the request completion
        sched_yield();
    }

    return retVal;

}

int
DcgmGFMHelper::getLWLinkDeviceInfoFromAllNodes(DcgmFabricParser *pConfig,
                                               GFMDevInfoMsgHdlr *devInfoMsgHdlr,
                                               DcgmGFMLWLinkDevRepo &linkDevRepo)
{
    std::map<uint32, uint64> requestIds;
    std::map<uint32, uint64>::iterator it;
    std::map <NodeKeyType, NodeConfig *>::iterator nodeit;
    int retVal = 0;

    PRINT_INFO("", "Getting LWLink Device Information from all nodes");

    for ( nodeit = pConfig->NodeCfg.begin(); nodeit!= pConfig->NodeCfg.end(); nodeit++ ) {
        NodeConfig *pNode = nodeit->second;
        uint64 requestId = 0;
        devInfoMsgHdlr->sendLWLinkDevInfoReq( pNode->nodeId, requestId );
        if (requestId) {
            requestIds.insert( std::make_pair(pNode->nodeId, requestId) );
        }
    }

    // wait for all the request to finish. Since requests has timeout, all the requests
    // should eventually finish.
    while (true) {
        for ( it = requestIds.begin(); it != requestIds.end(); ) {
            DevInfoReqResult reqResult = {0};
            if (devInfoMsgHdlr->isDevInfoReqComplete( it->second, reqResult )) {
                if (reqResult.status != FM_LWLINK_ST_SUCCESS) {
                    // mark the final status as failed, but continue with rest of the requests
                    retVal = reqResult.status;
                    PRINT_ERROR("%d", "Failed to get LWLink Device info from node %d", it->first);
                } else {
                    // one request is finished. copy the device information.
                    lwswitch::deviceInfoResponseMsg devInfoRsp = reqResult.devInfoRspMsg.devinforsp();
                    lwswitch::lwlinkDeviceInfoRsp lwlinkDevInfoRsp = devInfoRsp.lwlinkdevrsp();
                    linkDevRepo.addDeviceInfo( it->first, lwlinkDevInfoRsp );
                }
                requestIds.erase( it++ );
            } else {
                ++it;
            }
        }
        // one iteration is passed. exit if don't have any more request pending
        if ( requestIds.empty() ) {
            break;
        } else {
            // yield the CPU and try again for the request completion
            sched_yield();
        }
    }

    return retVal;
}                                       

int
DcgmGFMHelper::getLWSwitchDeviceInfoFromAllNodes(DcgmFabricParser *pConfig,
                                                 GFMDevInfoMsgHdlr *devInfoMsgHdlr,
                                                 DcgmFMLWSwitchInfoMap &lwswitchInfoMap)
{
    std::map<uint32, uint64> requestIds;
    std::map<uint32, uint64>::iterator it;
    std::map <NodeKeyType, NodeConfig *>::iterator nodeit;
    int retVal = 0;

    PRINT_INFO("", "Getting LWSwitch Device Information from all nodes");

    // then send get connection request to all the nodes.
    for ( nodeit = pConfig->NodeCfg.begin(); nodeit!= pConfig->NodeCfg.end(); nodeit++ ) {
        NodeConfig *pNode = nodeit->second;
        uint64 requestId = 0;
        devInfoMsgHdlr->sendLWSwitchDevInfoReq( pNode->nodeId, requestId );
        if (requestId) {
            requestIds.insert( std::make_pair(pNode->nodeId, requestId) );
        }
    }

    // wait for all the request to finish. Since requests has timeout, all the requests
    // should eventually finish.
    while (true) {
        for ( it = requestIds.begin(); it != requestIds.end(); ) {
            DevInfoReqResult reqResult = {0};
            if (devInfoMsgHdlr->isDevInfoReqComplete( it->second, reqResult )) {
                if (reqResult.status != FM_LWLINK_ST_SUCCESS) {
                    // mark the final status as failed, but continue with rest of the requests
                    retVal = reqResult.status;
                    PRINT_ERROR("%d", "Failed to get LWSwitch Device info from node %d", it->first);
                } else {
                    // one request is finished. copy the device information.
                    copyLWSwitchDeviceInfo(reqResult, lwswitchInfoMap);
                }
                requestIds.erase( it++ );
            } else {
                ++it;
            }
        }
        // one iteration is passed. exit if don't have any more request pending
        if ( requestIds.empty() ) {
            break;
        } else {
            // yield the CPU and try again for the request completion
            sched_yield();
        }
    }

    return retVal;
}                                       

int
DcgmGFMHelper::getGpuDeviceInfoFromNode(uint32 nodeId,
                                        GFMDevInfoMsgHdlr *devInfoMsgHdlr,
                                        DcgmFMGpuInfoMap &gpuInfoMap,
                                        DcgmFMGpuInfoMap &blacklistGpuInfoMap)
{
    uint64 requestId = 0;
    int retVal = 0;

    retVal = devInfoMsgHdlr->sendGpuDevInfoReq( nodeId, requestId );
    if ( !requestId ) {
        // unable to send gpu info request
        return retVal;
    }

    // wait for the request to finish
    while (true) {
        DevInfoReqResult reqResult = {0};
        if (devInfoMsgHdlr->isDevInfoReqComplete( requestId, reqResult )) {
            if (reqResult.status != FM_LWLINK_ST_SUCCESS) {
                // mark the final status as failed, but continue with rest of the requests
                retVal = reqResult.status;
                PRINT_ERROR("%d", "Failed to get GPU info from node %d", nodeId);
                return retVal;
            } else {
                // request finished. copy the device information.
                copyGpuDeviceInfo(reqResult, gpuInfoMap, blacklistGpuInfoMap);
                break;
            }
        }
        // yield the CPU and try again for the request completion
        sched_yield();
    }

    return retVal;
}

int
DcgmGFMHelper::getGpuDeviceInfoFromAllNodes(DcgmFabricParser *pConfig,
                                            GFMDevInfoMsgHdlr *devInfoMsgHdlr,
                                            DcgmFMGpuInfoMap &gpuInfoMap,
                                            DcgmFMGpuInfoMap &blacklistGpuInfoMap)
{
    std::map<uint32, uint64> requestIds;
    std::map<uint32, uint64>::iterator it;
    std::map <NodeKeyType, NodeConfig *>::iterator nodeit;
    int retVal = 0;

    PRINT_INFO("", "Getting GPU Device Information from all nodes");

    // then send get connection request to all the nodes.
    for ( nodeit = pConfig->NodeCfg.begin(); nodeit!= pConfig->NodeCfg.end(); nodeit++ ) {
        NodeConfig *pNode = nodeit->second;
        uint64 requestId = 0;
        devInfoMsgHdlr->sendGpuDevInfoReq( pNode->nodeId, requestId );
        if (requestId) {
            requestIds.insert( std::make_pair(pNode->nodeId, requestId) );
        }
    }

    // wait for all the request to finish. Since requests has timeout, all the requests
    // should eventually finish.
    while (true) {
        for ( it = requestIds.begin(); it != requestIds.end(); ) {
            DevInfoReqResult reqResult = {0};
            if (devInfoMsgHdlr->isDevInfoReqComplete( it->second, reqResult )) {
                if (reqResult.status != FM_LWLINK_ST_SUCCESS) {
                    // mark the final status as failed, but continue with rest of the requests
                    retVal = reqResult.status;
                    PRINT_ERROR("%d", "Failed to get GPU info from node %d", it->first);
                } else {
                    // one request is finished. copy the device information.
                    copyGpuDeviceInfo(reqResult, gpuInfoMap, blacklistGpuInfoMap);
                }
                requestIds.erase( it++ );
            } else {
                ++it;
            }
        }
        // one iteration is passed. exit if don't have any more request pending
        if ( requestIds.empty() ) {
            break;
        } else {
            // yield the CPU and try again for the request completion
            sched_yield();
        }
    }

    return retVal;
}                                       

int
DcgmGFMHelper::lwLinkInitializeNodes(NodeIDList &nodeIdList,
                                     DcgmFMLWLinkIntf *linkTrainIntf,
                                     DcgmGFMLWLinkDevRepo &linkDevRepo)
{
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
    if (lwLinkInitphase1ForNodes(nodeIdList, linkTrainIntf)) {
        PRINT_ERROR("", "Failed to Initphase1 for nodes");
        return 1;
    }

    if (lwLinkRxInitTermForNodes(nodeIdList, linkTrainIntf)) {
        PRINT_ERROR("", "Failed to RxInitTerm for nodes");
        return 1;
    }

    if (lwLinkSetRxDetectForNodes(nodeIdList, linkTrainIntf)) {
        PRINT_ERROR("", "Failed to SetRxDetect for nodes");
        return 1;
    }

    if (lwLinkGetRxDetectForNodes(nodeIdList, linkTrainIntf)) {
        PRINT_ERROR("", "Failed to GetRxDetect for nodes");
        return 1;
    }
#endif

    if (lwLinkEnableCommonModeForNodes(nodeIdList, linkTrainIntf)) {
        PRINT_ERROR("", "Failed to enable common mode for nodes");
        return 1;
    }

    if (lwLinkCalibrateNodes(nodeIdList, linkTrainIntf)) {
        PRINT_ERROR("", "Failed to calibrate nodes");
        return 1;        
    }

    if (lwLinkDisableCommonModeForNodes(nodeIdList, linkTrainIntf)) {
        PRINT_ERROR("", "Failed to disable common mode for nodes");
        return 1;        
    }

    if (lwLinkEnableDataForNodes(nodeIdList, linkTrainIntf)) {
        PRINT_ERROR("", "Failed to do enable data for nodes");
        return 1;
    }



    if (lwLinkInitLinkForNodes(nodeIdList, linkTrainIntf)) {
        PRINT_ERROR("", "Failed to start link initialize for nodes");
        return 1;
    }

    if (lwLinkGetLinkInitStatusForNodes(nodeIdList, linkTrainIntf, linkDevRepo)) {
        PRINT_ERROR("", "Failed to get link initialization status for nodes");
        return 1;
    }

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
    if (lwLinkInitnegotiateForNodes(nodeIdList, linkTrainIntf)) {
        PRINT_ERROR("", "Failed to Initnegotiate for nodes");
        return 1;
    }
#endif
    return 0;
}

int
DcgmGFMHelper::lwLinkEnableCommonModeForNodes(NodeIDList &nodeIdList,
                                              DcgmFMLWLinkIntf *linkTrainIntf)
{
    std::list<uint64> requestIds;
    NodeIDList::iterator nodeIt;
    int retVal = 0;

    PRINT_INFO("", "Enabling Common Mode for nodes");

    // first send request to all the active nodes
    for ( nodeIt = nodeIdList.begin(); nodeIt != nodeIdList.end(); nodeIt++ ) {
        uint32_t nodeId = (*nodeIt);
        uint64 requestId = 0;
        DcgmLWLinkReq linkReq = {{0}};
        linkReq.nodeInitReq.nodeId = nodeId;
        linkTrainIntf->sendEnableCommonModeReq( linkReq, requestId );
        if (requestId) {
            requestIds.push_back( requestId );
        }
    }

    retVal = waitForLinkRequestToComplete( linkTrainIntf, requestIds, "LWLink enable common mode" );
    return retVal;
}

int
DcgmGFMHelper::lwLinkCalibrateNodes(NodeIDList &nodeIdList,
                                    DcgmFMLWLinkIntf *linkTrainIntf)
{
    std::list<uint64> requestIds;
    NodeIDList::iterator nodeIt;
    int retVal = 0;    

    PRINT_INFO("", "Calibrating nodes");

    // first send request to all the active nodes
    for ( nodeIt = nodeIdList.begin(); nodeIt != nodeIdList.end(); nodeIt++ ) {
        uint32_t nodeId = (*nodeIt);
        uint64 requestId = 0;
        DcgmLWLinkReq linkReq = {{0}};
        linkReq.nodeInitReq.nodeId = nodeId;
        linkTrainIntf->sendCalibrateReq( linkReq, requestId );
        if (requestId) {
            requestIds.push_back( requestId );
        }
    }

    retVal = waitForLinkRequestToComplete( linkTrainIntf, requestIds, "LWLink calibration" );
    return retVal;
}

int
DcgmGFMHelper::lwLinkDisableCommonModeForNodes(NodeIDList &nodeIdList,
                                               DcgmFMLWLinkIntf *linkTrainIntf)
{
    std::list<uint64> requestIds;
    NodeIDList::iterator nodeIt;
    int retVal = 0;    

    PRINT_INFO("", "Disabling Common Mode for nodes");

    // first send request to all the active nodes
    for ( nodeIt = nodeIdList.begin(); nodeIt != nodeIdList.end(); nodeIt++ ) {
        uint32_t nodeId = (*nodeIt);
        uint64 requestId = 0;
        DcgmLWLinkReq linkReq = {{0}};
        linkReq.nodeInitReq.nodeId = nodeId;
        linkTrainIntf->sendDisableCommonModeReq( linkReq, requestId );
        if (requestId) {
            requestIds.push_back( requestId );
        }
    }

    retVal = waitForLinkRequestToComplete( linkTrainIntf, requestIds, "LWLink disable common mode" );
    return retVal;
}

int
DcgmGFMHelper::lwLinkEnableDataForNodes(NodeIDList &nodeIdList,
                                        DcgmFMLWLinkIntf *linkTrainIntf)
{
    std::list<uint64> requestIds;
    NodeIDList::iterator nodeIt;
    int retVal = 0;    

    PRINT_INFO("", "Enabling Data for nodes");

    // first send request to all the active nodes
    for ( nodeIt = nodeIdList.begin(); nodeIt != nodeIdList.end(); nodeIt++ ) {
        uint32_t nodeId = (*nodeIt);
        uint64 requestId = 0;
        DcgmLWLinkReq linkReq = {{0}};
        linkReq.nodeInitReq.nodeId = nodeId;
        linkTrainIntf->sendEnableDataReq( linkReq, requestId );
        if (requestId) {
            requestIds.push_back( requestId );
        }
    }

    retVal = waitForLinkRequestToComplete( linkTrainIntf, requestIds, "LWLink enable data" );
    return retVal;
}


#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
int
DcgmGFMHelper::lwLinkInitphase1ForNodes(NodeIDList &nodeIdList,
                                        DcgmFMLWLinkIntf *linkTrainIntf)
{
    std::list<uint64> requestIds;
    NodeIDList::iterator nodeIt;
    int retVal = 0;    

    PRINT_INFO("", "Initphase1 for nodes");

    // first send request to all the active nodes
    for ( nodeIt = nodeIdList.begin(); nodeIt != nodeIdList.end(); nodeIt++ ) {
        uint32_t nodeId = (*nodeIt);
        uint64 requestId = 0;
        DcgmLWLinkReq linkReq = {{0}};
        linkReq.nodeInitReq.nodeId = nodeId;
        linkTrainIntf->sendInitphase1Req( linkReq, requestId );
        if (requestId) {
            requestIds.push_back( requestId );
        }
    }

    retVal = waitForLinkRequestToComplete( linkTrainIntf, requestIds, "LWLink Initphase1" );
    return retVal;
}


int
DcgmGFMHelper::lwLinkRxInitTermForNodes(NodeIDList &nodeIdList,
                                        DcgmFMLWLinkIntf *linkTrainIntf)
{
    std::list<uint64> requestIds;
    NodeIDList::iterator nodeIt;
    int retVal = 0;    

    PRINT_INFO("", "Rx Init Term for nodes");

    // first send request to all the active nodes
    for ( nodeIt = nodeIdList.begin(); nodeIt != nodeIdList.end(); nodeIt++ ) {
        uint32_t nodeId = (*nodeIt);
        uint64 requestId = 0;
        DcgmLWLinkReq linkReq = {{0}};
        linkReq.nodeInitReq.nodeId = nodeId;
        linkTrainIntf->sendRxInitTermReq( linkReq, requestId );
        if (requestId) {
            requestIds.push_back( requestId );
        }
    }

    retVal = waitForLinkRequestToComplete( linkTrainIntf, requestIds, "LWLink Rx Init Term" );
    return retVal;
}


int
DcgmGFMHelper::lwLinkSetRxDetectForNodes(NodeIDList &nodeIdList,
                                         DcgmFMLWLinkIntf *linkTrainIntf)
{
    std::list<uint64> requestIds;
    NodeIDList::iterator nodeIt;
    int retVal = 0;    

    PRINT_INFO("", "Set RX detect for nodes");

    // first send request to all the active nodes
    for ( nodeIt = nodeIdList.begin(); nodeIt != nodeIdList.end(); nodeIt++ ) {
        uint32_t nodeId = (*nodeIt);
        uint64 requestId = 0;
        DcgmLWLinkReq linkReq = {{0}};
        linkReq.nodeInitReq.nodeId = nodeId;
        linkTrainIntf->sendSetRxDetectReq( linkReq, requestId );
        if (requestId) {
            requestIds.push_back( requestId );
        }
    }

    retVal = waitForLinkRequestToComplete( linkTrainIntf, requestIds, "LWLink Set RX detect" );
    return retVal;
}


int
DcgmGFMHelper::lwLinkGetRxDetectForNodes(NodeIDList &nodeIdList,
                                         DcgmFMLWLinkIntf *linkTrainIntf)
{
    std::list<uint64> requestIds;
    NodeIDList::iterator nodeIt;
    int retVal = 0;

    PRINT_INFO("", "Get RX detect for nodes");

    // first send request to all the active nodes
    for ( nodeIt = nodeIdList.begin(); nodeIt != nodeIdList.end(); nodeIt++ ) {
        uint32_t nodeId = (*nodeIt);
        uint64 requestId = 0;
        DcgmLWLinkReq linkReq = {{0}};
        linkReq.nodeInitReq.nodeId = nodeId;
        linkTrainIntf->sendGetRxDetectReq( linkReq, requestId );
        if (requestId) {
            requestIds.push_back( requestId );
        }
    }

    retVal = waitForLinkRequestToComplete( linkTrainIntf, requestIds, "LWLink Get RX detect" );
    return retVal;
}


int
DcgmGFMHelper::lwLinkInitnegotiateForNodes(NodeIDList &nodeIdList,
                                           DcgmFMLWLinkIntf *linkTrainIntf)
{
    std::list<uint64> requestIds;
    NodeIDList::iterator nodeIt;
    int retVal = 0;    

    PRINT_INFO("", "Initnegotiate nodes");

    // first send request to all the active nodes
    for ( nodeIt = nodeIdList.begin(); nodeIt != nodeIdList.end(); nodeIt++ ) {
        uint32_t nodeId = (*nodeIt);
        uint64 requestId = 0;
        DcgmLWLinkReq linkReq = {{0}};
        linkReq.nodeInitReq.nodeId = nodeId;
        linkTrainIntf->sendInitnegotiateReq( linkReq, requestId );
        if (requestId) {
            requestIds.push_back( requestId );
        }
    }

    retVal = waitForLinkRequestToComplete( linkTrainIntf, requestIds, "LWLink Initnegotiate" );
    return retVal;
}
#endif

int
DcgmGFMHelper::lwLinkInitLinkForNodes(NodeIDList &nodeIdList,
                                      DcgmFMLWLinkIntf *linkTrainIntf)
{
    std::list<uint64> requestIds;
    NodeIDList::iterator nodeIt;
    int retVal = 0;    

    PRINT_INFO("", "Starting link initialization on nodes");

    // first send request to all the active nodes
    for ( nodeIt = nodeIdList.begin(); nodeIt != nodeIdList.end(); nodeIt++ ) {
        uint32_t nodeId = (*nodeIt);
        uint64 requestId = 0;
        DcgmLWLinkReq linkReq = {{0}};
        linkReq.nodeInitReq.nodeId = nodeId;
        linkTrainIntf->sendLinkInitReq( linkReq, requestId );
        if (requestId) {
            requestIds.push_back( requestId );
        }
    }

    retVal = waitForLinkRequestToComplete( linkTrainIntf, requestIds, "LWLink link initialization" );
    return retVal;
}

int
DcgmGFMHelper::lwLinkGetLinkInitStatusForNodes(NodeIDList &nodeIdList,
                                               DcgmFMLWLinkIntf *linkTrainIntf,
                                               DcgmGFMLWLinkDevRepo &linkDevRepo)
{
    std::map<uint32, uint64> requestIds;
    std::map<uint32, uint64>::iterator it;
    NodeIDList::iterator nodeIt;

    PRINT_INFO("", "Checking link initialization status of nodes");

    // then send get connection request to all the nodes.
    for ( nodeIt = nodeIdList.begin(); nodeIt != nodeIdList.end(); nodeIt++ ) {
        uint32_t nodeId = (*nodeIt);
        uint64 requestId = 0;
        DcgmLWLinkReq linkReq = {{0}};
        linkReq.nodeInitStatusReq.nodeId = nodeId;
        linkTrainIntf->sendLinkInitStatusReq( linkReq, requestId );
        if (requestId) {
            requestIds.insert( std::make_pair(nodeId, requestId) );
        }
    }

    // wait for all the request to finish. Since requests has timeout, all the requests
    // should eventually finish.
    while (true) {
        for ( it = requestIds.begin(); it != requestIds.end(); ) {
            DcgmLWLinkReqResult reqResult = {0};
            if (linkTrainIntf->isLinkReqComplete( it->second, reqResult )) {
                if (reqResult.status == FM_LWLINK_ST_SUCCESS) {
                    // one request is finished. update our device repo based on that
                    linkDevRepo.setDeviceLinkInitStatus(it->first, reqResult.nodeInitStatusResp.statusInfo );
                }
                requestIds.erase( it++ );
            } else {
                ++it;
            }
        }
        // one iteration is passed. exit if don't have any more request pending
        if ( requestIds.empty() ) {
            break;
        } else {
            // yield the CPU and try again for the request completion
            sched_yield();
        }
    }

    return 0;
}

int
DcgmGFMHelper::lwLinkGetIntraNodeConns(NodeIDList &nodeIdList,
                                       DcgmFMLWLinkIntf *linkTrainIntf,
                                       DcgmFMLWLinkConnRepo &linkConnRepo)
{
    std::map<uint32, uint64> requestIds;
    std::map<uint32, uint64>::iterator it;
    NodeIDList::iterator nodeIt;

    // send get connection request to all the nodes.
    for ( nodeIt = nodeIdList.begin(); nodeIt != nodeIdList.end(); nodeIt++ ) {
        uint32_t nodeId = (*nodeIt);
        uint64 requestId = 0;
        DcgmLWLinkReq linkReq = {{0}};
        linkReq.getIntraNodeConnReq.nodeId = nodeId;
        linkTrainIntf->sendGetIntraNodeConnReq( linkReq, requestId );
        if (requestId) {
            requestIds.insert( std::make_pair(nodeId, requestId) );
        }
    }

    // wait for all the request to finish. Since requests has timeout, all the requests
    // should eventually finish.
    while (true) {
        for ( it = requestIds.begin(); it != requestIds.end(); ) {
            DcgmLWLinkReqResult reqResult = {0};
            if (linkTrainIntf->isLinkReqComplete( it->second, reqResult )) {
                // one request is finished. copy the connections to caller's map.
                if (reqResult.status != FM_LWLINK_ST_SUCCESS) {
                    PRINT_ERROR("%d %s", "Intranode connection discovery failed for node id %d with error: %s",
                                it->first, DcgmFMLWLinkError::getLinkErrorString(reqResult.status));
                } else {
                    // the request finished successfully, copy the connection information
                    linkConnRepo.addIntraConnections( it->first, reqResult.getIntraNodeConnResp.connInfo );
                }
                // remove the request from our tracking context anyway.
                requestIds.erase( it++ );
            } else {
                ++it;
            }
        }
        // one iteration is passed. exit if don't have any more request pending
        if ( requestIds.empty() ) {
            break;
        } else {
            // yield the CPU and try again for the request completion
            sched_yield();
        }
    }

    return 0;
}

int
DcgmGFMHelper::lwLinkSendDiscoverConnOnNodes(NodeIDList &nodeIdList,
                                             DcgmFMLWLinkIntf *linkTrainIntf)
{
    std::list<uint64> requestIds;
    NodeIDList::iterator nodeIt;
    int retVal = 0;

    // this routine will initiate LWLink connection discovery

    // first send request to all the active nodes
    for ( nodeIt = nodeIdList.begin(); nodeIt != nodeIdList.end(); nodeIt++ ) {
        uint32_t nodeId = (*nodeIt);
        uint64 requestId = 0;
        DcgmLWLinkReq linkReq = {{0}};
        linkReq.discoverIntraNodeConnReq.nodeId = nodeId;
        linkTrainIntf->sendDiscoverIntraNodeConnReq( linkReq, requestId );
        if (requestId) {
            requestIds.push_back( requestId );
        }
    }

    retVal = waitForLinkRequestToComplete( linkTrainIntf, requestIds, "LWLink discover connection" );
    return retVal;
}

int
DcgmGFMHelper::lwLinkWriteDiscoveryToken(uint32 nodeId,
                                         DcgmFMLWLinkIntf *linkTrainIntf,
                                         DcgmLWLinkWriteDiscoveryTokenResp &writeTokenResp)
{
    DcgmLWLinkReq linkReq = {{0}};
    DcgmLWLinkReqResult reqResult = {0};
    uint64 requestId = 0;
    int retVal;

    linkReq.writeDiscoveryTokenReq.nodeId = nodeId;
    retVal = linkTrainIntf->sendWriteDiscoveryReq( linkReq, requestId );
    if ( !requestId ) {
        // unable to send discovery token
        return retVal;
    }

    // wait for the request to finish
    while (true) {
        if (linkTrainIntf->isLinkReqComplete( requestId, reqResult )) {
            if (reqResult.status != FM_LWLINK_ST_SUCCESS) {
                PRINT_ERROR("%d %s", "LWLink write discovery token failed for node %d with error: %s",
                            nodeId, DcgmFMLWLinkError::getLinkErrorString(reqResult.status));
                return reqResult.status;
            }
            // copy the result and token information
            writeTokenResp = reqResult.writeDiscoveryTokenResp;
            break;
        }
        // yield the CPU and try again for the request completion
        sched_yield();
    }
    return 0;
}

int
DcgmGFMHelper::lwLinkReadDiscoveryToken(uint32 lwrWriteTokenNodeId,
                                        DcgmFMLWLinkIntf *linkTrainIntf,
                                        DcgmFabricParser *pConfig,
                                        std::map<uint32, DcgmLWLinkReadDiscoveryTokenResp> &readTokenResps)
{
    std::map<uint32, uint64> requestIds;
    std::map<uint32, uint64>::iterator it;
    std::map <NodeKeyType, NodeConfig *>::iterator nodeit;

    // first send request to all the active nodes
    for ( nodeit = pConfig->NodeCfg.begin(); nodeit!= pConfig->NodeCfg.end(); nodeit++ ) {
        NodeConfig *pNode = nodeit->second;
        if (pNode->nodeId == lwrWriteTokenNodeId) {
            // skip the node which wrote discovery token
            continue;
        }
        uint64 requestId = 0;
        DcgmLWLinkReq linkReq = {{0}};
        linkReq.readDiscoveryTokenReq.nodeId = pNode->nodeId;
        linkTrainIntf->sendReadDiscoveryReq( linkReq, requestId );
        if (requestId) {
            requestIds.insert( std::make_pair(pNode->nodeId, requestId) );
        }
    }

    // wait for all the request to finish. Since requests has timeout, all the requests
    // should eventually finish.
    while (true) {
        for ( it = requestIds.begin(); it != requestIds.end(); ) {
            DcgmLWLinkReqResult reqResult = {0};
            if (linkTrainIntf->isLinkReqComplete( it->second, reqResult )) {
                // one request is finished. copy the connections to caller's map.
                if (reqResult.status != FM_LWLINK_ST_SUCCESS) {
                    PRINT_ERROR("%d %s", "LWLink read discovery token failed for node %d with error: %s",
                                it->first, DcgmFMLWLinkError::getLinkErrorString(reqResult.status));
                } else {
                    readTokenResps.insert( std::make_pair(it->first, reqResult.readDiscoveryTokenResp) );
                }
                requestIds.erase( it++ );
            } else {
                ++it;
            }
        }
        // one iteration is passed. exit if don't have any more request pending
        if ( requestIds.empty() ) {
            break;
        } else {
            // yield the CPU and try again for the request completion
            sched_yield();
        }
    }
    return 0;
}

int
DcgmGFMHelper::lwLinkCorrelateConnections(DcgmLWLinkWriteDiscoveryTokenResp &writeTokenResp,
                                          std::map<uint32, DcgmLWLinkReadDiscoveryTokenResp> &readTokenResps,
                                          DcgmFMLWLinkConnRepo &linkConnRepo)
{
    // compare each discovery token wrote with the tokens read from other nodes

    DcgmLWLinkDiscoveryTokenList readTokenList;
    // create a list of read tokens for easy parsing
    std::map<uint32, DcgmLWLinkReadDiscoveryTokenResp>::iterator it;
    for ( it = readTokenResps.begin(); it != readTokenResps.end(); it++ ) {
        DcgmLWLinkReadDiscoveryTokenResp readResp = it->second;
        DcgmLWLinkDiscoveryTokenList ::iterator jit = readResp.tokenInfo.begin();
        for ( ; jit != readResp.tokenInfo.end(); jit++ ) {
            readTokenList.push_back( *jit );
        }
    }

    DcgmLWLinkDiscoveryTokenList::iterator wit;
    DcgmLWLinkDiscoveryTokenList::iterator rit;
    for ( wit = writeTokenResp.tokenInfo.begin(); wit != writeTokenResp.tokenInfo.end();) {
        bool bFound = false;
        for ( rit = readTokenList.begin(); rit != readTokenList.end();) {
            DcgmLinkDiscoveryTokenInfo writeDevInfo = (*wit);
            DcgmLinkDiscoveryTokenInfo readDevInfo = (*rit);
            if ( writeDevInfo.tokelwalue == readDevInfo.tokelwalue ) {
                // found a connection
                DcgmLWLinkConnInfo connInfo;
                connInfo.masterEnd.nodeId = writeDevInfo.nodeId;
                connInfo.masterEnd.linkIndex = writeDevInfo.linkIndex;
                connInfo.masterEnd.gpuOrSwitchId = writeDevInfo.gpuOrSwitchId;
                connInfo.slaveEnd.nodeId = readDevInfo.nodeId;
                connInfo.slaveEnd.linkIndex = readDevInfo.linkIndex;
                connInfo.slaveEnd.gpuOrSwitchId = readDevInfo.gpuOrSwitchId;
                // keep it if this is not a duplicate connection
                if (!lwLinkIsInterNodeConnectionExists(linkConnRepo, connInfo)) {
                    linkConnRepo.addInterConnections( connInfo );
                    rit = readTokenList.erase( rit );
                }
                bFound = true;
                break;
            } else {
                // move to next element
                rit++;
            }
        } //end of readTokenList iteration

        // if we found a match, remove this element from our writeToeknList.
        if (bFound == true) {
            wit = writeTokenResp.tokenInfo.erase( wit );
        } else {
            // fetch next write token and compare
            wit++;
        }
    }

    return 0;
}

bool
DcgmGFMHelper::lwLinkIsInterNodeConnectionExists(DcgmFMLWLinkConnRepo &linkConnRepo,
                                                 DcgmLWLinkConnInfo &newConn)
{
    // in inter-node connection, the discovery token can match for same connection
    // as we are reading and comparing on all nodes. ie match both ways. but 
    // there is only one connection.
    LWLinkInterNodeConns::iterator it;
    LWLinkInterNodeConns interConns = linkConnRepo.getInterConnections();
    for ( it = interConns.begin(); it != interConns.end(); it++ ) {
        DcgmFMLWLinkDetailedConnInfo *tempConn = *it;
        if ((tempConn->getMasterEndPointInfo()== newConn.masterEnd) &&
            (tempConn->getSlaveEndPointInfo() == newConn.slaveEnd)) {
            return true;
        }
        // do reverse comparison as well
        if ((tempConn->getMasterEndPointInfo() == newConn.slaveEnd) &&
            (tempConn->getSlaveEndPointInfo() == newConn.masterEnd)) {
            return true;
        }
   }
   return false;
}

int
DcgmGFMHelper::waitForLinkRequestToComplete(DcgmFMLWLinkIntf *linkTrainIntf,
                                            std::list<uint64> requestIds,
                                            std::string errorCtx)
{
    std::list<uint64>::iterator it;
    int retVal = 0;

    // wait for all the request to finish. Since requests has timeout, all the requests
    // should eventually finish.
    while (true) {
        for ( it = requestIds.begin(); it != requestIds.end(); ) {
            DcgmLWLinkReqResult reqResult = {0};
            if (linkTrainIntf->isLinkReqComplete( *it, reqResult )) {
                it = requestIds.erase( it );
                if (reqResult.status != FM_LWLINK_ST_SUCCESS) {
                    retVal = reqResult.status;
                    PRINT_ERROR("%s %s", "%s failed with error: %s",
                                errorCtx.c_str(), DcgmFMLWLinkError::getLinkErrorString(reqResult.status));
                }
            } else {
                ++it;
            }
        }

        // one iteration is passed. exit if don't have any more request pending
        if ( requestIds.empty() ) {
            break;
        } else {
            // yield the CPU and try again for the request completion
            sched_yield();
        }
    }

    return retVal;
}

void
DcgmGFMHelper::genAddInterNodeConnLinkReqMsg(DcgmGFMLWLinkDevRepo &linkDevRepo,
                                             DcgmLWLinkReq &linkReq,
                                             DcgmLWLinkEndPointInfo &localEndInfo,
                                             DcgmLWLinkEndPointInfo &remoteEndInfo)
{
    DcgmFMLWLinkDevInfo devInfo;
    lwlink_pci_dev_info devPciInfo;

    linkReq.addInterNodeConnReq.localEndInfo = localEndInfo;

    linkReq.addInterNodeConnReq.remoteEndInfo.nodeId = remoteEndInfo.nodeId;
    linkReq.addInterNodeConnReq.remoteEndInfo.linkIndex = remoteEndInfo.linkIndex;
    // get the remote device information from lwlink device repo
    linkDevRepo.getDeviceInfo( remoteEndInfo.nodeId, remoteEndInfo.gpuOrSwitchId, devInfo );
    devPciInfo = devInfo.getDevicePCIInfo();
    linkReq.addInterNodeConnReq.remoteEndInfo.pciDomain = devPciInfo.domain;
    linkReq.addInterNodeConnReq.remoteEndInfo.pciBus = devPciInfo.bus;
    linkReq.addInterNodeConnReq.remoteEndInfo.pciDevice = devPciInfo.device;
    linkReq.addInterNodeConnReq.remoteEndInfo.pciFunction = devPciInfo.function;
    linkReq.addInterNodeConnReq.remoteEndInfo.devType = devInfo.getDeviceType();
    memcpy( linkReq.addInterNodeConnReq.remoteEndInfo.uuid, devInfo.getDeviceUuid(),
            sizeof(linkReq.addInterNodeConnReq.remoteEndInfo.uuid) );
}

void
DcgmGFMHelper::copyLWSwitchDeviceInfo(DevInfoReqResult &reqResult,
                                      DcgmFMLWSwitchInfoMap &lwswitchInfoMap)
{
    uint32 nodeId = reqResult.devInfoRspMsg.nodeid();
    lwswitch::deviceInfoResponseMsg devInfoRsp = reqResult.devInfoRspMsg.devinforsp();
    lwswitch::lwswitchDeviceInfoRsp switchRsp = devInfoRsp.switchdevrsp();

    DcgmFMLWSwitchInfoList switchInfoList;
    // build the list of switch info for this node
    for (int idx = 0; idx < switchRsp.switchinfo_size(); idx++) {
        lwswitch::lwswitchDeviceInfoMsg infoMsg = switchRsp.switchinfo(idx);
        const lwswitch::devicePciInfo &pciInfoMsg = infoMsg.pciinfo();
        DcgmFMLWSwitchInfo switchInfo = {0};
        switchInfo.switchIndex = infoMsg.switchindex();
        switchInfo.physicalId = infoMsg.physicalid();
        switchInfo.enabledLinkMask = infoMsg.enabledlinkmask();
        switchInfo.pciInfo.domain = pciInfoMsg.domain();
        switchInfo.pciInfo.bus = pciInfoMsg.bus();
        switchInfo.pciInfo.device = pciInfoMsg.device();
        switchInfo.pciInfo.function = pciInfoMsg.function();        
        switchInfoList.push_back(switchInfo);
    }

    // add to the map of switch information, which is per node
    lwswitchInfoMap.insert( std::make_pair(nodeId, switchInfoList) );
}

void
DcgmGFMHelper::copyGpuDeviceInfo(DevInfoReqResult &reqResult,
                                 DcgmFMGpuInfoMap &gpuInfoMap,
                                 DcgmFMGpuInfoMap &blacklistGpuInfoMap)
{
    uint32 nodeId = reqResult.devInfoRspMsg.nodeid();
    lwswitch::deviceInfoResponseMsg devInfoRsp = reqResult.devInfoRspMsg.devinforsp();
    lwswitch::gpuDeviceInfoRsp gpuRsp = devInfoRsp.gpudevrsp();
    DcgmFMGpuInfoList gpuInfoList;
    DcgmFMGpuInfoList blacklistGpuInfoList;

    // build the list of gpu info for this node
    for (int idx = 0; idx < gpuRsp.gpuinfo_size(); idx++) {
        lwswitch::gpuDeviceInfoMsg infoMsg = gpuRsp.gpuinfo(idx);
        const lwswitch::devicePciInfo &pciInfoMsg = infoMsg.pciinfo();
        DcgmFMGpuInfo gpuInfo = {0};
        gpuInfo.gpuIndex = infoMsg.gpuindex();
        gpuInfo.pciInfo.domain = pciInfoMsg.domain();
        gpuInfo.pciInfo.bus = pciInfoMsg.bus();
        gpuInfo.pciInfo.device = pciInfoMsg.device();
        gpuInfo.pciInfo.function = pciInfoMsg.function();        
        if ( infoMsg.has_uuid() ) {
            strncpy( gpuInfo.uuid, infoMsg.uuid().c_str(), sizeof(gpuInfo.uuid) );
        }
        gpuInfoList.push_back(gpuInfo);
    }

    // add to the map of gpu information, which is per node
    gpuInfoMap.insert( std::make_pair(nodeId, gpuInfoList) );

    // build the list of blacklisted gpu info for this node
    for (int idx = 0; idx < gpuRsp.blacklistgpuinfo_size(); idx++) {
        lwswitch::gpuDeviceInfoMsg infoMsg = gpuRsp.blacklistgpuinfo(idx);
        const lwswitch::devicePciInfo &pciInfoMsg = infoMsg.pciinfo();
        DcgmFMGpuInfo gpuInfo = {0};
        gpuInfo.gpuIndex = infoMsg.gpuindex();
        gpuInfo.pciInfo.domain = pciInfoMsg.domain();
        gpuInfo.pciInfo.bus = pciInfoMsg.bus();
        gpuInfo.pciInfo.device = pciInfoMsg.device();
        gpuInfo.pciInfo.function = pciInfoMsg.function();
        if ( infoMsg.has_uuid() ) {
            strncpy( gpuInfo.uuid, infoMsg.uuid().c_str(), sizeof(gpuInfo.uuid) );
        }
        blacklistGpuInfoList.push_back(gpuInfo);
    }

    // add to the map of blacklisted gpu information, which is per node
    blacklistGpuInfoMap.insert( std::make_pair(nodeId, blacklistGpuInfoList) );
}

void
DcgmGFMHelper::updateDeviceAndConnLinkState(DcgmGFMLWLinkDevRepo &linkDevRepo,
                                            DcgmFMLWLinkDetailedConnInfo *connInfo,
                                            DcgmLWLinkConnTrainResp &connTrainResp)
{
    // update connection state
    connInfo->setLinkStateInfo( connTrainResp.masterState, connTrainResp.slaveState );

    // update corresponding device's link state in device repo
    DcgmLWLinkEndPointInfo masterEnd = connInfo->getMasterEndPointInfo();
    DcgmLWLinkEndPointInfo slaveEnd = connInfo->getSlaveEndPointInfo();

    linkDevRepo.setDeviceLinkState(masterEnd.nodeId, masterEnd.gpuOrSwitchId,
                                   masterEnd.linkIndex, connTrainResp.masterState);

    linkDevRepo.setDeviceLinkState(slaveEnd.nodeId, slaveEnd.gpuOrSwitchId,
                                   slaveEnd.linkIndex, connTrainResp.slaveState);
}

const char*
DcgmGFMHelper::getLWLinkTrainTypeString(DcgmLWLinkTrainType trainType)
{
    switch (trainType) {
        case LWLINK_TRAIN_OFF_TO_SAFE:
            return "safe";
        case LWLINK_TRAIN_SAFE_TO_HIGH:
            return "high speed";
        case LWLINK_TRAIN_TO_OFF:
            return "off";
        case LWLINK_TRAIN_HIGH_TO_SAFE:
            return "high speed to safe";
        case LWLINK_TRAIN_SAFE_TO_OFF:
            return "safe to off";
    }

    // no train type case matched. shouldn't happen
    return "Unknown train type";
}
