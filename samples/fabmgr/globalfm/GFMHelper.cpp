/*
 *  Copyright 2018-2022 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */
#include <sstream>
#include <stdexcept>
#include "fm_log.h"
#include "GFMHelper.h"
#include "ctrl_dev_lwswitch.h"
#include <g_lwconfig.h>
#include "lwos.h"
#include "FMVersion.h"
#include <unistd.h>
#include <climits>

int
GFMHelper::lwLinkInitializeAllNodes(GlobalFMLWLinkIntf *linkTrainIntf,
                                    FMFabricParser* pConfig,
                                    GlobalFMLWLinkDevRepo &linkDevRepo)
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
GFMHelper::lwLinkInitializeNode(uint32 nodeId,
                                GlobalFMLWLinkIntf *linkTrainIntf,
                                GlobalFMLWLinkDevRepo &linkDevRepo)
{
    // only one node to initialize
    NodeIDList nodeIdList;
    nodeIdList.push_back( nodeId );
    // call the helper to initialize the node
    return lwLinkInitializeNodes(nodeIdList, linkTrainIntf, linkDevRepo);
}

int
GFMHelper::lwLinkGetAllNodeLinkInitStatus(GlobalFMLWLinkIntf *linkTrainIntf,
                                          FMFabricParser* pConfig,
                                          GlobalFMLWLinkDevRepo &linkDevRepo)
{
    NodeIDList nodeIdList;
    std::map <NodeKeyType, NodeConfig *>::iterator nodeit;

    // create a list of node IDs from the parsed fabric config
    for ( nodeit = pConfig->NodeCfg.begin(); nodeit!= pConfig->NodeCfg.end(); nodeit++ ) {
        NodeConfig *pNode = nodeit->second;
        nodeIdList.push_back( pNode->nodeId );
    }

    // call the helper to initialize the nodes
    return lwLinkGetLinkInitStatusForNodes(nodeIdList, linkTrainIntf, linkDevRepo);
}

int
GFMHelper::lwLinkGetAllNodeDeviceLwlinkState(GlobalFMLWLinkIntf *linkTrainIntf,
                                             FMFabricParser* pConfig,
                                             GlobalFMLWLinkDevRepo &linkDevRepo)
{
    NodeIDList nodeIdList;
    std::map <NodeKeyType, NodeConfig *>::iterator nodeit;

    // create a list of node IDs from the parsed fabric config
    for ( nodeit = pConfig->NodeCfg.begin(); nodeit!= pConfig->NodeCfg.end(); nodeit++ ) {
        NodeConfig *pNode = nodeit->second;
        nodeIdList.push_back( pNode->nodeId );
    }

    // call the helper to initialize the nodes
    return lwLinkGetDeviceLwlinkStateForNodes(nodeIdList, linkTrainIntf, linkDevRepo);
}

int
GFMHelper::lwLinkDiscoverIntraNodeConnOnNodes(FMFabricParser *pConfig,
                                              GlobalFMLWLinkIntf *linkTrainIntf,
                                              GlobalFMLWLinkConnRepo &linkConnRepo)
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

    int retVal = lwLinkSendDiscoverConnOnNodes(nodeIdList, linkTrainIntf);
    if (retVal) {
        FM_LOG_ERROR("failed to initiate LWLink device discovery");
        return retVal;
    }

    // call the helper to query all the discovered connections
    return lwLinkGetIntraNodeConns(nodeIdList, linkTrainIntf, linkConnRepo);

}

int
GFMHelper::lwLinkDiscoverIntraNodeConnOnNode(uint32 nodeId,
                                             GlobalFMLWLinkIntf *linkTrainIntf,
                                             GlobalFMLWLinkConnRepo &linkConnRepo)
{
    // this routine will initiate LWLink connection discovery and report
    // all the connections local to the node. It is assumed that the links
    // are in SAFE/HS mode prior to initiating discovery.

    // only one node to initialize
    NodeIDList nodeIdList;
    nodeIdList.push_back( nodeId );

    // first send device discovery request to the node
    int retVal = lwLinkSendDiscoverConnOnNodes(nodeIdList, linkTrainIntf);
    if (retVal) {
        FM_LOG_ERROR("failed to initiate LWLink device discovery");
        return retVal;
    }

    // call the helper to query all the discovered connections
    return lwLinkGetIntraNodeConns(nodeIdList, linkTrainIntf, linkConnRepo);
}

int
GFMHelper::lwLinkGetIntraNodeConnOnNodes(uint32 nodeId,
                                         GlobalFMLWLinkIntf *linkTrainIntf,
                                         GlobalFMLWLinkConnRepo &linkConnRepo)
{
    // this routine will NOT initiate LWLink connection discovery. But query
    // and populate lready discovered connections from the specified node.

    // only one node to query
    NodeIDList nodeIdList;
    nodeIdList.push_back( nodeId );

    // call the helper to query all the discovered connections
    return lwLinkGetIntraNodeConns(nodeIdList, linkTrainIntf, linkConnRepo);
}

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
int
GFMHelper::lwLinkReadInterNodeLinkSids( GlobalFMLWLinkIntf *linkTrainIntf,
                                        FMFabricParser *pConfig,
                                        GlobalFMLWLinkConnRepo &linkConnRepo )
{
    FM_LOG_DEBUG( "GFMHelper::lwLinkReadInterNodeLinkSids" );
    int retVal = 0;

    std::map < NodeKeyType, NodeConfig * >::iterator nodeit;
    if ( pConfig->NodeCfg.size() == 1 ) {
        // we have only one node. there are no inter node connections
        // to be discovered.
        return 0;
    }

    FMLWLinkSidList interNodeLinkSidList;
    std::map< uint64, uint32 > sidToNodeIdMap;
    std::map< uint64, uint64 > sidToGpuOrSwitchIdMap;

    // Read local and remote sid for all links on all nodes
    retVal = lwLinkReadLinkSids( linkTrainIntf, pConfig, interNodeLinkSidList, sidToNodeIdMap, sidToGpuOrSwitchIdMap );
    if (retVal) {
        // read Link Sids failed for the node, bail out
        return retVal;
    }

    // correlate link SIDS
    lwLinkCorrelateLinkSids( interNodeLinkSidList, sidToNodeIdMap, sidToGpuOrSwitchIdMap, linkConnRepo );

    return retVal;
}

int
GFMHelper::lwLinkDiscoverInterNodeConnections( GlobalFMLWLinkIntf *linkTrainIntf,
                                               FMFabricParser *pConfig,
                                               GlobalFMLWLinkConnRepo &linkConnRepo )
{
    FM_LOG_DEBUG( "GFMHelper::lwLinkDiscoverInterNodeConnections" );
    std::map <NodeKeyType, NodeConfig *>::iterator nodeit;
    int retVal = 0;

    if ( pConfig->NodeCfg.size() == 1 ) {
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
        FMLWLinkWriteDiscoveryTokenResp writeTokenResp;
        std::map< uint32, FMLWLinkReadDiscoveryTokenResp > readTokenResps;

        retVal = lwLinkWriteDiscoveryToken( pNode->nodeId, linkTrainIntf, writeTokenResp );
        if ( retVal ) {
            // write discovery token failed for the node, bail out
            return retVal;
        }

        // read discovery token from all node except the current node which wrote the tokens
        retVal = lwLinkReadDiscoveryToken( pNode->nodeId, linkTrainIntf, pConfig, readTokenResps );
        if (retVal) {
            // read discovery token failed for the node, bail out
            return retVal;
        }

        // identify the connections by matching the tokens
        lwLinkCorrelateConnections( writeTokenResp, readTokenResps, linkConnRepo );

        // for two nodes, we only need to write from one node and read from another node.
        // no need to repeat write/read discovery token for both nodes.
        if ( pConfig->NodeCfg.size() == 2 ) {
            break;
        }
    }
    return retVal;
}

int
GFMHelper::lwLinkTrainWaitForReponsesParallelInternode(GlobalFMLWLinkIntf *linkTrainIntf,
                                                       GlobalFMLWLinkDevRepo &linkDevRepo,
                                                       FMLWLinkTrainType trainTo,
                                                       std::map<uint64, FMLWLinkDetailedConnInfoList*> &requestIds)
{
    int retVal = FM_LWLINK_ST_SUCCESS;
    const char* strTrainTo = getLWLinkTrainTypeString( trainTo );

    // wait for all the request to finish. Since requests have timeouts, all the requests
    // should eventually finish.
    while (true) {
        for ( auto reqIt = requestIds.begin(); reqIt != requestIds.end(); ) {
            FMLWLinkReqResult reqResult = {0};
            if (linkTrainIntf->isLinkReqComplete( reqIt->first, reqResult )) {
                // update the training/link status information to the connection
                FMLWLinkDetailedConnInfoList *connInfoListPerRequest = reqIt->second;
                if (reqResult.status != FM_LWLINK_ST_SUCCESS) {

                    // TODO: IOCTL lwrrently returns generic failure if even one link fails. This will be changing soon such
                    // that there will be different failures returned if IOCTL itself fails vs if only training for one link fails.
                    // Once the driver side changes are in the different failures need to be handled in all four parallel calls
                    // i.e. IOCTL_LWLINK_TRAIN_INTRANODE_CONNS_PARALLEL, IOCTL_LWLINK_TRAIN_INTERNODE_LINKS_INITOPTIMIZE,
                    // IOCTL_LWLINK_TRAIN_INTERNODE_LINKS_POST_INITOPTIMIZE and IOCTL_LWLINK_TRAIN_INTERNODE_CONNS_PARALLEL
                    // final correct behavior is to set the error code when error oclwrred and continue,
                    // so that error from one link would not affect other links which are trained in parallel together
                    retVal = reqResult.status;
                    // mark the final status as failed, but continue with rest of the requests
                    FM_LOG_ERROR("LWLink connection training to %s failed with error: %s\n",
                                 strTrainTo, FMLWLinkError::getLinkErrorString(reqResult.status));

                    //The request that failed was for the following list fof connections
                    for ( unsigned int i = 0; i < reqResult.connTrainParallelResp.size(); i++ ) {
                        FM_LOG_DEBUG("parallel training to high speed request failed for (node, device, link) %u,%llu,%u ",
                                     reqResult.connTrainParallelResp[i].masterNodeId,
                                     reqResult.connTrainParallelResp[i].masterGpuOrSwitchId, reqResult.connTrainParallelResp[i].masterLinkIndex
                                     );
                    }

                    delete( connInfoListPerRequest );
                    requestIds.erase( reqIt++ );
                    continue;
                }
                // update the training/link status information to the connection/device repo
                for(auto connIt = connInfoListPerRequest->begin(); connIt != connInfoListPerRequest->end(); connIt++) {
                    FMLWLinkConnTrainResp trainResp;
                    memset(&trainResp, 0, sizeof(trainResp));
                    bool isMaster = true;
                    FMLWLinkQualityInfo linkQualityInfo;
                    FMLWLinkFomValues fomValues;
                    FMLWLinkGradingValues gradingValues;
                    for ( unsigned int i = 0; i < reqResult.connTrainParallelResp.size(); i++ ) {
                        if ((reqResult.connTrainParallelResp[i].masterNodeId == (*connIt)->getMasterEndPointInfo().nodeId) &&
                            (reqResult.connTrainParallelResp[i].masterGpuOrSwitchId == (*connIt)->getMasterEndPointInfo().gpuOrSwitchId) &&
                            (reqResult.connTrainParallelResp[i].masterLinkIndex == (*connIt)->getMasterEndPointInfo().linkIndex)
                            ) {
                            trainResp.masterState = reqResult.connTrainParallelResp[i].masterState;
                            linkQualityInfo = reqResult.connTrainParallelResp[i].masterQualityInfo;
                            fomValues = reqResult.connTrainParallelResp[i].fomValues;
                            gradingValues = reqResult.connTrainParallelResp[i].gradingValues;
                            isMaster = true;

                            FM_LOG_DEBUG("parallel training master(node, device, link, mode) %u,%llu,%u ,%u",
                                         reqResult.connTrainParallelResp[i].masterNodeId,
                                         reqResult.connTrainParallelResp[i].masterGpuOrSwitchId, reqResult.connTrainParallelResp[i].masterLinkIndex,
                                         trainResp.masterState.linkMode);

                            break;
                        } else if ((reqResult.connTrainParallelResp[i].masterNodeId == (*connIt)->getSlaveEndPointInfo().nodeId) &&
                                   (reqResult.connTrainParallelResp[i].masterGpuOrSwitchId == (*connIt)->getSlaveEndPointInfo().gpuOrSwitchId) &&
                                   (reqResult.connTrainParallelResp[i].masterLinkIndex == (*connIt)->getSlaveEndPointInfo().linkIndex)
                                   ) {
                            isMaster = false;
                            trainResp.slaveState = reqResult.connTrainParallelResp[i].masterState;
                            linkQualityInfo = reqResult.connTrainParallelResp[i].masterQualityInfo;
                            fomValues = reqResult.connTrainParallelResp[i].fomValues;
                            gradingValues = reqResult.connTrainParallelResp[i].gradingValues;
                            FM_LOG_DEBUG("parallel training slave(node, device, link, mode) %u,%llu,%u ,%u",
                                         reqResult.connTrainParallelResp[i].masterNodeId,
                                         reqResult.connTrainParallelResp[i].masterGpuOrSwitchId, reqResult.connTrainParallelResp[i].masterLinkIndex,
                                         trainResp.slaveState.linkMode);
                            break;
                        }

                    }

                    updateDeviceAndConnEndPointState( linkDevRepo, *connIt, trainResp, isMaster );
                    updateConnEndPointLinkQualityInfo( *connIt, linkQualityInfo, isMaster );
                    updateConnEndPointFomValues( *connIt, fomValues, isMaster );
                    updateConnEndPointGradingValues( *connIt, gradingValues, isMaster );
                }
                delete( connInfoListPerRequest );
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
            lwosThreadYield();
        }
    }
    return retVal;
}

void
GFMHelper::getGradingAndFomValues( GlobalFabricManager *gfm,
                                   GlobalFMLWLinkIntf *linkTrainIntf,
                                   GlobalFMLWLinkConnRepo &failedConnections,
                                   GlobalFMLWLinkDevRepo &linkDevRepo )
{
    FMFabricParser *pConfig = gfm->mpParser;
    lwLinkTrainInterNodeConnectionsParallelGetStatus(linkTrainIntf, failedConnections, linkDevRepo,
                                                     pConfig, LWLINK_TRAIN_INTERNODE_GET_GRADING_AND_FOM_VALUES);
}

#endif

int
GFMHelper::lwLinkTrainWaitForReponsesParallel(GlobalFabricManager *gfm,
                                              GlobalFMLWLinkIntf *linkTrainIntf,
                                              GlobalFMLWLinkDevRepo &linkDevRepo,
                                              FMLWLinkTrainType trainTo,
                                              std::map<uint64, FMLWLinkDetailedConnInfoList*> &requestIds)
{
    FMFabricParser *pConfig = gfm->mpParser;
    int retVal = FM_LWLINK_ST_SUCCESS;
    const char* strTrainTo = getLWLinkTrainTypeString( trainTo );

    // wait for all the request to finish. Since requests have timeouts, all the requests
    // should eventually finish.
    while (true) {
        for ( auto reqIt = requestIds.begin(); reqIt != requestIds.end(); ) {
            FMLWLinkReqResult reqResult = {0};
            if (linkTrainIntf->isLinkReqComplete( reqIt->first, reqResult )) {
                // update the training/link status information to the connection
                FMLWLinkDetailedConnInfoList *connInfoListPerRequest = reqIt->second;
                if (reqResult.status != FM_LWLINK_ST_SUCCESS) {
                    // set the error code when error oclwrred and continue,
                    // so that error from one link would not affect other links which are
                    // trained in parallel together
                    retVal = reqResult.status;
                    // mark the final status as failed, but continue with rest of the requests
                    FM_LOG_ERROR("LWLink connection training to %s failed with error: %s\n",
                                 strTrainTo, FMLWLinkError::getLinkErrorString(reqResult.status));
                }
                // update the training/link status information to the connection/device repo
                for(auto connIt = connInfoListPerRequest->begin(); connIt != connInfoListPerRequest->end(); connIt++) {
                    FMLWLinkConnTrainResp trainResp;
                    for ( unsigned int i = 0; i < reqResult.connTrainParallelResp.size(); i++ ) {
                        if ((reqResult.connTrainParallelResp[i].masterNodeId == (*connIt)->getMasterEndPointInfo().nodeId) &&
                            (reqResult.connTrainParallelResp[i].masterGpuOrSwitchId == (*connIt)->getMasterEndPointInfo().gpuOrSwitchId) &&
                            (reqResult.connTrainParallelResp[i].masterLinkIndex == (*connIt)->getMasterEndPointInfo().linkIndex)
                            ) {
                            trainResp.masterState = reqResult.connTrainParallelResp[i].masterState;
                            trainResp.slaveState = reqResult.connTrainParallelResp[i].slaveState;
                            FM_LOG_DEBUG("parallel training %llu,%u <====> %llu,%u master.linkMode=%u slave.linkMode=%u",
                                         reqResult.connTrainParallelResp[i].masterGpuOrSwitchId, reqResult.connTrainParallelResp[i].masterLinkIndex,
                                         reqResult.connTrainParallelResp[i].slaveGpuOrSwitchId, reqResult.connTrainParallelResp[i].slaveLinkIndex,
                                         trainResp.masterState.linkMode, trainResp.slaveState.linkMode);

                            std::stringstream outStr;
                            (*connIt)->dumpConnInfo( &outStr, gfm, linkDevRepo);
                            FM_LOG_DEBUG("following LWLink connection training to %s with result: %s\n %s",
                                         strTrainTo, FMLWLinkError::getLinkErrorString(reqResult.status), outStr.str().c_str());
                            updateDeviceAndConnLinkState( linkDevRepo, *connIt, trainResp );
                            break;
                        }
                    }

                }
                delete( connInfoListPerRequest );
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
            lwosThreadYield();
        }
    }
    return retVal;
}

int
GFMHelper::lwLinkTrainIntraNodeConnectionsParallel(GlobalFabricManager *gfm,
                                                   GlobalFMLWLinkIntf *linkTrainIntf,
                                                   GlobalFMLWLinkConnRepo &linkConnRepo,
                                                   GlobalFMLWLinkDevRepo &linkDevRepo,
                                                   FMLWLinkTrainType trainTo)
{
    FMFabricParser *pConfig = gfm->mpParser;
    std::map<uint64, FMLWLinkDetailedConnInfoList*> requestIds;
    LWLinkIntraConnMap::iterator it;
    LWLinkIntraConnMap intraConnMap = linkConnRepo.getIntraConnections();
    const char* strTrainTo = getLWLinkTrainTypeString( trainTo );

    FM_LOG_INFO( "training all LWLink connections to %s", strTrainTo );

    // first send train request for all the connections
    for ( it = intraConnMap.begin(); it != intraConnMap.end(); it++ ) {
        FMLWLinkDetailedConnInfoList &connList = it->second;
        FMLWLinkDetailedConnInfoList::iterator jit;
        uint32_t nodeId = it->first;
        // Train all connections of a switch in parallel
        int requestConnCount = 0;
        FMLWLinkDetailedConnInfoList *connListPerRequest;
        FMLWLinkReq linkParallelReq;
        for ( jit = connList.begin(); jit != connList.end(); jit++ ) {

            if (requestConnCount == 0 ) {
                connListPerRequest = new FMLWLinkDetailedConnInfoList;
                linkParallelReq.connTrainParallelReq.clear();
            }
            requestConnCount++;
            FMLWLinkDetailedConnInfo *conn = *jit;
            // fill master end information of the connection
            FMLWLinkEndPointInfo masterEnd = conn->getMasterEndPointInfo();
            // fill slave end information of the connection
            FMLWLinkEndPointInfo slaveEnd = conn->getSlaveEndPointInfo();

            FMLWLinkConnTrainReq connTrainReq;
            connTrainReq.masterNodeId = masterEnd.nodeId;
            connTrainReq.masterGpuOrSwitchId = masterEnd.gpuOrSwitchId;
            connTrainReq.masterLinkIndex = masterEnd.linkIndex;
            connTrainReq.slaveNodeId = slaveEnd.nodeId;
            connTrainReq.slaveGpuOrSwitchId = slaveEnd.gpuOrSwitchId;
            connTrainReq.slaveLinkIndex = slaveEnd.linkIndex;
            // fill the train to state, High speed state
            connTrainReq.trainTo = trainTo;

            linkParallelReq.connTrainParallelReq.push_back(connTrainReq);

            connListPerRequest->push_back(conn);
            if (requestConnCount == LWLINK_MAX_PARALLEL_CONNS_TRAIN_COUNT) {
                requestConnCount = 0;

                uint64 requestId = 0;
                linkTrainIntf->sendConnTrainParallelReq( linkParallelReq, requestId );
                if (requestId) {
                    requestIds.insert( std::make_pair(requestId, connListPerRequest) );
                }
            }
        }
        if ( requestConnCount != 0 ) {
            uint64 requestId = 0;
            linkTrainIntf->sendConnTrainParallelReq( linkParallelReq, requestId );
            if (requestId) {
                requestIds.insert( std::make_pair(requestId, connListPerRequest) );
            }
        }
    }

    return lwLinkTrainWaitForReponsesParallel(gfm, linkTrainIntf, linkDevRepo, trainTo, requestIds);
}

int
GFMHelper::lwLinkTrainIntraNodeConnections(GlobalFabricManager *gfm,
                                           GlobalFMLWLinkIntf *linkTrainIntf,
                                           GlobalFMLWLinkConnRepo &linkConnRepo,
                                           GlobalFMLWLinkDevRepo &linkDevRepo,
                                           FMLWLinkTrainType trainTo)
{
    FMFabricParser *pConfig = gfm->mpParser;
    std::map<uint64, FMLWLinkDetailedConnInfo*> requestIds;
    std::map<uint64, FMLWLinkDetailedConnInfo*>::iterator reqIt;
    LWLinkIntraConnMap::iterator it;
    int retVal = 0;
    LWLinkIntraConnMap intraConnMap = linkConnRepo.getIntraConnections();
    const char* strTrainTo = getLWLinkTrainTypeString( trainTo );

    FM_LOG_INFO( "training LWLink connections to %s", strTrainTo );

    // first send train request for all the connections
    for ( it = intraConnMap.begin(); it != intraConnMap.end(); it++ ) {
        FMLWLinkDetailedConnInfoList &connList = it->second;
        FMLWLinkDetailedConnInfoList::iterator jit;
        for ( jit = connList.begin(); jit != connList.end(); jit++ ) {
            uint64 requestId = 0;
            FMLWLinkReq linkReq = {{0}};
            FMLWLinkDetailedConnInfo *conn = *jit;
            // fill master end information of the connection
            FMLWLinkEndPointInfo masterEnd = conn->getMasterEndPointInfo();
            linkReq.connTrainReq.masterNodeId = masterEnd.nodeId;
            linkReq.connTrainReq.masterGpuOrSwitchId = masterEnd.gpuOrSwitchId;
            linkReq.connTrainReq.masterLinkIndex = masterEnd.linkIndex;
            // fill slave end information of the connection
            FMLWLinkEndPointInfo slaveEnd = conn->getSlaveEndPointInfo();
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

    // wait for all the request to finish. Since requests has timeout, all the requests
    // should eventually finish.
    while (true) {
        for ( reqIt = requestIds.begin(); reqIt != requestIds.end(); ) {
            FMLWLinkReqResult reqResult = {0};
            if (linkTrainIntf->isLinkReqComplete( reqIt->first, reqResult )) {
                // update the training/link status information to the connection
                FMLWLinkDetailedConnInfo *connInfo = reqIt->second;
                if (reqResult.status != FM_LWLINK_ST_SUCCESS) {
                    // mark the final status as failed, but continue with rest of the requests
                    std::stringstream outStr;
                    connInfo->dumpConnInfo( &outStr, gfm, linkDevRepo);
                    retVal = reqResult.status;
                    FM_LOG_ERROR("following LWLink connection training to %s failed with error: %s\n %s",
                                strTrainTo, FMLWLinkError::getLinkErrorString(reqResult.status), outStr.str().c_str());
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
            lwosThreadYield();
        }
    }

    return retVal;
}

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
int
GFMHelper::lwLinkTrainInterNodeConnectionsParallelDoStep( GlobalFMLWLinkIntf *linkTrainIntf,
                                                          GlobalFMLWLinkConnRepo &linkConnRepo,
                                                          GlobalFMLWLinkDevRepo &linkDevRepo,
                                                          FMFabricParser *pConfig,
                                                          FMLWLinkTrainType trainTo)
{
    FM_LOG_INFO( "training all connections to %s", getLWLinkTrainTypeString( trainTo ) );

    // Next do POST INITOPTIMIZE on all nodes.
    std::list<uint64> requestIdsInitoptimize;

    std::map <NodeKeyType, NodeConfig *>::iterator nodeit;
    for ( nodeit = pConfig->NodeCfg.begin(); nodeit!= pConfig->NodeCfg.end(); nodeit++ ) {
        NodeConfig *pNode = nodeit->second;
        LWLinkInterNodeConns::iterator it;
        uint64 requestId = 0;
        FMLWLinkReq *linkReq = new( FMLWLinkReq ) ;
        linkReq->connTrainParallelReq.clear();

        LWLinkInterNodeConns interConnList = linkConnRepo.getInterConnections();

        for ( it = interConnList.begin(); it != interConnList.end(); it++ ) {
            FMLWLinkDetailedConnInfo *conn = *it;
            FMLWLinkEndPointInfo masterEnd = conn->getMasterEndPointInfo();
            FMLWLinkEndPointInfo slaveEnd = conn->getSlaveEndPointInfo();
            if ( ( masterEnd.nodeId != pNode->nodeId ) && ( slaveEnd.nodeId != pNode->nodeId ) )
                continue;
            FMLWLinkConnTrainReq connTrainReq;
            memset(&connTrainReq, 0, sizeof(connTrainReq));
            // Note that slave end is ignored for initoptmize requests
            if ( masterEnd.nodeId == pNode->nodeId ) {
                // fill master end information of the connection
                connTrainReq.masterNodeId = masterEnd.nodeId;
                connTrainReq.masterGpuOrSwitchId = masterEnd.gpuOrSwitchId;
                connTrainReq.masterLinkIndex = masterEnd.linkIndex;
            } else {
                // fill slave end information of the connection
                connTrainReq.masterNodeId = slaveEnd.nodeId;
                connTrainReq.masterGpuOrSwitchId = slaveEnd.gpuOrSwitchId;
                connTrainReq.masterLinkIndex = slaveEnd.linkIndex;
            }
            connTrainReq.trainTo = trainTo;
            linkReq->connTrainParallelReq.push_back( connTrainReq );
        }

        if (linkReq->connTrainParallelReq.size() > 0)
            linkTrainIntf->sendConnTrainParallelReq( *linkReq, requestId );

        delete linkReq;
        if (requestId) {
            requestIdsInitoptimize.push_back( requestId );
        }
    }

    return waitForLinkRequestToComplete( linkTrainIntf, requestIdsInitoptimize, "LWLink post initoptimize" );
}

// train the main link to HS for all nodes. All links for a node are trained in parallel. Requests is sent to
// all nodes before waiting for replies
int
GFMHelper::lwLinkTrainInterNodeConnectionsParallelGetStatus( GlobalFMLWLinkIntf *linkTrainIntf,
                                                             GlobalFMLWLinkConnRepo &linkConnRepo,
                                                             GlobalFMLWLinkDevRepo &linkDevRepo,
                                                             FMFabricParser *pConfig,
                                                             FMLWLinkTrainType trainTo)
{
    FM_LOG_INFO( "training all connections to %s", getLWLinkTrainTypeString( trainTo ) );

    std::map<uint64, FMLWLinkDetailedConnInfoList*> requestIds;

    std::map <NodeKeyType, NodeConfig *>::iterator nodeit;
    //Iterate over list of nodes
    for ( nodeit = pConfig->NodeCfg.begin(); nodeit!= pConfig->NodeCfg.end(); nodeit++ ) {
        NodeConfig *pNode = nodeit->second;
        uint64 requestId = 0;
        FMLWLinkReq *linkReq = new( FMLWLinkReq ) ;

        // list of connections per request
        int requestConnCount = 0;
        FMLWLinkDetailedConnInfoList *connListPerRequest = nullptr;
        LWLinkInterNodeConns interConnList = linkConnRepo.getInterConnections();

        // Iterate over all connections and choose the nodes with maching node ID
        // The message will be repeated for both ends of the connections. This for loop chooses a node say X and
        // then chooses all the end points that are in node X (whether master or slave) and sends a message to train.
        LWLinkInterNodeConns::iterator it;
        for ( it = interConnList.begin(); it != interConnList.end(); it++ ) {
            if (requestConnCount == 0 ) {
                 connListPerRequest = new FMLWLinkDetailedConnInfoList;
                 linkReq->connTrainParallelReq.clear();
            }
            requestConnCount++;
            FMLWLinkDetailedConnInfo *conn = *it;
            FMLWLinkEndPointInfo masterEnd = conn->getMasterEndPointInfo();
            FMLWLinkEndPointInfo slaveEnd = conn->getSlaveEndPointInfo();
            if ( ( masterEnd.nodeId != pNode->nodeId ) && ( slaveEnd.nodeId != pNode->nodeId ) )
                continue;

            FMLWLinkConnTrainReq connTrainReq;
            memset(&connTrainReq, 0 , sizeof(connTrainReq));
            // Note that slave end is ignored for initoptmize to high requests
            // For each link connTrainReq.slaveNodeId is set to true if it is the master end
            // otherwise connTrainReq.slaveNodeId is set to false

            if ( masterEnd.nodeId == pNode->nodeId ) {
                // fill master end information of the connection
                connTrainReq.masterNodeId = masterEnd.nodeId;
                connTrainReq.masterGpuOrSwitchId = masterEnd.gpuOrSwitchId;
                connTrainReq.masterLinkIndex = masterEnd.linkIndex;
                // true for master end of connection. TODO: Add new field for this?
                connTrainReq.slaveNodeId = true;
            } else {
                // fill slave end information of the connection
                connTrainReq.masterNodeId = slaveEnd.nodeId;
                connTrainReq.masterGpuOrSwitchId = slaveEnd.gpuOrSwitchId;
                connTrainReq.masterLinkIndex = slaveEnd.linkIndex;
                // false for slave end of connection. TODO: Add new field for this?
                connTrainReq.slaveNodeId = false;

            }
            connTrainReq.trainTo = trainTo;
            linkReq->connTrainParallelReq.push_back( connTrainReq );
            connListPerRequest->push_back(conn);
            // TODO : handle case when more that LWLINK_MAX_PARALLEL_CONNS_TRAIN_COUNT connections exist on one node
            // if (requestConnCount == LWLINK_MAX_PARALLEL_CONNS_TRAIN_COUNT)
        }

        if (linkReq->connTrainParallelReq.size() > 0) {
            linkTrainIntf->sendConnTrainParallelReq( *linkReq, requestId );
            FM_LOG_DEBUG("lwLinkTrainInterNodeConnectionsParallelGetStatus size=%lu", linkReq->connTrainParallelReq.size());
        }

        delete linkReq;
        if (requestId && (connListPerRequest != nullptr)) {
            requestIds.insert( std::make_pair(requestId, connListPerRequest) );
        } else {
            delete connListPerRequest;
        }
    }
    return lwLinkTrainWaitForReponsesParallelInternode(linkTrainIntf, linkDevRepo, trainTo, requestIds);
}

void
GFMHelper::lwLinkTrainInterNodeConnectionsRemoveHighEomLinks(GlobalFMLWLinkConnRepo &linkConnRepo)
{
    // get a reference to the connection list as we need to remove good EOM reporting conns.
    LWLinkInterNodeConns &interConnList = linkConnRepo.getInterConnections();
    LWLinkInterNodeConns::iterator it;
    for (it = interConnList.begin(); it != interConnList.end();) {
        FMLWLinkDetailedConnInfo *tempConnInfo = *it;
        if ((false == tempConnInfo->getMasterLinkEomLow()) && (false == tempConnInfo->getSlaveLinkEomLow())) {
            // good link, remove it
            it = interConnList.erase(it);
            // the list maintains a pointer to detailed conn info, so free the same.
            delete tempConnInfo;
        } else {
            ++it;
        }
    }
}

int
GFMHelper::lwLinkTrainInterNodeConnectionsParallelDoForceEQ(GlobalFabricManager *gfm,
                                                            GlobalFMLWLinkIntf *linkTrainIntf,
                                                            GlobalFMLWLinkConnRepo &linkConnRepo,
                                                            GlobalFMLWLinkDevRepo &linkDevRepo)
{
    FMFabricParser *pConfig = gfm->mpParser;

    //
    // This optical training step ilwolves the following
    // 1. Check whether any link is reporting EOM low
    // 2. For those links, force equalization which will sets all the forces up and sets the overrides
    // 3. Sleep for 500ms (for nodes)
    // 4. Release equalization, which will clear/release all the overrides
    // 5. Repeat this at least 3 times for remaining low EOM links
    //

    //
    // build a temporary map of LWLink conn repo for us to operate. this map will later get trimmed
    // for links which are reporting good EOM values
    //
    int retVal = 0;
    GlobalFMLWLinkConnRepo tempLWLinkConnRepo;
    LWLinkInterNodeConns interConnList = linkConnRepo.getInterConnections();
    LWLinkInterNodeConns::iterator it;
    for (it = interConnList.begin(); it != interConnList.end(); it++) {
        FMLWLinkDetailedConnInfo *tempConnInfo = *it;
        FMLWLinkConnInfo lwLinkConn;
        lwLinkConn.masterEnd = tempConnInfo->getMasterEndPointInfo();
        lwLinkConn.slaveEnd = tempConnInfo->getSlaveEndPointInfo();
        tempLWLinkConnRepo.addInterConnections(lwLinkConn);
    }

    //
    // now we have copy of LWLinkConnRepo with all the inter node connections
    // get EOM values of corresponding links
    //
    interConnList = tempLWLinkConnRepo.getInterConnections();
    lwLinkTrainInterNodeConnectionsParallelGetStatus(linkTrainIntf, tempLWLinkConnRepo, linkDevRepo,
                                                     pConfig, LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_CHECK_EOM_STATUS);

    // filter low eom links by removing links which are reporting good EOMs
    lwLinkTrainInterNodeConnectionsRemoveHighEomLinks(tempLWLinkConnRepo);

    // check whether we have any connections with low EOM
    interConnList = tempLWLinkConnRepo.getInterConnections();
    if (interConnList.empty()) {
        FM_LOG_INFO("all the LWLink connections are reporting good EOM values");
        return retVal;
    }

    //
    // we have some links reporting low EOM values. Try enable, disable forceEQ sequence
    // Note: Even if only one endpoint of the connection report low EOM, we will do the
    // ForceEQ sequence on both endpoints. While repeating forceEQ on good EOM endpoint is
    // not necessary, h/w is fine with it for now. If it must be for the specific endpoint,
    // then some endpoint specific filtering is required.
    //

    int retryCnt = 0;
    do {
        FM_LOG_INFO("doing enable and disable force equalization on following links, iteration %d", retryCnt+1);
        for (it = interConnList.begin(); it != interConnList.end(); it++) {
            FMLWLinkDetailedConnInfo *tempConnInfo = *it;
            std::stringstream outStr;
            tempConnInfo->dumpConnInfo(&outStr, gfm, linkDevRepo);
            FM_LOG_INFO("%s", outStr.str().c_str());
        }
        // enable and disable EQ for these links
        lwLinkTrainInterNodeConnectionsParallelDoStep(linkTrainIntf, tempLWLinkConnRepo, linkDevRepo,
                                                      pConfig, LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_FORCE_EQ);
        // TODO check this sleep. 500ms or 1s okay.
        sleep(1);
        lwLinkTrainInterNodeConnectionsParallelDoStep(linkTrainIntf, tempLWLinkConnRepo, linkDevRepo,
                                                      pConfig, LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_DISABLE_FORCE_EQ);
        // get link EOM values again
        lwLinkTrainInterNodeConnectionsParallelGetStatus(linkTrainIntf, tempLWLinkConnRepo, linkDevRepo,
                                                         pConfig, LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_CHECK_EOM_STATUS);

        // filter low eom links by removing links which are reporting good EOMs
        lwLinkTrainInterNodeConnectionsRemoveHighEomLinks(tempLWLinkConnRepo);

        // check whether we have any connections reporting low EOM agian
        interConnList = tempLWLinkConnRepo.getInterConnections();
        if (interConnList.empty()) {
            FM_LOG_INFO("all the LWLink connections are now reporting good EOM values");
            return retVal;
        }
        // we still have some more connections with low EOM. repeat the step
        retryCnt++;
    } while(retryCnt < FM_LWLINK_OPTICAL_FORCE_EQ_RETRY_CNT);

    // after all the retries, if we have some more connection, then log it
    FM_LOG_WARNING("following connections are reporting low EOM values after force equalization attempt");
    for (it = interConnList.begin(); it != interConnList.end(); it++) {
        FMLWLinkDetailedConnInfo *tempConnInfo = *it;
        std::stringstream outStr;
        tempConnInfo->dumpConnInfo(&outStr, gfm, linkDevRepo);
        FM_LOG_INFO("%s", outStr.str().c_str());
    }

    //
    // Note: the EOM values on actual LWLink connections in linkConnRepo which GFM is maintaining
    // is not updated as we were operating on a copy. Do a get EOM query with linkConnRepo if
    // EOM values need to be populated on master LWLink conn copy.
    //
    return retVal;
}

int
GFMHelper::lwLinkTrainInterNodeConnectionsParallel(GlobalFabricManager *gfm,
                                                   GlobalFMLWLinkIntf *linkTrainIntf,
                                                   GlobalFMLWLinkConnRepo &linkConnRepo,
                                                   GlobalFMLWLinkDevRepo &linkDevRepo)
{
    FMFabricParser *pConfig = gfm->mpParser;
    int retVal = 0;

    if ( pConfig->NodeCfg.size() == 1 ) {
        // we have only one node. there is no inter node connections
        // to be discovered.
        return 0;
    }
    FM_LOG_DEBUG("LWLINK_TRAIN_SAFE_TO_INITOPTIMIZE start");
    lwLinkTrainInterNodeConnectionsParallelDoStep( linkTrainIntf, linkConnRepo, linkDevRepo, pConfig, LWLINK_TRAIN_SAFE_TO_INITOPTIMIZE);
    FM_LOG_DEBUG("LWLINK_TRAIN_SAFE_TO_INITOPTIMIZE end");

    sleep( 5 );

    FM_LOG_DEBUG("LWLINK_TRAIN_POST_INITOPTIMIZE start");
    lwLinkTrainInterNodeConnectionsParallelDoStep( linkTrainIntf, linkConnRepo, linkDevRepo, pConfig, LWLINK_TRAIN_POST_INITOPTIMIZE);
    FM_LOG_DEBUG("LWLINK_TRAIN_POST_INITOPTIMIZE end");

    FM_LOG_DEBUG("LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_INF_MODE start");
    lwLinkTrainInterNodeConnectionsParallelDoStep( linkTrainIntf, linkConnRepo, linkDevRepo, pConfig, LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_INF_MODE);
    FM_LOG_DEBUG("LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_INF_MODE end");

    FM_LOG_DEBUG("LWLINK_TRAIN_INTERNODE_PARALLEL_INITOPTIMIZE_TO_HIGH start");
    retVal = lwLinkTrainInterNodeConnectionsParallelGetStatus( linkTrainIntf, linkConnRepo, linkDevRepo, pConfig, LWLINK_TRAIN_INTERNODE_PARALLEL_INITOPTIMIZE_TO_HIGH);
    FM_LOG_DEBUG("LWLINK_TRAIN_INTERNODE_PARALLEL_INITOPTIMIZE_TO_HIGH end");

    // wait for RXEQ adaptation to complete
    sleep( 5 );

    FM_LOG_DEBUG("LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_MAINTENANCE_TX start");
    lwLinkTrainInterNodeConnectionsParallelDoStep( linkTrainIntf, linkConnRepo, linkDevRepo, pConfig, LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_MAINTENANCE_TX);
    FM_LOG_DEBUG("LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_MAINTENANCE_TX end");

    sleep( 5 );

    FM_LOG_DEBUG("LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_MAINTENANCE_RX start");
    lwLinkTrainInterNodeConnectionsParallelDoStep( linkTrainIntf, linkConnRepo, linkDevRepo, pConfig, LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_MAINTENANCE_RX);
    FM_LOG_DEBUG("LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_MAINTENANCE_RX end");

    // wait for xcvr maintenance to settle
    sleep( 10 );

    FM_LOG_DEBUG("lwLinkTrainInterNodeConnectionsParallelDoForceEQ start");
    lwLinkTrainInterNodeConnectionsParallelDoForceEQ(gfm, linkTrainIntf, linkConnRepo, linkDevRepo);
    FM_LOG_DEBUG("lwLinkTrainInterNodeConnectionsParallelDoForceEQ end");

    FM_LOG_DEBUG("LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_DISABLE_INF_MODE start");
    lwLinkTrainInterNodeConnectionsParallelDoStep( linkTrainIntf, linkConnRepo, linkDevRepo, pConfig, LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_DISABLE_INF_MODE);
    FM_LOG_DEBUG("LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_DISABLE_INF_MODE end");
    // Wait for link to complete transition to active
    sleep( 5 );
    FM_LOG_DEBUG("LWLINK_TRAIN_INTERNODE_PARALLEL_GET_LINK_STATE start");
    retVal = lwLinkTrainInterNodeConnectionsParallelGetStatus( linkTrainIntf, linkConnRepo, linkDevRepo, pConfig, LWLINK_TRAIN_INTERNODE_PARALLEL_GET_LINK_STATE);
    FM_LOG_DEBUG("LWLINK_TRAIN_INTERNODE_PARALLEL_GET_LINK_STATE end");

    return retVal;
}

int
GFMHelper::lwlinkAddInterNodeConnections(GlobalFMLWLinkIntf *linkTrainIntf,
                                         GlobalFMLWLinkConnRepo &linkConnRepo,
                                         GlobalFMLWLinkDevRepo &linkDevRepo)
{
    std::list<uint64> requestIds;
    int retVal = 0;
    LWLinkInterNodeConns::iterator it;
    LWLinkInterNodeConns interConnList = linkConnRepo.getInterConnections();

    FM_LOG_INFO("adding LWLink connections");

    for ( it = interConnList.begin(); it != interConnList.end(); it++ ) {
        uint64 requestId = 0;
        FMLWLinkReq linkReq = {{0}};
        FMLWLinkDetailedConnInfo *tempInterConn = *it;
        FMLWLinkEndPointInfo localEndInfo = tempInterConn->getMasterEndPointInfo();
        FMLWLinkEndPointInfo remoteEndInfo = tempInterConn->getSlaveEndPointInfo();

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
    retVal = waitForLinkRequestToComplete( linkTrainIntf, requestIds, "LWLink add connection" );

    return retVal;
}

int
GFMHelper::lwLinkTrainInterNodeConnections(GlobalFabricManager *gfm,
                                           GlobalFMLWLinkIntf *linkTrainIntf,
                                           GlobalFMLWLinkConnRepo &linkConnRepo,
                                           GlobalFMLWLinkDevRepo &linkDevRepo,
                                           FMLWLinkTrainType trainTo)
{
    FMFabricParser *pConfig = gfm->mpParser;
    std::map<uint64, FMLWLinkDetailedConnInfo*> requestIds;
    int retVal = 0;
    std::map<uint64, uint64> requestIdPairs;
    std::map <NodeKeyType, NodeConfig *>::iterator nodeit;
    LWLinkInterNodeConns interConnList = linkConnRepo.getInterConnections();
    const char* strTrainTo = getLWLinkTrainTypeString( trainTo );

    FM_LOG_INFO( "training all connections to %s", strTrainTo );

    vector<FMLWLinkTrainType> trainTypes;
    getTrainTypeVector(trainTo, trainTypes);

    // for each connection, send a request to each node associated with it
    // first send request to do Sublink training, then for main link training
    for (unsigned int i = 0; i < trainTypes.size(); i++) {
        sendRequestToTrainLinks(interConnList, linkTrainIntf, requestIds, requestIdPairs, trainTypes[i]);
        
        // wait for all the request to finish. Since requests has timeout, all the requests
        // should eventually finish.
        waitForTrainingLinkReqs(gfm, linkTrainIntf, requestIds, requestIdPairs, linkDevRepo, trainTo);
    }

    return retVal;
}

void
GFMHelper::sendRequestToTrainLinks(LWLinkInterNodeConns interConnList, GlobalFMLWLinkIntf *linkTrainIntf,
                                   std::map<uint64, FMLWLinkDetailedConnInfo*> &requestIds, 
                                   std::map<uint64, uint64> &requestIdPairs,
                                   FMLWLinkTrainType trainType)
{
    LWLinkInterNodeConns::iterator it;
    int retVal = 0;
    for (it = interConnList.begin(); it != interConnList.end(); it++) {
        FMLWLinkReq linkReq = {{0}};
        FMLWLinkDetailedConnInfo *conn = *it;
        uint64 masterReqId = 0;
        uint64 slaveReqId = 0;

        FMLWLinkEndPointInfo masterEnd = conn->getMasterEndPointInfo();
        FMLWLinkEndPointInfo slaveEnd = conn->getSlaveEndPointInfo();

        // fill the train to state, High speed state
        linkReq.connTrainReq.trainTo = trainType;

        // fill master end information of the connection
        linkReq.connTrainReq.masterNodeId = masterEnd.nodeId;
        linkReq.connTrainReq.masterGpuOrSwitchId = masterEnd.gpuOrSwitchId;
        linkReq.connTrainReq.masterLinkIndex = masterEnd.linkIndex;
        // fill slave end information of the connection    
        linkReq.connTrainReq.slaveNodeId = slaveEnd.nodeId;
        linkReq.connTrainReq.slaveGpuOrSwitchId = slaveEnd.gpuOrSwitchId;
        linkReq.connTrainReq.slaveLinkIndex = slaveEnd.linkIndex;

        linkTrainIntf->sendConnTrainReq( linkReq, masterReqId );
        if (masterReqId) {
            requestIds.insert( std::make_pair(masterReqId, conn) );                
        }

        // fill master end information of the connection
        // this is taken as INT_MAX so that lfm knows it is slave side
        linkReq.connTrainReq.masterNodeId = INT_MAX;
        linkReq.connTrainReq.masterGpuOrSwitchId = masterEnd.gpuOrSwitchId;
        linkReq.connTrainReq.masterLinkIndex = masterEnd.linkIndex;
        // fill slave end information of the connection    
        linkReq.connTrainReq.slaveNodeId = slaveEnd.nodeId;
        linkReq.connTrainReq.slaveGpuOrSwitchId = slaveEnd.gpuOrSwitchId;
        linkReq.connTrainReq.slaveLinkIndex = slaveEnd.linkIndex;

        linkTrainIntf->sendConnTrainReq( linkReq, slaveReqId );
        if (slaveReqId) {
            requestIds.insert( std::make_pair(slaveReqId, conn) );                
        }

        requestIdPairs[masterReqId] = slaveReqId;
    }
}

void 
GFMHelper::waitForTrainingLinkReqs(GlobalFabricManager *gfm,
                                   GlobalFMLWLinkIntf *linkTrainIntf,
                                   std::map<uint64, FMLWLinkDetailedConnInfo*> &requestIds, 
                                   std::map<uint64, uint64> &requestIdPairs,
                                   GlobalFMLWLinkDevRepo &linkDevRepo,
                                   FMLWLinkTrainType trainTo)
{
    FMFabricParser *pConfig = gfm->mpParser;
    std::map<uint64, FMLWLinkDetailedConnInfo*>::iterator reqIt;
    map<uint64, FMLWLinkReqResult> reqIdsComplete;
    int retVal = 0;
    const char* strTrainTo = getLWLinkTrainTypeString( trainTo );
    while (true) {
        for ( reqIt = requestIds.begin(); reqIt != requestIds.end(); ) {
            FMLWLinkReqResult reqResult = {0};
            // do not check for slave requests. Check for master requests
            // and then find the corresponding slave pair to check if req is complete
            if (requestIdPairs.find(reqIt->first) == requestIdPairs.end()) {
                    ++reqIt;
                    continue;
            }
            uint64 masterReqId = reqIt->first;
            uint64 slaveReqId = requestIdPairs[reqIt->first];

            if (linkTrainIntf->isLinkReqComplete(masterReqId, reqResult)) {
                reqIdsComplete.insert(make_pair(masterReqId, reqResult));
            }

            if (linkTrainIntf->isLinkReqComplete(slaveReqId, reqResult)) {
                reqIdsComplete.insert(make_pair(slaveReqId, reqResult));
            }
            // check if the corresponding pair of master and slave requests are complete and process them
            if (reqIdsComplete.find(masterReqId) != reqIdsComplete.end() && reqIdsComplete.find(slaveReqId) != reqIdsComplete.end()) {
                FMLWLinkDetailedConnInfo *connInfo = reqIt->second;
                FMLWLinkReqResult masterReqResult = reqIdsComplete[masterReqId];
                FMLWLinkReqResult slaveReqResult = reqIdsComplete[slaveReqId];
                if (masterReqResult.status != FM_LWLINK_ST_SUCCESS || slaveReqResult.status != FM_LWLINK_ST_SUCCESS) {
                    // mark the final status as failed, but continue with rest of the requests
                    std::stringstream outStr;
                    connInfo->dumpConnInfo( &outStr, gfm, linkDevRepo);
                    retVal = masterReqResult.status != FM_LWLINK_ST_SUCCESS ? masterReqResult.status : slaveReqResult.status;
                    FM_LOG_ERROR("following LWLink connection training to %s failed with error: %s\n %s",
                                 strTrainTo, FMLWLinkError::getLinkErrorString((LWLinkErrorCodes)retVal), outStr.str().c_str());
                    // log connection training failures to syslog
                    FM_SYSLOG_ERR("failed to train LWLink connection: %s", outStr.str().c_str());
                }
                //copy over slave response of linkState
                masterReqResult.connTrainResp.slaveState = slaveReqResult.connTrainResp.slaveState;
                // update the training/link status information to the connection/device repo
                updateDeviceAndConnLinkState( linkDevRepo, connInfo, masterReqResult.connTrainResp );
                requestIds.erase( reqIt++ );
                requestIds.erase( slaveReqId );
                reqIdsComplete.erase(slaveReqId);
                reqIdsComplete.erase(masterReqId);
            } else {
                ++reqIt;
            }
        }
        // one iteration is passed. exit if don't have any more request pending
        if ( requestIds.empty() ) {
            break;
        } else {
            // yield the CPU and try again for the request completion
            lwosThreadYield();
        }
    }
}

#endif
int
GFMHelper::trainLWLinkConnection(GlobalFabricManager *gfm,
                                 GlobalFMLWLinkIntf *linkTrainIntf,
                                 GlobalFMLWLinkConnRepo &linkConnRepo,
                                 GlobalFMLWLinkDevRepo &linkDevRepo,
                                 FMLWLinkDetailedConnInfo *connInfo,
                                 FMLWLinkTrainType trainTo)
{
    FMFabricParser *pConfig = gfm->mpParser;
    uint64 requestId = 0;
    int retVal = 0;
    FMLWLinkReq linkReq = {{0}};

    // first send train request for the connection

    // fill master end information of the connection
    FMLWLinkEndPointInfo masterEnd = connInfo->getMasterEndPointInfo();
    linkReq.connTrainReq.masterNodeId = masterEnd.nodeId;
    linkReq.connTrainReq.masterGpuOrSwitchId = masterEnd.gpuOrSwitchId;
    linkReq.connTrainReq.masterLinkIndex = masterEnd.linkIndex;
    // fill slave end information of the connection
    FMLWLinkEndPointInfo slaveEnd = connInfo->getSlaveEndPointInfo();
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
        FMLWLinkReqResult reqResult = {0};
        if (linkTrainIntf->isLinkReqComplete( requestId, reqResult )) {
            if (reqResult.status != FM_LWLINK_ST_SUCCESS) {
                // mark the final status as failed.
                std::stringstream outStr;
                connInfo->dumpConnInfo( &outStr, gfm, linkDevRepo);
                retVal = reqResult.status;
                FM_LOG_ERROR("following LWLink connection training failed with error: %s\n %s",
                            FMLWLinkError::getLinkErrorString(reqResult.status), outStr.str().c_str()) ;
                // log connection training failures to syslog
                FM_SYSLOG_ERR("failed to train LWLink connection: %s", outStr.str().c_str());
            }
            // update the training/link status information to the connection/device repo
            updateDeviceAndConnLinkState( linkDevRepo, connInfo, reqResult.connTrainResp );
            break;
        }
        // one iteration is passed. yield the CPU and try again for the request completion
        lwosThreadYield();
    }

    return 0;
}

int
GFMHelper::lwLinkSendResetSwitchLinks(uint32 nodeId,
                                      uint64 switchPhysicalId,
                                      uint64 linkMask,
                                      GlobalFMLWLinkIntf *linkTrainIntf)
{
    FMLWLinkReq linkReq = {{0}};
    FMLWLinkReqResult reqResult = {0};
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

    // wait for the request to finish
    while (true) {
        if (linkTrainIntf->isLinkReqComplete( requestId, reqResult )) {
            if (reqResult.status != FM_LWLINK_ST_SUCCESS) {
                FM_LOG_ERROR("LWLink request to reset LWSwitch links for " NODE_ID_LOG_STR " %d failed with error: %s",
                              nodeId, FMLWLinkError::getLinkErrorString(reqResult.status));
                return reqResult.status;
            }
            break;
        }
        // yield the CPU and try again for the request completion
        lwosThreadYield();
    }

    return 0;
}

int
GFMHelper::lwLinkResetAllSwitchLinks(uint32 nodeId, GlobalFMLWLinkIntf *linkTrainIntf)
{
    FMLWLinkReq linkReq = {{0}};
    FMLWLinkReqResult reqResult = {0};
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
                FM_LOG_ERROR("request to reset all LWSwitch links failed for " NODE_ID_LOG_STR " %d with error: %s",
                              nodeId, FMLWLinkError::getLinkErrorString(reqResult.status));
                return reqResult.status;
            }
            break;
        }
        // yield the CPU and try again for the request completion
        lwosThreadYield();
    }

    return 0;
}

int
GFMHelper::getLWLinkDeviceInfoFromNode(uint32 nodeId,
                                       GlobalFMDevInfoMsgHdlr *devInfoMsgHdlr,
                                       GlobalFMLWLinkDevRepo &linkDevRepo)
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
                FM_LOG_ERROR("failed to get LWLink device information for " NODE_ID_LOG_STR " %d", nodeId);
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
        lwosThreadYield();
    }

    return retVal;

}

int
GFMHelper::getLWLinkDeviceInfoFromAllNodes(FMFabricParser *pConfig,
                                           GlobalFMDevInfoMsgHdlr *devInfoMsgHdlr,
                                           GlobalFMLWLinkDevRepo &linkDevRepo)
{
    std::map<uint32, uint64> requestIds;
    std::map<uint32, uint64>::iterator it;
    std::map <NodeKeyType, NodeConfig *>::iterator nodeit;
    int retVal = 0;

    FM_LOG_INFO("getting LWLink device information");

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
                    FM_LOG_ERROR("failed to get LWLink device information from " NODE_ID_LOG_STR " %d", it->first);
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
            lwosThreadYield();
        }
    }

    return retVal;
}


int
GFMHelper::getLWSwitchDeviceInfoFromNode(uint32 nodeId,
                                         GlobalFMDevInfoMsgHdlr *devInfoMsgHdlr,
                                         FMLWSwitchInfoMap &lwswitchInfoMap,
                                         FMExcludedLWSwitchInfoMap &excludedLwswitchInfoMap)
{
    // get LWSwitch info from one node only
    NodeIDList nodeIdList;
    nodeIdList.push_back( nodeId );

    // call the helper to get info from node
    return getLWSwitchDeviceInfoFromNodes(nodeIdList, devInfoMsgHdlr,
                                          lwswitchInfoMap, excludedLwswitchInfoMap);
}

int
GFMHelper::getLWSwitchDeviceInfoFromAllNodes(FMFabricParser *pConfig,
                                             GlobalFMDevInfoMsgHdlr *devInfoMsgHdlr,
                                             FMLWSwitchInfoMap &lwswitchInfoMap,
                                             FMExcludedLWSwitchInfoMap &excludedLwswitchInfoMap)
{
    NodeIDList nodeIdList;
    std::map <NodeKeyType, NodeConfig *>::iterator nodeit;

    // create a list of node IDs from the parsed fabric config
    for ( nodeit = pConfig->NodeCfg.begin(); nodeit!= pConfig->NodeCfg.end(); nodeit++ ) {
        NodeConfig *pNode = nodeit->second;
        nodeIdList.push_back( pNode->nodeId );
    }

    // call the helper to get info from node
    return getLWSwitchDeviceInfoFromNodes(nodeIdList, devInfoMsgHdlr,
                                          lwswitchInfoMap, excludedLwswitchInfoMap);
}

int
GFMHelper::getGpuDeviceInfoFromNode(uint32 nodeId,
                                    GlobalFMDevInfoMsgHdlr *devInfoMsgHdlr,
                                    FMGpuInfoMap &gpuInfoMap,
                                    FMExcludedGpuInfoMap &excludedGpuInfoMap)
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
                FM_LOG_ERROR("failed to get GPU information from " NODE_ID_LOG_STR " %d", nodeId);
                return retVal;
            } else {
                // request finished. copy the device information.
                copyGpuDeviceInfo(reqResult, gpuInfoMap, excludedGpuInfoMap);
                break;
            }
        }
        // yield the CPU and try again for the request completion
        lwosThreadYield();
    }

    return retVal;
}

bool
GFMHelper::getFMVersionInfoForAllNodes(FMFabricParser *pConfig,
                                       GlobalFMDevInfoMsgHdlr *devInfoMsgHdlr)
{
    std::map<uint32, uint64> requestIds;
    std::map<uint32, uint64>::iterator it;
    std::map <NodeKeyType, NodeConfig *>::iterator nodeit;
    int retVal = 0;

    FM_LOG_INFO("Getting fabric node FM version info");

    // then send get version info request to all the nodes.
    for ( nodeit = pConfig->NodeCfg.begin(); nodeit!= pConfig->NodeCfg.end(); nodeit++ ) {
        NodeConfig *pNode = nodeit->second;
        uint64 requestId = 0;
        devInfoMsgHdlr->sendFMVersionInfoReq( pNode->nodeId, requestId );
        if (requestId) {
            requestIds.insert( std::make_pair(pNode->nodeId, requestId) );
        }
    }

    // wait for all the request to finish. Since requests has timeout, all the requests
    // should eventually finish.
    bool mismatch = false;
    while (true) {
        for ( it = requestIds.begin(); it != requestIds.end(); ) {
            DevInfoReqResult reqResult = {0};
            if (devInfoMsgHdlr->isDevInfoReqComplete( it->second, reqResult )) {
                if (reqResult.status != FM_LWLINK_ST_SUCCESS) {
                    // mark the final status as failed, but continue with rest of the requests
                    retVal = reqResult.status;
                    FM_LOG_ERROR("failed to get fabric node FM version info from " NODE_ID_LOG_STR " %d", it->first);
                } else {
                    // one request is finished. copy the device information.
                    lwswitch::deviceInfoResponseMsg devInfoRsp = reqResult.devInfoRspMsg.devinforsp();
                    lwswitch::nodeVersionInfoRsp versionInfoRsp = devInfoRsp.versioninforsp();
                    std::string recvdVersion = versionInfoRsp.versionstring();

                    if (recvdVersion != FM_VERSION_STRING ) {
                        mismatch = true;
                        FM_LOG_ERROR("global fabric manager version %s is not compatible with local fabric manager "
                                     " version %s running on " NODE_ID_LOG_STR " %d", FM_VERSION_STRING,
                                     recvdVersion.c_str(), it->first);
                        FM_SYSLOG_ERR("global fabric manager version %s is not compatible with local fabric manager "
                                     " version %s running on " NODE_ID_LOG_STR " %d", FM_VERSION_STRING,
                                     recvdVersion.c_str(), it->first);
                    }
                }
                requestIds.erase( it++ );
            }
            else {
                ++it;
            }
        }
        // one iteration is passed. exit if don't have any more request pending
        if ( requestIds.empty() ) {
            break;
        } else {
            // yield the CPU and try again for the request completion
            lwosThreadYield();
        }
    }

    return mismatch;
}

int
GFMHelper::getGpuDeviceInfoFromAllNodes(FMFabricParser *pConfig,
                                        GlobalFMDevInfoMsgHdlr *devInfoMsgHdlr,
                                        FMGpuInfoMap &gpuInfoMap,
                                        FMExcludedGpuInfoMap &excludedGpuInfoMap)
{
    std::map<uint32, uint64> requestIds;
    std::map<uint32, uint64>::iterator it;
    std::map <NodeKeyType, NodeConfig *>::iterator nodeit;
    int retVal = 0;

    FM_LOG_INFO("getting GPU device information");

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
                    FM_LOG_ERROR("failed to get GPU information from " NODE_ID_LOG_STR " %d", it->first);
                } else {
                    // one request is finished. copy the device information.
                    copyGpuDeviceInfo(reqResult, gpuInfoMap, excludedGpuInfoMap);
                }
                requestIds.erase( it++ );
            }
            else {
                ++it;
            }
        }
        // one iteration is passed. exit if don't have any more request pending
        if ( requestIds.empty() ) {
            break;
        } else {
            // yield the CPU and try again for the request completion
            lwosThreadYield();
        }
    }

    return retVal;
}

int
GFMHelper::getGpuLWLinkSpeedInfoFromAllNodes(FMFabricParser *pConfig,
                                             GlobalFMDevInfoMsgHdlr *devInfoMsgHdlr,
                                             FMGpuLWLinkSpeedInfoMap &gpuLinkSpeedInfoMap)
{
    std::map<uint32, uint64> requestIds;
    std::map<uint32, uint64>::iterator it;
    std::map <NodeKeyType, NodeConfig *>::iterator nodeit;
    int retVal = 0;

    FM_LOG_INFO("getting GPU LWLink speed information");

    for ( nodeit = pConfig->NodeCfg.begin(); nodeit!= pConfig->NodeCfg.end(); nodeit++ ) {
        NodeConfig *pNode = nodeit->second;
        uint64 requestId = 0;
        devInfoMsgHdlr->sendGpuLWLinkSpeedInfoReq( pNode->nodeId, requestId );
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
                    FM_LOG_ERROR("failed to get GPU LWLink speed information from " NODE_ID_LOG_STR " %d", it->first);
                } else {
                    // one request is finished. copy the device information.
                    copyGpuLWLinkSpeedInfo(reqResult, gpuLinkSpeedInfoMap);
                }
                requestIds.erase( it++ );
            }
            else {
                ++it;
            }
        }
        // one iteration is passed. exit if don't have any more request pending
        if ( requestIds.empty() ) {
            break;
        } else {
            // yield the CPU and try again for the request completion
            lwosThreadYield();
        }
    }

    return retVal;
}

void
GFMHelper::updateDeviceAndConnEndPointState(GlobalFMLWLinkDevRepo &linkDevRepo,
                                            FMLWLinkDetailedConnInfo *connInfo,
                                            FMLWLinkConnTrainResp &connTrainResp,
                                            bool isMaster)
{

    FMLWLinkEndPointInfo masterEnd = connInfo->getMasterEndPointInfo();
    FMLWLinkEndPointInfo slaveEnd = connInfo->getSlaveEndPointInfo();

    if(isMaster == true) {
        // update connection state
        connInfo->setLinkStateInfoMaster( connTrainResp.masterState);

        // update corresponding device's link state in device repo
        linkDevRepo.setDeviceLinkState(masterEnd.nodeId, masterEnd.gpuOrSwitchId,
                                       masterEnd.linkIndex, connTrainResp.masterState);
        FM_LOG_DEBUG("updateDeviceAndConnEndPointState Master %u,%llu,%u  %u,%u,%u",
                      masterEnd.nodeId, masterEnd.gpuOrSwitchId,masterEnd.linkIndex,
                      connTrainResp.masterState.linkMode, connTrainResp.masterState.txSubLinkMode,
                      connTrainResp.masterState.rxSubLinkMode);
    } else {
        // update connection state
        connInfo->setLinkStateInfoSlave( connTrainResp.slaveState);

        // update corresponding device's link state in device repo
        linkDevRepo.setDeviceLinkState(slaveEnd.nodeId, slaveEnd.gpuOrSwitchId,
                                       slaveEnd.linkIndex, connTrainResp.slaveState);
        FM_LOG_DEBUG("updateDeviceAndConnEndPointState Slave %u,%llu,%u  %u,%u,%u",
                      slaveEnd.nodeId, slaveEnd.gpuOrSwitchId, slaveEnd.linkIndex,
                      connTrainResp.slaveState.linkMode, connTrainResp.slaveState.txSubLinkMode,
                      connTrainResp.slaveState.rxSubLinkMode);
    }
}

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
void
GFMHelper::updateConnEndPointFomValues(FMLWLinkDetailedConnInfo *connInfo,
                                       FMLWLinkFomValues &fomValues,
                                       bool isMaster)
{
    if (isMaster == true) {
        connInfo->setMasterLinkFomValues(fomValues);
    } else {
        connInfo->setSlaveLinkFomValues(fomValues);
    }
}

void
GFMHelper::updateConnEndPointGradingValues(FMLWLinkDetailedConnInfo *connInfo,
                                           FMLWLinkGradingValues &gradingValues,
                                           bool isMaster)
{
    if (isMaster == true) {
        connInfo->setMasterLinkGradingValues(gradingValues);
    } else {
        connInfo->setSlaveLinkGradingValues(gradingValues);
    }
}
#endif

void
GFMHelper::updateConnEndPointLinkQualityInfo(FMLWLinkDetailedConnInfo *connInfo,
                                             FMLWLinkQualityInfo &linkQualityInfo,
                                             bool isMaster)
{
    if (isMaster == true) {
        connInfo->setMasterLinkQualityInfo(linkQualityInfo);
    } else {
        connInfo->setSlaveLinkQualityInfo(linkQualityInfo);
    }
}

int
GFMHelper::lwLinkGetDeviceLwlinkStateForNodes(NodeIDList &nodeIdList,
                                              GlobalFMLWLinkIntf *linkTrainIntf,
                                              GlobalFMLWLinkDevRepo &linkDevRepo)
{
    std::map<uint32, uint64> requestIds;
    std::map<uint32, uint64>::iterator it;
    NodeIDList::iterator nodeIt;

    // first send request to all the active nodes
    for ( nodeIt = nodeIdList.begin(); nodeIt != nodeIdList.end(); nodeIt++ ) {
        uint32_t nodeId = (*nodeIt);
        uint64 requestId = 0;
        FMLWLinkReq linkReq = {{0}};
        linkReq.getDeviceLwlinkStateReq.nodeId = nodeId;
        linkTrainIntf->sendGetDeviceLwlinkStateReq( linkReq, requestId );
        if (requestId) {
            requestIds.insert( std::make_pair(nodeId, requestId) );
        }
    }

    // wait for all the request to finish. Since requests has timeout, all the requests
    // should eventually finish.
    while (true) {
        for (it = requestIds.begin(); it != requestIds.end(); ) {
            FMLWLinkReqResult reqResult = {0};
            if (linkTrainIntf->isLinkReqComplete( it->second, reqResult )) {
                // one request is finished. copy the link state information to caller's map.
                if (reqResult.status != FM_LWLINK_ST_SUCCESS) {
                    FM_LOG_ERROR("getting device LWLink state information failed for " NODE_ID_LOG_STR " %d"
                                 "with error: %s", it->first, FMLWLinkError::getLinkErrorString(reqResult.status));
                } else {
                    // update corresponding device's link state in device repo
                    updateDeviceLinkState(linkDevRepo, reqResult.getDeviceLwlinkStateResp);
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
            lwosThreadYield();
        }
    }

    return 0;
}

void
GFMHelper::updateDeviceAndConnLinkState(GlobalFMLWLinkDevRepo &linkDevRepo,
                                        FMLWLinkDetailedConnInfo *connInfo,
                                        FMLWLinkConnTrainResp &connTrainResp)
{
    // update connection state
    connInfo->setLinkStateInfo( connTrainResp.masterState, connTrainResp.slaveState );

    // update corresponding device's link state in device repo
    FMLWLinkEndPointInfo masterEnd = connInfo->getMasterEndPointInfo();
    FMLWLinkEndPointInfo slaveEnd = connInfo->getSlaveEndPointInfo();

    linkDevRepo.setDeviceLinkState(masterEnd.nodeId, masterEnd.gpuOrSwitchId,
                                   masterEnd.linkIndex, connTrainResp.masterState);

    linkDevRepo.setDeviceLinkState(slaveEnd.nodeId, slaveEnd.gpuOrSwitchId,
                                   slaveEnd.linkIndex, connTrainResp.slaveState);
}

// construct linkConnRepoByDriverId from connections with matching lwlinkDriverId
// in linkConnRepo
void
GFMHelper::getLWLinkConnsRepoByLWLinkDriverId(uint32_t nodeId, uint64_t lwlinkDriverId,
                                              GlobalFMLWLinkConnRepo &linkConnRepo,
                                              GlobalFMLWLinkConnRepo &linkConnRepoByDriverId)
{
    LWLinkIntraConnMap &intraConnMap = linkConnRepo.getIntraConnections();
    LWLinkIntraConnMap::iterator it = intraConnMap.find(nodeId);
    if (it == intraConnMap.end()) {
        // no entry for the specified node
        return;
    }

    FMLWLinkDetailedConnInfoList &intraConnList = it->second;

    // iterate over all the connections in the node
    FMLWLinkDetailedConnInfoList::iterator connIt;
    FMLWLinkConnList driverIdConnList;
    driverIdConnList.clear();

    for (connIt = intraConnList.begin(); connIt != intraConnList.end(); connIt++) {
        FMLWLinkDetailedConnInfo *lwLinkConn = *connIt;

        if ((lwLinkConn->getMasterEndPointInfo().gpuOrSwitchId == lwlinkDriverId) ||
            (lwLinkConn->getSlaveEndPointInfo().gpuOrSwitchId == lwlinkDriverId)) {

            FMLWLinkConnInfo connInfo;
            connInfo.masterEnd = lwLinkConn->getMasterEndPointInfo();
            connInfo.slaveEnd = lwLinkConn->getSlaveEndPointInfo();
            driverIdConnList.push_back(connInfo);
        }
    }

    linkConnRepoByDriverId.addIntraConnections(nodeId, driverIdConnList);
}

uint32_t
GFMHelper::getNumBaseboard(uint32_t nodeId,
                           FMLWSwitchInfoMap &switchInfoMap,
                           FMExcludedLWSwitchInfoMap &excludedSwitchInfoMap)
{
    std::set<uint32_t> boardInfoSet;
    boardInfoSet.clear();

    // get the board info from switches
    FMLWSwitchInfoMap::iterator switchMapIt = switchInfoMap.find(nodeId);
    if (switchMapIt != switchInfoMap.end()) {
        FMLWSwitchInfoList switchInfoList = switchMapIt->second;
        FMLWSwitchInfoList::iterator it;
        for (it = switchInfoList.begin(); it != switchInfoList.end(); it++) {
            FMLWSwitchInfo switchInfo = *it;
            boardInfoSet.insert(getBaseboardSlotNumberFromSwitchPhysicaId(switchInfo.physicalId));
        }
    }

    // get the board info from excluded switches
    FMExcludedLWSwitchInfoMap::iterator excludedSwitchMapIt = excludedSwitchInfoMap.find(nodeId);
    if (excludedSwitchMapIt != excludedSwitchInfoMap.end()) {
        FMExcludedLWSwitchInfoList excludedSwitchInfoList = excludedSwitchMapIt->second;
        FMExcludedLWSwitchInfoList::iterator it;
        for (it = excludedSwitchInfoList.begin(); it != excludedSwitchInfoList.end(); it++) {
            FMExcludedLWSwitchInfo_t switchInfo = *it;
            boardInfoSet.insert(getBaseboardSlotNumberFromSwitchPhysicaId(switchInfo.physicalId));
        }
    }

    return boardInfoSet.size();
}

int
GFMHelper::lwLinkSendSwitchTrainingFailedLinkInfo(uint32 nodeId,
                                                  uint64 switchPhysicalId,
                                                  uint64 attemptedLinkMask0,
                                                  uint64 failedLinkMask0,
                                                  GlobalFMLWLinkIntf *linkTrainIntf)
{
    FMLWLinkReq linkReq = {{0}};
    FMLWLinkReqResult reqResult = {0};
    uint64 requestId = 0;
    int retVal;

    linkReq.switchTrainingFailedLinkInfoReq.nodeId = nodeId;
    linkReq.switchTrainingFailedLinkInfoReq.switchId = switchPhysicalId;
    linkReq.switchTrainingFailedLinkInfoReq.attemptedMask0 = attemptedLinkMask0;
    linkReq.switchTrainingFailedLinkInfoReq.failedMask0 = failedLinkMask0;
    retVal = linkTrainIntf->sendSwitchTrainingFailedLinkInfo( linkReq, requestId );
    if ( !requestId ) {
        // unable to send the request
        return retVal;
    }

    // wait for the request to finish
    while (true) {
        if (linkTrainIntf->isLinkReqComplete( requestId, reqResult )) {
            if (reqResult.status != FM_LWLINK_ST_SUCCESS) {
                FM_LOG_ERROR("request to set LWSwitch link training failed information failed for " NODE_ID_LOG_STR " %d with error: %s",
                              nodeId, FMLWLinkError::getLinkErrorString(reqResult.status));
                return reqResult.status;
            }
            break;
        }
        // yield the CPU and try again for the request completion
        lwosThreadYield();
    }

    return 0;
}

int
GFMHelper::lwLinkInitializeNodes(NodeIDList &nodeIdList,
                                 GlobalFMLWLinkIntf *linkTrainIntf,
                                 GlobalFMLWLinkDevRepo &linkDevRepo)
{
    int retVal;
    bool isMods= false;

#if defined(LW_MODS) && !defined(LW_MODS_GDM_BUILD)
    // The sleep() calls here are for optical link training.
    // Optical link training need to be done only for multi-node systems
    // except in the special case for single node MODS with loopback/loopout optical links
    isMods = true;
#endif


#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    FM_LOG_DEBUG("Init optical links");
    retVal = lwLinkInitDoStepForNodes(nodeIdList, linkTrainIntf, lwswitch::FM_LWLINK_OPTICAL_INIT_LINKS);
    if (retVal) {
        FM_LOG_ERROR( "request to do LWLink initialization failed");
        return retVal;
    }

    FM_LOG_DEBUG("Enable IOBIST");
    retVal = lwLinkInitDoStepForNodes(nodeIdList, linkTrainIntf, lwswitch::FM_LWLINK_OPTICAL_ENABLE_IOBIST);
    if (retVal) {
        FM_LOG_ERROR( "request to enable IOBIST for LWLinks failed");
        return retVal;
    }

    FM_LOG_DEBUG("Start pre-train TX");
    retVal = lwLinkInitDoStepForNodes(nodeIdList, linkTrainIntf, lwswitch::FM_LWLINK_OPTICAL_START_PRETRAIN_TX);
    if (retVal) {
        FM_LOG_ERROR( "request to start TX pre-training for LWLinks failed");
        return retVal;
    }

    if ((nodeIdList.size() > 1 )  || ( isMods == true)) {
        // 3 second sleep needed to allow TX pre-training to complete
        FM_LOG_DEBUG("sleeping %d seconds to allow TX pre-training to complete", 3);
        sleep(3);
    }

    FM_LOG_DEBUG("Check pre-train TX");
    retVal = lwLinkInitDoStepForNodes(nodeIdList, linkTrainIntf, lwswitch::FM_LWLINK_OPTICAL_CHECK_PRETRAIN_TX);
    if (retVal) {
        FM_LOG_ERROR( "request to check TX pre-training completion for LWLinks failed");
        return retVal;
    }

    if ((nodeIdList.size() > 1 )  || ( isMods == true)) {
        FM_LOG_DEBUG("sleeping %d seconds", 1);
        sleep(1);
    }
    FM_LOG_DEBUG("Start pre-train RX");
    retVal = lwLinkInitDoStepForNodes(nodeIdList, linkTrainIntf, lwswitch::FM_LWLINK_OPTICAL_START_PRETRAIN_RX);
    if (retVal) {
        FM_LOG_ERROR( "request to start RX pre-training for LWLinks failed");
        return retVal;
    }

    if ((nodeIdList.size() > 1 )  || ( isMods == true)) {
        // 10 second sleep needed to allow RX pre-training to complete
        FM_LOG_DEBUG("sleeping %d seconds to allow RX pre-training to complete", 10);
        sleep(10);
    }

    FM_LOG_DEBUG("Check pre-train RX");
    retVal = lwLinkInitDoStepForNodes(nodeIdList, linkTrainIntf, lwswitch::FM_LWLINK_OPTICAL_CHECK_PRETRAIN_RX);
    if (retVal) {
        FM_LOG_ERROR( "request to check RX pre-training completion for LWLinks failed");
        return retVal;
    }

    FM_LOG_DEBUG("Stop pre-train");
    retVal = lwLinkInitDoStepForNodes(nodeIdList, linkTrainIntf, lwswitch::FM_LWLINK_OPTICAL_STOP_PRETRAIN);
    if (retVal) {
        FM_LOG_ERROR( "request to stop pre-training for LWLinks failed");
        return retVal;
    }

    if ((nodeIdList.size() > 1 )  || ( isMods == true)) {
        // One second sleep required before disabling IOBIST
        FM_LOG_DEBUG("sleeping %d secondsi, required before disabling IOBIST", 1);
        sleep(1);
    }

    FM_LOG_DEBUG("Disable IOBIST");
    retVal = lwLinkInitDoStepForNodes(nodeIdList, linkTrainIntf, lwswitch::FM_LWLINK_OPTICAL_DISABLE_IOBIST);
    if (retVal) {
        FM_LOG_ERROR( "request to disable IOBIST for LWLinks failed");
        return retVal;
    }
#endif
    FM_LOG_DEBUG("lwLinkInitphase1ForNodes start");
    retVal = lwLinkInitDoStepForNodes(nodeIdList, linkTrainIntf, lwswitch::FM_LWLINK_INITPHASE1);
    if (retVal) {
        FM_LOG_ERROR( "request to do LWLink initialization phase1 failed");
        return retVal;
    }
    FM_LOG_DEBUG("lwLinkRxInitTermForNodes start");
    retVal = lwLinkInitDoStepForNodes(nodeIdList, linkTrainIntf, lwswitch::FM_LWLINK_RX_INIT_TERM);
    if (retVal) {
        FM_LOG_ERROR( "request to do LWLink receiver termination failed");
        return retVal;
    }

    FM_LOG_DEBUG("lwLinkSetRxDetectForNodes start");
    retVal = lwLinkInitDoStepForNodes(nodeIdList, linkTrainIntf, lwswitch::FM_LWLINK_SET_RX_DETECT);
    if (retVal) {
        FM_LOG_ERROR( "request to do LWLink receiver detect failed");
        return retVal;
    }

    FM_LOG_DEBUG("lwLinkGetRxDetectForNodes start");
    retVal = lwLinkInitDoStepForNodes(nodeIdList, linkTrainIntf, lwswitch::FM_LWLINK_GET_RX_DETECT);
    if (retVal) {
        FM_LOG_ERROR( "failed to query output of LWLink receiver detect");
        return retVal;
    }

    FM_LOG_DEBUG("lwLinkEnableCommonModeForNodes start");
    retVal = lwLinkInitDoStepForNodes(nodeIdList, linkTrainIntf, lwswitch::FM_LWLINK_ENABLE_TX_COMMON_MODE);
    if (retVal) {
        FM_LOG_ERROR("request to do LWLink enable common mode failed");
        return retVal;
    }

    FM_LOG_DEBUG("lwLinkCalibrateNodes start");
    retVal = lwLinkInitDoStepForNodes(nodeIdList, linkTrainIntf, lwswitch::FM_LWLINK_CALIBRATE);
    if (retVal) {
        FM_LOG_ERROR("request to do LWLink receiver calibration failed");
        return retVal;
    }
    FM_LOG_DEBUG("lwLinkDisableCommonModeForNodes start");
    retVal = lwLinkInitDoStepForNodes(nodeIdList, linkTrainIntf, lwswitch::FM_LWLINK_DISABLE_TX_COMMON_MODE);
    if (retVal) {
        FM_LOG_ERROR("reques to do LWLink disable common mode failed");
        return retVal;
    }

    FM_LOG_DEBUG("lwLinkInitphase5ForNodes start");
    retVal = lwLinkInitDoStepForNodes(nodeIdList, linkTrainIntf, lwswitch::FM_LWLINK_INITPHASE5);
    if (retVal) {
        FM_LOG_ERROR("request to do LWLink initialization phase 5 failed");
        return retVal;
    }

    FM_LOG_DEBUG("lwLinkEnableDataForNodes start");
    retVal = lwLinkInitDoStepForNodes(nodeIdList, linkTrainIntf, lwswitch::FM_LWLINK_ENABLE_DATA);
    if (retVal) {
        FM_LOG_ERROR("request to do LWLink enable data failed");
        return retVal;
    }
    FM_LOG_DEBUG("lwLinkInitLinkForNodes start");
    retVal = lwLinkInitDoStepForNodes(nodeIdList, linkTrainIntf, lwswitch::FM_LWLINK_INIT);
    if (retVal) {
        FM_LOG_ERROR("request to do start link initialize failed");
        return retVal;
    }

    FM_LOG_DEBUG("lwLinkGetLinkInitStatusForNodes start");
    retVal = lwLinkGetLinkInitStatusForNodes(nodeIdList, linkTrainIntf, linkDevRepo);
    if (retVal) {
        FM_LOG_ERROR("request to get link initialization status failed");
        return retVal;
    }

    FM_LOG_DEBUG("lwLinkInitnegotiateForNodes start");
    retVal = lwLinkInitDoStepForNodes(nodeIdList, linkTrainIntf, lwswitch::FM_LWLINK_INITNEGOTIATE);
    if (retVal) {
        FM_LOG_ERROR( "request to do LWLink init negotiate failed");
        return retVal;
    }
    FM_LOG_DEBUG("lwLinkInitnegotiateForNodes end");

    return 0;
}
const char*
GFMHelper::getLWLinkInitTypeString( lwswitch::FabricManagerMessageType msgType )
{

    switch( msgType ) {
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        case lwswitch::FM_LWLINK_OPTICAL_INIT_LINKS:
            return "LWLink initialization";
        case lwswitch::FM_LWLINK_OPTICAL_ENABLE_IOBIST:
            return "enable IOBIST";
        case lwswitch::FM_LWLINK_OPTICAL_START_PRETRAIN_TX:
            return "start TX pre-training";
        case lwswitch::FM_LWLINK_OPTICAL_CHECK_PRETRAIN_TX:
            return "check TX pre-training completion";
        case lwswitch::FM_LWLINK_OPTICAL_START_PRETRAIN_RX:
            return "start RX pre-training";
        case lwswitch::FM_LWLINK_OPTICAL_CHECK_PRETRAIN_RX:
            return "check RX pre-training completion";
        case lwswitch::FM_LWLINK_OPTICAL_STOP_PRETRAIN:
            return "stop pre-training";
        case lwswitch::FM_LWLINK_OPTICAL_DISABLE_IOBIST:
            return "disable IOBIST";
#endif
        case lwswitch::FM_LWLINK_INITPHASE1:
            return "initialization phase 1";
        case lwswitch::FM_LWLINK_RX_INIT_TERM:
            return "receiver termination";
        case lwswitch::FM_LWLINK_SET_RX_DETECT:
            return "receiver detect";
        case lwswitch::FM_LWLINK_GET_RX_DETECT:
            return "query receiver detect";
        case lwswitch::FM_LWLINK_ENABLE_TX_COMMON_MODE:
            return "enable common mode";
        case lwswitch::FM_LWLINK_CALIBRATE:
            return "receiver calibration";
        case lwswitch::FM_LWLINK_DISABLE_TX_COMMON_MODE:
            return "disable common mode";
        case lwswitch::FM_LWLINK_INITPHASE5:
            return "initialization phase 5";
        case lwswitch::FM_LWLINK_ENABLE_DATA:
            return "enable data";
        case lwswitch::FM_LWLINK_INIT:
            return "start link initialize";
        case lwswitch::FM_LWLINK_INITNEGOTIATE:
            return "init negotiate";
        default:
            // no train type case matched. shouldn't happen
            return "Unknown init type";
    }
}

int
GFMHelper::lwLinkInitDoStepForNodes(NodeIDList &nodeIdList,
                                    GlobalFMLWLinkIntf *linkTrainIntf,
                                    lwswitch::FabricManagerMessageType msgType)
{
    std::list<uint64> requestIds;
    NodeIDList::iterator nodeIt;
    int retVal = 0;

    FM_LOG_INFO("sending request to do LWLink initialization step %s", getLWLinkInitTypeString(msgType) );

    // first send request to all the active nodes
    for ( nodeIt = nodeIdList.begin(); nodeIt != nodeIdList.end(); nodeIt++ ) {
        uint32_t nodeId = (*nodeIt);
        uint64 requestId = 0;
        FMLWLinkReq linkReq = {{0}};
        linkReq.nodeInitReq.nodeId = nodeId;
        linkTrainIntf->sendTrainRequest(msgType, linkReq, requestId);
        if (requestId) {
            requestIds.push_back( requestId );
        }
    }

    retVal = waitForLinkRequestToComplete( linkTrainIntf, requestIds, "LWLink initialization step" );
    return retVal;
}

int
GFMHelper::lwLinkGetLinkInitStatusForNodes(NodeIDList &nodeIdList,
                                           GlobalFMLWLinkIntf *linkTrainIntf,
                                           GlobalFMLWLinkDevRepo &linkDevRepo)
{
    std::map<uint32, uint64> requestIds;
    std::map<uint32, uint64>::iterator it;
    NodeIDList::iterator nodeIt;

    FM_LOG_INFO("checking LWLink initialization/safe mode transition status");

    // then send get connection request to all the nodes.
    for ( nodeIt = nodeIdList.begin(); nodeIt != nodeIdList.end(); nodeIt++ ) {
        uint32_t nodeId = (*nodeIt);
        uint64 requestId = 0;
        FMLWLinkReq linkReq = {{0}};
        linkReq.nodeInitStatusReq.nodeId = nodeId;
        linkTrainIntf->sendTrainRequest(lwswitch::FM_LWLINK_INIT_STATUS, linkReq, requestId );
        if (requestId) {
            requestIds.insert( std::make_pair(nodeId, requestId) );
        }
    }

    // wait for all the request to finish. Since requests has timeout, all the requests
    // should eventually finish.
    while (true) {
        for ( it = requestIds.begin(); it != requestIds.end(); ) {
            FMLWLinkReqResult reqResult = {0};
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
            lwosThreadYield();
        }
    }

    return 0;
}

int
GFMHelper::lwLinkGetIntraNodeConns(NodeIDList &nodeIdList,
                                   GlobalFMLWLinkIntf *linkTrainIntf,
                                   GlobalFMLWLinkConnRepo &linkConnRepo)
{
    std::map<uint32, uint64> requestIds;
    std::map<uint32, uint64>::iterator it;
    NodeIDList::iterator nodeIt;

    // send get connection request to all the nodes.
    for ( nodeIt = nodeIdList.begin(); nodeIt != nodeIdList.end(); nodeIt++ ) {
        uint32_t nodeId = (*nodeIt);
        uint64 requestId = 0;
        FMLWLinkReq linkReq = {{0}};
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
            FMLWLinkReqResult reqResult = {0};
            if (linkTrainIntf->isLinkReqComplete( it->second, reqResult )) {
                // one request is finished. copy the connections to caller's map.
                if (reqResult.status != FM_LWLINK_ST_SUCCESS) {
                    FM_LOG_ERROR("LWLink connection discovery failed for " NODE_ID_LOG_STR " %d with error: %s",
                                it->first, FMLWLinkError::getLinkErrorString(reqResult.status));
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
            lwosThreadYield();
        }
    }

    return 0;
}

int
GFMHelper::lwLinkSendDiscoverConnOnNodes(NodeIDList &nodeIdList,
                                         GlobalFMLWLinkIntf *linkTrainIntf)
{
    std::list<uint64> requestIds;
    NodeIDList::iterator nodeIt;
    int retVal = 0;

    // this routine will initiate LWLink connection discovery

    // first send request to all the active nodes
    for ( nodeIt = nodeIdList.begin(); nodeIt != nodeIdList.end(); nodeIt++ ) {
        uint32_t nodeId = (*nodeIt);
        uint64 requestId = 0;
        FMLWLinkReq linkReq = {{0}};
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
GFMHelper::lwLinkWriteDiscoveryToken(uint32 nodeId,
                                     GlobalFMLWLinkIntf *linkTrainIntf,
                                     FMLWLinkWriteDiscoveryTokenResp &writeTokenResp)
{
    FMLWLinkReq linkReq = {{0}};
    FMLWLinkReqResult reqResult = {0};
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
                FM_LOG_ERROR("LWLink write discovery token failed for " NODE_ID_LOG_STR " %d with error: %s",
                            nodeId, FMLWLinkError::getLinkErrorString(reqResult.status));
                return reqResult.status;
            }
            // copy the result and token information
            writeTokenResp = reqResult.writeDiscoveryTokenResp;
            break;
        }
        // yield the CPU and try again for the request completion
        lwosThreadYield();
    }
    return 0;
}

int
GFMHelper::lwLinkReadLinkSids( GlobalFMLWLinkIntf *linkTrainIntf,
                               FMFabricParser *pConfig,
                               FMLWLinkSidList &interNodeLinkSidList,
                               std::map<uint64, uint32> &sidToNodeIdMap,
                               std::map<uint64, uint64> &sidToGpuOrSwitchIdMap )
{
    std::map< uint32, uint64 > requestIds;
    std::map< uint32, uint64 >::iterator it;
    std::map < NodeKeyType, NodeConfig * >::iterator nodeit;
    int retVal = 0;

    // first send requests to all the active nodes
    for ( nodeit = pConfig->NodeCfg.begin(); nodeit!= pConfig->NodeCfg.end(); nodeit++ ) {
        NodeConfig *pNode = nodeit->second;
        uint64 requestId = 0;
        FMLWLinkReq linkReq = { { 0 } };
        linkReq.readSidReq.nodeId = pNode->nodeId;
        FM_LOG_DEBUG("sendReadSidReq");
        linkTrainIntf->sendReadSidReq( linkReq, requestId );
        if ( requestId ) {
            requestIds.insert(std::make_pair( pNode->nodeId, requestId ) );
        }
    }

    // wait for all the requests to finish. Since requests have timeouts, all the requests
    // should eventually finish.
    while ( true ) {
        for ( it = requestIds.begin(); it != requestIds.end(); ) {
            FMLWLinkReqResult reqResult = { 0 };
            if ( linkTrainIntf->isLinkReqComplete( it->second, reqResult ) ) {
                // one request is finished. copy the connections to caller's map.
                if ( reqResult.status != FM_LWLINK_ST_SUCCESS ) {
                    retVal = reqResult.status;
                    FM_LOG_ERROR( "LWLink read SID failed for " NODE_ID_LOG_STR " %d with error: %s",
                                 it->first, FMLWLinkError::getLinkErrorString( reqResult.status ) );
                } else {
                    FMLWLinkSidList::iterator it;
                    for ( it = reqResult.readSidResp.sidList.begin(); it != reqResult.readSidResp.sidList.end(); it++ ) {
                        interNodeLinkSidList.push_back( *it );
                        sidToNodeIdMap[ it->nearSid ] = it->nodeId;
                        sidToGpuOrSwitchIdMap[ it->nearSid ] = it->gpuOrSwitchId;
                        FM_LOG_DEBUG( "sid=%llu: nodeId=%d", it->nearSid, it->nodeId );
                        FM_LOG_DEBUG( "sid=%llu: gpuOrSwitchId=%llu", it->nearSid, it->gpuOrSwitchId );
                    }

                }
                requestIds.erase( it++ );
            } else {
                ++it;
            }
        }
        // one iteration is passed. exit if no more requests are pending
        if ( requestIds.empty() ) {
            break;
        } else {
            // yield the CPU and check again for request completion
            lwosThreadYield();
        }
    }
    return retVal;
}

int
GFMHelper::lwLinkReadDiscoveryToken(uint32 lwrWriteTokenNodeId,
                                    GlobalFMLWLinkIntf *linkTrainIntf,
                                    FMFabricParser *pConfig,
                                    std::map<uint32, FMLWLinkReadDiscoveryTokenResp> &readTokenResps)
{
    int retVal = FM_LWLINK_ST_SUCCESS;
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
        FMLWLinkReq linkReq = {{0}};
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
            FMLWLinkReqResult reqResult = {0};
            if (linkTrainIntf->isLinkReqComplete( it->second, reqResult )) {
                // one request is finished. copy the connections to caller's map.
                if (reqResult.status != FM_LWLINK_ST_SUCCESS) {
                    retVal = reqResult.status;
                    FM_LOG_ERROR("LWLink read discovery token failed for " NODE_ID_LOG_STR " %d with error: %s",
                                it->first, FMLWLinkError::getLinkErrorString(reqResult.status));
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
            lwosThreadYield();
        }
    }
    return retVal;
}

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
void
GFMHelper::lwLinkCorrelateLinkSids( FMLWLinkSidList &interNodeLinkSidList,
                                    std::map<uint64, uint32> &sidToNodeIdMap,
                                    std::map<uint64, uint64> &sidToGpuOrSwitchIdMap,
                                    GlobalFMLWLinkConnRepo &linkConnRepo )
{
    for ( auto it = interNodeLinkSidList.begin(); it != interNodeLinkSidList.end(); it++ ) {
        FMLWLinkConnInfo connInfo;
        connInfo.masterEnd.nodeId = sidToNodeIdMap[ it->nearSid ];
        connInfo.slaveEnd.nodeId = sidToNodeIdMap[ it->farSid ];
        connInfo.masterEnd.gpuOrSwitchId = sidToGpuOrSwitchIdMap[ it->nearSid ];
        connInfo.slaveEnd.gpuOrSwitchId = sidToGpuOrSwitchIdMap[ it->farSid ];
        connInfo.masterEnd.linkIndex = it->nearLinkIndex ;
        connInfo.slaveEnd.linkIndex = it->farLinkIndex ;
        if ( !lwLinkIsInterNodeConnectionExists( linkConnRepo, connInfo ) ) {
            if(connInfo.masterEnd.nodeId != connInfo.slaveEnd.nodeId) {
                linkConnRepo.addInterConnections( connInfo );
                FM_LOG_DEBUG("addInterConnection (mnodeId=%d, mgpuOrSwitchId= %llu mLink= %d) (snodeId=%d, sgpuOrSwitchId=%llu mLink= %d)",
                             connInfo.masterEnd.nodeId, connInfo.masterEnd.gpuOrSwitchId, connInfo.masterEnd.linkIndex, connInfo.slaveEnd.nodeId, connInfo.slaveEnd.gpuOrSwitchId, connInfo.slaveEnd.linkIndex );
            }
        }
    }
}

void
GFMHelper::lwLinkCorrelateConnections(FMLWLinkWriteDiscoveryTokenResp &writeTokenResp,
                                      std::map<uint32, FMLWLinkReadDiscoveryTokenResp> &readTokenResps,
                                      GlobalFMLWLinkConnRepo &linkConnRepo)
{
    // compare each discovery token wrote with the tokens read from other nodes

    FMLWLinkDiscoveryTokenList readTokenList;
    // create a list of read tokens for easy parsing
    std::map<uint32, FMLWLinkReadDiscoveryTokenResp>::iterator it;
    for ( it = readTokenResps.begin(); it != readTokenResps.end(); it++ ) {
        FMLWLinkReadDiscoveryTokenResp readResp = it->second;
        FMLWLinkDiscoveryTokenList ::iterator jit = readResp.tokenInfo.begin();
        for ( ; jit != readResp.tokenInfo.end(); jit++ ) {
            readTokenList.push_back( *jit );
        }
    }

    FMLWLinkDiscoveryTokenList::iterator wit;
    FMLWLinkDiscoveryTokenList::iterator rit;
    for ( wit = writeTokenResp.tokenInfo.begin(); wit != writeTokenResp.tokenInfo.end();) {
        bool bFound = false;
        for ( rit = readTokenList.begin(); rit != readTokenList.end();) {
            FMLinkDiscoveryTokenInfo writeDevInfo = (*wit);
            FMLinkDiscoveryTokenInfo readDevInfo = (*rit);
            if ( writeDevInfo.tokelwalue == readDevInfo.tokelwalue ) {
                // found a connection
                FMLWLinkConnInfo connInfo;
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
        } // end of readTokenList iteration

        // if we found a match, remove this element from our writeToeknList.
        if (bFound == true) {
            wit = writeTokenResp.tokenInfo.erase( wit );
        } else {
            // fetch next write token and compare
            wit++;
        }
    }
}

bool
GFMHelper::lwLinkIsInterNodeConnectionExists(GlobalFMLWLinkConnRepo &linkConnRepo,
                                             FMLWLinkConnInfo &newConn)
{
    // in inter-node connection, the discovery token can match for same connection
    // as we are reading and comparing on all nodes. ie match both ways. but
    // there is only one connection.
    LWLinkInterNodeConns::iterator it;
    LWLinkInterNodeConns interConns = linkConnRepo.getInterConnections();
    for ( it = interConns.begin(); it != interConns.end(); it++ ) {
        FMLWLinkDetailedConnInfo *tempConn = *it;
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
#endif

int
GFMHelper::waitForLinkRequestToComplete(GlobalFMLWLinkIntf *linkTrainIntf,
                                        std::list<uint64> requestIds,
                                        std::string errorCtx)
{
    std::list<uint64>::iterator it;
    int retVal = 0;

    // wait for all the request to finish. Since requests has timeout, all the requests
    // should eventually finish.
    while (true) {
        for ( it = requestIds.begin(); it != requestIds.end(); ) {
            FMLWLinkReqResult reqResult = {0};
            if (linkTrainIntf->isLinkReqComplete( *it, reqResult )) {
                it = requestIds.erase( it );
                if (reqResult.status != FM_LWLINK_ST_SUCCESS) {
                    retVal = reqResult.status;
                    FM_LOG_ERROR("%s failed with error: %s",
                                errorCtx.c_str(), FMLWLinkError::getLinkErrorString(reqResult.status));
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
            lwosThreadYield();
        }
    }

    return retVal;
}

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
void
GFMHelper::genAddInterNodeConnLinkReqMsg(GlobalFMLWLinkDevRepo &linkDevRepo,
                                         FMLWLinkReq &linkReq,
                                         FMLWLinkEndPointInfo &localEndInfo,
                                         FMLWLinkEndPointInfo &remoteEndInfo)
{
    FMLWLinkDevInfo devInfo;
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
#endif

void
GFMHelper::copyLWSwitchDeviceInfo(DevInfoReqResult &reqResult,
                                  FMLWSwitchInfoMap &lwswitchInfoMap,
                                  FMExcludedLWSwitchInfoMap &excludedLwswitchInfoMap)
{
    uint32 nodeId = reqResult.devInfoRspMsg.nodeid();
    lwswitch::deviceInfoResponseMsg devInfoRsp = reqResult.devInfoRspMsg.devinforsp();
    lwswitch::lwswitchDeviceInfoRsp switchRsp = devInfoRsp.switchdevrsp();

    FMLWSwitchInfoList switchInfoList;
    FMExcludedLWSwitchInfoList excludedLwswitchInfoList;
    switchInfoList.clear();
    // build the list of switch info for this node
    for (int idx = 0; idx < switchRsp.switchinfo_size(); idx++) {
        lwswitch::lwswitchDeviceInfoMsg infoMsg = switchRsp.switchinfo(idx);
        const lwswitch::devicePciInfo &pciInfoMsg = infoMsg.pciinfo();
        FMLWSwitchInfo switchInfo = {0};
        switchInfo.switchIndex = infoMsg.switchindex();
        switchInfo.physicalId = infoMsg.physicalid();
        switchInfo.enabledLinkMask = infoMsg.enabledlinkmask();
        switchInfo.archType = infoMsg.archtype();
        switchInfo.pciInfo.domain = pciInfoMsg.domain();
        switchInfo.pciInfo.bus = pciInfoMsg.bus();
        switchInfo.pciInfo.device = pciInfoMsg.device();
        switchInfo.pciInfo.function = pciInfoMsg.function();
        snprintf(switchInfo.pciInfo.busId, FM_DEVICE_PCI_BUS_ID_BUFFER_SIZE,
                 FM_DEVICE_PCI_BUS_ID_FMT, FM_DEVICE_PCI_BUS_ID_FMT_ARGS(&switchInfo.pciInfo));
        if ( infoMsg.has_uuid() ) {
            memset(switchInfo.uuid.bytes, 0, sizeof(switchInfo.uuid.bytes));
            strncpy( switchInfo.uuid.bytes, infoMsg.uuid().c_str(), sizeof(switchInfo.uuid) - 1);
        }
        switchInfoList.push_back(switchInfo);
    }

    // add to the map of switch information, which is per node
    lwswitchInfoMap.insert( std::make_pair(nodeId, switchInfoList) );

    // build the list of excluded lwswitch info for this node
    excludedLwswitchInfoList.clear();
    for (int idx = 0; idx < switchRsp.excludedswitchinfo_size(); idx++) {
        lwswitch::lwswitchDeviceInfoMsg infoMsg = switchRsp.excludedswitchinfo(idx);
        const lwswitch::devicePciInfo &pciInfoMsg = infoMsg.pciinfo();
        FMExcludedLWSwitchInfo_t switchInfo;
        memset(&switchInfo, 0, sizeof(FMExcludedLWSwitchInfo_t));
        switchInfo.physicalId = infoMsg.has_physicalid() ? infoMsg.physicalid() : infoMsg.switchindex();
        switchInfo.pciInfo.domain = pciInfoMsg.domain();
        switchInfo.pciInfo.bus = pciInfoMsg.bus();
        switchInfo.pciInfo.device = pciInfoMsg.device();
        switchInfo.pciInfo.function = pciInfoMsg.function();
        snprintf(switchInfo.pciInfo.busId, FM_DEVICE_PCI_BUS_ID_BUFFER_SIZE,
                 FM_DEVICE_PCI_BUS_ID_FMT, FM_DEVICE_PCI_BUS_ID_FMT_ARGS(&switchInfo.pciInfo));
        if ( infoMsg.has_uuid() ) {
            memset(switchInfo.uuid.bytes, 0, sizeof(switchInfo.uuid.bytes));
            strncpy( switchInfo.uuid.bytes, infoMsg.uuid().c_str(), sizeof(switchInfo.uuid) - 1);
        }
        switchInfo.excludedReason = infoMsg.excludedreason();
        excludedLwswitchInfoList.push_back(switchInfo);
    }

    // add to the map of excluded lwswitch information, which is per node
    excludedLwswitchInfoMap.insert( std::make_pair(nodeId, excludedLwswitchInfoList) );
}

void
GFMHelper::copyGpuDeviceInfo(DevInfoReqResult &reqResult,
                             FMGpuInfoMap &gpuInfoMap,
                             FMExcludedGpuInfoMap &excludedGpuInfoMap)
{
    uint32 nodeId = reqResult.devInfoRspMsg.nodeid();
    lwswitch::deviceInfoResponseMsg devInfoRsp = reqResult.devInfoRspMsg.devinforsp();
    lwswitch::gpuDeviceInfoRsp gpuRsp = devInfoRsp.gpudevrsp();
    FMGpuInfoList gpuInfoList;
    FMExcludedGpuInfoList excludedGpuInfoList;

    // build the list of gpu info for this node
    gpuInfoList.clear();
    for (int idx = 0; idx < gpuRsp.gpuinfo_size(); idx++) {
        lwswitch::gpuDeviceInfoMsg infoMsg = gpuRsp.gpuinfo(idx);
        const lwswitch::devicePciInfo &pciInfoMsg = infoMsg.pciinfo();
        FMGpuInfo_t gpuInfo = {0};
        gpuInfo.gpuIndex = infoMsg.gpuindex();
        gpuInfo.pciInfo.domain = pciInfoMsg.domain();
        gpuInfo.pciInfo.bus = pciInfoMsg.bus();
        gpuInfo.pciInfo.device = pciInfoMsg.device();
        gpuInfo.pciInfo.function = pciInfoMsg.function();
        snprintf(gpuInfo.pciInfo.busId, FM_DEVICE_PCI_BUS_ID_BUFFER_SIZE,
                 FM_DEVICE_PCI_BUS_ID_FMT, FM_DEVICE_PCI_BUS_ID_FMT_ARGS(&gpuInfo.pciInfo));
        if ( infoMsg.has_uuid() ) {
            memset(gpuInfo.uuid.bytes, 0, sizeof(gpuInfo.uuid.bytes));
            strncpy( gpuInfo.uuid.bytes, infoMsg.uuid().c_str(), sizeof(gpuInfo.uuid) - 1);
        }
        gpuInfo.discoveredLinkMask = infoMsg.discoveredlinkmask();
        gpuInfo.enabledLinkMask = infoMsg.enabledlinkmask();
        gpuInfo.archType = infoMsg.archtype();
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
        gpuInfo.isEgmCapable = infoMsg.isegmcapable();
        gpuInfo.isSpaCapable = infoMsg.isspacapable();
        if ( gpuInfo.isSpaCapable ) {
            gpuInfo.spaAddress = infoMsg.spaaddress();
        }
#endif
        gpuInfoList.push_back(gpuInfo);
    }

    // add to the map of gpu information, which is per node
    gpuInfoMap.insert( std::make_pair(nodeId, gpuInfoList) );

    // build the list of excluded gpu info for this node
    excludedGpuInfoList.clear();
    for (int idx = 0; idx < gpuRsp.excludedgpuinfo_size(); idx++) {
        lwswitch::gpuDeviceInfoMsg infoMsg = gpuRsp.excludedgpuinfo(idx);
        const lwswitch::devicePciInfo &pciInfoMsg = infoMsg.pciinfo();
        FMExcludedGpuInfo_t gpuInfo = {{0}};
        // gpuInfo.gpuIndex = infoMsg.gpuindex();
        gpuInfo.pciInfo.domain = pciInfoMsg.domain();
        gpuInfo.pciInfo.bus = pciInfoMsg.bus();
        gpuInfo.pciInfo.device = pciInfoMsg.device();
        gpuInfo.pciInfo.function = pciInfoMsg.function();
        snprintf(gpuInfo.pciInfo.busId, FM_DEVICE_PCI_BUS_ID_BUFFER_SIZE,
                 FM_DEVICE_PCI_BUS_ID_FMT, FM_DEVICE_PCI_BUS_ID_FMT_ARGS(&gpuInfo.pciInfo));
        if ( infoMsg.has_uuid() ) {
            memset(gpuInfo.uuid.bytes, 0, sizeof(gpuInfo.uuid.bytes));
            strncpy( gpuInfo.uuid.bytes, infoMsg.uuid().c_str(), sizeof(gpuInfo.uuid) - 1);
        }
        excludedGpuInfoList.push_back(gpuInfo);
    }

    // add to the map of excluded gpu information, which is per node
    excludedGpuInfoMap.insert( std::make_pair(nodeId, excludedGpuInfoList) );
}

void
GFMHelper::copyGpuLWLinkSpeedInfo(DevInfoReqResult &reqResult,
                                  FMGpuLWLinkSpeedInfoMap &gpuLinkSpeedInfoMap)
{
    uint32 nodeId = reqResult.devInfoRspMsg.nodeid();
    lwswitch::deviceInfoResponseMsg devInfoRsp = reqResult.devInfoRspMsg.devinforsp();
    lwswitch::gpuLWLinkSpeedInfoRsp gpuSpeedInfoRsp = devInfoRsp.gpulinkspeedrsp();

    FMGpuLWLinkSpeedInfoList speedInfoList;
    speedInfoList.clear();
    // build the list of switch info for this node
    for ( int idx = 0; idx < gpuSpeedInfoRsp.gpulinkspeedinfo_size(); idx++ ) {
        lwswitch::gpuLWLinkSpeedInfoMsg speedInfoMsg = gpuSpeedInfoRsp.gpulinkspeedinfo(idx);
        FMGpuLWLinkSpeedInfo speedInfo;
        if ( speedInfoMsg.has_uuid() ) {
            memset(speedInfo.uuid.bytes, 0, sizeof(speedInfo.uuid.bytes));
            strncpy( speedInfo.uuid.bytes, speedInfoMsg.uuid().c_str(), sizeof(speedInfo.uuid) - 1);
        }
        for ( int j = 0; j < speedInfoMsg.speedinfo_size(); j++ ) {
            lwswitch::lwLinkSpeedInfoMsg infoMsg = speedInfoMsg.speedinfo(j);
            FMLWLinkSpeedInfo tempSpeedInfo;
            tempSpeedInfo.linkIndex = infoMsg.linkindex();
            tempSpeedInfo.linkLineRateMBps = infoMsg.linklineratembps();
            tempSpeedInfo.linkClockMhz = infoMsg.linkclockmhz();
            tempSpeedInfo.linkClockType = infoMsg.linkclocktype();
            tempSpeedInfo.linkDataRateKiBps = infoMsg.linkdataratekibps();
            speedInfo.linkSpeedInfo.push_back( tempSpeedInfo );
        }
        speedInfoList.push_back( speedInfo );
    }

    // add to the map of switch information, which is per node
    gpuLinkSpeedInfoMap.insert( std::make_pair(nodeId, speedInfoList) );
}


const char*
GFMHelper::getLWLinkTrainTypeString(FMLWLinkTrainType trainType)
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
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        case LWLINK_TRAIN_SAFE_TO_INITOPTIMIZE:
            return "safe to initoptimize";
        case LWLINK_TRAIN_POST_INITOPTIMIZE:
            return "post initoptimize";
        case LWLINK_TRAIN_INTERNODE_PARALLEL_INITOPTIMIZE_TO_HIGH:
            return "initoptimize to high speed parallel";
        case LWLINK_TRAIN_INTERNODE_PARALLEL_TO_OFF:
            return "turn links off";
        case LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_INF_MODE:
            return "enable infinite PRBS";
        case LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_MAINTENANCE_TX:
            return "enable maintenance TX";
        case LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_MAINTENANCE_RX:
            return "enable maintenance RX";
        case LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_DISABLE_INF_MODE:
            return "disable infinite PRBS";
        case LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_FORCE_EQ:
            return "enable force eq";
        case LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_DISABLE_FORCE_EQ:
            return "disable force eq";
        case LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_CHECK_EOM_STATUS:
            return "get link EOM states";
        case LWLINK_TRAIN_INTERNODE_GET_GRADING_AND_FOM_VALUES:
            return "get grading and FOM values";
        case LWLINK_TRAIN_INTERNODE_PARALLEL_GET_LINK_STATE:
            return "get link states";
#endif
		default:
            return "Unknown train type";
    }
    // no train type case matched. shouldn't happen
    return "Unknown train type";
}

void
GFMHelper::getTrainTypeVector(FMLWLinkTrainType trainType, vector<FMLWLinkTrainType> &trainTypes)
{
    switch (trainType) {
        case LWLINK_TRAIN_OFF_TO_SAFE: {
            trainTypes = {LWLINK_TRAIN_OFF_TO_SAFE_SUBLINK, LWLINK_TRAIN_OFF_TO_SAFE_MAINLINK};
            return;
        }
        case LWLINK_TRAIN_SAFE_TO_HIGH: {
            trainTypes = {LWLINK_TRAIN_SAFE_TO_HIGH_SUBLINK, LWLINK_TRAIN_SAFE_TO_HIGH_MAINLINK};
            return;
        }
        case LWLINK_TRAIN_HIGH_TO_SAFE: {
            trainTypes = {LWLINK_TRAIN_HIGH_TO_SAFE_SUBLINK, LWLINK_TRAIN_HIGH_TO_SAFE_MAINLINK};
            return;
        }
        default:
            break;
    }
}

int
GFMHelper::getLWSwitchDeviceInfoFromNodes(NodeIDList &nodeIdList,
                                          GlobalFMDevInfoMsgHdlr *devInfoMsgHdlr,
                                          FMLWSwitchInfoMap &lwswitchInfoMap,
                                          FMExcludedLWSwitchInfoMap &excludedLwswitchInfoMap)
{
    std::map<uint32, uint64> requestIds;
    std::map<uint32, uint64>::iterator it;
    NodeIDList::iterator nodeIt;
    int retVal = 0;

    FM_LOG_INFO("getting LWSwitch device information");

    // then send get connection request to all the nodes.
    for ( nodeIt = nodeIdList.begin(); nodeIt != nodeIdList.end(); nodeIt++ ) {
        uint32_t nodeId = (*nodeIt);
        uint64 requestId = 0;
        devInfoMsgHdlr->sendLWSwitchDevInfoReq( nodeId, requestId );
        if (requestId) {
            requestIds.insert( std::make_pair(nodeId, requestId) );
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
                    FM_LOG_ERROR("failed to get LWSwitch Device info from " NODE_ID_LOG_STR " %d", it->first);
                } else {
                    // one request is finished. copy the device information.
                    copyLWSwitchDeviceInfo(reqResult, lwswitchInfoMap, excludedLwswitchInfoMap);
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
            lwosThreadYield();
        }
    }

    return retVal;
}

uint32_t
GFMHelper::getBaseboardSlotNumberFromSwitchPhysicaId(uint32_t physicalId)
{
    // based on current document
    // LWSwitch GPIO pysicalId bits[6:3] is "Position/Type" of board

    // TODO if above changes in the future platforms, this helper functions need to change accordingly
    uint32_t boardInfoMask  = 0x78;
    uint32_t boardInfoShift = 3;
    uint32_t boardInfo = (physicalId & boardInfoMask) >> boardInfoShift;

    return boardInfo;
}

void
GFMHelper::updateDeviceLinkState(GlobalFMLWLinkDevRepo &linkDevRepo, FMLWLinkGetDeviceLwlinkStateResp &deviceLwlinkStateList)
{
    FMLWLinkGetDeviceLwlinkStateResp::iterator jit;

    for ( jit = deviceLwlinkStateList.begin(); jit != deviceLwlinkStateList.end(); jit++ ) {
        FMLWLinkGetDeviceLwlinkStateRespDetailed deviceLwlinkStateInfo = (*jit);
        linkDevRepo.setDeviceLinkState(deviceLwlinkStateInfo.lwEndInfo.nodeId, deviceLwlinkStateInfo.lwEndInfo.gpuOrSwitchId,
                                       deviceLwlinkStateInfo.lwEndInfo.linkIndex, deviceLwlinkStateInfo.stateInfo);
    }
}

// Log the error and throw an exception when the errorOclwrred is the same as errorCode
void
GFMHelper::logErrAndThrowException(int errorOclwrred, int errCode, const char *errMessage)
{
    if (errorOclwrred == errCode) {
        FM_LOG_ERROR("%s", errMessage);
        FM_SYSLOG_ERR("%s", errMessage);
        throw std::runtime_error(errMessage);
    }
}


// colwert LWSWITCH_GET_INFO_INDEX_ARCH_X defined in ctrl_dev_lwswitch.h
// to lwSwitchArchType
lwSwitchArchType
GFMHelper::driverToLwSwitchArchType(uint32_t driverArchType)
{
    lwSwitchArchType arch;

    switch (driverArchType)
    {
    case LWSWITCH_GET_INFO_INDEX_ARCH_SV10:
        arch = LWSWITCH_ARCH_TYPE_SV10;
        break;

    case LWSWITCH_GET_INFO_INDEX_ARCH_LR10:
        arch = LWSWITCH_ARCH_TYPE_LR10;
        break;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    case LWSWITCH_GET_INFO_INDEX_ARCH_LS10:
        arch = LWSWITCH_ARCH_TYPE_LS10;
        break;
#endif

    default:
        arch = LWSWITCH_ARCH_TYPE_ILWALID;
        break;
    }

    return arch;
}

bool
GFMHelper::getArchNameForArchType( lwSwitchArchType archType, const char **name )
{
    switch ( archType )
    {
        case LWSWITCH_ARCH_TYPE_SV10:
            *name = "SV10";
            return true;
        case LWSWITCH_ARCH_TYPE_LR10:
            *name = "LR10";
            return true;
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
        case LWSWITCH_ARCH_TYPE_LS10:
            *name = "LS10";
            return true;
#endif
        default:
            return false;
    }
}
