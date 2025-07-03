#include "socket_interface.h"
#include "lwlink_lib_ioctl.h"
#include "helper.h"
#include <map>
#define MULIT_NODE_TRAINING_SOCKET_PORT_NUM 7866

int isServer;
SocketServer gServerSocket; // server side socket object
SocketClient* gAcceptedSocket; //on server side, indicate an accepted connection
SocketClient gClientSocket;

LWLinkConnList internodeConnListServer; //internode connection list on server side.
LWLinkConnList internodeConnListClient; //internode connection list on client side.
FMLWLinkConnList fmInternodeConnListServer;
FMLWLinkConnList fmInternodeConnListClient;

static void train_link_to_high_speed_server(unsigned int connIdx)
{
    MsgHdr msgReq = {0};
    MsgHdr msgReply = {0};

    // first get the required connection from index
    if (fmInternodeConnListServer.size() < connIdx) {
        std::cout <<"Invalid internode connection index " << std::endl;
        return;
    }

    FMLWLinkConnList::iterator it = fmInternodeConnListServer.begin();
    std::advance(it, connIdx);
    fm_lwlink_conn_info fmConnInfo = *it;
    lwlink_connection_info connInfo = fmConnInfo.connInfo;

    std::cout << "\tTraining the following internode connection \n";
    std::cout << " \tnodeId  " << connInfo.srcEndPoint.nodeId;
    std::cout << " linkIndex " << connInfo.srcEndPoint.linkIndex;
    std::cout << " domain " << (int)connInfo.srcEndPoint.pciInfo.domain;
    std::cout << " bus " << (int)connInfo.srcEndPoint.pciInfo.bus;
    std::cout << " device " << (int)connInfo.srcEndPoint.pciInfo.device;
    std::cout << " function " << (int)connInfo.srcEndPoint.pciInfo.function;
    std::cout << "<======>";
    std::cout << " nodeId = " << connInfo.dstEndPoint.nodeId;
    std::cout << " linkIndex " << connInfo.dstEndPoint.linkIndex;
    std::cout << " domain " << (int)connInfo.dstEndPoint.pciInfo.domain;
    std::cout << " bus " << (int)connInfo.dstEndPoint.pciInfo.bus;
    std::cout << " device " << (int)connInfo.dstEndPoint.pciInfo.device;
    std::cout << " function " << (int)connInfo.dstEndPoint.pciInfo.function;
    std::cout << std::endl;

    if (getArch() != LWSWITCH_ARCH_TYPE_SV10) {
        
    }

    // first ask slave to start
    msgReq.msgCmd = CMD_START_SAFE_TO_HS;
    msgReq.option = connIdx;
    gAcceptedSocket->SendBytes((char*) &msgReq, sizeof (msgReq));
    
    // try for high speed
    std::cout << "setting sublink to LWLINK_TRAIN_LINK_SAFE_TO_HS\n";
    set_sublink_state(lwlink_train_sublink_safe_to_hs, true, connInfo.srcEndPoint);

    // wait for response from client
    gAcceptedSocket->ReadBytes((char*) &msgReply, sizeof (msgReply));
    
    // now set the main link
    std::cout << "setting mainlink to LWLINK_TRAIN_LINK_SAFE_TO_HS\n";
    set_mainlink_state(lwlink_train_link_swcfg_to_active, true, connInfo.srcEndPoint);
}

static void train_link_to_high_speed_client(unsigned int connIdx)
{
    MsgHdr msgReply = {0};

    // first get the required connection from index
    if (internodeConnListClient.size() < connIdx) {
        std::cout <<"Invalid internode connection index " << std::endl;
        return;
    }

    LWLinkConnList::iterator it = internodeConnListClient.begin();
    std::advance(it, connIdx);
    lwlink_connection_info connInfo = *it;

    std::cout << "\tTraining the following internode connection \n";
    std::cout << " \tnodeId  " << connInfo.srcEndPoint.nodeId;
    std::cout << " linkIndex " << connInfo.srcEndPoint.linkIndex;
    std::cout << " domain " << (int)connInfo.srcEndPoint.pciInfo.domain;
    std::cout << " bus " << (int)connInfo.srcEndPoint.pciInfo.bus;
    std::cout << " device " << (int)connInfo.srcEndPoint.pciInfo.device;
    std::cout << " function " << (int)connInfo.srcEndPoint.pciInfo.function;
    std::cout << "<======>";
    std::cout << " nodeId = " << connInfo.dstEndPoint.nodeId;
    std::cout << " linkIndex " << connInfo.dstEndPoint.linkIndex;
    std::cout << " domain " << (int)connInfo.dstEndPoint.pciInfo.domain;
    std::cout << " bus " << (int)connInfo.dstEndPoint.pciInfo.bus;
    std::cout << " device " << (int)connInfo.dstEndPoint.pciInfo.device;
    std::cout << " function " << (int)connInfo.dstEndPoint.pciInfo.function;
    std::cout << std::endl;

    // try for high speed
    std::cout << "setting sublink to LWLINK_TRAIN_LINK_SAFE_TO_HS\n";
    set_sublink_state(lwlink_train_sublink_safe_to_hs, false, connInfo.dstEndPoint);

    msgReply.msgCmd = CMD_RESP_SAFE_TO_HS_SUBLINK_DONE;
    gClientSocket.SendBytes((char*) &msgReply, sizeof (msgReply));

    // now set the main link
    std::cout << "setting mainlink to LWLINK_TRAIN_LINK_SAFE_TO_HS\n";
    set_mainlink_state(lwlink_train_link_swcfg_to_active, false, connInfo.dstEndPoint);
}

static void correlate_internode_conns(DiscoveryTokenList &writeList,
                                            DiscoveryTokenList &readList)
{
    DiscoveryTokenList::iterator wit;
    DiscoveryTokenList::iterator rit;
    for ( wit = writeList.begin(); wit != writeList.end(); wit++) {
        for ( rit = readList.begin(); rit != readList.end();) {
            DiscoveryTokenInfo writeInfo = (*wit);
            DiscoveryTokenInfo readInfo = (*rit);
            if ( writeInfo.tokelwalue == readInfo.tokelwalue ) {
                // found a connection
                fm_lwlink_conn_info fmConnInfo;
                fmConnInfo.connInfo.srcEndPoint.nodeId = writeInfo.nodeId;
                fmConnInfo.connInfo.srcEndPoint.linkIndex = writeInfo.linkIndex;
                fmConnInfo.connInfo.srcEndPoint.pciInfo.domain = writeInfo.domain;
                fmConnInfo.connInfo.srcEndPoint.pciInfo.bus = writeInfo.bus;
                fmConnInfo.connInfo.srcEndPoint.pciInfo.device = writeInfo.device;
                fmConnInfo.connInfo.srcEndPoint.pciInfo.function =  writeInfo.function;
                fmConnInfo.nearDevType = writeInfo.devType;
                
                fmConnInfo.connInfo.dstEndPoint.nodeId = readInfo.nodeId;
                fmConnInfo.connInfo.dstEndPoint.linkIndex = readInfo.linkIndex;
                fmConnInfo.connInfo.dstEndPoint.pciInfo.domain = readInfo.domain;
                fmConnInfo.connInfo.dstEndPoint.pciInfo.bus = readInfo.bus;
                fmConnInfo.connInfo.dstEndPoint.pciInfo.device = readInfo.device;
                fmConnInfo.connInfo.dstEndPoint.pciInfo.function =  readInfo.function;
                setBusToPhyId(readInfo.nodeId, readInfo.bus, readInfo.phyId);
                fmConnInfo.farDevType = readInfo.devType;
                fmInternodeConnListServer.push_back( fmConnInfo );
                rit = readList.erase( rit );
                break;
            }else{
                // move to next read token
                rit++;
            }
        }
    }
}


static bool connectionDoesNotExist(lwlink_connection_info conn)
{
    FMLWLinkConnList::iterator it = fmInternodeConnListServer.begin();
    for (it = fmInternodeConnListServer.begin(); it != fmInternodeConnListServer.end(); it++) {
        // first check
        fm_lwlink_conn_info tempInfo = *it;
        lwlink_connection_info tempConn = tempInfo.connInfo;

        if ( ((conn.srcEndPoint.nodeId == tempConn.srcEndPoint.nodeId) &&
             (conn.srcEndPoint.linkIndex == tempConn.srcEndPoint.linkIndex) &&
             (conn.srcEndPoint.pciInfo.domain == tempConn.srcEndPoint.pciInfo.domain) && 
             (conn.srcEndPoint.pciInfo.bus == tempConn.srcEndPoint.pciInfo.bus) &&
             (conn.srcEndPoint.pciInfo.device == tempConn.srcEndPoint.pciInfo.device) &&
             (conn.srcEndPoint.pciInfo.function == tempConn.srcEndPoint.pciInfo.function)) &&
             
             ((conn.dstEndPoint.nodeId == tempConn.dstEndPoint.nodeId) &&
             (conn.dstEndPoint.linkIndex == tempConn.dstEndPoint.linkIndex) &&
             (conn.dstEndPoint.pciInfo.domain == tempConn.dstEndPoint.pciInfo.domain) && 
             (conn.dstEndPoint.pciInfo.bus == tempConn.dstEndPoint.pciInfo.bus) &&
             (conn.dstEndPoint.pciInfo.device == tempConn.dstEndPoint.pciInfo.device) &&
             (conn.dstEndPoint.pciInfo.function == tempConn.dstEndPoint.pciInfo.function) ))
             
        {
            std::cout << "\tskipping the following connection to list\n";
            std::cout << " \tnodeId  " << conn.srcEndPoint.nodeId;
            std::cout << " linkIndex " << conn.srcEndPoint.linkIndex;
            std::cout << " domain " << (int)conn.srcEndPoint.pciInfo.domain;
            std::cout << " bus " << (int)conn.srcEndPoint.pciInfo.bus;
            std::cout << " device " << (int)conn.srcEndPoint.pciInfo.device;
            std::cout << " function " << (int)conn.srcEndPoint.pciInfo.function;
            std::cout << "<======>";
            std::cout << " nodeId = " << conn.dstEndPoint.nodeId;
            std::cout << " linkIndex " << conn.dstEndPoint.linkIndex;
            std::cout << " domain " << (int)conn.dstEndPoint.pciInfo.domain;
            std::cout << " bus " << (int)conn.dstEndPoint.pciInfo.bus;
            std::cout << " device " << (int)conn.dstEndPoint.pciInfo.device;
            std::cout << " function " << (int)conn.dstEndPoint.pciInfo.function;
            std::cout << std::endl;
            return false;
        }

        if ( ((conn.srcEndPoint.nodeId == tempConn.dstEndPoint.nodeId) &&
             (conn.srcEndPoint.linkIndex == tempConn.dstEndPoint.linkIndex) &&
             (conn.srcEndPoint.pciInfo.domain == tempConn.dstEndPoint.pciInfo.domain) && 
             (conn.srcEndPoint.pciInfo.bus == tempConn.dstEndPoint.pciInfo.bus) &&
             (conn.srcEndPoint.pciInfo.device == tempConn.dstEndPoint.pciInfo.device) &&
             (conn.srcEndPoint.pciInfo.function == tempConn.dstEndPoint.pciInfo.function)) &&
             
             ((conn.dstEndPoint.nodeId == tempConn.srcEndPoint.nodeId) &&
             (conn.dstEndPoint.linkIndex == tempConn.srcEndPoint.linkIndex) &&
             (conn.dstEndPoint.pciInfo.domain == tempConn.srcEndPoint.pciInfo.domain) && 
             (conn.dstEndPoint.pciInfo.bus == tempConn.srcEndPoint.pciInfo.bus) &&
             (conn.dstEndPoint.pciInfo.device == tempConn.srcEndPoint.pciInfo.device) &&
             (conn.dstEndPoint.pciInfo.function == tempConn.srcEndPoint.pciInfo.function) ))
             
        {
            std::cout << "\tskipping the following connection to list\n";
            std::cout << " \tnodeId  " << conn.srcEndPoint.nodeId;
            std::cout << " linkIndex " << conn.srcEndPoint.linkIndex;
            std::cout << " domain " << (int)conn.srcEndPoint.pciInfo.domain;
            std::cout << " bus " << (int)conn.srcEndPoint.pciInfo.bus;
            std::cout << " device " << (int)conn.srcEndPoint.pciInfo.device;
            std::cout << " function " << (int)conn.srcEndPoint.pciInfo.function;
            std::cout << "<======>";
            std::cout << " nodeId = " << conn.dstEndPoint.nodeId;
            std::cout << " linkIndex " << conn.dstEndPoint.linkIndex;
            std::cout << " domain " << (int)conn.dstEndPoint.pciInfo.domain;
            std::cout << " bus " << (int)conn.dstEndPoint.pciInfo.bus;
            std::cout << " device " << (int)conn.dstEndPoint.pciInfo.device;
            std::cout << " function " << (int)conn.dstEndPoint.pciInfo.function;
            std::cout << std::endl;
            return false;
        }
    }

    return true;
}

static void 
correlate_internode_conns_with_SIDs(SidInfoList &sidList,
                                                std::map<uint64, uint32> &sidToNodeIdMap,
                                                std::map<uint64, uint64> &sidToGpuOrSwitchIdMap,
                                                std::map<uint64, LwU16> sidToDomainMap,
                                                std::map<uint64, LwU8> sidToBusMap,
                                                std::map<uint64, LwU8> sidToDeviceMap,
                                                std::map<uint64, LwU8> sidToFunctionMap,
                                                std::map<uint64, LwU32> sidToDevTypeMap)
{
    std::cout << "num of sid list values is " << sidList.size() << std::endl;
    int idx = 0;
    for (auto it = sidList.begin(); it != sidList.end(); it++ ) {
        fm_lwlink_conn_info fmConnInfo;
        lwlink_connection_info connInfo;
        connInfo.srcEndPoint.nodeId = sidToNodeIdMap[ it->nearSid ];
        connInfo.srcEndPoint.linkIndex = it->nearLinkIndex;
        connInfo.srcEndPoint.pciInfo.domain = sidToDomainMap[it->nearSid];
        connInfo.srcEndPoint.pciInfo.bus = sidToBusMap[it->nearSid];
        connInfo.srcEndPoint.pciInfo.device = sidToDeviceMap[it->nearSid];
        connInfo.srcEndPoint.pciInfo.function =  sidToFunctionMap[it->nearSid];

        if (sidToNodeIdMap.find(it->farSid) == sidToNodeIdMap.end())
        {
            idx++;
            continue;
        }
        
        connInfo.dstEndPoint.nodeId = sidToNodeIdMap[ it->farSid ];
        connInfo.dstEndPoint.linkIndex = it->farLinkIndex;
        connInfo.dstEndPoint.pciInfo.domain = sidToDomainMap[it->farSid];
        connInfo.dstEndPoint.pciInfo.bus = sidToBusMap[it->farSid];
        connInfo.dstEndPoint.pciInfo.device = sidToDeviceMap[it->farSid];
        connInfo.dstEndPoint.pciInfo.function =  sidToFunctionMap[it->farSid];

        fmConnInfo.connInfo = connInfo;
        fmConnInfo.nearDevType = sidToDevTypeMap[it->nearSid];
        fmConnInfo.farDevType = sidToDevTypeMap[it->farSid];
        //setBusToPhyId(readInfo.nodeId, readInfo.bus, readInfo.phyId);
        if (connectionDoesNotExist(fmConnInfo.connInfo)) {
            if(connInfo.srcEndPoint.nodeId != connInfo.dstEndPoint.nodeId) {
                fmInternodeConnListServer.push_back( fmConnInfo );
            }
        }
    }
    std::cout << "skipped " << idx << " connections " << std::endl;
    fprintf(stderr, "size of list %d \n", (int) fmInternodeConnListServer.size());
}
                                            
static void read_discovery_token_client(void)
{
    //printf("read_discovery_token_client called\n");
    DiscoveryTokenList readList;
    DiscoveryTokenList::iterator it;
    int idx = 0;
    MsgHdr msgReply = {0};
    
    // read tokens locally
    read_discovery_tokens(readList);
    
    // send the response to server
    msgReply.msgCmd = CMD_RESP_READ_DISCOVERY_TOKEN;
    for ( it = readList.begin(); it != readList.end(); it++ ) {
        DiscoveryTokenInfo tempInfo = *it;
        msgReply.readDiscTokens.readDiscToken[idx].nodeId = tempInfo.nodeId;
        msgReply.readDiscTokens.readDiscToken[idx].domain = tempInfo.domain;
        msgReply.readDiscTokens.readDiscToken[idx].bus = tempInfo.bus;
        msgReply.readDiscTokens.readDiscToken[idx].device = tempInfo.device;
        msgReply.readDiscTokens.readDiscToken[idx].function = tempInfo.function;
        msgReply.readDiscTokens.readDiscToken[idx].linkIndex = tempInfo.linkIndex;
        msgReply.readDiscTokens.readDiscToken[idx].tokelwalue = tempInfo.tokelwalue;
        msgReply.readDiscTokens.readDiscToken[idx].devType = tempInfo.devType;
        //msgReply.readDiscTokens.readDiscToken[idx].phyId = tempInfo.phyId;
        msgReply.readDiscTokens.readDiscToken[idx].phyId = getBusToPhyId(tempInfo.nodeId, (int) tempInfo.bus);;
        idx++;
    }
    msgReply.readDiscTokens.count = idx;
    gClientSocket.SendBytes((char*) &msgReply, sizeof (msgReply));
}

static void read_SIDs_client(void)
{
    SidInfoList sidList;
    SidInfoList::iterator it;
    int idx = 0;
    MsgHdr msgReply = {0};

    read_SIDs(sidList);

    msgReply.msgCmd = CMD_RESP_READ_SID;
    for (it = sidList.begin(); it != sidList.end(); it++) {
        SidNodeConnectionInfo sidInfo = *it;
        msgReply.interNodeSidInfo.connInfo[idx].nodeId = sidInfo.nodeId;
        msgReply.interNodeSidInfo.connInfo[idx].gpuOrSwitchId = sidInfo.gpuOrSwitchId;
        msgReply.interNodeSidInfo.connInfo[idx].nearSid = sidInfo.nearSid;
        msgReply.interNodeSidInfo.connInfo[idx].nearLinkIndex = sidInfo.nearLinkIndex;
        msgReply.interNodeSidInfo.connInfo[idx].farSid = sidInfo.farSid;
        msgReply.interNodeSidInfo.connInfo[idx].farLinkIndex = sidInfo.farLinkIndex;
        msgReply.interNodeSidInfo.connInfo[idx].domain = sidInfo.domain;
        msgReply.interNodeSidInfo.connInfo[idx].bus = sidInfo.bus;
        msgReply.interNodeSidInfo.connInfo[idx].device = sidInfo.device;
        msgReply.interNodeSidInfo.connInfo[idx].function = sidInfo.function;
        msgReply.interNodeSidInfo.connInfo[idx].devType = sidInfo.devType;
        idx++;
    }

    msgReply.interNodeSidInfo.count = idx;
    gClientSocket.SendBytes((char*) &msgReply, sizeof(msgReply));
}

static void discover_internode_connections_server_sid(void)
{
    MsgHdr msgReq = {0};
    MsgHdr msgReply = {0};
    SidInfoList sidList;
    std::map<uint64, uint32> sidToNodeIdMap;
    std::map<uint64, uint64> sidToGpuOrSwitchIdMap;
    std::map<uint64, LwU16> sidToDomainMap;
    std::map<uint64, LwU8> sidToBusMap;
    std::map<uint64, LwU8> sidToDeviceMap;
    std::map<uint64, LwU8> sidToFunctionMap;
    std::map<uint64, LwU32> sidToDevTypeMap;

    SidInfoList::iterator it;
    read_SIDs(sidList);

    for (it = sidList.begin(); it != sidList.end(); it++) {
        SidNodeConnectionInfo sidInfo = *it;
        sidToNodeIdMap[sidInfo.nearSid] = sidInfo.nodeId;
        sidToGpuOrSwitchIdMap[sidInfo.nearSid] = sidInfo.gpuOrSwitchId;
        sidToDomainMap[sidInfo.nearSid] = sidInfo.domain;
        sidToBusMap[sidInfo.nearSid] = sidInfo.bus;
        sidToFunctionMap[sidInfo.nearSid] = sidInfo.function;
        sidToDeviceMap[sidInfo.nearSid] = sidInfo.device;
        sidToDevTypeMap[sidInfo.nearSid] = sidInfo.devType;
    }
    msgReq.msgCmd = CMD_START_READ_SID;
    gAcceptedSocket->SendBytes((char*) &msgReq, sizeof(msgReq));

    //wait here for response and read it
    gAcceptedSocket->ReadBytes((char*) &msgReply, sizeof(msgReply));

    for (unsigned int idx = 0; idx < msgReply.interNodeSidInfo.count; idx++) {
        SidNodeConnectionInfo sidInfo;
        sidInfo.nodeId = msgReply.interNodeSidInfo.connInfo[idx].nodeId; 
        sidInfo.gpuOrSwitchId = msgReply.interNodeSidInfo.connInfo[idx].gpuOrSwitchId;
        sidInfo.nearSid = msgReply.interNodeSidInfo.connInfo[idx].nearSid;
        sidInfo.nearLinkIndex = msgReply.interNodeSidInfo.connInfo[idx].nearLinkIndex;
        sidInfo.farSid = msgReply.interNodeSidInfo.connInfo[idx].farSid;
        sidInfo.farLinkIndex = msgReply.interNodeSidInfo.connInfo[idx].farLinkIndex;
        sidInfo.domain = msgReply.interNodeSidInfo.connInfo[idx].domain;
        sidInfo.bus = msgReply.interNodeSidInfo.connInfo[idx].bus;
        sidInfo.device = msgReply.interNodeSidInfo.connInfo[idx].device;
        sidInfo.function = msgReply.interNodeSidInfo.connInfo[idx].function;
        sidInfo.devType = msgReply.interNodeSidInfo.connInfo[idx].devType;
        sidToNodeIdMap[sidInfo.nearSid] = sidInfo.nodeId;
        sidToGpuOrSwitchIdMap[sidInfo.nearSid] = sidInfo.gpuOrSwitchId;
        sidToDomainMap[sidInfo.nearSid] = sidInfo.domain;
        sidToBusMap[sidInfo.nearSid] = sidInfo.bus;
        sidToFunctionMap[sidInfo.nearSid] = sidInfo.function;
        sidToDeviceMap[sidInfo.nearSid] = sidInfo.device;
        sidToDevTypeMap[sidInfo.nearSid] = sidInfo.devType;
        sidList.push_back(sidInfo);
    }

    correlate_internode_conns_with_SIDs(sidList, sidToNodeIdMap, sidToGpuOrSwitchIdMap,
                                        sidToDomainMap, sidToBusMap, sidToFunctionMap,
                                        sidToDeviceMap, sidToDevTypeMap);
}

static void discover_internode_connections_server_discovery_token(void)
{
    MsgHdr msgReq = {0};
    MsgHdr msgReply = {0};
    DiscoveryTokenList readList;
    DiscoveryTokenList writeList;
    DiscoveryTokenList::iterator readIt;
    DiscoveryTokenList::iterator writeIt;

    // first write discovery tokens locally
    write_discovery_tokens(writeList);

    // now tell client to read discovery tokens
    msgReq.msgCmd = CMD_START_READ_DISCOVERY_TOKEN;
    gAcceptedSocket->SendBytes((char*) &msgReq, sizeof (msgReq));

    //std::cout << "Waiting for token information from client " << std::endl;    
    // client will respond with number of tokens read as well
    gAcceptedSocket->ReadBytes((char*) &msgReply, sizeof (msgReply));

    // now compare and create internode connections
    //std::cout << "total token written count " << writeList.size() << std::endl;    
    
    for (unsigned int idx = 0; idx < msgReply.readDiscTokens.count; idx++ ) {
        DiscoveryTokenInfo info;
        info.nodeId = msgReply.readDiscTokens.readDiscToken[idx].nodeId;
        info.domain = msgReply.readDiscTokens.readDiscToken[idx].domain;
        info.bus = msgReply.readDiscTokens.readDiscToken[idx].bus;
        info.phyId = msgReply.readDiscTokens.readDiscToken[idx].phyId;
        info.device = msgReply.readDiscTokens.readDiscToken[idx].device;
        info.function = msgReply.readDiscTokens.readDiscToken[idx].function;
        info.linkIndex = msgReply.readDiscTokens.readDiscToken[idx].linkIndex;
        info.tokelwalue = msgReply.readDiscTokens.readDiscToken[idx].tokelwalue;
        info.devType = msgReply.readDiscTokens.readDiscToken[idx].devType;
        readList.push_back(info);
    }

    correlate_internode_conns(writeList, readList);
    // dump the connection information
    //std::cout << "Total number of inter-node connections = " << internodeConnListServer.size() << std::endl;
    FMLWLinkConnList::iterator it = fmInternodeConnListServer.begin();
    int connIdx = 0;
    //std::cout << "nodeId\t(d::b:d.f)\tphyId\tlinkIndex\tnodeIdFar\t(d::b:d.f)Far\tphyIdFar\tlinkIndexFar\n";

    while ( it != fmInternodeConnListServer.end() ) {
        fm_lwlink_conn_info fmConnInfo = *it;
        lwlink_connection_info connInfo = fmConnInfo.connInfo;
        connIdx++;
        std::cout << connInfo.srcEndPoint.nodeId;
        std::cout << "\t(" << (int)connInfo.srcEndPoint.pciInfo.domain;
        std::cout << ":" << (int)connInfo.srcEndPoint.pciInfo.bus;
        std::cout << ":" << (int)connInfo.srcEndPoint.pciInfo.device;
        std::cout << "." << (int)connInfo.srcEndPoint.pciInfo.function <<")";
        //TODO
        int32_t phyId = getBusToPhyId(connInfo.srcEndPoint.nodeId, (int)connInfo.srcEndPoint.pciInfo.bus);
        std::cout << "\t" << phyId;
        std::cout << "\t" << connInfo.srcEndPoint.linkIndex;
        if (phyId > 0)
            std::cout << "\t\t" << 0;
        else
            std::cout << "\t\t" << 1;
        std::cout << "\t" << connInfo.dstEndPoint.nodeId;
        std::cout << "\t\t(" << (int)connInfo.dstEndPoint.pciInfo.domain;
        std::cout << ":" << (int)connInfo.dstEndPoint.pciInfo.bus;
        std::cout << ":" << (int)connInfo.dstEndPoint.pciInfo.device;
        std::cout << "." << (int)connInfo.dstEndPoint.pciInfo.function<<")";
        //TODO
        int32_t phyIdFar = getBusToPhyId(connInfo.dstEndPoint.nodeId, (int)connInfo.dstEndPoint.pciInfo.bus);
        std::cout << "\t" << phyIdFar;
        std::cout << "\t\t" << connInfo.dstEndPoint.linkIndex;
        if (phyIdFar > 0)
            std::cout << "\t\t" << 0;
        else
            std::cout << "\t\t" << 1;
        std::cout << std::endl;

        it++;
    }
}

static void add_internode_connections_server()
{
    
    FMLWLinkConnList::iterator it = fmInternodeConnListServer.begin();
    int idx = 0;
    for ( it = fmInternodeConnListServer.begin(); it != fmInternodeConnListServer.end(); it++) {
        fm_lwlink_conn_info fmConnInfo = *it;
        // since when the internode connection is created, srcEndPoint is treated as
        // local endpoint. So use src as localEndPoint
        int myNodeId = getNodeId();

        if (myNodeId != fmConnInfo.connInfo.srcEndPoint.nodeId && myNodeId != fmConnInfo.connInfo.dstEndPoint.nodeId) 
            continue;

        idx++;
        if (getNodeId() == fmConnInfo.connInfo.dstEndPoint.nodeId) {
            lwlink_remote_endpoint_info remoteEndPoint;
            remoteEndPoint.pciInfo = fmConnInfo.connInfo.srcEndPoint.pciInfo;
            remoteEndPoint.linkIndex = fmConnInfo.connInfo.srcEndPoint.linkIndex;
            remoteEndPoint.nodeId = fmConnInfo.connInfo.srcEndPoint.nodeId;
            remoteEndPoint.devType = fmConnInfo.nearDevType;
            add_internode_connections(fmConnInfo.connInfo.dstEndPoint, remoteEndPoint);
        } else {
            lwlink_remote_endpoint_info remoteEndPoint;
            remoteEndPoint.pciInfo = fmConnInfo.connInfo.dstEndPoint.pciInfo;
            remoteEndPoint.linkIndex = fmConnInfo.connInfo.dstEndPoint.linkIndex;
            remoteEndPoint.nodeId = fmConnInfo.connInfo.dstEndPoint.nodeId;
            remoteEndPoint.devType = fmConnInfo.farDevType;
            add_internode_connections(fmConnInfo.connInfo.srcEndPoint, remoteEndPoint);
        }

        // add_internode_connections(fmConnInfo.connInfo.srcEndPoint, remoteEndPoint);
    }
    std::cout << "added " << idx << " num of connections " << std::endl;
}

static void send_add_internode_connections_to_client()
{
    MsgHdr msgReq = {0};
    int idx = 0;
    
    msgReq.msgCmd = CMD_START_INTERNODE_CONN_ADD;
    
    FMLWLinkConnList::iterator it = fmInternodeConnListServer.begin();
    for ( it = fmInternodeConnListServer.begin(); it != fmInternodeConnListServer.end(); it++) {
        fm_lwlink_conn_info connInfo = *it;
        // source node information
        msgReq.interNodeConns.connInfo[idx].srcEndPoint.linkIndex = connInfo.connInfo.srcEndPoint.linkIndex;
        msgReq.interNodeConns.connInfo[idx].srcEndPoint.nodeId = connInfo.connInfo.srcEndPoint.nodeId;
        msgReq.interNodeConns.connInfo[idx].srcEndPoint.pciInfo.domain = connInfo.connInfo.srcEndPoint.pciInfo.domain;
        msgReq.interNodeConns.connInfo[idx].srcEndPoint.pciInfo.bus = connInfo.connInfo.srcEndPoint.pciInfo.bus;
        msgReq.interNodeConns.connInfo[idx].srcEndPoint.pciInfo.device = connInfo.connInfo.srcEndPoint.pciInfo.device;
        msgReq.interNodeConns.connInfo[idx].srcEndPoint.pciInfo.function = connInfo.connInfo.srcEndPoint.pciInfo.function;
        msgReq.interNodeConns.connInfo[idx].nearDevType = connInfo.nearDevType;
        // dest node information
        msgReq.interNodeConns.connInfo[idx].dstEndPoint.linkIndex = connInfo.connInfo.dstEndPoint.linkIndex;
        msgReq.interNodeConns.connInfo[idx].dstEndPoint.nodeId = connInfo.connInfo.dstEndPoint.nodeId;
        msgReq.interNodeConns.connInfo[idx].dstEndPoint.pciInfo.domain = connInfo.connInfo.dstEndPoint.pciInfo.domain;
        msgReq.interNodeConns.connInfo[idx].dstEndPoint.pciInfo.bus = connInfo.connInfo.dstEndPoint.pciInfo.bus;
        msgReq.interNodeConns.connInfo[idx].dstEndPoint.pciInfo.device = connInfo.connInfo.dstEndPoint.pciInfo.device;
        msgReq.interNodeConns.connInfo[idx].dstEndPoint.pciInfo.function = connInfo.connInfo.dstEndPoint.pciInfo.function;
        msgReq.interNodeConns.connInfo[idx].dstDevType = connInfo.farDevType;
        idx++;
    }

    msgReq.interNodeConns.count = idx;
    // now tell client to add connections
    gAcceptedSocket->SendBytes((char*) &msgReq, sizeof (msgReq));
}

static void handle_add_internode_connections_client(MsgHdr &addMsg)
{
    std::cout << "adding " << addMsg.interNodeConns.count << " num of connections " << std::endl;
    for (unsigned int idx = 0; idx < addMsg.interNodeConns.count; idx++ ) {
        InterNodeConnectionInfo connInfo = addMsg.interNodeConns.connInfo[idx];
        std::cout << "\tAdding following internode connction locally\n";
        std::cout << " nodeId  " << connInfo.srcEndPoint.nodeId;
        std::cout << " linkIndex " << connInfo.srcEndPoint.linkIndex;
        std::cout << " domain " << (int)connInfo.srcEndPoint.pciInfo.domain;
        std::cout << " bus " << (int)connInfo.srcEndPoint.pciInfo.bus;
        std::cout << " device " << (int)connInfo.srcEndPoint.pciInfo.device;
        std::cout << " function " << (int)connInfo.srcEndPoint.pciInfo.function;
        std::cout << "<======>";
        std::cout << " nodeId = " << connInfo.dstEndPoint.nodeId;
        std::cout << " linkIndex " << connInfo.dstEndPoint.linkIndex;
        std::cout << " domain " << (int)connInfo.dstEndPoint.pciInfo.domain;
        std::cout << " bus " << (int)connInfo.dstEndPoint.pciInfo.bus;
        std::cout << " device " << (int)connInfo.dstEndPoint.pciInfo.device;
        std::cout << " function " << (int)connInfo.dstEndPoint.pciInfo.function;
        std::cout << std::endl;

        int myNodeId = getNodeId();

        if (myNodeId != connInfo.srcEndPoint.nodeId && myNodeId != connInfo.dstEndPoint.nodeId) 
            continue;

        if (getNodeId() == connInfo.dstEndPoint.nodeId) {
            lwlink_remote_endpoint_info remoteEndPoint;
            remoteEndPoint.pciInfo = connInfo.srcEndPoint.pciInfo;
            remoteEndPoint.linkIndex = connInfo.srcEndPoint.linkIndex;
            remoteEndPoint.nodeId = connInfo.srcEndPoint.nodeId;
            remoteEndPoint.devType = connInfo.nearDevType;
            add_internode_connections(connInfo.dstEndPoint, remoteEndPoint);
        } else {
            lwlink_remote_endpoint_info remoteEndPoint;
            remoteEndPoint.pciInfo = connInfo.dstEndPoint.pciInfo;
            remoteEndPoint.linkIndex = connInfo.dstEndPoint.linkIndex;
            remoteEndPoint.nodeId = connInfo.dstEndPoint.nodeId;
            remoteEndPoint.devType = connInfo.dstDevType;
            add_internode_connections(connInfo.srcEndPoint, remoteEndPoint);
        }

        // add the connection to our list
        lwlink_connection_info tempConnInfo;
        tempConnInfo.srcEndPoint = connInfo.srcEndPoint;
        tempConnInfo.dstEndPoint = connInfo.dstEndPoint;
        internodeConnListClient.push_back(tempConnInfo);
    }
}
void start_client_command_parsing()
{
    MsgHdr msgReq = {0};

    while (1) {
        memset(&msgReq, 0, sizeof(msgReq));
        gClientSocket.ReadBytes((char*) &msgReq, sizeof (msgReq));
        switch(msgReq.msgCmd) {
            case CMD_START_OFF_TO_SAFE: {
                break;
            }
            case CMD_START_SAFE_TO_HS: {
                unsigned int connIdx = msgReq.option;
                train_link_to_high_speed_client(connIdx);
                break;
            }
            case CMD_START_HS_TO_SAFE: {
                break;
            }
            case CMD_START_SAFE_TO_OFF: {
                break;
            }
            case CMD_START_READ_DISCOVERY_TOKEN: {
                //std::cout <<" CMD_START_READ_DISCOVERY_TOKEN\n";
                read_discovery_token_client();
                break;
            }
            case CMD_START_INTERNODE_CONN_ADD: {
                handle_add_internode_connections_client(msgReq);
                break;
            }
            case CMD_START_READ_SID: {
                read_SIDs_client();
                break;
            }
            case 0: {
                //std::cout <<" Exiting \n";
                return;
            }
        }
    }

}

void run_multi_node_server()
{
    int option;
    SocketServer serverSocket; // server side socket object
    SocketClient* acceptedSocket; //on server side, indicate an accepted connection

    //std::cout <<" Staring server and waiting for connection \n";
    serverSocket.BindAndListen(MULIT_NODE_TRAINING_SOCKET_PORT_NUM);
    acceptedSocket = serverSocket.Accept();

    gServerSocket = serverSocket;
    gAcceptedSocket = acceptedSocket;

    if (getArch() == LWSWITCH_ARCH_TYPE_SV10)
        discover_internode_connections_server_discovery_token();
    else 
        discover_internode_connections_server_sid();
    return;
}

void run_multi_node_client(std::string ipAddress)
{
    int option;
    SocketServer serverSocket; // server side socket object
    SocketClient* acceptedSocket; //on server side, indicate an accepted connection
    SocketClient clientSocket;

    clientSocket.ConnectTo(ipAddress, MULIT_NODE_TRAINING_SOCKET_PORT_NUM);

    gClientSocket = clientSocket;

    start_client_command_parsing();
    return;
}

void show_multi_node_training_options()
{
    int option;
    SocketServer serverSocket; // server side socket object
    SocketClient* acceptedSocket = NULL; //on server side, indicate an accepted connection

    SocketClient clientSocket;
    std::string ipAddress;

    std::cout <<" Note: Start one node as Server and other as client. Always start server first \n";
    
    std::cout <<" Is this node acting as server? (enter 1 for yes and 0 for no) \n";
    std::cin >> isServer;
    if (isServer == 1) {
        std::cout <<" Staring server and waiting for connection \n";
        serverSocket.BindAndListen(MULIT_NODE_TRAINING_SOCKET_PORT_NUM);
        acceptedSocket = serverSocket.Accept();
    } else {
        std::cout <<"Enter server node IP address ";
        std::cin >> ipAddress;
        clientSocket.ConnectTo(ipAddress, MULIT_NODE_TRAINING_SOCKET_PORT_NUM);
    }

    gServerSocket = serverSocket;
    gAcceptedSocket = acceptedSocket;
    gClientSocket = clientSocket;

    // for server, show the menu options
    // client will just follow commands from server
    if (!isServer) {
        start_client_command_parsing();
        return;
    }
    
    while (1) {
        std::cout <<" 1. Discover Internode Connections \n";
        std::cout <<" 2. Add Discovered Internode Connections Locally \n";
        std::cout <<" 3. Add Discovered Internode Connections on Client Node \n"; 
        std::cout <<" 4. Train an internode connection to HS \n";
        std::cout <<" 0. Exit \n";

        std::cout <<"Enter your choice ";
        std::cin >> option;

        switch (option) {
            case 0: {
                return;
            }
            case 1: {
                 if (getArch() == LWSWITCH_ARCH_TYPE_SV10)
                        discover_internode_connections_server_discovery_token();
                 else 
                        discover_internode_connections_server_sid();
                 break;
            }
            case 2: {
                add_internode_connections_server();
                break;
            }
            case 3: {
                send_add_internode_connections_to_client();
                break;
            }
            case 4: {
                unsigned int connIdx;
                std::cout <<" Enter internode connection index ";
                std::cin >> connIdx;
                train_link_to_high_speed_server(connIdx);
                break;
            }
        }
    }
}

