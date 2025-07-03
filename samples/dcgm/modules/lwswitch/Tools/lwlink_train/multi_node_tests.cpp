#include "socket_interface.h"
#include "lwlink_lib_ioctl.h"
#include "helper.h"

#define MULIT_NODE_TRAINING_SOCKET_PORT_NUM 7850

int isServer;
SocketServer gServerSocket; // server side socket object
SocketClient* gAcceptedSocket; //on server side, indicate an accepted connection
SocketClient gClientSocket;

LWLinkConnList internodeConnListServer; //internode connection list on server side.
LWLinkConnList internodeConnListClient; //internode connection list on client side.

static void train_link_to_high_speed_server(unsigned int connIdx)
{
    MsgHdr msgReq = {0};
    MsgHdr msgReply = {0};

    // first get the required connection from index
    if (internodeConnListServer.size() < connIdx) {
        std::cout <<"Invalid internode connection index " << std::endl;
        return;
    }

    LWLinkConnList::iterator it = internodeConnListServer.begin();
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
                lwlink_connection_info connInfo;
                connInfo.srcEndPoint.nodeId = writeInfo.nodeId;
                connInfo.srcEndPoint.linkIndex = writeInfo.linkIndex;
                connInfo.srcEndPoint.pciInfo.domain = writeInfo.domain;
                connInfo.srcEndPoint.pciInfo.bus = writeInfo.bus;
                connInfo.srcEndPoint.pciInfo.device = writeInfo.device;
                connInfo.srcEndPoint.pciInfo.function =  writeInfo.function;
                
                connInfo.dstEndPoint.nodeId = readInfo.nodeId;
                connInfo.dstEndPoint.linkIndex = readInfo.linkIndex;
                connInfo.dstEndPoint.pciInfo.domain = readInfo.domain;
                connInfo.dstEndPoint.pciInfo.bus = readInfo.bus;
                connInfo.dstEndPoint.pciInfo.device = readInfo.device;
                connInfo.dstEndPoint.pciInfo.function =  readInfo.function;
                setBusToPhyId(readInfo.nodeId, readInfo.bus, readInfo.phyId);
                internodeConnListServer.push_back( connInfo );
                rit = readList.erase( rit );
                break;
            }else{
                // move to next read token
                rit++;
            }
        }
    }
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
        //msgReply.readDiscTokens.readDiscToken[idx].phyId = tempInfo.phyId;
        msgReply.readDiscTokens.readDiscToken[idx].phyId = getBusToPhyId(tempInfo.nodeId, (int) tempInfo.bus);;
        idx++;
    }
    msgReply.readDiscTokens.count = idx;
    gClientSocket.SendBytes((char*) &msgReply, sizeof (msgReply));
}
static void discover_internode_connections_server(void)
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

    std::cout << "Waiting for token information from client " << std::endl;    
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
        readList.push_back(info);
    }

    correlate_internode_conns(writeList, readList);
    // dump the connection information
    //std::cout << "Total number of inter-node connections = " << internodeConnListServer.size() << std::endl;
    LWLinkConnList::iterator it = internodeConnListServer.begin();
    int connIdx = 0;
    //std::cout << "nodeId\t(d::b:d.f)\tphyId\tlinkIndex\tnodeIdFar\t(d::b:d.f)Far\tphyIdFar\tlinkIndexFar\n";

    while ( it != internodeConnListServer.end() ) {
        lwlink_connection_info connInfo = *it;
        connIdx++;
        std::cout << connInfo.srcEndPoint.nodeId;
        std::cout << "\t(" << (int)connInfo.srcEndPoint.pciInfo.domain;
        std::cout << ":" << (int)connInfo.srcEndPoint.pciInfo.bus;
        std::cout << ":" << (int)connInfo.srcEndPoint.pciInfo.device;
        std::cout << "." << (int)connInfo.srcEndPoint.pciInfo.function <<")";
        //TODO
        std::cout << "\t" << getBusToPhyId(connInfo.srcEndPoint.nodeId, (int)connInfo.srcEndPoint.pciInfo.bus);
        std::cout << "\t" << connInfo.srcEndPoint.linkIndex;
        std::cout << "\t" << connInfo.dstEndPoint.nodeId;
        std::cout << "\t(" << (int)connInfo.dstEndPoint.pciInfo.domain;
        std::cout << ":" << (int)connInfo.dstEndPoint.pciInfo.bus;
        std::cout << ":" << (int)connInfo.dstEndPoint.pciInfo.device;
        std::cout << "." << (int)connInfo.dstEndPoint.pciInfo.function<<")";
        //TODO
        std::cout << "\t" << getBusToPhyId(connInfo.dstEndPoint.nodeId, (int)connInfo.dstEndPoint.pciInfo.bus);
        std::cout << "\t" << connInfo.dstEndPoint.linkIndex;
        std::cout << std::endl;

        it++;
    }
}

static void add_internode_connections_server()
{
    LWLinkConnList::iterator it = internodeConnListServer.begin();
    for ( it = internodeConnListServer.begin(); it != internodeConnListServer.end(); it++) {
        lwlink_connection_info connInfo = *it;
        // since when the internode connection is created, srcEndPoint is treated as
        // local endpoint. So use src as localEndPoint
        lwlink_remote_endpoint_info remoteEndPoint;
        remoteEndPoint.pciInfo = connInfo.dstEndPoint.pciInfo;
        remoteEndPoint.linkIndex = connInfo.dstEndPoint.linkIndex;
        remoteEndPoint.nodeId = connInfo.dstEndPoint.nodeId;
        remoteEndPoint.devType = 2; // GPU - TODO
        add_internode_connections(connInfo.srcEndPoint, remoteEndPoint);
    }
}

static void send_add_internode_connections_to_client()
{
    MsgHdr msgReq = {0};
    int idx = 0;
    
    msgReq.msgCmd = CMD_START_INTERNODE_CONN_ADD;
    
    LWLinkConnList::iterator it = internodeConnListServer.begin();
    for ( it = internodeConnListServer.begin(); it != internodeConnListServer.end(); it++) {
        lwlink_connection_info connInfo = *it;
        // source node information
        msgReq.interNodeConns.connInfo[idx].srcEndPoint.linkIndex = connInfo.srcEndPoint.linkIndex;
        msgReq.interNodeConns.connInfo[idx].srcEndPoint.nodeId = connInfo.srcEndPoint.nodeId;
        msgReq.interNodeConns.connInfo[idx].srcEndPoint.pciInfo.domain = connInfo.srcEndPoint.pciInfo.domain;
        msgReq.interNodeConns.connInfo[idx].srcEndPoint.pciInfo.bus = connInfo.srcEndPoint.pciInfo.bus;
        msgReq.interNodeConns.connInfo[idx].srcEndPoint.pciInfo.device = connInfo.srcEndPoint.pciInfo.device;
        msgReq.interNodeConns.connInfo[idx].srcEndPoint.pciInfo.function = connInfo.srcEndPoint.pciInfo.function;
        // dest node information
        msgReq.interNodeConns.connInfo[idx].dstEndPoint.linkIndex = connInfo.dstEndPoint.linkIndex;
        msgReq.interNodeConns.connInfo[idx].dstEndPoint.nodeId = connInfo.dstEndPoint.nodeId;
        msgReq.interNodeConns.connInfo[idx].dstEndPoint.pciInfo.domain = connInfo.dstEndPoint.pciInfo.domain;
        msgReq.interNodeConns.connInfo[idx].dstEndPoint.pciInfo.bus = connInfo.dstEndPoint.pciInfo.bus;
        msgReq.interNodeConns.connInfo[idx].dstEndPoint.pciInfo.device = connInfo.dstEndPoint.pciInfo.device;
        msgReq.interNodeConns.connInfo[idx].dstEndPoint.pciInfo.function = connInfo.dstEndPoint.pciInfo.function;
        idx++;
    }
    msgReq.interNodeConns.count = idx;
    // now tell client to add connections
    gAcceptedSocket->SendBytes((char*) &msgReq, sizeof (msgReq));
}

static void handle_add_internode_connections_client(MsgHdr &addMsg)
{
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

        // since when the internode connection is created, srcEndPoint is treated as
        // local endpoint. So use destEndpoint as localEndPoint for client
        lwlink_remote_endpoint_info remoteEndPoint;
        remoteEndPoint.pciInfo = connInfo.srcEndPoint.pciInfo;
        remoteEndPoint.linkIndex = connInfo.srcEndPoint.linkIndex;
        remoteEndPoint.nodeId = connInfo.srcEndPoint.nodeId;
        remoteEndPoint.devType = 2; // GPU - TODO
        add_internode_connections(connInfo.dstEndPoint, remoteEndPoint);

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
                std::cout <<" CMD_START_READ_DISCOVERY_TOKEN\n";
                read_discovery_token_client();
                break;
            }
            case CMD_START_INTERNODE_CONN_ADD: {
                handle_add_internode_connections_client(msgReq);
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

    discover_internode_connections_server();
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
    SocketClient* acceptedSocket; //on server side, indicate an accepted connection
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
                 discover_internode_connections_server();
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

