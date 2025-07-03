#include <stdio.h>
#include <iostream>
#include <map>
#include <fstream>

#include "master.h"
#include "helper.h"

using namespace std;

static int num_nodes;
LWLinkConnList internodeConnListServer; //internode connection list on server side.
std::map<int, LWLinkConnList> intranodeConnListServerMap;
std::map<uint64, uint32> sidToNodeIdMap;
std::map<uint64, uint64> sidToGpuOrSwitchIdMap;
std::map<uint64, LwU16> sidToDomainMap;
std::map<uint64, LwU8> sidToBusMap;
std::map<uint64, LwU8> sidToDeviceMap;
std::map<uint64, LwU8> sidToFunctionMap;
std::map<uint64, LwU32> sidToDevTypeMap;
std::map<int, lwlink_get_devices_info> deviceInfoMap;

void startMaster(int nodes) {
	num_nodes = nodes;
	doAndSendDeviceInfoReq();
	printDeviceInfo();
	doAndSendSetInitphase1Req();
    doAndSendRxInitTerm();
    doAndSendSetRxDetect();
    doAndSendGetRxDetect();
    doAndSendEnableCommonMode();
    doAndSendCalibrateDevices();
    doAndSendDisableCommonMode();
    doAndSendEnableDevicesData();
    doAndSendSetInitphase5Req();
    doAndSendDoLinkInit();
    doAndSendDoInitNegotiate();
    doAndSendDiscoverIntraConnections();
    getAllIntraConnections();
    printIntraNodeConns();

    //do multi-node stuff
    discoverInterNodeConnections();
    displayInterNodeConnections();
    addInterNodeConnections();

    // do link training
    doAndSendIntraNodeTraining(lwlink_train_conn_swcfg_to_active);
    doInterNodeTraining(lwlink_train_conn_swcfg_to_active);
}

int setNumNodes(int nodes)
{
	num_nodes = nodes;
}

void doAndSendDeviceInfoReq()
{
	lwlink_get_devices_info deviceInfo = get_device_information();

	if (deviceInfo.status != LWL_SUCCESS) {
		//send failure response
		PRINT_VERBOSE_ERRORS << "fail"<<endl;
	}

	deviceInfoMap.insert(make_pair(getNodeId(), deviceInfo));

    for (int node = 1; node < num_nodes; node++) {
		sendMasterReqMsg(GET_DEVICE_INFO, node);

	    //wait for response
	    std::string recvdMsg = recvMessage(node);
		parseMessage(recvdMsg);
	}
}

std::string getDeviceName(int node, lwlink_pci_dev_info pciInfo)
{
	lwlink_get_devices_info deviceInfo = deviceInfoMap[node];
	for (int idx = 0; idx < deviceInfo.numDevice; idx++) {
		if (deviceInfo.devInfo[idx].pciInfo.domain == pciInfo.domain
			&& deviceInfo.devInfo[idx].pciInfo.bus == pciInfo.bus
			&& deviceInfo.devInfo[idx].pciInfo.device == pciInfo.device
			&& deviceInfo.devInfo[idx].pciInfo.function == pciInfo.function) 
		{
			return deviceInfo.devInfo[idx].deviceName;
		}

	}

	return "";
}

void printDeviceInfo()
{
	for (int node = 0; node < num_nodes; node++) {
		PRINT_VERBOSE << "printing device info for node " << node << endl;
		lwlink_get_devices_info deviceInfo = deviceInfoMap[node];
		print_device_information(deviceInfo);
	}
}

void sendMasterReqMsg(int init_action, int node, void *arg)
{
	Json::Value initReqMsg(Json::objectValue);
	switch (init_action) {
		case GET_DEVICE_INFO:
			{
				initReqMsg["type"] = MSG_GET_DEVICES_INFO_REQ;		
				break;
			}
		case INIT_PHASE1:
			{
				initReqMsg["type"] = MSG_SET_INIT_PHASE1_REQ;		
				break;
			}
		case RX_INIT_TERM:
			{
				initReqMsg["type"] = MSG_SET_RX_INIT_TERM_REQ;
				break;
			}
		case SET_RX_DETECT:
			{
				initReqMsg["type"] = MSG_SET_RX_DETECT_REQ;
				break;
			}
		case GET_RX_DETECT:
			{
				initReqMsg["type"] = MSG_GET_RX_DETECT_REQ;
				break;
			}
		case ENABLE_COMMON_MODE:
			{
				initReqMsg["type"] = MSG_ENABLE_COMMON_MODE_REQ;
				break;
			}
		case CALIBRATE_DEVICES:
			{
				initReqMsg["type"] = MSG_CALIBRATE_DEVICES_REQ;
				break;
			}
		case DISABLE_COMMON_MODE:
			{
				initReqMsg["type"] = MSG_DISABLE_COMMON_MODE_REQ;
				break;
			}
		case ENABLE_DEVICES_DATA:
			{
				initReqMsg["type"] = MSG_ENABLE_DEVICES_DATA_REQ;
				break;
			}
		case INIT_PHASE5:
			{
				initReqMsg["type"] = MSG_SET_INIT_PHASE5_REQ;
				break;
			}
		case DO_LINK_INIT:
			{
				initReqMsg["type"] = MSG_DO_LINK_INIT_REQ;
				break;
			}
		case DO_INITNEGOTIATE:
			{
				initReqMsg["type"] = MSG_DO_INITNEGOTIATE_REQ;
				break;
			}
		case DISCOVER_INTRA_CONNECTIONS:
			{
				initReqMsg["type"] = MSG_DISCOVER_INTRA_CONNECTIONS_REQ;
				break;
			}
		case WRITE_DISCOVERY_TOKENS:
			{
				initReqMsg["type"] = MSG_WRITE_DISCOVERY_TOKENS_REQ;
				break;
			}
		case READ_DISCOVERY_TOKENS:
			{
				initReqMsg["type"] = MSG_READ_DISCOVERY_TOKENS_REQ;
				break;
			}
		case DISPLAY_INTRA_NODE_CONNS:
			{
				initReqMsg["type"] = MSG_DISPLAY_INTRA_NODE_CONNS_REQ;
				break;
			}
		case READ_SIDs:
			{
				initReqMsg["type"] = MSG_READ_SIDs_REQ;
				break;
			}
		case GET_INTRANODE_CONNS:
			{
				initReqMsg["type"] = MSG_GET_INTRANODE_CONNS_REQ;
				break;
			}
		case INTRA_NODE_CONNS_TRAIN:
			{
				initReqMsg["type"] = MSG_INTRA_NODE_CONNS_TRAIN_REQ;
				int *trainTo = (int*) arg; 
				initReqMsg["trainTo"] = *trainTo;
				break;
			}
		case APP_EXIT:
			{
				initReqMsg["type"] = MSG_APP_EXIT_REQ;
				break;
			}	
		default: 
			{
				PRINT_VERBOSE_ERRORS << "No such message type from Master " << init_action << endl;
				return;
			}
	}

	sendMessage(initReqMsg, node);
}

void doAndSendSetInitphase1Req()
{
	// do init phase 1 for this node
	if (!set_initphase1()) {
		PRINT_VERBOSE << "Setting initphase1 failed for node " << getNodeId();
	}

	for (int node = 1; node < num_nodes; node++) {
		sendMasterReqMsg(INIT_PHASE1, node);

	    //wait for response
	    std::string recvdMsg = recvMessage(node);
		parseMessage(recvdMsg);
	}

	PRINT_VERBOSE << "Setting initphase1 successfully for all nodes"<<endl;
}

void doAndSendRxInitTerm()
{
	// do set rx init for this node
	if (!rx_init_term()) {
		PRINT_VERBOSE << "Setting rx init term failed for node " << getNodeId();
	}

	for (int node = 1; node < num_nodes; node++) {
		sendMasterReqMsg(RX_INIT_TERM, node);

	    //wait for response
	    std::string recvdMsg = recvMessage(node);
		parseMessage(recvdMsg);
	}

	PRINT_VERBOSE << "Setting rx init term successfully for all nodes"<<endl;
}

void doAndSendSetRxDetect()
{
	// do set rx detect for this node
	if (!set_rx_detect()) {
		PRINT_VERBOSE << "Setting rx detect failed for node " << getNodeId();
	}

	for (int node = 1; node < num_nodes; node++) {
		sendMasterReqMsg(SET_RX_DETECT, node);

	    //wait for response
	    std::string recvdMsg = recvMessage(node);
		parseMessage(recvdMsg);
	}

	PRINT_VERBOSE << "Setting rx detect successfully for all nodes"<<endl;
}

void doAndSendGetRxDetect()
{
	// do get rx detect for this node
	if (!get_rx_detect()) {
		PRINT_VERBOSE_ERRORS << "Getting rx detect failed for node " << getNodeId();
	}

	for (int node = 1; node < num_nodes; node++) {
		sendMasterReqMsg(GET_RX_DETECT, node);

	    //wait for response
	    std::string recvdMsg = recvMessage(node);
		parseMessage(recvdMsg);
	}

	PRINT_VERBOSE << "Getting rx detect successfully for all nodes"<<endl;
}

void doAndSendEnableCommonMode()
{
	// do get rx detect for this node
	if (!enable_devices_common_mode()) {
		PRINT_VERBOSE_ERRORS << "enabling common mode failed for node " << getNodeId();
	}

	for (int node = 1; node < num_nodes; node++) {
		sendMasterReqMsg(ENABLE_COMMON_MODE, node);

	    //wait for response
	    std::string recvdMsg = recvMessage(node);
		parseMessage(recvdMsg);
	}

	PRINT_VERBOSE << "enabling common mode successfully for all nodes"<<endl;
}

void doAndSendCalibrateDevices()
{
	// do get rx detect for this node
	if (!calibrate_devices()) {
		PRINT_VERBOSE_ERRORS << "calibrating devices failed for node " << getNodeId();
	}

	for (int node = 1; node < num_nodes; node++) {
		sendMasterReqMsg(CALIBRATE_DEVICES, node);

	    //wait for response
	    std::string recvdMsg = recvMessage(node);
		parseMessage(recvdMsg);
	}

	PRINT_VERBOSE << "calibrating successfully for all nodes"<<endl;
}

void doAndSendDisableCommonMode()
{
	// do get rx detect for this node
	if (!disable_devices_common_mode()) {
		PRINT_VERBOSE_ERRORS << "disable common mode failed for node " << getNodeId();
	}

	for (int node = 1; node < num_nodes; node++) {
		sendMasterReqMsg(DISABLE_COMMON_MODE, node);

	    //wait for response
	    std::string recvdMsg = recvMessage(node);
		parseMessage(recvdMsg);
	}

	PRINT_VERBOSE << "disable common mode successfully for all nodes"<<endl;
}

void doAndSendEnableDevicesData()
{
	// do get rx detect for this node
	if (!enable_devices_data()) {
		PRINT_VERBOSE_ERRORS << "enable devices data failed for node " << getNodeId();
	}

	for (int node = 1; node < num_nodes; node++) {
		sendMasterReqMsg(ENABLE_DEVICES_DATA, node);

	    //wait for response
	    std::string recvdMsg = recvMessage(node);
		parseMessage(recvdMsg);
	}

	PRINT_VERBOSE << "enable devices data successfully for all nodes"<<endl;
}

void doAndSendSetInitphase5Req()
{
    // do init phase 5 for this node
    if (!set_initphase5()) {
        PRINT_VERBOSE << "Setting initphase5 failed for node " << getNodeId();
    }

    for (int node = 1; node < num_nodes; node++) {
        sendMasterReqMsg(INIT_PHASE5, node);

        //wait for response
        std::string recvdMsg = recvMessage(node);
        parseMessage(recvdMsg);
    }

    PRINT_VERBOSE << "Setting initphase5 successfully for all nodes"<<endl;
}

void doAndSendDoLinkInit()
{
	// do get rx detect for this node
	if (!do_link_init()) {
		PRINT_VERBOSE_ERRORS << "link init failed for node " << getNodeId();
	}

	for (int node = 1; node < num_nodes; node++) {
		sendMasterReqMsg(DO_LINK_INIT, node);

	    //wait for response
	    std::string recvdMsg = recvMessage(node);
		parseMessage(recvdMsg);
	}

	PRINT_VERBOSE << "link init successful for all nodes"<<endl;
}

void doAndSendDoInitNegotiate()
{	
	uint64 startTime = lwrrent_timestamp();
	for (int node = 1; node < num_nodes; node++) {
		sendMasterReqMsg(DO_INITNEGOTIATE, node);
	}

	// do init_negotiate for this node
	if (!do_initnegotiate()) {
		PRINT_VERBOSE_ERRORS << "init_negotiate failed for node " << getNodeId();
	}


	for (int node = 1; node < num_nodes; node++) {
		std::string recvdMsg = recvMessage(node);
		parseMessage(recvdMsg);
	}
	uint64 endTime = lwrrent_timestamp();
	PRINT_VERBOSE << "time taken is " << endTime - startTime << " milli seconds " << std::endl;
	PRINT_VERBOSE << "init_negotiate successful for all nodes"<<endl;
}

void doAndSendDiscoverIntraConnections()
{
	// do get rx detect for this node
	if (!discover_intra_connections()) {
		PRINT_VERBOSE_ERRORS << "discover_intra_connections failed for node " << getNodeId();
	}

	for (int node = 1; node < num_nodes; node++) {
		sendMasterReqMsg(DISCOVER_INTRA_CONNECTIONS, node);

	    //wait for response
	    std::string recvdMsg = recvMessage(node);
		parseMessage(recvdMsg);
	}

	PRINT_VERBOSE << "discover_intra_connections successful for all nodes"<<endl;
}

void getAllIntraConnections()
{
	for (int node = 0; node < num_nodes; node++) {
		if ( node == 0 ) {
			LWLinkConnList connList = getIntraConns();
			intranodeConnListServerMap[node] = connList;
			continue;
		}

		sendMasterReqMsg(GET_INTRANODE_CONNS, node);

		//wait for response
	    std::string recvdMsg = recvMessage(node);
		parseMessage(recvdMsg);
	}

	PRINT_VERBOSE << "getAllIntraConnections successful for all nodes"<<endl;
}

void printIntraNodeConns()
{
	std::map<int, LWLinkConnList>::iterator it;
	for (it = intranodeConnListServerMap.begin(); it != intranodeConnListServerMap.end(); it++) {
		int nodeId = it->first;
		LWLinkConnList connList = it->second;
		PRINT_VERBOSE << "Displaying intra node connections for node " << nodeId << endl;
		display_connections(connList);
	}
}

void discoverInterNodeConnections()
{
	if (getArch() == LWSWITCH_ARCH_TYPE_SV10) {
		if (num_nodes == 1) return;
		// first write from one node and read from all nodes
		DiscoveryTokenList writeList;
		for (int node = 0; node < num_nodes; node++) {
			writeDiscoveryTokenReq(node, writeList);
		}

		std::map<int, DiscoveryTokenList> readListMap;
		for (int node = 0; node < num_nodes; node++) {
			readDiscoveryTokenReq(node, readListMap);
		}

		correlateDiscoveryTokens(writeList, readListMap);
	} 
	else {
		SidInfoList sidList;
		//read sids for all nodes
		readLinkSids(sidList);

		correlateInterNodeConnsWithSids(sidList);
	}
}

void writeDiscoveryTokenReq(int nodeId, DiscoveryTokenList &writeList)
{
	if (nodeId == 0) {
		if (!write_discovery_tokens(writeList)) {
			PRINT_VERBOSE_ERRORS << "write_discovery_token failed" << endl;
			exit(0);
		}
		return;
	}

	// send request to node and recieve write_discovery_token_info
	sendMasterReqMsg(WRITE_DISCOVERY_TOKENS, nodeId);

	//wait for response
    std::string recvdMsg = recvMessage(nodeId);
	parseMessage(recvdMsg, (void*) &writeList);
}

void readDiscoveryTokenReq(int nodeId, std::map<int, DiscoveryTokenList> &readListMap)
{
	DiscoveryTokenList readList;
	if (nodeId == 0) {
		if (!read_discovery_tokens(readList)) {
			PRINT_VERBOSE_ERRORS << "read_discovery_token failed" << endl;
			exit(0);
		}
		return;
	}

	// send request to node and recieve write_discovery_token_info
	sendMasterReqMsg(READ_DISCOVERY_TOKENS, nodeId);

	//wait for response
    std::string recvdMsg = recvMessage(nodeId);
	parseMessage(recvdMsg, (void*) &readList);

	readListMap[nodeId] = readList;
	// // iterate through all nodes and read discovery tokens
	// for (int node = 0; node < num_nodes; node++) {
	// 	if (node == nodeId) {
	// 		continue;
	// 	}

	// 	DiscoveryTokenList readList;
	// 	if (node == 0) {
	// 		if (!read_discovery_tokens(readList)) {
	// 			PRINT_VERBOSE_ERRORS << "read_discovery_token failed" << endl;
	// 			exit(0);
	// 		}
	// 	} else {
	// 		// send request to node and recieve write_discovery_token_info
	// 		sendMasterReqMsg(READ_DISCOVERY_TOKENS, node);

	// 		//wait for response
	// 	    std::string recvdMsg = recvMessage(node);
	// 		parseMessage(recvdMsg, (void*) &readList);
	// 	}

	// 	readListMap[nodeId] = readList;
	// }
}

void readLinkSids(SidInfoList &sidList)
{
	for (int node = 0; node < num_nodes; node++) {
		if (node == 0) {
			read_SIDs(sidList);
			continue;
		}

		sendMasterReqMsg(READ_SIDs, node);

		//wait for response
	    std::string recvdMsg = recvMessage(node);
		parseMessage(recvdMsg, (void*) &sidList);
	}

	SidInfoList::iterator it;
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
}

void correlateDiscoveryTokens(DiscoveryTokenList &writeList,
							  std::map<int, DiscoveryTokenList> &readListMap)
{
	DiscoveryTokenList readList;
	std::map<int, DiscoveryTokenList>::iterator it;
	for (it = readListMap.begin(); it != readListMap.end(); it++) {
		DiscoveryTokenList tempList = it->second;
		DiscoveryTokenList::iterator jit;
		for (jit = tempList.begin(); jit != tempList.end(); jit++) {
			readList.push_back(*jit);
		}
	}

	DiscoveryTokenList::iterator wit;
	DiscoveryTokenList::iterator rit;

	for (wit = writeList.begin(); wit != writeList.end();) {
		bool bFound = false;
		for (rit = readList.begin(); rit != readList.end();) {
			DiscoveryTokenInfo writeInfo = *wit;
			DiscoveryTokenInfo readInfo = *rit;

			if (readInfo.tokelwalue == writeInfo.tokelwalue) {
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
				bFound = true;
				break;
			} else {
				rit++;
			}
		}

		if (bFound == true) {
			wit = writeList.erase(wit);
		} else {
			wit++;
		}
	}
}

static bool connectionDoesNotExist(lwlink_connection_info conn)
{
    LWLinkConnList::iterator it = internodeConnListServer.begin();
    for (it = internodeConnListServer.begin(); it != internodeConnListServer.end(); it++) {
        // first check
        lwlink_connection_info tempConn = *it;

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
          	return false;
        }
    }

    return true;
}

void correlateInterNodeConnsWithSids(SidInfoList &sidList)
{
    for (auto it = sidList.begin(); it != sidList.end(); it++ ) {
		lwlink_connection_info connInfo;
		connInfo.srcEndPoint.nodeId = sidToNodeIdMap[ it->nearSid ];
		connInfo.srcEndPoint.linkIndex = it->nearLinkIndex;
		connInfo.srcEndPoint.pciInfo.domain = sidToDomainMap[it->nearSid];
		connInfo.srcEndPoint.pciInfo.bus = sidToBusMap[it->nearSid];
		connInfo.srcEndPoint.pciInfo.device = sidToDeviceMap[it->nearSid];
		connInfo.srcEndPoint.pciInfo.function =  sidToFunctionMap[it->nearSid];

		if (sidToNodeIdMap.find(it->farSid) == sidToNodeIdMap.end())
		{
			continue;
		}

		connInfo.dstEndPoint.nodeId = sidToNodeIdMap[ it->farSid ];
		connInfo.dstEndPoint.linkIndex = it->farLinkIndex;
		connInfo.dstEndPoint.pciInfo.domain = sidToDomainMap[it->farSid];
		connInfo.dstEndPoint.pciInfo.bus = sidToBusMap[it->farSid];
		connInfo.dstEndPoint.pciInfo.device = sidToDeviceMap[it->farSid];
		connInfo.dstEndPoint.pciInfo.function =  sidToFunctionMap[it->farSid];

        // setBusToPhyId(sidToNodeIdMap[ it->nearSid ], sidToBusMap[it->nearSid], readInfo.phyId);
        if (!isDuplicateConnection(connInfo, internodeConnListServer)) {
            if(connInfo.srcEndPoint.nodeId != connInfo.dstEndPoint.nodeId) {
                /*PRINT_VERBOSE << "\tadding the following internode connection \n";
                printConnInfo(connInfo);
                */
                internodeConnListServer.push_back( connInfo );
            }
        }
    }
}

void displayInterNodeConnections()
{
	PRINT_VERBOSE << "Display Inter node connections"<<endl;
	display_connections(internodeConnListServer);
}

void addInterNodeConnections()
{
	LWLinkConnList::iterator it;
	for (it = internodeConnListServer.begin(); it != internodeConnListServer.end(); it++) {
		lwlink_connection_info connInfo = *it;

		// add to master end
		int toNodeId = connInfo.srcEndPoint.nodeId;
		sendAddConnectionInfoToNode(toNodeId, connInfo.srcEndPoint, connInfo.dstEndPoint);

		//add to slave end
		toNodeId = connInfo.dstEndPoint.nodeId;
		sendAddConnectionInfoToNode(toNodeId, connInfo.dstEndPoint, connInfo.srcEndPoint);
	}

	PRINT_VERBOSE << "added all connections successfully"<<endl;
}

void sendAddConnectionInfoToNode(int nodeId, lwlink_endpoint srcEndPoint, lwlink_endpoint dstEndPoint)
{
	if (nodeId == 0) {
		lwlink_remote_endpoint_info remoteEndPoint;
		remoteEndPoint.nodeId = dstEndPoint.nodeId;
		remoteEndPoint.linkIndex = dstEndPoint.linkIndex;
		remoteEndPoint.pciInfo.domain = dstEndPoint.pciInfo.domain;
		remoteEndPoint.pciInfo.bus = dstEndPoint.pciInfo.bus;
		remoteEndPoint.pciInfo.device = dstEndPoint.pciInfo.device;
		remoteEndPoint.pciInfo.function = dstEndPoint.pciInfo.function;
		remoteEndPoint.devType = getDevType(deviceInfoMap[dstEndPoint.nodeId], dstEndPoint.pciInfo); 
		if (!add_internode_connections(srcEndPoint, remoteEndPoint)) {
			PRINT_VERBOSE_ERRORS << "add internode conns failed for node " << nodeId;
			exit(0);
		}
	} else {
		Json::Value interNodeConnInfoMsg(Json::objectValue);
		Json::Value srcEndPointInfo(Json::objectValue);
		Json::Value remoteEndPointInfo(Json::objectValue);
	    
	    srcEndPointInfo["nodeId"] = srcEndPoint.nodeId;
	    srcEndPointInfo["linkIndex"] = srcEndPoint.linkIndex;
	    srcEndPointInfo["pciInfo"] = getPciInfoJson(srcEndPoint);

	    remoteEndPointInfo["nodeId"] = dstEndPoint.nodeId;
	    remoteEndPointInfo["linkIndex"] = dstEndPoint.linkIndex;
	    remoteEndPointInfo["pciInfo"] = getPciInfoJson(dstEndPoint);
	    remoteEndPointInfo["devType"] = getDevType(deviceInfoMap[dstEndPoint.nodeId], dstEndPoint.pciInfo); 

	    interNodeConnInfoMsg["type"] = MSG_ADD_INTERNODE_CONNS_REQ;
	    interNodeConnInfoMsg["src_end_point"] = srcEndPointInfo;
	    interNodeConnInfoMsg["remote_end_point"] = remoteEndPointInfo;
	    sendMessage(interNodeConnInfoMsg, nodeId);

	    //wait for response
	    std::string recvdMsg = recvMessage(nodeId);
		parseMessage(recvdMsg);
	}
}

void doAndSendIntraNodeTraining(lwlink_conn_train_type trainTo)
{
	if (getArch() == LWSWITCH_ARCH_TYPE_SV10) {
		for (int node = 0; node < num_nodes; node++) {
			if (node == 0) {
				train_all_intra_connections(trainTo);
				continue;
			}

			sendMasterReqMsg(INTRA_NODE_CONNS_TRAIN, node, (void*) &trainTo);

			//wait for response
		    std::string recvdMsg = recvMessage(node);
			parseMessage(recvdMsg);
		}
	} else {
		doIntraNodeParallelTraining(trainTo);
	}

	PRINT_VERBOSE << "all intra connections trained to " << trainTo << " successfully"<<endl; 
}

void doIntraNodeParallelTraining(lwlink_conn_train_type trainTo)
{
	for (int node = 0; node < num_nodes; node++) {
		if (node == 0) {
			train_intra_conn_parallel(trainTo, intranodeConnListServerMap[node]);
			continue;
		}

		sendParallelTrainLinkMessage(node, trainTo);
	}

}

void sendParallelTrainLinkMessage(int node, lwlink_conn_train_type trainTo)
{
	int idx = 0;
	Json::Value parallelLinkTrainMessage(Json::objectValue);
	Json::Value parallelLinkTrainConnList(Json::arrayValue);
	Json::Value parallelLinkTrainConn(Json::objectValue);
	parallelLinkTrainMessage["type"] = MSG_PARALLEL_LINK_TRAIN_REQ;
	parallelLinkTrainMessage["nodeId"] = node;
	parallelLinkTrainMessage["trainTo"] = trainTo;

	LWLinkConnList connList = intranodeConnListServerMap[node];
	LWLinkConnList::iterator it;
	for (it = connList.begin(); it != connList.end(); it++) {
		lwlink_connection_info connInfo = *it;
		colwertConnInfoToJson(connInfo.srcEndPoint, connInfo.dstEndPoint, parallelLinkTrainConn);
		parallelLinkTrainConnList.append(parallelLinkTrainConn);
		idx++;

		if (idx == LWLINK_MAX_PARALLEL_CONNS_TRAIN_COUNT) {
			idx = 0;
			parallelLinkTrainMessage["parallel_link_train_list"] = parallelLinkTrainConnList;
			sendMessage(parallelLinkTrainMessage, node);
			//wait for response
		    std::string recvdMsg = recvMessage(node);
			parseMessage(recvdMsg);
			parallelLinkTrainConnList.clear();
		}
	}

	if (idx != 0) {
		parallelLinkTrainMessage["parallel_link_train_list"] = parallelLinkTrainConnList;
		sendMessage(parallelLinkTrainMessage, node);

		//wait for response
	    std::string recvdMsg = recvMessage(node);
		parseMessage(recvdMsg);
	}
}

void doInterNodeTraining(lwlink_conn_train_type trainTo)
{
	if (getArch() == LWSWITCH_ARCH_TYPE_SV10) {
		interNodeTrain(trainTo, false);
	} else {
		//do safe to initoptimzie first
		sendParallelOptimizeReq("INIT_OPTIMIZE");
		sleep( WAIT_BETWEEN_PARELLEL_LINK_TRAINING_STEPS );
		//do post initoptimize
		sendParallelOptimizeReq("POST_INITOPTIMIZE");
		//do link training
		interNodeTrain(trainTo, true);
	}
}

void interNodeTrain(lwlink_conn_train_type trainTo, bool skipSublinkTraining)
{
	LWLinkConnList::iterator it;
	int idx = 1;
	for (it = internodeConnListServer.begin(); it != internodeConnListServer.end(); it++) {
		lwlink_connection_info connInfo = *it;
		int srcNodeId = connInfo.srcEndPoint.nodeId;
		int dstNodeId = connInfo.dstEndPoint.nodeId;

		if (!skipSublinkTraining)
			sendLinkConnTrainMessage(connInfo, SUBLINK_INTERNODE_CONN_TRAIN, trainTo);

		sendLinkConnTrainMessage(connInfo, MAINLINK_INTERNODE_CONN_TRAIN, trainTo);
		// int toNodeId = connInfo.srcEndPoint.nodeId;
		// if (toNodeId == 0) {
		// 	sendTrainMessageAndWaitForConf(connInfo, trainTo);
		// 	continue;
		// } 
		// sendStartInterNodeTrainReqMessage(connInfo, trainTo);
		// idx++;
	}
}

void sendLinkConnTrainMessage(lwlink_connection_info connInfo, int train_type, lwlink_conn_train_type trainTo)
{
	int srcNodeId = connInfo.srcEndPoint.nodeId;
	int dstNodeId = connInfo.dstEndPoint.nodeId;

	PRINT_VERBOSE_DEBUG << "src node id " << srcNodeId << " , dstNodeId " << dstNodeId << endl;

	Json::Value interNodeConn(Json::objectValue);
	Json::Value interNodeConnMsgReq(Json::objectValue);

	//printConnInfo(connInfo);

	colwertConnInfoToJson(connInfo.srcEndPoint, connInfo.dstEndPoint, interNodeConn);

	interNodeConnMsgReq["conn_info"] = interNodeConn;
	switch (train_type) {
		case SUBLINK_INTERNODE_CONN_TRAIN:
			{
				interNodeConnMsgReq["type"] = MSG_SUBLINK_INTERNODE_CONN_TRAIN_REQ;
				interNodeConnMsgReq["trainTo"] = lwlink_train_sublink_safe_to_hs;
				break;
			}
		case MAINLINK_INTERNODE_CONN_TRAIN:
			{
				interNodeConnMsgReq["type"] = MSG_MAINLINK_INTERNODE_CONN_TRAIN_REQ;
				interNodeConnMsgReq["trainTo"] = trainTo;
				break;
			}
		default: 
			{
				PRINT_VERBOSE << "no such train option " << endl;
				break;
			}
	}

	//send message to src
	if (srcNodeId == 0) {
		if (train_type == SUBLINK_INTERNODE_CONN_TRAIN)
			set_sublink_state(lwlink_train_sublink_safe_to_hs, true, connInfo.srcEndPoint);
		else
			set_mainlink_state((lwlink_link_train_type)trainTo, true, connInfo.srcEndPoint);
	} else {
		interNodeConnMsgReq["isMasterEnd"] = true;
		sendMessage(interNodeConnMsgReq, connInfo.srcEndPoint.nodeId);

		//wait for response
	    std::string recvdMsg = recvMessage(connInfo.srcEndPoint.nodeId);
		parseMessage(recvdMsg);
	}

	if (dstNodeId == 0) {
		if (train_type == SUBLINK_INTERNODE_CONN_TRAIN)
			set_sublink_state(lwlink_train_sublink_safe_to_hs, false, connInfo.dstEndPoint);
		else
			set_mainlink_state((lwlink_link_train_type)trainTo, false, connInfo.dstEndPoint);
	} else {
		interNodeConnMsgReq["isMasterEnd"] = false;
		sendMessage(interNodeConnMsgReq, connInfo.dstEndPoint.nodeId);

		// wait for response
	    std::string recvdMsg = recvMessage(connInfo.dstEndPoint.nodeId);
		parseMessage(recvdMsg);
	}
}

void sendTrainMessageAndWaitForConf(lwlink_connection_info connInfo, lwlink_conn_train_type trainTo)
{
	Json::Value interNodeConn(Json::objectValue);
	Json::Value interNodeConnMsgReq(Json::objectValue);

	printConnInfo(connInfo);

	colwertConnInfoToJson(connInfo.srcEndPoint, connInfo.dstEndPoint, interNodeConn);

	interNodeConnMsgReq["conn_info"] = interNodeConn;
	interNodeConnMsgReq["type"] = MSG_SUBLINK_INTERNODE_CONN_TRAIN_MASTER_REQ;
	interNodeConnMsgReq["trainTo"] = trainTo;

	sendMessage(interNodeConnMsgReq, connInfo.dstEndPoint.nodeId);

	//wait for response
	int recvConf = 0;
    std::string recvdMsg = recvMessage(connInfo.dstEndPoint.nodeId);
	parseMessage(recvdMsg, (void*) &recvConf);

	if (recvConf == SLAVE_SUBLINK_STATE_REQ_RECVD) {
		set_sublink_state(lwlink_train_sublink_safe_to_hs, true, connInfo.srcEndPoint);
	} 

	//now send request to train to high speed
	interNodeConnMsgReq["type"] = MSG_MAINLINK_INTERNODE_CONN_TRAIN_MASTER_REQ;
	sendMessage(interNodeConnMsgReq, connInfo.dstEndPoint.nodeId);

	recvdMsg = recvMessage(connInfo.dstEndPoint.nodeId);
	parseMessage(recvdMsg, (void*) &recvConf);

	if (recvConf == SLAVE_MAINLINK_STATE_REQ_RECVD) {
		set_mainlink_state(lwlink_train_link_swcfg_to_active, true, connInfo.srcEndPoint);
	}
}

void sendStartInterNodeTrainReqMessage(lwlink_connection_info connInfo, lwlink_conn_train_type trainTo)
{
	Json::Value startInterNodeTrainReqMessage(Json::objectValue);
	Json::Value interNodeConn(Json::objectValue);
	startInterNodeTrainReqMessage["type"] = MSG_START_INTERNODE_TRAIN_REQ;

	printConnInfo(connInfo);

	colwertConnInfoToJson(connInfo.srcEndPoint, connInfo.dstEndPoint, interNodeConn);

	startInterNodeTrainReqMessage["conn_info"] = interNodeConn;
	startInterNodeTrainReqMessage["trainTo"] = trainTo;

	sendMessage(startInterNodeTrainReqMessage, connInfo.srcEndPoint.nodeId);

	int reqComplete = 0;
	std::string recvdMsg = recvMessage(connInfo.dstEndPoint.nodeId);
	parseMessage(recvdMsg, (void*) &reqComplete);

	if (reqComplete == 0) {
		PRINT_VERBOSE << "training link failed " <<endl;
	}
}

void sendParallelOptimizeReq(std::string optimizeType)
{
	for (int node = 0; node < num_nodes; node++) {
		LWLinkConnList::iterator it;
		Json::Value optimizeLinkTrainEndPoint(Json::objectValue);
		Json::Value optimizeLinkTrainList(Json::arrayValue);
		Json::Value optimizeReqMsg(Json::objectValue);
		int endPointCount = 0;
		for (it = internodeConnListServer.begin(); it != internodeConnListServer.end(); it++) 
		{
			lwlink_connection_info connInfo = *it;
			if (connInfo.srcEndPoint.nodeId != node && connInfo.dstEndPoint.nodeId != node) {
				continue;
			}
			lwlink_endpoint ePoint;
			ePoint = connInfo.srcEndPoint.nodeId == node ? connInfo.srcEndPoint : connInfo.dstEndPoint;

			optimizeLinkTrainEndPoint["master_node_id"] = ePoint.nodeId;
			optimizeLinkTrainEndPoint["pci_info"] = getPciInfoJson(ePoint);
			optimizeLinkTrainEndPoint["link_index"] = ePoint.linkIndex;
			optimizeLinkTrainList.append(optimizeLinkTrainEndPoint);
			endPointCount++;
		}

		optimizeReqMsg["optimize_type"] = optimizeType;
		optimizeReqMsg["type"] = MSG_PARALLEL_LINK_TRAIN_OPTIMIZE_REQ;
		optimizeReqMsg["nodeId"] = node;
		optimizeReqMsg["end_point_count"] = endPointCount;
		optimizeReqMsg["end_point_list"] = optimizeLinkTrainList;

		if (node == 0) {
			bool status = doOptimizeReq(optimizeReqMsg);
		} else {
			sendMessage(optimizeReqMsg, node);

			std::string recvdMsg = recvMessage(node);
			parseMessage(recvdMsg);
		}
	}
}

void doTrainIntraConnection(lwlink_conn_train_type trainTo, int numNode, int connIdx)
{
	if (numNode == 0) {
		train_intra_connection(trainTo, connIdx);
		return;
	}

	Json::Value trainConnectionMsg(Json::objectValue);
	trainConnectionMsg["connIdx"] = connIdx;
	trainConnectionMsg["trainTo"] = trainTo;
	trainConnectionMsg["type"] = MSG_SINGLE_INTRANODE_CONN_TRAIN_REQ;
	
	sendMessage(trainConnectionMsg, numNode);
	
	std::string recvdMsg = recvMessage(numNode);
	parseMessage(recvdMsg);
	return;
}

void sendExitMessage()
{
	for (int node = 1; node < num_nodes; node++) {
		sendMasterReqMsg(APP_EXIT, node);
	}
}

static void parseMessage(std::string recvMsg, void *infoFromMsg) {
	Json::Reader reader;
	Json::Value message(Json::objectValue);
	if (reader.parse(recvMsg, message) == false) {
		PRINT_VERBOSE_ERRORS <<"Message received but could not be parsed"<<endl;
	}
	int msg_type = message["type"].asInt();
	switch (msg_type) {
		case MSG_GET_DEVICES_INFO_RESP: 
			{
				handleGetDeviceInfoRespMsg(message);
				break;
			}
		case MSG_SET_INIT_PHASE1_RESP:
			{
				handleInitRespMessage(message, "init_phase1");
				break;
			}
		case MSG_SET_RX_INIT_TERM_RESP: 
			{
				handleInitRespMessage(message, "set_rx_init");
				break;
			}
		case MSG_SET_RX_DETECT_RESP:
			{
				handleInitRespMessage(message, "set_rx_detect");
				break;
			}
		case MSG_GET_RX_DETECT_RESP: 
			{
				handleInitRespMessage(message, "get_rx_detect");
				break;
			}
		case MSG_ENABLE_COMMON_MODE_RESP:
			{
				handleInitRespMessage(message, "enable_common_mode");
				break;
			}
		case MSG_CALIBRATE_DEVICES_RESP:
			{
				handleInitRespMessage(message, "calibrate_devices");
				break;
			}
		case MSG_DISABLE_COMMON_MODE_RESP:
			{
				handleInitRespMessage(message, "disable_common_mode");
				break;
			}
		case MSG_ENABLE_DEVICES_DATA_RESP:
			{
				handleInitRespMessage(message, "enable_devices_data");
				break;
			}
		case MSG_SET_INIT_PHASE5_RESP:
			{
				handleInitRespMessage(message, "init_phase5");
				break;
			}
		case MSG_DO_LINK_INIT_RESP:
			{
				handleInitRespMessage(message, "do_link_init");
				break;
			}
		case MSG_DO_INITNEGOTIATE_RESP:
			{
				handleInitRespMessage(message, "do_initnegotiate");
				break;
			}
		case MSG_DISCOVER_INTRA_CONNECTIONS_RESP:
			{
				handleInitRespMessage(message, "discover_intra_connections");
				break;
			}
		case MSG_WRITE_DISCOVERY_TOKENS_RESP:
			{
				handleWriteDiscoveryTokensMessage(message, infoFromMsg);
				break;
			}
		case MSG_READ_DISCOVERY_TOKENS_RESP:
			{
				handleReadDiscoveryTokensMessage(message, infoFromMsg);
				break;
			}
		case MSG_ADD_INTERNODE_CONNS_RESP:
			{
				handleAddInterNodeConnsMessage(message, "add_internode_connections");
				break;
			}
		case MSG_DISPLAY_INTRA_NODE_CONNS_RESP:
			{
				handleInitRespMessage(message, "display_intra_node_conns");
				break;
			}
		case MSG_READ_SIDs_RESP:
			{
				handleSidTokenInfoRespMessage(message, infoFromMsg);
				break;
			}
		case MSG_GET_INTRANODE_CONNS_RESP:
			{
				handleGetIntraNodeConnsRespMessage(message);
				break;
			}
		 case MSG_INTRA_NODE_CONNS_TRAIN_RESP:
		 	{
		 		handleInitRespMessage(message, "intra_node_train");
		 		break;
		 	}
		case MSG_SUBLINK_INTERNODE_CONN_TRAIN_SLAVE_RESP:
		case MSG_MAINLINK_INTERNODE_CONN_TRAIN_SLAVE_RESP:
			{
				handleFollowerInterNodeTrainResp(msg_type, infoFromMsg);
				break;
			}
		case MSG_INTERNODE_CONN_TRAIN_REQ_COMPLETE:
			{
				handleInterNodeTrainReqComplete(message, infoFromMsg);
				break;
			}
		case MSG_PARALLEL_LINK_TRAIN_RESP:
			{
				handleInitRespMessage(message, "parallel_link_training");
				break;
			}
		case MSG_SUBLINK_INTERNODE_CONN_TRAIN_RESP:
		case MSG_MAINLINK_INTERNODE_CONN_TRAIN_RESP:
			{
				std::string init_action = msg_type == MSG_SUBLINK_INTERNODE_CONN_TRAIN_RESP ? "sublink_train" : "mainlink_train";
				handleInitRespMessage(message, init_action);
				break;
			}
		case MSG_PARALLEL_LINK_TRAIN_OPTIMIZE_RESP:
			{
				handleInitRespMessage(message, "parallel_optimize");
				break;
			}
		case MSG_SINGLE_INTRANODE_CONN_TRAIN_RESP:
			{
				handleInitRespMessage(message, "single_intranode_train");
				break;
			}
		default:
			PRINT_VERBOSE_ERRORS << "Wrong message type received by Master "<< msg_type <<endl;
	}

	return;
}

void handleGetDeviceInfoRespMsg(Json::Value message)
{
	lwlink_get_devices_info deviceInfo;

    Json::Value deviceInfoList(Json::arrayValue);
    int recvNodeId = message["nodeId"].asInt();

    if (message["status"].asInt() == STATUS_FAILED) {
    	PRINT_VERBOSE << "Could not get device information from node " << recvNodeId;
    }

    deviceInfoList = message["device_info_list"];

    deviceInfo.numDevice = message["num_devices"].asInt();
    int idx = 0;

    for (auto itr : deviceInfoList) {
    	lwlink_detailed_dev_info devInfo;
    	lwlink_pci_dev_info pciInfo;
    	pciInfo.domain = itr["domain"].asInt();
    	pciInfo.bus = itr["bus"].asInt();
    	pciInfo.device = itr["device"].asInt();
    	pciInfo.function = itr["function"].asInt();
    	devInfo.pciInfo = pciInfo;
    	devInfo.devType = itr["devType"].asInt();
    	devInfo.numLinks = itr["num_links"].asInt();
    	strncpy(devInfo.deviceName, itr["device_name"].asString().c_str(), LWLINK_DEVICE_NAME_LEN_MAX);
    	deviceInfo.devInfo[idx] = devInfo;
    	idx++;
    }

    deviceInfoMap.insert(make_pair(recvNodeId, deviceInfo));
}

void handleInitRespMessage(Json::Value message, std::string init_action)
{
	// if status fails, display
	if (message["status"].asInt() == STATUS_FAILED) {
		PRINT_VERBOSE_ERRORS << init_action << " failed for node " << message["nodeId"].asInt();
		exit(0);
	}
}

void handleWriteDiscoveryTokensMessage(Json::Value message, void *infoFromMsg)
{
	DiscoveryTokenList *writeList = (DiscoveryTokenList*) infoFromMsg;
	if (message["status"].asInt() == STATUS_FAILED) {
		PRINT_VERBOSE_ERRORS << "write_discovery_token failed for node " << message["nodeId"].asInt();
		exit(0);
	}

	Json::Value writeDiscoveryTokenInfo(Json::objectValue);
    Json::Value writeDiscoveryTokenInfoList(Json::arrayValue);

    writeDiscoveryTokenInfoList = message["write_discovery_token_info_list"];

   	for (auto itr : writeDiscoveryTokenInfoList) {
   		DiscoveryTokenInfo info;
   		info.nodeId = itr["nodeId"].asInt();
        info.domain = itr["domain"].asInt();
        info.bus = itr["bus"].asInt();
        info.device = itr["device"].asInt();
        info.function = itr["function"].asInt();
        info.linkIndex = itr["linkIndex"].asInt();
        info.tokelwalue = getUint64FromString(itr["tokelwalue"].asString());
        writeList->push_back(info);
   	}
}

void handleReadDiscoveryTokensMessage(Json::Value message, void *infoFromMsg)
{
	DiscoveryTokenList *readList = (DiscoveryTokenList*) infoFromMsg;
	if (message["status"].asInt() == STATUS_FAILED) {
		PRINT_VERBOSE_ERRORS << "read_discovery_token failed for node " << message["nodeId"].asInt();
		exit(0);
	}

	Json::Value readDiscoveryTokenInfo(Json::objectValue);
    Json::Value readDiscoveryTokenInfoList(Json::arrayValue);

    readDiscoveryTokenInfo = message["read_discovery_token_info_list"];

   	for (auto itr : readDiscoveryTokenInfo) {
   		DiscoveryTokenInfo info;
   		info.nodeId = itr["nodeId"].asInt();
        info.domain = itr["domain"].asInt();
        info.bus = itr["bus"].asInt();
        info.device = itr["device"].asInt();
        info.function = itr["function"].asInt();
        info.linkIndex = itr["linkIndex"].asInt();
        info.tokelwalue = getUint64FromString(itr["tokelwalue"].asString());
        info.phyId = itr["phyId"].asInt();
        readList->push_back(info);
   	}
}

void handleAddInterNodeConnsMessage(Json::Value message, std::string init_action)
{
	// if status fails, display
	if (message["status"].asInt() == STATUS_FAILED) {
		PRINT_VERBOSE_ERRORS << init_action << " failed for node " << message["nodeId"].asInt();
		exit(0);
	}
}

void handleSidTokenInfoRespMessage(Json::Value message, void *infoFromMsg)
{
	SidInfoList *infoList = (SidInfoList*) infoFromMsg;
	if (message["status"].asInt() == STATUS_FAILED) {
		PRINT_VERBOSE_ERRORS << "read_SIDs failed for node " << message["nodeId"].asInt();
		exit(0);
	}

	Json::Value readSIDTokenInfo(Json::objectValue);
    Json::Value readSIDTokenInfoList(Json::arrayValue);

    readSIDTokenInfoList = message["read_SIDs_info_list"];

   	for (auto itr : readSIDTokenInfoList) {
   		SidNodeConnectionInfo info;
   		info.nodeId = itr["nodeId"].asInt();
   		info.gpuOrSwitchId = getUint64FromString(itr["gpuOrSwitchId"].asString());
        info.nearSid = getUint64FromString(itr["nearSid"].asString());
        info.nearLinkIndex = itr["nearLinkIndex"].asInt();;
        info.farSid = getUint64FromString(itr["farSid"].asString());
        info.farLinkIndex = itr["farLinkIndex"].asInt();;
        info.domain = itr["domain"].asInt();
        info.bus = itr["bus"].asInt();
        info.device = itr["device"].asInt();
        info.function = itr["function"].asInt();
        info.devType = itr["devType"].asInt();
        infoList->push_back(info);
   	}
}

void handleGetIntraNodeConnsRespMessage(Json::Value message)
{
	Json::Value intraNodeConnList(Json::arrayValue);
	Json::Value intraNodeConn(Json::objectValue);

	int nodeId = message["nodeId"].asInt();
	intraNodeConnList = message["intra_node_conn_list"];
	LWLinkConnList connList;
	for (auto itr : intraNodeConnList) {
		lwlink_connection_info connInfo;
		colwertJsonToConnInfo(connInfo.srcEndPoint, connInfo.dstEndPoint, itr);
		connList.push_back(connInfo);
	}

	intranodeConnListServerMap[nodeId] = connList;
}

void handleFollowerInterNodeTrainResp(int msg_type, void *infoFromMsg)
{
	int *recvConf = (int*) infoFromMsg;
	switch (msg_type) {
		case MSG_SUBLINK_INTERNODE_CONN_TRAIN_SLAVE_RESP:
			{
				*recvConf = SLAVE_SUBLINK_STATE_REQ_RECVD;
				break;
			}
		case MSG_MAINLINK_INTERNODE_CONN_TRAIN_SLAVE_RESP:
			{
				*recvConf = SLAVE_MAINLINK_STATE_REQ_RECVD;
				break;
			}
	}
}

void handleInterNodeTrainReqComplete(Json::Value message, void *infoFromMsg)
{
	int *reqComplete = (int*) infoFromMsg;
	if (message["status"].asInt() == STATUS_SUCCESS)
		*reqComplete = 1;
	else 
		*reqComplete = 0;
}

