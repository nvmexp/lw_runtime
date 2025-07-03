#include <stdio.h>
#include <iostream>
#include <pthread.h>

#include "slave.h"
#include "message_types.h"
#include "helper.h"

#include "lwlink.h"
#include "lwlink_lib_ioctl.h"
#include "ioctl_lwswitch.h"
#include "ioctl_dev_lwswitch.h"
#include "lwos.h"

extern "C"
{
#include "lwswitch_user_linux.h"
}

using namespace std;

static int num_nodes;
bool stop_slave;

void startSlave(int nodes, int rank)
{
	num_nodes = nodes;
	stop_slave = false;
	pthread_t threads[num_nodes];

	for (int node = 0; node < num_nodes; node++) {
		if (node == rank) continue;
		int *arg = (int*) malloc (sizeof(int));
		*arg = node;
		pthread_create(&threads[node], NULL, startListening, arg);
	}

	for (int node = 0; node < num_nodes; node++) {
		if (node != rank)
			pthread_join(threads[node], NULL);
	}

	return;
}

void *startListening(void *nodeId)
{
	int nId = *((int*) nodeId);

	while (!stop_slave) {
		std::string recvMsg = recvMessage(nId);
		parseMessageSlave(recvMsg);
	}

	pthread_exit(NULL);
}

static void parseMessageSlave(std::string recvMsg, void *infoMsg) {
	Json::Reader reader;
	Json::Value message(Json::objectValue);
	if (reader.parse(recvMsg, message) == false) {
		PRINT_VERBOSE_ERRORS<<"Message received but could not be parsed"<<endl;
	}
	int msg_type = message["type"].asInt();

	switch (msg_type) {
		case MSG_GET_DEVICES_INFO_REQ: 
			{
				handleGetDeviceInfoReqMsg();
				break;
			}

		case MSG_SET_INIT_PHASE1_REQ:
			{
				handleSetInitPhase1ReqMsg();
				break;
			}
		case MSG_SET_RX_INIT_TERM_REQ: 
			{
				handleSetRxInitTermReqMsg();
				break;
			}
		case MSG_SET_RX_DETECT_REQ:
			{
				handleSetRxDetectReqMsg();
				break;
			}
		case MSG_GET_RX_DETECT_REQ:
			{
				handleGetRxDetectReqMsg();
				break;
			}
		case MSG_ENABLE_COMMON_MODE_REQ:
			{
				handleEnableCommonModeReqMsg();
				break;
			}
		case MSG_CALIBRATE_DEVICES_REQ:
			{
				handleCalibrateDevicesReqMsg();
				break;
			}
		case MSG_DISABLE_COMMON_MODE_REQ:
			{
				handleDisableCommonModeReqMsg();
				break;
			}
		case MSG_ENABLE_DEVICES_DATA_REQ:
			{
				handleEnableDevicesDataReqMsg();
				break;
			}
		case MSG_SET_INIT_PHASE5_REQ:
			{
				handleSetInitPhase5ReqMsg();
				break;
			}
		case MSG_DO_LINK_INIT_REQ:
			{
				handleDoLinkInitReqMsg();
				break;
			}
		case MSG_DO_INITNEGOTIATE_REQ:
			{
				handleDoInitNegotiateReqMsg();
				break;
			}
		case MSG_DISCOVER_INTRA_CONNECTIONS_REQ:
			{
				handleDiscoverIntraConnectionsReqMsg();
				break;
			}
		case MSG_WRITE_DISCOVERY_TOKENS_REQ:
			{
				handleWriteDiscoveryReqMsg();
				break;
			}
		case MSG_READ_DISCOVERY_TOKENS_REQ:
			{
				handleReadDiscoveryReqMsg();
				break;
			}
		case MSG_ADD_INTERNODE_CONNS_REQ:
			{
				handleAddInterNodeConnsReqMsg(message);
				break;
			}
		case MSG_DISPLAY_INTRA_NODE_CONNS_REQ:
			{
				handeDisplayIntraNodeConnsReq();
				break;
			}
		case MSG_READ_SIDs_REQ:
			{
				handleReadSidTokenReq();
				break;
			}
		case MSG_GET_INTRANODE_CONNS_REQ:
			{
				handleGetIntraNodeConnReq();
				break;
			}
		case MSG_INTRA_NODE_CONNS_TRAIN_REQ:
			{
				handleIntraNodeTrainingReq(message);
				break;
			}
		case MSG_SUBLINK_INTERNODE_CONN_TRAIN_MASTER_REQ:
		case MSG_MAINLINK_INTERNODE_CONN_TRAIN_MASTER_REQ:
			{
				handleInterNodeTrainingReq(message);
				break;
			}
		case MSG_START_INTERNODE_TRAIN_REQ:
			{
				handleStartInterNodeTrainingReq(message);
				break;
			}
		// These cases are used only in cases when peer to peer LFM messages are required to be exchanged
		// for internode link training purposes. In this application internode link training is done without
		// this peer to peer communication and are coordinated only by master. These cases are left as 
		// is in case of future needs, arent lwrrently used.
		case MSG_SUBLINK_INTERNODE_CONN_TRAIN_SLAVE_RESP:
		case MSG_MAINLINK_INTERNODE_CONN_TRAIN_SLAVE_RESP:
			{
				handlePeerSlaveInterNodeTrainResp(msg_type, infoMsg);
				break;
			}
		case MSG_PARALLEL_LINK_TRAIN_REQ:
			{
				handleParallelLinkTrainingReq(message);
			}
		case MSG_SUBLINK_INTERNODE_CONN_TRAIN_REQ:
		case MSG_MAINLINK_INTERNODE_CONN_TRAIN_REQ:
			{
				handleLinkTrainConnReq(message);
				break;
			}
		case MSG_PARALLEL_LINK_TRAIN_OPTIMIZE_REQ:
			{
				handleInitOptimizeReq(message);
				break;
			}
		case MSG_SINGLE_INTRANODE_CONN_TRAIN_REQ:
			{
				handleSingleTrainReq(message);
				break;
			}
		case MSG_APP_EXIT_REQ:
			{
				stop_slave = true;
				break;
			}
		default:
			PRINT_VERBOSE_ERRORS<< "Wrong message type from Slave " << msg_type <<endl;
	}

	return;
}

void sendSlaveResp(int init_action, int node, int status)
{
	Json::Value initRespMsg(Json::objectValue);
	initRespMsg["nodeId"] = getNodeId();
	initRespMsg["status"] = status;
	switch (init_action) {
		case GET_DEVICE_INFO:
			{
				initRespMsg["type"] = MSG_GET_DEVICES_INFO_RESP;		
				break;
			}
		case INIT_PHASE1:
			{
				initRespMsg["type"] = MSG_SET_INIT_PHASE1_RESP;		
				break;
			}
		case RX_INIT_TERM:
			{
				initRespMsg["type"] = MSG_SET_RX_INIT_TERM_RESP;
				break;
			}
		case SET_RX_DETECT:
			{
				initRespMsg["type"] = MSG_SET_RX_DETECT_RESP;
				break;
			}
		case GET_RX_DETECT:
			{
				initRespMsg["type"] = MSG_GET_RX_DETECT_RESP;
				break;
			}
		case ENABLE_COMMON_MODE:
			{
				initRespMsg["type"] = MSG_ENABLE_COMMON_MODE_RESP;
				break;
			}
		case CALIBRATE_DEVICES:
			{
				initRespMsg["type"] = MSG_CALIBRATE_DEVICES_RESP;
				break;
			}
		case DISABLE_COMMON_MODE:
			{
				initRespMsg["type"] = MSG_DISABLE_COMMON_MODE_RESP;
				break;
			}
		case ENABLE_DEVICES_DATA:
			{
				initRespMsg["type"] = MSG_ENABLE_DEVICES_DATA_RESP;
				break;
			}
		case INIT_PHASE5:
			{
				initRespMsg["type"] = MSG_SET_INIT_PHASE5_RESP;
				break;
			}
		case DO_LINK_INIT:
			{
				initRespMsg["type"] = MSG_DO_LINK_INIT_RESP;
				break;
			}
		case DO_INITNEGOTIATE:
			{
				initRespMsg["type"] = MSG_DO_INITNEGOTIATE_RESP;
				break;
			}
		case DISCOVER_INTRA_CONNECTIONS:
			{
				initRespMsg["type"] = MSG_DISCOVER_INTRA_CONNECTIONS_RESP;
				break;
			}
		case ADD_INTERNODE_CONNS:
			{
				initRespMsg["type"] = MSG_ADD_INTERNODE_CONNS_RESP;
				break;
			}
		case DISPLAY_INTRA_NODE_CONNS:
			{
				initRespMsg["type"] = MSG_DISPLAY_INTRA_NODE_CONNS_RESP;
				break;
			}
		case INTRA_NODE_CONNS_TRAIN:
			{
				initRespMsg["type"] = MSG_INTRA_NODE_CONNS_TRAIN_RESP;
				break;
			}
		case SUBLINK_INTERNODE_CONN_TRAIN_SLAVE:
			{
				initRespMsg["type"] = MSG_SUBLINK_INTERNODE_CONN_TRAIN_SLAVE_RESP;
				break;
			}
		case MAINLINK_INTERNODE_CONN_TRAIN_SLAVE:
			{
				initRespMsg["type"] = MSG_MAINLINK_INTERNODE_CONN_TRAIN_SLAVE_RESP;
				break;
			}
		case INTERNODE_CONN_TRAIN_REQ_COMPLETE:
			{
				initRespMsg["type"] = MSG_INTERNODE_CONN_TRAIN_REQ_COMPLETE;
				break;
			}
		case INTRA_CONN_PARALLEL_TRAINING:
		 	{
		 		initRespMsg["type"] = MSG_PARALLEL_LINK_TRAIN_RESP;
		 		break;
		 	}
		case SUBLINK_INTERNODE_CONN_TRAIN:
			{
				initRespMsg["type"] = MSG_SUBLINK_INTERNODE_CONN_TRAIN_RESP;
				break;
			} 
		case MAINLINK_INTERNODE_CONN_TRAIN:
			{
				initRespMsg["type"] = MSG_MAINLINK_INTERNODE_CONN_TRAIN_RESP;
				break;
			}
		case OPTIMIZE_CONN_TRAIN:
			{
				initRespMsg["type"] = MSG_PARALLEL_LINK_TRAIN_OPTIMIZE_RESP;
				break;
			}
		case SINGLE_INTRANODE_CONN_TRAIN:
			{
				initRespMsg["type"] = MSG_SINGLE_INTRANODE_CONN_TRAIN_RESP;
				break;
			}
		default: 
			{
				PRINT_VERBOSE_ERRORS << "No such message type from Slave" << init_action << endl;
			}
	}

	sendMessage(initRespMsg, node);
}

void handleGetDeviceInfoReqMsg()
{
	// lwlink_get_devices_info deviceInfo;
	Json::Value devicesInfo(Json::objectValue);
    Json::Value deviceInfoList(Json::arrayValue);
    Json::Value getDeviceInfoRspMsg(Json::objectValue);

	getDeviceInfoRspMsg["type"] = MSG_GET_DEVICES_INFO_RESP;
	getDeviceInfoRspMsg["nodeId"] = getNodeId();

	lwlink_get_devices_info deviceInfo = get_device_information();
	if (deviceInfo.status != LWL_SUCCESS) {
		//send failure response
		getDeviceInfoRspMsg["status"] = STATUS_FAILED;
		sendMessage(getDeviceInfoRspMsg, 0);
	}

	for (unsigned int idx = 0; idx < deviceInfo.numDevice; idx++ ) {
		devicesInfo["device_name"] = deviceInfo.devInfo[idx].deviceName;
		devicesInfo["domain"] = (int)deviceInfo.devInfo[idx].pciInfo.domain;
		devicesInfo["bus"] = (int)deviceInfo.devInfo[idx].pciInfo.bus;;
		devicesInfo["device"] = (int)deviceInfo.devInfo[idx].pciInfo.device;;
		devicesInfo["function"] = (int)deviceInfo.devInfo[idx].pciInfo.function;
		devicesInfo["devType"] = (int)deviceInfo.devInfo[idx].devType;
		devicesInfo["num_links"] = (int)deviceInfo.devInfo[idx].numLinks;
		deviceInfoList.append(devicesInfo);
	}

	getDeviceInfoRspMsg["status"] = STATUS_SUCCESS;
	getDeviceInfoRspMsg["num_devices"] = deviceInfo.numDevice;
	getDeviceInfoRspMsg["device_info_list"] = deviceInfoList;

	sendMessage(getDeviceInfoRspMsg, 0);
}

void handleSetInitPhase1ReqMsg()
{
	int status = set_initphase1() ? STATUS_SUCCESS : STATUS_FAILED;
	sendSlaveResp(INIT_PHASE1, 0, status);
}

void handleSetRxInitTermReqMsg()
{
	int status = rx_init_term() ? STATUS_SUCCESS : STATUS_FAILED;
	sendSlaveResp(RX_INIT_TERM, 0, status);
}

void handleSetRxDetectReqMsg()
{
	int status = set_rx_detect() ? STATUS_SUCCESS : STATUS_FAILED;
	sendSlaveResp(SET_RX_DETECT, 0, status);
}

void handleGetRxDetectReqMsg()
{
	int status = get_rx_detect() ? STATUS_SUCCESS : STATUS_FAILED;
	sendSlaveResp(GET_RX_DETECT, 0, status);
}

void handleEnableCommonModeReqMsg()
{
	int status = enable_devices_common_mode() ? STATUS_SUCCESS : STATUS_FAILED;
	sendSlaveResp(ENABLE_COMMON_MODE, 0, status);
}

void handleCalibrateDevicesReqMsg()
{
	int status = calibrate_devices() ? STATUS_SUCCESS : STATUS_FAILED;
	sendSlaveResp(CALIBRATE_DEVICES, 0, status);
}

void handleDisableCommonModeReqMsg()
{
	int status = disable_devices_common_mode() ? STATUS_SUCCESS : STATUS_FAILED;
	sendSlaveResp(DISABLE_COMMON_MODE, 0, status);
}

void handleEnableDevicesDataReqMsg()
{
	int status = enable_devices_data() ? STATUS_SUCCESS : STATUS_FAILED;
	sendSlaveResp(ENABLE_DEVICES_DATA, 0, status);
}

void handleSetInitPhase5ReqMsg()
{
    int status = set_initphase5() ? STATUS_SUCCESS : STATUS_FAILED;
    sendSlaveResp(INIT_PHASE5, 0, status);
}

void handleDoLinkInitReqMsg()
{
	int status = do_link_init() ? STATUS_SUCCESS : STATUS_FAILED;
	sendSlaveResp(DO_LINK_INIT, 0, status);
}

void handleDoInitNegotiateReqMsg()
{
	PRINT_VERBOSE_DEBUG << "recevied request to do do_initnegotiate" << std::endl;
	int status = do_initnegotiate() ? STATUS_SUCCESS : STATUS_FAILED;
	sendSlaveResp(DO_INITNEGOTIATE, 0, status);
}

void handleDiscoverIntraConnectionsReqMsg()
{
	int status = discover_intra_connections() ? STATUS_SUCCESS : STATUS_FAILED;
	sendSlaveResp(DISCOVER_INTRA_CONNECTIONS, 0, status);
}

void handleWriteDiscoveryReqMsg()
{	
	DiscoveryTokenList writeList;
	int status = write_discovery_tokens(writeList) ? STATUS_SUCCESS : STATUS_FAILED;

	Json::Value writeDiscoveryTokenInfo(Json::objectValue);
    Json::Value writeDiscoveryTokenInfoList(Json::arrayValue);
    Json::Value writeDiscoveryTokenRespMsg(Json::objectValue);
    Json::Value tokelwalue(Json::UInt64);

    PRINT_VERBOSE_DEBUG << "received write discovery message " << endl;

    writeDiscoveryTokenRespMsg["status"] = status;
    writeDiscoveryTokenRespMsg["nodeId"] = getNodeId();
    writeDiscoveryTokenRespMsg["type"] = MSG_WRITE_DISCOVERY_TOKENS_RESP;

    DiscoveryTokenList::iterator it;
    PRINT_VERBOSE_DEBUG << "write list size from Slave " << writeList.size() << endl;
    for (it = writeList.begin(); it != writeList.end(); it++) {
		DiscoveryTokenInfo info = *it;
		writeDiscoveryTokenInfo["nodeId"] = info.nodeId;
		writeDiscoveryTokenInfo["domain"] = info.domain;
		writeDiscoveryTokenInfo["bus"] = info.bus;
		writeDiscoveryTokenInfo["device"] = info.device;
		writeDiscoveryTokenInfo["function"] = info.function;
		writeDiscoveryTokenInfo["linkIndex"] = info.linkIndex;
		writeDiscoveryTokenInfo["tokelwalue"] = getStringFromUint64(info.tokelwalue);
		writeDiscoveryTokenInfoList.append(writeDiscoveryTokenInfo);
    }

	writeDiscoveryTokenRespMsg["write_discovery_token_info_list"] = writeDiscoveryTokenInfoList;
	sendMessage(writeDiscoveryTokenRespMsg, 0);
}

void handleReadDiscoveryReqMsg()
{
	DiscoveryTokenList readList;
	int status = read_discovery_tokens(readList) ? STATUS_SUCCESS : STATUS_FAILED;

	Json::Value readDiscoveryTokenInfo(Json::objectValue);
    Json::Value readDiscoveryTokenInfoList(Json::arrayValue);
    Json::Value readDiscoveryTokenRespMsg(Json::objectValue);
    Json::Value tokelwalue(Json::UInt64);

    PRINT_VERBOSE_DEBUG << "received read discovery message " << endl;

    readDiscoveryTokenRespMsg["status"] = status;
    readDiscoveryTokenRespMsg["nodeId"] = getNodeId();
    readDiscoveryTokenRespMsg["type"] = MSG_READ_DISCOVERY_TOKENS_RESP;

    DiscoveryTokenList::iterator it;
    PRINT_VERBOSE_DEBUG << "read list size from Slave " << readList.size() << endl;
    for (it = readList.begin(); it != readList.end(); it++) {
		DiscoveryTokenInfo info = *it;
		readDiscoveryTokenInfo["nodeId"] = info.nodeId;
		readDiscoveryTokenInfo["domain"] = info.domain;
		readDiscoveryTokenInfo["bus"] = info.bus;
		readDiscoveryTokenInfo["device"] = info.device;
		readDiscoveryTokenInfo["function"] = info.function;
		readDiscoveryTokenInfo["linkIndex"] = info.linkIndex;
		readDiscoveryTokenInfo["tokelwalue"] = getStringFromUint64(info.tokelwalue);
		readDiscoveryTokenInfo["phyId"] = info.phyId;
		readDiscoveryTokenInfoList.append(readDiscoveryTokenInfo);
    }

	readDiscoveryTokenRespMsg["read_discovery_token_info_list"] = readDiscoveryTokenInfoList;
	sendMessage(readDiscoveryTokenRespMsg, 0);
}

void handleAddInterNodeConnsReqMsg(Json::Value interNodeConnectionMsg)
{
    lwlink_endpoint srcEndPoint;
    lwlink_remote_endpoint_info remoteEndPoint;

    Json::Value srcEndPointInfo(Json::objectValue);
	Json::Value remoteEndPointInfo(Json::objectValue);
	Json::Value pciInfo(Json::objectValue);
	srcEndPointInfo = interNodeConnectionMsg["src_end_point"];
	remoteEndPointInfo = interNodeConnectionMsg["remote_end_point"];

	srcEndPoint.nodeId = srcEndPointInfo["nodeId"].asInt();
	srcEndPoint.linkIndex = srcEndPointInfo["linkIndex"].asInt();
	pciInfo = srcEndPointInfo["pciInfo"];
	srcEndPoint.pciInfo.domain = pciInfo["domain"].asInt();
	srcEndPoint.pciInfo.bus = pciInfo["bus"].asInt();
	srcEndPoint.pciInfo.device = pciInfo["device"].asInt();
	srcEndPoint.pciInfo.function = pciInfo["function"].asInt();

	remoteEndPoint.nodeId = remoteEndPointInfo["nodeId"].asInt();
	remoteEndPoint.linkIndex = remoteEndPointInfo["linkIndex"].asInt();
	pciInfo = remoteEndPointInfo["pciInfo"];
	remoteEndPoint.pciInfo.domain = pciInfo["domain"].asInt();
	remoteEndPoint.pciInfo.bus = pciInfo["bus"].asInt();
	remoteEndPoint.pciInfo.device = pciInfo["device"].asInt();
	remoteEndPoint.pciInfo.function = pciInfo["function"].asInt();
	remoteEndPoint.devType = remoteEndPointInfo["devType"].asInt();

	int status = add_internode_connections(srcEndPoint, remoteEndPoint) ? STATUS_SUCCESS : STATUS_FAILED;

	sendSlaveResp(ADD_INTERNODE_CONNS, 0, status);
}

void handeDisplayIntraNodeConnsReq()
{
	PRINT_VERBOSE_DEBUG << "Displaying intra node connections for node " << getNodeId() << endl;
	LWLinkConnList connList = getIntraConns();
	display_connections(connList);

	int status = STATUS_SUCCESS;
	sendSlaveResp(DISPLAY_INTRA_NODE_CONNS, 0, status);
}

void handleReadSidTokenReq()
{
	SidInfoList infoList;

	int status = read_SIDs(infoList) ? STATUS_SUCCESS : STATUS_FAILED;

	Json::Value readSIDTokenInfo(Json::objectValue);
    Json::Value readSIDTokenInfoList(Json::arrayValue);
    Json::Value readSIDTokenInfoRespMsg(Json::objectValue);
    Json::Value tokelwalue(Json::UInt64);

    PRINT_VERBOSE_DEBUG << "received read discovery message " << endl;

    readSIDTokenInfoRespMsg["status"] = status;
    readSIDTokenInfoRespMsg["nodeId"] = getNodeId();
    readSIDTokenInfoRespMsg["type"] = MSG_READ_SIDs_RESP;

    SidInfoList::iterator it;
    for (it = infoList.begin(); it != infoList.end(); it++) {
    	SidNodeConnectionInfo info = *it;
    	readSIDTokenInfo["nodeId"] = info.nodeId;
    	readSIDTokenInfo["gpuOrSwitchId"] = getStringFromUint64(info.gpuOrSwitchId);
    	readSIDTokenInfo["nearSid"] = getStringFromUint64(info.nearSid);
		readSIDTokenInfo["nearLinkIndex"] = info.nearLinkIndex;
		readSIDTokenInfo["farSid"] = getStringFromUint64(info.farSid);
		readSIDTokenInfo["farLinkIndex"] = info.farLinkIndex;
		readSIDTokenInfo["domain"] = info.domain;
		readSIDTokenInfo["bus"] = info.bus;
		readSIDTokenInfo["device"] = info.device;
		readSIDTokenInfo["function"] = info.function;
		readSIDTokenInfo["devType"] = info.devType;
		readSIDTokenInfoList.append(readSIDTokenInfo);
    }

    readSIDTokenInfoRespMsg["read_SIDs_info_list"] = readSIDTokenInfoList;
    Json::StyledWriter styledWriter;
    std::string sStyled = styledWriter.write(readSIDTokenInfoRespMsg);
    PRINT_VERBOSE << "size of message " << sStyled.size() << endl;
	sendMessage(readSIDTokenInfoRespMsg, 0);
}

void handleGetIntraNodeConnReq()
{
	Json::Value intraNodeConnList(Json::arrayValue);
	Json::Value intraNodeConn(Json::objectValue);
	Json::Value getIntraNodeConnMsg(Json::objectValue);

	getIntraNodeConnMsg["status"] = STATUS_SUCCESS;
	getIntraNodeConnMsg["nodeId"] = getNodeId();
	getIntraNodeConnMsg["type"] = MSG_GET_INTRANODE_CONNS_RESP;

	LWLinkConnList intraConns = getIntraConns();

	LWLinkConnList::iterator it;
	for (it = intraConns.begin(); it != intraConns.end(); it++) {
		lwlink_connection_info connInfo = *it;
		colwertConnInfoToJson(connInfo.srcEndPoint, connInfo.dstEndPoint, intraNodeConn);
		intraNodeConnList.append(intraNodeConn);
	}

	getIntraNodeConnMsg["intra_node_conn_list"] = intraNodeConnList;
	sendMessage(getIntraNodeConnMsg, 0);
}

void handleIntraNodeTrainingReq(Json::Value message)
{
	int status = train_all_intra_connections((lwlink_conn_train_type)message["trainTo"].asInt()) ? STATUS_SUCCESS : STATUS_FAILED;
	sendSlaveResp(INTRA_NODE_CONNS_TRAIN, 0, status);
}

void handleInterNodeTrainingReq(Json::Value message)
{
	int type = message["type"].asInt();
	lwlink_link_train_type trainTo = (lwlink_link_train_type)message["trainTo"].asInt();
	int status;
	lwlink_connection_info connInfo;
	Json::Value interNodeConn(Json::objectValue);
	interNodeConn = message["conn_info"];
	colwertJsonToConnInfo(connInfo.dstEndPoint, connInfo.srcEndPoint, interNodeConn);

	switch (type) {
		case MSG_SUBLINK_INTERNODE_CONN_TRAIN_MASTER_REQ:
			{
				status = set_sublink_state(lwlink_train_sublink_safe_to_hs, false, connInfo.srcEndPoint) ? STATUS_SUCCESS : STATUS_FAILED;
				sendSlaveResp(SUBLINK_INTERNODE_CONN_TRAIN_SLAVE, connInfo.dstEndPoint.nodeId, status);
				break;
			}
		case MSG_MAINLINK_INTERNODE_CONN_TRAIN_MASTER_REQ:
			{
				status = set_mainlink_state(lwlink_train_link_swcfg_to_active, false, connInfo.srcEndPoint) ? STATUS_SUCCESS : STATUS_FAILED;
				sendSlaveResp(MAINLINK_INTERNODE_CONN_TRAIN_SLAVE, connInfo.dstEndPoint.nodeId, status);
				break;
			}
	}

	return;
}

void handleStartInterNodeTrainingReq(Json::Value message)
{
	message["type"] = MSG_SUBLINK_INTERNODE_CONN_TRAIN_MASTER_REQ;
	Json::Value interNodeConn(Json::objectValue);
	interNodeConn = message["conn_info"];
	int dstNodeId = interNodeConn["dst_nodeId"].asInt();
	sendMessage(message, dstNodeId);

	//wait for response
	int recvConf = 0;
    std::string recvdMsg = recvMessage(dstNodeId);
	parseMessageSlave(recvdMsg, (void*) &recvConf);

	lwlink_endpoint srcEndPoint;
	srcEndPoint.nodeId = interNodeConn["src_nodeId"].asInt();
	srcEndPoint.linkIndex = interNodeConn["src_link_index"].asInt();
	srcEndPoint.pciInfo.domain = interNodeConn["src_domain"].asInt();
	srcEndPoint.pciInfo.bus = interNodeConn["src_bus"].asInt();
	srcEndPoint.pciInfo.device = interNodeConn["src_device"].asInt();
	srcEndPoint.pciInfo.function = interNodeConn["src_function"].asInt();

	if (recvConf == SLAVE_SUBLINK_STATE_REQ_RECVD) {
		set_sublink_state(lwlink_train_sublink_safe_to_hs, true, srcEndPoint);
	} 

	//now send request to train to high speed
	message["type"] = MSG_MAINLINK_INTERNODE_CONN_TRAIN_MASTER_REQ;
	sendMessage(message, dstNodeId);

	recvdMsg = recvMessage(dstNodeId);
	parseMessageSlave(recvdMsg, (void*) &recvConf);

	if (recvConf == SLAVE_MAINLINK_STATE_REQ_RECVD) {
		set_mainlink_state(lwlink_train_link_swcfg_to_active, true, srcEndPoint);
	}

	int status = STATUS_SUCCESS;
	sendSlaveResp(INTERNODE_CONN_TRAIN_REQ_COMPLETE, 0, status);
}

void handlePeerSlaveInterNodeTrainResp(int msg_type, void *infoFromMsg)
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

void handleParallelLinkTrainingReq(Json::Value message)
{
	LWLinkConnList connList;
	Json::Value parallelLinkTrainingConnList(Json::arrayValue);
	parallelLinkTrainingConnList = message["parallel_link_train_list"];

	for (auto itr:parallelLinkTrainingConnList) {
		lwlink_connection_info connInfo;
		colwertJsonToConnInfo(connInfo.srcEndPoint, connInfo.dstEndPoint, itr);
		connList.push_back(connInfo);
	}

	int status = train_intra_conn_parallel((lwlink_conn_train_type) message["trainTo"].asInt(), connList) ? STATUS_SUCCESS : STATUS_FAILED;
	sendSlaveResp(INTRA_CONN_PARALLEL_TRAINING, 0, status);
}

void handleLinkTrainConnReq(Json::Value message)
{
	int type = message["type"].asInt();
	lwlink_link_train_type trainTo = (lwlink_link_train_type)message["trainTo"].asInt();
	int status;
	lwlink_connection_info connInfo;
	Json::Value interNodeConn(Json::objectValue);
	interNodeConn = message["conn_info"];
	bool isMasterEnd = message["isMasterEnd"].asBool();

	if (isMasterEnd)
		colwertJsonToConnInfo(connInfo.srcEndPoint, connInfo.dstEndPoint, interNodeConn);
	else
		colwertJsonToConnInfo(connInfo.dstEndPoint, connInfo.srcEndPoint, interNodeConn);

	switch (type) {
		case MSG_SUBLINK_INTERNODE_CONN_TRAIN_REQ:
			{
				status = set_sublink_state(lwlink_train_sublink_safe_to_hs, isMasterEnd, connInfo.srcEndPoint) ? STATUS_SUCCESS : STATUS_FAILED;
				sendSlaveResp(SUBLINK_INTERNODE_CONN_TRAIN, 0, status);
				sleep(1);
				break;
			}
		case MSG_MAINLINK_INTERNODE_CONN_TRAIN_REQ:
			{
				status = set_mainlink_state(lwlink_train_link_swcfg_to_active, isMasterEnd, connInfo.srcEndPoint) ? STATUS_SUCCESS : STATUS_FAILED;
				sendSlaveResp(MAINLINK_INTERNODE_CONN_TRAIN, 0, status);
				break;
			}
	}

	return;
}

void handleInitOptimizeReq(Json::Value message)
{
	int status = doOptimizeReq(message) ? STATUS_SUCCESS : STATUS_FAILED;
	sendSlaveResp(OPTIMIZE_CONN_TRAIN, 0, status);
}

void handleSingleTrainReq(Json::Value message)
{
	lwlink_conn_train_type trainTo = (lwlink_conn_train_type) message["trainTo"].asInt();
	int connIdx = message["connIdx"].asInt();
	int status = train_intra_connection(trainTo, connIdx) ? STATUS_SUCCESS : STATUS_FAILED;
	sendSlaveResp(SINGLE_INTRANODE_CONN_TRAIN, 0, status);
}
