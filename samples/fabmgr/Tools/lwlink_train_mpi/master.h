#pragma once 

#include <stdio.h>
#include <iostream>
#include <string>

#include "json/json.h"
#include <mpi.h>
#include "message_types.h"
#include "helper.h"
#include "logging.h"

using namespace std;

void startMaster(int num_nodes);
static void parseMessage(std::string recvMsg, void *infoFromMsg=NULL);
void sendMasterReqMsg(int init_action, int node, void *arg=NULL);
int setNumNodes(int nodes);
std::string getDeviceName(int node, lwlink_pci_dev_info pciInfo);

// functions to send messages to local
void doAndSendDeviceInfoReq();
void printDeviceInfo();
void doAndSendSetInitphase1Req();
void doAndSendRxInitTerm();
void doAndSendSetRxDetect();
void doAndSendGetRxDetect();
void doAndSendEnableCommonMode();
void doAndSendCalibrateDevices();
void doAndSendDisableCommonMode();
void doAndSendEnableDevicesData();
void doAndSendSetInitphase5Req();
void doAndSendDoLinkInit();
void doAndSendDoInitNegotiate();
void doAndSendDiscoverIntraConnections();
void doAndSendIntraNodeTraining(lwlink_conn_train_type trainTo);
void interNodeTrain(lwlink_conn_train_type trainTo, bool skipSublinkTraining=false);
void getAllIntraConnections();
void printIntraNodeConns();
void discoverInterNodeConnections();
void doInterNodeTraining(lwlink_conn_train_type trainTo);
void doIntraNodeParallelTraining(lwlink_conn_train_type trainTo);
void doTrainIntraConnection(lwlink_conn_train_type trainTo, int numNode, int connIdx);
void sendExitMessage();

void writeDiscoveryTokenReq(int nodeId, DiscoveryTokenList &writeList);
void readDiscoveryTokenReq(int nodeId, std::map<int, DiscoveryTokenList> &readListMap);
void readLinkSids(SidInfoList &sidList);
void correlateDiscoveryTokens(DiscoveryTokenList &writeList,
							  std::map<int, DiscoveryTokenList> &readListMap);
void correlateInterNodeConnsWithSids(SidInfoList &sidList);
void displayInterNodeConnections();
void addInterNodeConnections();
void sendAddConnectionInfoToNode(int nodeId, lwlink_endpoint srcEndPoint, lwlink_endpoint dstEndPoint);
void sendTrainMessageAndWaitForConf(lwlink_connection_info connInfo, lwlink_conn_train_type trainTo);
void sendStartInterNodeTrainReqMessage(lwlink_connection_info connInfo, lwlink_conn_train_type trainTo);
void sendParallelTrainLinkMessage(int node, lwlink_conn_train_type trainTo);
void sendLinkConnTrainMessage(lwlink_connection_info connInfo, int train_type, lwlink_conn_train_type trainTo);
void sendParallelOptimizeReq(std::string optimizeType);

// functions to handle messages from local
void handleInitRespMessage(Json::Value message, std::string init_action);
void handleGetDeviceInfoRespMsg(Json::Value message);
void handleWriteDiscoveryTokensMessage(Json::Value message, void *infoFromMsg);
void handleReadDiscoveryTokensMessage(Json::Value message, void *infoFromMsg);
void handleAddInterNodeConnsMessage(Json::Value message, std::string init_action);
void handleSidTokenInfoRespMessage(Json::Value message, void *infoFromMsg);
void handleGetIntraNodeConnsRespMessage(Json::Value message);
void handleFollowerInterNodeTrainResp(int msg_type, void *infoFromMsg);
void handleInterNodeTrainReqComplete(Json::Value message, void *infoFromMsg);
