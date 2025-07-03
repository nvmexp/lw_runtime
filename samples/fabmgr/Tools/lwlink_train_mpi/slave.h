#pragma once

#include <stdio.h>
#include <iostream>
#include <string>

#include "json/json.h"
#include "logging.h"
#include <mpi.h>

using namespace std;

void startSlave(int size, int rank);
static void parseMessageSlave(std::string recvMsg, void *infoMsg=NULL);
void sendSlaveResp(int init_action, int node);
void *startListening(void *nodeId);

//functions to handle messages from global
void handleGetDeviceInfoReqMsg();
void handleSetInitPhase1ReqMsg();
void handleSetRxInitTermReqMsg();
void handleSetRxDetectReqMsg();
void handleGetRxDetectReqMsg();
void handleEnableCommonModeReqMsg();
void handleCalibrateDevicesReqMsg();
void handleDisableCommonModeReqMsg();
void handleEnableDevicesDataReqMsg();
void handleSetInitPhase5ReqMsg();
void handleDoLinkInitReqMsg();
void handleDoInitNegotiateReqMsg();
void handleDiscoverIntraConnectionsReqMsg();
void handleWriteDiscoveryReqMsg();
void handleReadDiscoveryReqMsg();
void handleAddInterNodeConnsReqMsg(Json::Value message);
void handeDisplayIntraNodeConnsReq();
void handleReadSidTokenReq();
void handleGetIntraNodeConnReq();
void handleIntraNodeTrainingReq(Json::Value message);
void handleInterNodeTrainingReq(Json::Value message);
void handleStartInterNodeTrainingReq(Json::Value message);
void handlePeerSlaveInterNodeTrainResp(int msg_type, void *infoFromMsg);
void handleParallelLinkTrainingReq(Json::Value message);
void handleLinkTrainConnReq(Json::Value message);
void handleInitOptimizeReq(Json::Value message);
void handleSingleTrainReq(Json::Value message);
