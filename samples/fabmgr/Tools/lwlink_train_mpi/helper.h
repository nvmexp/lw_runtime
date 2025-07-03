#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>
#include <iostream>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <pthread.h>
#include <list>
#include <string>
#include "lwlink_train_cmd_parser.h"

#include "lwlink.h"
#include "lwlink_lib_ioctl.h"
#include "ioctl_lwswitch.h"
#include "ioctl_dev_lwswitch.h"
#include "lwos.h"

#include "json/json.h"
#include "message_types.h"
#include "FMCommonTypes.h"
#include "logging.h"

extern "C"
{
#include "lwswitch_user_linux.h"
}

#define SLAVE_SUBLINK_STATE_REQ_RECVD 1
#define SLAVE_MAINLINK_STATE_REQ_RECVD 2

typedef std::list<lwlink_connection_info> LWLinkConnList;
typedef std::list<DiscoveryTokenInfo> DiscoveryTokenList;
typedef std::list<SidNodeConnectionInfo> SidInfoList;

uint64_t lwrrent_timestamp();
int open_lwlinklib_driver(unsigned short nodeId);
void show_menu(int num_nodes);

int getNodeId();
void sendMessage(Json::Value message, int dest);
std::string getStringFromUint64(uint64_t val);
uint64_t getUint64FromString(std::string strVal);
std::string recvMessage(int src);
Json::Value getPciInfoJson(lwlink_endpoint endPoint);
lwlink_pci_dev_info getJsonPciInfo(Json::Value pci_info);
LWLinkConnList getIntraConns();

lwlink_get_devices_info get_device_information();
bool set_initphase1();
bool rx_init_term();
bool set_rx_detect();
bool get_rx_detect();
bool enable_devices_common_mode();
bool disable_devices_common_mode();
bool calibrate_devices();
bool enable_devices_data();
bool set_initphase5();
bool do_initnegotiate();
bool do_link_init();
bool discover_intra_connections();
bool isDuplicateConnection(lwlink_connection_info conn, LWLinkConnList &connectionList);
void display_connections(LWLinkConnList connList);
bool write_discovery_tokens(DiscoveryTokenList &writeList);
bool read_discovery_tokens(DiscoveryTokenList &readList);
bool read_SIDs(SidInfoList &sidList);
bool add_internode_connections(lwlink_endpoint &localEndPoint, lwlink_remote_endpoint_info &remoteEndPoint);
bool train_intra_connection(lwlink_conn_train_type trainTo, int connIdx);
bool train_all_intra_connections(lwlink_conn_train_type trainTo);
bool train_intra_conn_parallel(lwlink_conn_train_type trainTo, LWLinkConnList connInfoList);
bool set_mainlink_state(lwlink_link_train_type trainTo, int isMasterEnd, lwlink_endpoint localEndPoint);
bool set_sublink_state(lwlink_sublink_train_type trainTo, int isMasterEnd, lwlink_endpoint localEndPoint);
bool doOptimizeReq(Json::Value message);
bool doInitOptimize(Json::Value optimizeLinkTrainList, int endPointCount);
bool doPostInitOptimize(Json::Value optimizeLinkTrainList, int endPointCount);
std::string getMainLinkStateString(int linkMode);
std::string getTxSubLinkStateString(int txSubLinkMode);
std::string getRxSubLinkStateString(int rxSubLinkMode);

void setBusToPhyId(int node_id, int bus_id, int phy_id);
int getBusToPhyId(int node_id, int bus_id);
int open_switch_dev(std::string dev_name);
static int getSwitchPhyId(int fd);
static int getSwitchArch(int fd);
int getArch();
void save_switch_phy_id(std::string device_name, int bus_id);
uint64_t getDeviceId(lwlink_detailed_dev_info devInfo);
int getDevType(lwlink_get_devices_info deviceInfo, lwlink_pci_dev_info pciInfo);
void print_device_information(lwlink_get_devices_info deviceInfo);

void colwertConnInfoToJson(lwlink_endpoint srcEndPoint, lwlink_endpoint dstEndPoint, Json::Value &message);
void colwertJsonToConnInfo(lwlink_endpoint &srcEndPoint, lwlink_endpoint &dstEndPoint, Json::Value message);
void printConnInfo(lwlink_connection_info connInfo);
