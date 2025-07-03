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

#include "FMCommonTypes.h"

extern "C"
{
#include "lwswitch_user_linux.h"
}

typedef unsigned long long uint64;
typedef std::list<lwlink_connection_info> LWLinkConnList;
typedef std::list<fm_lwlink_conn_info> FMLWLinkConnList;
typedef std::list<DiscoveryTokenInfo> DiscoveryTokenList;
typedef std::list<SidNodeConnectionInfo> SidInfoList;

uint64 lwrrent_timestamp();
int open_lwlinklib_driver(unsigned short nodeId);
void get_device_information();
uint64 getDeviceId(lwlink_detailed_dev_info devInfo);
void enable_devices_common_mode();
void disable_devices_common_mode();
void rx_init_term();
void set_rx_detect();
void get_rx_detect();
void calibrate_devices();
void set_initphase1();
void enable_devices_data();
void set_initphase5();
void do_link_init();
void do_initnegotiate();
void discover_intra_connections();
void train_intra_connection(lwlink_conn_train_type trainTo, unsigned int connIdx);
void train_all_intra_connections(lwlink_conn_train_type trainTo);

void set_mainlink_state(lwlink_link_train_type trainTo, int isMasterEnd, lwlink_endpoint localEndPoint);
void set_sublink_state(lwlink_sublink_train_type trainTo, int isMasterEnd, lwlink_endpoint localEndPoint);

void write_discovery_tokens(DiscoveryTokenList &writeList);
void read_discovery_tokens(DiscoveryTokenList &readList);
void read_SIDs(SidInfoList &sidList);
void add_internode_connections(lwlink_endpoint &localEndPoint, lwlink_remote_endpoint_info &remoteEndPoint);

void show_train_menu();
void run_training_steps(ntCmdParser_t *pCmdParser);
void run_multi_node_server();
void run_multi_node_client(std::string ipAddress);

void show_multi_node_training_options();

std::string getMainLinkStateString(LwU32 linkMode);
std::string getTxSubLinkStateString(LwU32 txSubLinkMode);
std::string getRxSubLinkStateString(LwU32 rxSubLinkMode);
int getBusToPhyId(int node_id, int bus_id);
void setBusToPhyId(int node_id, int bus_id, int phy_id);
void discover_intra_node_connections_all_steps();
void list_steps();
int getArch();
int getNodeId();
