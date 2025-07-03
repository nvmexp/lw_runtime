#pragma once

#include <sys/types.h>
#include <vector>


//#define ROUTING_TABLE_ENTRIES   LWSWITCH_INGRESS_REQUEST_ENTRIES_MAX  //256
#define ROUTING_TABLE_ENTRIES   8192
#define DEST_UNREACHABLE        INT_MAX
#define DEST_ERROR              -1


typedef struct {
    int egress_port_id; //egress switch port to which this table entry points
    int count;          //number of entries with matching egress port id
} naReqEgressPort_t;

//per-port request table
typedef std::vector<naReqEgressPort_t> naPortRequestTable_t;

//per-port response table
typedef std::vector<int> naPortResponseTable_t;

//all request tables for a switch
typedef std::vector<naPortRequestTable_t> naSwitchRequestTables_t;

//all response tables for a switch
typedef std::vector<naPortResponseTable_t> naSwitchResponseTables_t;

//all request tables for all switches on a node
typedef std::vector<naSwitchRequestTables_t> naNodeRequestTables_t;

//all response tables for all switches on a node
typedef std::vector<naSwitchResponseTables_t> naNodeResponseTables_t;


//read all request tables for all ports on all switches on node
bool naReadRequestTables(naNodeRequestTables_t &req_tables, int num_switches, uint32_t num_table_entries);

//read all response tables for all ports on all switches on node
bool naReadResponseTables(naNodeResponseTables_t &res_tables, int num_switches, uint32_t num_table_entries);

//For request packet compute the final egress switch/port for a starting switch/port/dest_gpu_id 
bool naComputeReqEgressPort( naNodeRequestTables_t &req_tables,  int switch_id,
                         uint32_t switch_port,  int dest_gpu_id, int &egress_switch_id, int &egress_port_id);

//For response packet compute the final egress switch/port for a starting switch/port/requestor_link_id 
bool naComputeResEgressPort( naNodeResponseTables_t &res_tables,  int switch_id,
                     uint32_t switch_port,  int req_link_id, int &egress_switch_id, int &egress_port_id);

//check if the final (egress_port_id, egress_switch_ID) is connected to the specified dest_gpu_id
int naCheckReqEgressPort( int dest_gpu_id,  int egress_switch_id,  int egress_port_id);

//set request entry
bool naSetRequestEntry(int switch_id, int switch_port, int dest_gpu_id, int valid, int egress_port);

//set response entry
bool naSetResponseEntry(int switch_id, int switch_port, int dest_rlid, int valid, int egress_port);
