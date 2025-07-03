
#ifndef DCGM_FM_COMMON_H
#define DCGM_FM_COMMON_H

#include <stdint.h>
#include <map>
#include <list>
#include <set>
#include "dcgm_structs.h"
#include <g_lwconfig.h>

typedef unsigned long long uint64;
typedef unsigned int       uint32;
typedef unsigned short     uint16;
typedef unsigned char      uint8;

/* Version number used on all fabric manager messages                        */

#define FABRIC_MANAGER_VERSION 0x00100

/* defaults and options */

#define DEFAULT_TOPOLOGY_FILE          "/usr/share/lwpu/lwswitch/topology"
#define DEPRECATED_TOPOLOGY_FILE       "/usr/share/lwpu/topology.proto.bin"

#define DEFAULT_TOPOLOGY_CONF_FILE     "/usr/share/lwpu/lwswitch/topology_conf.txt"
#define DEFAULT_RUNNING_TOPOLOGY_FILE  "/usr/share/lwpu/lwswitch/running.topology"

#define OPT_DISABLE_GPU             "--disable-gpu"
#define OPT_DISABLE_SWITCH          "--disable-switch"
#define OPT_PORT_LOOPBACK           "--loopback-port"



/* number of worker threads in LwcmServer work queue */
#define FM_DCGM_SERVER_WORKER_NUM_THREADS 1

/* Beyond a certain number of total nodes, we will need to introduce         */
/* "Regional" Fabric Managers to aggregate results. This node count is       */
/* not in the plan for Willow, and not likely for Lime Rock                  */

#define MAX_LOCAL_SUBORDINATES 1024

/* Default to listening on localhost only */
#define FM_DEFAULT_BIND_INTERFACE "127.0.0.1"

/* Default port numbers */
#define FM_CONTROL_CONN_PORT       16000
#define PEER_LFM_COORDINATION_PORT_OFFSET 1

#define GLOBAL_FM_CMD_SERVER_PORT  17000
#define LOCAL_FM_CMD_SERVER_PORT   18000

/* Default host ID is HOME IPv4 address */
#define DCGM_HOME_IP "127.0.0.1"

// VC Remap options
#define PASS_VC     0
#define ILWERT_VC   1
#define USE_VC_ZERO 2
#define USE_VC_ONE  3

// Valid bit options
#define ACCESS_ILWALID  0
#define ACCESS_VALID    1

typedef enum
{
    STOPPED         = 1,
    STOP_PENDING    = 2,
    RUNNING         = 3,
} RunState;

#define NUM_PORTS_PER_LWSWITCH          36
#define NUM_LWLINKS_PER_GPU             6
#define NUM_INGR_REQ_ENTRIES_PER_GPU    4    // up to 4 entries, each entry represents a 16G memory region
#define NUM_INGR_RESP_ENTRIES_PER_GPU   NUM_LWLINKS_PER_GPU

#define INGRESS_REQ_TABLE_SIZE          8192
#define INGRESS_RESP_TABLE_SIZE         8192
#define GANGED_LINK_TABLE_SIZE          256

#define MAX_NUM_NODES                   8
#define MAX_NUM_GPUS_PER_NODE           16
#define MAX_NUM_WILLOWS_PER_NODE        12

#define IS_NODE_VALID(nodeId)      ( (nodeId >= 0) && (nodeId < MAX_NUM_NODES) )
// TODO fix this IS_WILLOW_VALID based on Explorer GPIO Ids
// Note: Hardcode to return true always for now.
#define IS_WILLOW_VALID(index)     true
#define IS_PORT_VALID(index)       ( (index >= 0) && (index < NUM_PORTS_PER_LWSWITCH) )
#define IS_GPU_VALID(index)        ( (index >= 0) && (index < MAX_NUM_GPUS_PER_NODE) )
#define IS_INGR_REQ_VALID(index)   ( (index >=0 ) && (index < INGRESS_REQ_TABLE_SIZE) )
#define IS_INGR_RESP_VALID(index)  ( (index >=0 ) && (index < INGRESS_RESP_TABLE_SIZE) )
#define IS_GANGED_LINK_ENTRY_VALID(index)  ( (index >=0 ) && (index < GANGED_LINK_TABLE_SIZE) )

#define GPU_ENDPOINT_ID(nodeId, physicalid)          ( nodeId * MAX_NUM_GPUS_PER_NODE + physicalid )
#define GPU_PHYSICAL_ID(gpuEndPointId)               ( gpuEndPointId % MAX_NUM_GPUS_PER_NODE )

#define GPU_FABRIC_DEFAULT_ADDR_RANGE                ( 1LL << 36 )    // 64G
#define GPU_FABRIC_DEFAULT_ADDR_BASE(gpuEndPointId)  ( (uint64_t)gpuEndPointId << 36 )
#define GPU_ENDPOINT_ID_FROM_ADDR_BASE(gpuAddrBase)  ( (uint64_t)gpuAddrBase >> 36 )

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)        //LimeRock specific block

#define NUM_LWLINKS_PER_AMPERE              12
#define NUM_INGR_RMAP_ENTRIES_PER_AMPERE    1   // each entry represents a 64G memory region
#define NUM_INGR_RID_ENTRIES_PER_AMPERE     1   // LimeRock IDs target a GPU
#define NUM_INGR_RLAN_ENTRIES_PER_AMPERE    1   // RLAN entries, per RID, target a specific port on last egress
#define LR_INGRESS_RMAP_TABLE_SIZE          2048
#define LR_INGRESS_RID_TABLE_SIZE           512
#define LR_INGRESS_RLAN_TABLE_SIZE          512
#define LR_GANGED_LINK_TABLE_SIZE           256

#define LR_MAX_NUM_NODES                    16  // current thinking. may be subject to change.
#define LR_MAX_NUM_GPUS_PER_NODE            16  // LW will build 8 GPU nodes. OEMs may build 16 
#define LR_MAX_NUM_LIMEROCKS_PER_NODE       12  // LW will build 6 LR per node. OEMs may build 12

// GPA addressing will only be done within a node, and will have the first set of map slots.
// We will reserve space in the RMAP table for the max possible within a node.
// FLA addressing will be done inter-node and intra-node, and will start at LR_FIRST_FLA_RMAP_SLOT
#define LR_FIRST_FLA_RMAP_SLOT              LR_MAX_NUM_GPUS_PER_NODE * NUM_INGR_RMAP_ENTRIES_PER_AMPERE

#define LR_IS_NODE_VALID(nodeId)      ( (nodeId >= 0) && (nodeId < MAX_NUM_NODES) )
// TODO fix this IS_LIMEROCK_VALID based on Explorer GPIO Ids
// Note: Hardcode to return true always for now.
#define LR_IS_LIMEROCK_VALID(index)     true
#define LR_IS_PORT_VALID(index)       ( (index >= 0) && (index < NUM_PORTS_PER_LWSWITCH) )
#define LR_IS_GPU_VALID(index)        ( (index >= 0) && (index < MAX_NUM_GPUS_PER_NODE) )
#define LR_IS_INGR_RMAP_VALID(index)  ( (index >=0 ) && (index < INGRESS_REQ_TABLE_SIZE) )
#define LR_IS_INGR_RID_VALID(index)   ( (index >=0 ) && (index < INGRESS_RESP_TABLE_SIZE) )
#define LR_IS_INGR_RLAN_VALID(index)  ( (index >=0 ) && (index < INGRESS_RESP_TABLE_SIZE) )
#define LR_IS_GANGED_LINK_ENTRY_VALID(index)  ( (index >=0 ) && (index < GANGED_LINK_TABLE_SIZE) )

#define LR_GPU_ENDPOINT_ID(nodeId, physicalid)       ( nodeId * MAX_NUM_GPUS_PER_NODE + physicalid )
#define LR_GPU_PHYSICAL_ID(gpuEndPointId)            ( gpuEndPointId % MAX_NUM_GPUS_PER_NODE )

// Note the below numbers were derived allowing for 64 Ampere per node, 48 GB frame buffer.
// POR in first release is 16 Ampere per node, 48 GB frame buffer. If we build a huge system,
// (max is 2048 GPUs) with small nodes, or if we build Ampere with 96 GB, the below numbers will need to change.

#define LR_GPU_FABRIC_DEFAULT_ADDR_RANGE                ( 1LL << 36 )    // 64G
#define LR_GPU_MAX_MEM                                  ( 3LL << 34 )    // 48G
#define LR_GPU_FABRIC_DEFAULT_FLA_BASE(gpuEndPointId)   ( ( 1LL << 42 ) + ( (uint64_t)gpuEndPointId << 36 ) )
#define LR_GPU_FABRIC_DEFAULT_GPA_BASE(gpuEndPointId)   ( (uint64_t)gpuEndPointId << 36 ) 
#define LR_GPU_ENDPOINT_ID_FROM_FLA_BASE(gpuAddrBase)   ( ( (uint64_t)gpuAddrBase - ( 1LL << 42 ) ) >> 36 )
#define LR_GPU_ENDPOINT_ID_FROM_GPA_BASE(gpuAddrBase)   ( (uint64_t)gpuAddrBase >> 36 )

#endif // end of LimeRock macros

#define ILWALID_FABRIC_PARTITION_ID 0xFFFFFFFF

#define FM_SYSLOG_ERR(fmt, ...) \
        SYSLOG_ERROR("Error: fabricmanager: " fmt, ##__VA_ARGS__)

#define FM_SYSLOG_NOTICE(fmt, ...)  \
        SYSLOG_NOTICE("fabricmanager: " fmt,  ##__VA_ARGS__)

// common stats defines
typedef struct {
    uint64_t   txCounter0;
    uint64_t   rxCounter0;
    uint64_t   txCounter1;
    uint64_t   rxCounter1;
} LwlinkCounter_t;    // LWSWITCH_GET_LWLIPT_COUNTERS

typedef struct {
    uint64_t   elapsedTimeMsec;
    uint64_t   low;
    uint64_t   med;
    uint64_t   high;
    uint64_t   panic;
} PortLatencyHist_t;  // LWSWITCH_GET_INTERNAL_LATENCY

typedef struct
{
    uint32_t domain;
    uint32_t bus;
    uint32_t device;
    uint32_t function;
} DcgmFMPciInfo;

typedef struct {
    uint32_t gpuIndex;
    DcgmFMPciInfo pciInfo;
    char uuid[DCGM_DEVICE_UUID_BUFFER_SIZE];
} DcgmFMGpuInfo;

typedef struct {
    uint32_t switchIndex;
    uint32_t physicalId;
    DcgmFMPciInfo pciInfo;
    uint64_t ecid;
    uint64 enabledLinkMask;
} DcgmFMLWSwitchInfo;

// list of device information for a node
typedef std::list<DcgmFMGpuInfo> DcgmFMGpuInfoList;
typedef std::list<DcgmFMLWSwitchInfo> DcgmFMLWSwitchInfoList;
// map of device information for all the nodes in GFM
typedef std::map<uint32, DcgmFMGpuInfoList>  DcgmFMGpuInfoMap;
typedef std::map<uint32, DcgmFMLWSwitchInfoList> DcgmFMLWSwitchInfoMap;

#endif
