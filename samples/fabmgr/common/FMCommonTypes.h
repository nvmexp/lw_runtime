/*
 *  Copyright 2018-2020 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */

#pragma once

#include <stdint.h>
#include <map>
#include <list>
#include <set>
#include <string.h>
#include <g_lwconfig.h>
#include "FMErrorCodesInternal.h"

/* Version number used on all fabric manager messages                        */
#define FABRIC_MANAGER_VERSION 0x00100

/* defining these just once, instead of using #ifdef _WINDOWS for every oclwrence */
#ifdef _WINDOWS
#define inet_aton(a,b) inet_pton(AF_INET, a, b)
#define usleep(a) Sleep(a/1000);
#define poll(a, b, c) WaitForMultipleObjects((DWORD)b, (HANDLE*)a, 0, c)
#define bcopy(src, dest, len) memmove(dest, src, len)
#define snprintf _snprintf
#define strtok_r strtok_s
#define strncasecmp _strnicmp
typedef long long ssize_t;
#endif

#define MULTI_NODE_LWLINK_CONN_DUMP_FILE "/var/log/fabricmanager_links_discovered.csv"

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
#define NODE_ID_LOG_STR "NodeId"
#else
#ifdef DEBUG
#define NODE_ID_LOG_STR "NodeId"
#else
#define NODE_ID_LOG_STR "fid"
#endif
#endif

/* number of worker threads in FmServer work queue */
#define FM_SERVER_WORKER_NUM_THREADS 1

/* defaults and options */
#define DGX2_HGX2_TOPOLOGY_FILENAME             "dgx2_hgx2_topology"
#define DGXA100_HGXA100_TOPOLOGY_FILENAME       "dgxa100_hgxa100_topology"
// TODO: rename once product name is finalized for Vulcan/Viking
#define DGXH100_HGXH100_TOPOLOGY_FILENAME       "dgxh100_hgxh100_topology"

#ifdef DEBUG
#define DEFAULT_TOPOLOGY_CONF_FILE     "/usr/share/lwpu/lwswitch/topology_conf.txt"
#define DEFAULT_RUNNING_TOPOLOGY_FILE  "/usr/share/lwpu/lwswitch/running.topology"
#endif

#define OPT_DISABLE_GPU             "--disable-gpu"
#define OPT_DISABLE_SWITCH          "--disable-switch"
#define OPT_PORT_LOOPBACK           "--loopback-port"
#define OPT_DISABLE_PARTITION       "--disable-partition"


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


/* Error Polling Timeouts */
#define FATAL_ERROR_POLLING_TIMEOUT 10  // 10 seconds

/**
 * PCI format string for ::busId
 */
#define FM_DEVICE_PCI_BUS_ID_FMT                  "%08X:%02X:%02X.0"

/**
 * Utility macro for filling the pci bus id format from a lwmlPciInfo_t
 */
#define FM_DEVICE_PCI_BUS_ID_FMT_ARGS(pciInfo)      (pciInfo)->domain, \
                                                    (pciInfo)->bus,    \
                                                    (pciInfo)->device

typedef unsigned int fm_connection_id_t;

#define FM_CONNECTION_ID_NONE ((fm_connection_id_t)0)                           ///< Invalid or no connection
#define FM_CONNECTION_ID_START ((fm_connection_id_t) FM_CONNECTION_ID_NONE + 1) ///< Lowest valid connection id

typedef enum
{
    STOPPED         = 1,
    STOP_PENDING    = 2,
    RUNNING         = 3,
} RunState;

#define MAX_PORTS_PER_LWSWITCH          64
#define MAX_LWLINKS_PER_GPU             18

#ifdef LW_MODS
#define MAX_NUM_GPUS_PER_NODE           64
#else
#define MAX_NUM_GPUS_PER_NODE           16
#endif
#define MAX_NUM_LWSWITCH_PER_NODE       12

#define IS_PORT_VALID(index)       ( (index >= 0) && (index < MAX_PORTS_PER_LWSWITCH) )
#define IS_GPU_VALID(index)        ( (index >= 0) && (index < MAX_NUM_GPUS_PER_NODE) )

#define GPU_ENDPOINT_ID(nodeId, physicalid)          ( nodeId * MAX_NUM_GPUS_PER_NODE + physicalid )
#define GPU_PHYSICAL_ID(gpuEndPointId)               ( gpuEndPointId % MAX_NUM_GPUS_PER_NODE )
#define GPU_TARGET_ID(nodeId, physicalid)            GPU_ENDPOINT_ID(nodeId, physicalid)

#define ILWALID_FABRIC_PARTITION_ID 0xFFFFFFFF
#define FABRIC_PARTITION_VERSION FABRIC_MANAGER_VERSION

// FM REQ RESP timeouts for Simulation/Emulation
// GPU/LWSwitch will take more time to open/program compared to real Silicon and it is expected
#define FM_REQ_RESP_TIME_INTRVL_SIM    30 // seconds simulation to accommodate worse case when all GPUs
                                          // are attach/detach request response time.
#define FM_REQ_RESP_TIME_THRESHOLD_SIM 10 // total number of iterations
                                          // total wait time is FM_REQ_RESP_TIME_INTRVL_SIM * FM_REQ_RESP_TIME_THRESHOLD_SIM

// FM REQ RESP timeouts for real Silicon
#define FM_REQ_RESP_TIME_INTRVL        20 // seconds, to accommodate worse case when all GPUs
                                          // are attach/detach request response time.
#define FM_REQ_RESP_TIME_THRESHOLD     4  // total number of iterations.
                                          // total wait time is FM_REQ_RESP_TIME_INTRVL * FM_REQ_RESP_TIME_THRESHOLD

// configuration timeouts
#define FM_CONFIG_MSG_TIMEOUT_SIM 1800 // 1800 second, timeout for node GPU, LWSwitch configuration in simulation
#define FM_CONFIG_MSG_TIMEOUT     10   // 10 second, timeout for node, GPU, LWSwitch configuration

#define HEARTBEAT_INTERVAL       10    ///< Interval in seconds for sending heartbeats to LFM
#define HEARTBEAT_THRESHOLD      6     ///< Max # of LFM heartbeats missed after which it is considered a Fatal error
#define PARTITION_CFG_TIMEOUT_MS 30000 // 30000ms, 30s, timeout for partition configuration

/**
 * Buffer size guaranteed to be large enough for pci bus id
 */
#define FM_DEVICE_PCI_BUS_ID_BUFFER_SIZE  32

typedef struct {
    unsigned int domain;             // PCI domain on which the device's bus resides, 0 to 0xffffffff
    unsigned int bus;                // bus on which the device resides, 0 to 0xff
    unsigned int device;             // device's id on the bus, 0 to 31
    unsigned int function;           // PCI function information
    char busId[FM_DEVICE_PCI_BUS_ID_BUFFER_SIZE]; //the tuple domain:bus:device PCI identifier (&amp; NULL terminator)
} FMPciInfo_t;

#if !defined(LW_MODS) || !defined(INCLUDE_FM)
typedef unsigned long long uint64;
typedef unsigned int       uint32;
typedef unsigned short     uint16;
typedef unsigned char      uint8;
#endif

typedef uint32      FMNodeId_t;
typedef uint64_t    FMSwitchPortMask_t;
typedef uint64      FMGpuOrSwitchId_t;
typedef uint32      FMPhysicalId_t;
typedef uint32      FMLwlinkIndex_t;

#define FM_UUID_BUFFER_SIZE 80 // ref LWML

typedef struct FMUuid_st{
    char bytes[FM_UUID_BUFFER_SIZE];
    bool operator==(const FMUuid_st& rhs)
    {
        if (0 == memcmp(bytes, rhs.bytes, sizeof(FMUuid_st))) {
            return true;
        }
        // no match
        return false;
    };
    bool operator<(const FMUuid_st& rhs) const
    {
        if (0 > memcmp(bytes, rhs.bytes, sizeof(FMUuid_st))) {
            return true;
        }
        // no match
        return false;
    }
} FMUuid_t;

typedef struct {
    unsigned int gpuIndex;       // GPU Index
    uint32_t discoveredLinkMask; // discovered links at h/w level (aka supported links)
    uint32_t enabledLinkMask;    // lwrrently enabled link mask
    FMPciInfo_t pciInfo;         // The PCI BDF information
    uint32_t archType;           // GPU arch
    FMUuid_t uuid;               // UUID (ASCII string format)
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    bool     isEgmCapable;       // Indicate whether the GPU support EGM feature
    bool     isSpaCapable;       // Indicate whether the GPU support SPA address feature
    uint64_t spaAddress;         // GPU's SPA address if the GPU is SPA capable
#endif
} FMGpuInfo_t;

typedef struct {
    FMPciInfo_t pciInfo;        // The PCI information for the excluded GPU
    FMUuid_t uuid;              // The ASCII string UUID for the excluded GPU
} FMExcludedGpuInfo_t;

typedef struct {
    uint32_t switchIndex;
    uint32_t physicalId;
    FMPciInfo_t pciInfo;
    uint64_t ecid;
    uint64 enabledLinkMask;
    uint32_t archType;
    FMUuid_t uuid;
} FMLWSwitchInfo;

typedef struct {
    uint32_t physicalId;        // GPIO based physical Id
    FMPciInfo_t pciInfo;        // The PCI information for the excluded LWSwitch
    FMUuid_t uuid;              // The ASCII string UUID for the excluded LWSwitch
    uint32_t excludedReason;   // reason this switch was excluded
} FMExcludedLWSwitchInfo_t;

typedef std::list<FMGpuInfo_t> FMGpuInfoList;
typedef std::list<FMExcludedGpuInfo_t> FMExcludedGpuInfoList;
typedef std::list<FMLWSwitchInfo> FMLWSwitchInfoList;
typedef std::list<FMExcludedLWSwitchInfo_t> FMExcludedLWSwitchInfoList;

// map of device information for all the nodes in GFM
typedef std::map<uint32, FMGpuInfoList>  FMGpuInfoMap;
typedef std::map<uint32, FMExcludedGpuInfoList> FMExcludedGpuInfoMap;
typedef std::map<uint32, FMLWSwitchInfoList> FMLWSwitchInfoMap;
typedef std::map<uint32, FMExcludedLWSwitchInfoList> FMExcludedLWSwitchInfoMap;

typedef struct {
    uint32_t linkIndex;
    uint32_t linkLineRateMBps;
    uint32_t linkClockMhz;
    uint32_t linkClockType;
    uint32_t linkDataRateKiBps;
} FMLWLinkSpeedInfo;

typedef std::list<FMLWLinkSpeedInfo> FMLWLinkSpeedInfoList;

typedef struct {
    FMUuid_t uuid; // storing only GPU UUID instead of FMGpuInfo_t itself.
    FMLWLinkSpeedInfoList linkSpeedInfo;
} FMGpuLWLinkSpeedInfo;

typedef std::list<FMGpuLWLinkSpeedInfo> FMGpuLWLinkSpeedInfoList;
// map of device information for all the nodes in GFM
typedef std::map<uint32, FMGpuLWLinkSpeedInfoList>  FMGpuLWLinkSpeedInfoMap;

