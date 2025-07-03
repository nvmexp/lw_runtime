/*
 *  Copyright 2018-2022 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */
#pragma once
#include <g_lwconfig.h>
// config option attributes

#define FM_CONFIG_MAX_ITEM_NAME_LEN     40
#define FM_CONFIG_MAX_STRING_ITEM_LEN   256
#define FM_CONFIG_MAX_ITEM_LINE_LEN     320   //item name len + item value len + accounting for any white spaces 
#define MAX_IP_ADDR_LEN                 32
#define FM_CONFIG_ITEM_TYPE_STRING  's'
#define FM_CONFIG_ITEM_TYPE_NUMBER  'n'

#define FM_VAR_RUNTIME_DATA_PATH     "/var/run/lwpu-fabricmanager"
#define FM_DEFAULT_PID_FILE_PATH     FM_VAR_RUNTIME_DATA_PATH "/lw-fabricmanager.pid"

typedef enum fmConfig {
    // This list must start at zero, and be in the same order
    // as the struct all_args list, below:
    LOG_LEVEL = 0,
    LOG_FILE_NAME,
    LOG_APPEND_TO_LOG,
    LOG_FILE_MAX_SIZE,
    LOG_USE_SYSLOG,
    ENABLE_GLOBALFM,
    ENABLE_LOCALFM,
    STARTING_TCP_PORT,
    FABRIC_MODE,
    SHARED_FABRIC_MODE,
    UNIX_SOCKET_PATH,
    STATE_FILE_NAME,
    DAEMONIZE,
    PID_FILE_PATH,
    BIND_INTERFACE_IP,
    FM_CMD_BIND_INTERFACE,
    FM_CMD_UNIX_SOCKET_PATH,
    FM_CMD_PORT,
    FM_CONTINUE_RUN_WITH_FAILURE,
    ACCESS_LINK_FAILURE_MODE,
    TRUNK_LINK_FAILURE_MODE,
    LWSWITCH_FAILURE_MODE,
    ENABLE_TOPOLOGY_VALIDATION,
    FABRIC_PARTITION_DEFINITION_FILE,
    SHARED_PARTITION_DEFINITION_FILE,
    FABRIC_MODE_RESTART,
    SHARED_FABRIC_MODE_RESTART,
    ABORT_LWDA_JOBS_ON_FM_EXIT,
    SWITCH_HEARTBEAT_TIMEOUT,
    TOPOLOGY_FILE_PATH,
    DISABLE_DEGRADED_MODE,
    GFM_WAIT_TIMEOUT,
    SIMULATION_MODE,
    FABRIC_NODE_CONFIG_FILE,
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    MULTI_NODE_TOPOLOGY,
#endif
    IMEX_REQ_TIMEOUT,
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    DISABLE_LWLINK_ALI,
#endif
    FM_LWLINK_RETRAIN_COUNT
} fmConfigEnum_t;

// Different FM operating modes 
typedef enum {
    FM_MODE_BAREMETAL = 0x0,         // Baremetal or Full pass through virtualization mode
    FM_MODE_SHARED_LWSWITCH = 0x1,   // Shared LWSwitch multitenancy mode
    FM_MODE_VGPU = 0x2,              // vGPU based multitenancy mode
    FM_MODE_MAX
} FM_OPER_MODE_TYPE;

typedef struct {
    char            configName[FM_CONFIG_MAX_ITEM_NAME_LEN];
    void*           configValue;   // pointer to the config value
    char            configValType; // one of FM_CONFIG_ITEM_TYPE
    fmConfigEnum_t  configEnum;
} FMConfigItem_t;

typedef struct {
    unsigned int logLevel;                                      // logging level
    char logFileName[FM_CONFIG_MAX_STRING_ITEM_LEN];            // log file name
    unsigned int appendToLogFile;                               // append to an existing log file or overwrite it
    unsigned int maxLogFileSize;                                // size/cap for logfile (in MB)
    unsigned int useSysLog;                                     // redirect all the logs to syslog instead of file
    unsigned int enableGlobalFM;                                // enable Global Fabric Manager
    unsigned int enableLocalFM;                                 // enable Local Fabric Manager
    unsigned int fmStartingTcpPort;                             // Starting TCP port number for FM control connections
    char fmUnixSockPath[FM_CONFIG_MAX_STRING_ITEM_LEN];         // FM Unix domain socket path
    unsigned int fabricMode;                                    // Fabric manager opertaing mode
    char fmStateFileName[FM_CONFIG_MAX_STRING_ITEM_LEN];        // Filename to save Fabric manager states. Only valid in multilatency mode
    unsigned int fmDaemonize;                                   // To or not to daemonize
    char fmPidFilePath[FM_CONFIG_MAX_STRING_ITEM_LEN];          // File path to store pid           
    char bindInterfaceIp[FM_CONFIG_MAX_STRING_ITEM_LEN];        // IP address of the network interface that the FM should listen on
    char fmLibCmdBindInterface[FM_CONFIG_MAX_STRING_ITEM_LEN];  // IP address of the network interface that the GFM Lib server should listen on
    char fmLibCmdUnixSockPath[FM_CONFIG_MAX_STRING_ITEM_LEN];   // FM Lib Unix domain socket path
    unsigned int fmLibPortNumber;                               // FM Lib Port Number
    unsigned int continueWithFailures;                          // FM continues to run when facing failures
    unsigned int accessLinkFailureMode;                         // Degraded mode option when GPU access links fails
    unsigned int trunkLinkFailureMode;                          // Degraded mode option when LWSwitch trunk links fails
    unsigned int lwswitchFailureMode;                           // Degraded mode option when LWSwitch fails
    unsigned int enableTopologyValidation;                      // Enable strict topology validation
    char fabricPartitionDefFile[FM_CONFIG_MAX_STRING_ITEM_LEN]; // To override the default fabric partition definition
                                                                // information in the topology file
    unsigned int fabricModeRestart;                             // restart fabric manager and flow shared LWSwitch or vGPU based resiliency/restart sequence.
    unsigned int abortLwdaJobsOnFmExit;                         // control running LWCA jobs behavior when Fabric Manager exits.
    unsigned int switchHeartbeatTimeout;                        // hidden config option for keep alive time, in millisecond
    char topologyFilePath[FM_CONFIG_MAX_STRING_ITEM_LEN];       // directory of topology file path
    unsigned int disableDegradedMode;                           // Disabled degraded mode processing
    int gfmWaitTimeout;                                         // Time that GFM waits for LFM to come up before giving up. Negative value denotes infinite wait.
    unsigned int simMode;                                       // run in simulation or emulation
    char fabricNodeConfigFile[FM_CONFIG_MAX_STRING_ITEM_LEN];   // File location containing IP addresses of all nodes in a multi node setup
    char multiNodeTopology[FM_CONFIG_MAX_STRING_ITEM_LEN];      // specifies multinode topology file name. Presents of this option override default single node mode    
    unsigned int imexReqTimeout;                                // low threshold timeout for importer requests in seconds
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    unsigned int disableLwlinkAli;                              // disable Lwlink ALI (Autonomous Link Initialization) training method
#endif
    int fmLwLinkRetrainCount;                                   // FM LWLink Retrain Count
} FMConfigOptions_t;
extern FMConfigOptions_t gFMConfigOptions;

int
fabricManagerLoadConfigOptions(char* configFileName);

