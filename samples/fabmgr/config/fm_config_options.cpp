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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>
#include <errno.h>
#include <climits>
#include <map>
#include "fm_config_options.h"
#include "fm_log.h"
#include <g_lwconfig.h>

// structure to hold all the Fabric Manager config file options
FMConfigOptions_t gFMConfigOptions = {0};

// list of config items
FMConfigItem_t fmConfigItemList[] =
{
    // config item/name                   value holder                               item data type         
    { "LOG_LEVEL",                        &gFMConfigOptions.logLevel,                FM_CONFIG_ITEM_TYPE_NUMBER,      LOG_LEVEL                   },
    { "LOG_FILE_NAME",                    &gFMConfigOptions.logFileName,             FM_CONFIG_ITEM_TYPE_STRING,      LOG_FILE_NAME               },
    { "LOG_APPEND_TO_LOG",                &gFMConfigOptions.appendToLogFile,         FM_CONFIG_ITEM_TYPE_NUMBER,      LOG_APPEND_TO_LOG           },
    { "LOG_FILE_MAX_SIZE",                &gFMConfigOptions.maxLogFileSize,          FM_CONFIG_ITEM_TYPE_NUMBER,      LOG_FILE_MAX_SIZE           },
    { "LOG_USE_SYSLOG",                   &gFMConfigOptions.useSysLog,               FM_CONFIG_ITEM_TYPE_NUMBER,      LOG_USE_SYSLOG              },
    { "ENABLE_GLOBALFM",                  &gFMConfigOptions.enableGlobalFM,          FM_CONFIG_ITEM_TYPE_NUMBER,      ENABLE_GLOBALFM             },
    { "ENABLE_LOCALFM",                   &gFMConfigOptions.enableLocalFM,           FM_CONFIG_ITEM_TYPE_NUMBER,      ENABLE_LOCALFM              },
    { "STARTING_TCP_PORT",                &gFMConfigOptions.fmStartingTcpPort,       FM_CONFIG_ITEM_TYPE_NUMBER,      STARTING_TCP_PORT           },
    { "FABRIC_MODE",                      &gFMConfigOptions.fabricMode,              FM_CONFIG_ITEM_TYPE_NUMBER,      FABRIC_MODE                 },
    { "SHARED_FABRIC_MODE",               &gFMConfigOptions.fabricMode,              FM_CONFIG_ITEM_TYPE_NUMBER,      SHARED_FABRIC_MODE          },
    { "UNIX_SOCKET_PATH",                 &gFMConfigOptions.fmUnixSockPath,          FM_CONFIG_ITEM_TYPE_STRING,      UNIX_SOCKET_PATH            },
    { "STATE_FILE_NAME",                  &gFMConfigOptions.fmStateFileName,         FM_CONFIG_ITEM_TYPE_STRING,      STATE_FILE_NAME             },
    { "DAEMONIZE",                        &gFMConfigOptions.fmDaemonize,             FM_CONFIG_ITEM_TYPE_NUMBER,      DAEMONIZE                   },
    { "PID_FILE_PATH",                    &gFMConfigOptions.fmPidFilePath,           FM_CONFIG_ITEM_TYPE_STRING,      PID_FILE_PATH               },
    { "BIND_INTERFACE_IP",                &gFMConfigOptions.bindInterfaceIp,         FM_CONFIG_ITEM_TYPE_STRING,      BIND_INTERFACE_IP           },
    { "FM_CMD_BIND_INTERFACE",            &gFMConfigOptions.fmLibCmdBindInterface,   FM_CONFIG_ITEM_TYPE_STRING,      FM_CMD_BIND_INTERFACE       },
    { "FM_CMD_UNIX_SOCKET_PATH",          &gFMConfigOptions.fmLibCmdUnixSockPath,    FM_CONFIG_ITEM_TYPE_STRING,      FM_CMD_UNIX_SOCKET_PATH     },
    { "FM_CMD_PORT_NUMBER",               &gFMConfigOptions.fmLibPortNumber,         FM_CONFIG_ITEM_TYPE_NUMBER,      FM_CMD_PORT                 },
    { "FM_STAY_RESIDENT_ON_FAILURES",     &gFMConfigOptions.continueWithFailures,    FM_CONFIG_ITEM_TYPE_NUMBER,      FM_CONTINUE_RUN_WITH_FAILURE},
    { "ACCESS_LINK_FAILURE_MODE",         &gFMConfigOptions.accessLinkFailureMode,   FM_CONFIG_ITEM_TYPE_NUMBER,      ACCESS_LINK_FAILURE_MODE    },
    { "TRUNK_LINK_FAILURE_MODE",          &gFMConfigOptions.trunkLinkFailureMode,    FM_CONFIG_ITEM_TYPE_NUMBER,      TRUNK_LINK_FAILURE_MODE     },
    { "LWSWITCH_FAILURE_MODE",            &gFMConfigOptions.lwswitchFailureMode,     FM_CONFIG_ITEM_TYPE_NUMBER,      LWSWITCH_FAILURE_MODE       },
    { "ENABLE_TOPOLOGY_VALIDATION",       &gFMConfigOptions.enableTopologyValidation,FM_CONFIG_ITEM_TYPE_NUMBER,      ENABLE_TOPOLOGY_VALIDATION  },
    { "FABRIC_PARTITION_DEFINITION_FILE", &gFMConfigOptions.fabricPartitionDefFile,  FM_CONFIG_ITEM_TYPE_STRING,      FABRIC_PARTITION_DEFINITION_FILE},
    { "SHARED_PARTITION_DEFINITION_FILE", &gFMConfigOptions.fabricPartitionDefFile,  FM_CONFIG_ITEM_TYPE_STRING,      SHARED_PARTITION_DEFINITION_FILE},
    { "FABRIC_MODE_RESTART",              &gFMConfigOptions.fabricModeRestart,       FM_CONFIG_ITEM_TYPE_NUMBER,      FABRIC_MODE_RESTART         },
    { "SHARED_FABRIC_MODE_RESTART",       &gFMConfigOptions.fabricModeRestart,       FM_CONFIG_ITEM_TYPE_NUMBER,      SHARED_FABRIC_MODE_RESTART  },
    { "ABORT_LWDA_JOBS_ON_FM_EXIT",       &gFMConfigOptions.abortLwdaJobsOnFmExit,   FM_CONFIG_ITEM_TYPE_NUMBER,      ABORT_LWDA_JOBS_ON_FM_EXIT  },
    { "SWITCH_HEARTBEAT_TIMEOUT",         &gFMConfigOptions.switchHeartbeatTimeout,  FM_CONFIG_ITEM_TYPE_NUMBER,      SWITCH_HEARTBEAT_TIMEOUT    },
    { "TOPOLOGY_FILE_PATH",               &gFMConfigOptions.topologyFilePath,        FM_CONFIG_ITEM_TYPE_STRING,      TOPOLOGY_FILE_PATH          },
    { "DISABLE_DEGRADED_MODE",            &gFMConfigOptions.disableDegradedMode,     FM_CONFIG_ITEM_TYPE_NUMBER,      DISABLE_DEGRADED_MODE       },
    { "GFM_WAIT_TIMEOUT",                 &gFMConfigOptions.gfmWaitTimeout,          FM_CONFIG_ITEM_TYPE_NUMBER,      GFM_WAIT_TIMEOUT            },
    { "SIMULATION_MODE",                  &gFMConfigOptions.simMode,                 FM_CONFIG_ITEM_TYPE_NUMBER,      SIMULATION_MODE             },
    { "FABRIC_NODE_CONFIG_FILE",          &gFMConfigOptions.fabricNodeConfigFile,    FM_CONFIG_ITEM_TYPE_STRING,      FABRIC_NODE_CONFIG_FILE     },
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    { "MULTI_NODE_TOPOLOGY",              &gFMConfigOptions.multiNodeTopology,       FM_CONFIG_ITEM_TYPE_STRING,      MULTI_NODE_TOPOLOGY         },
#endif
    { "IMEX_REQ_TIMEOUT",                 &gFMConfigOptions.imexReqTimeout,          FM_CONFIG_ITEM_TYPE_NUMBER,      IMEX_REQ_TIMEOUT            },
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    { "DISABLE_LWLINK_ALI",               &gFMConfigOptions.disableLwlinkAli,        FM_CONFIG_ITEM_TYPE_NUMBER,      DISABLE_LWLINK_ALI          },
#endif
    { "FM_LWLINK_RETRAIN_COUNT",          &gFMConfigOptions.fmLwLinkRetrainCount,    FM_CONFIG_ITEM_TYPE_NUMBER,      FM_LWLINK_RETRAIN_COUNT     }
};

FILE* gConfigFileHandle = NULL;
int gConfigItemCount = sizeof(fmConfigItemList)/sizeof(FMConfigItem_t);

static void
fabricManagerSetDefaultConfigOptions()
{
    // log related default values
    gFMConfigOptions.logLevel = 4; //INFO logging by default
    gFMConfigOptions.appendToLogFile = 1;
    gFMConfigOptions.maxLogFileSize = 1024; // 1024 MB by default 
    gFMConfigOptions.useSysLog = 0; // by default use file based logging
    strncpy(gFMConfigOptions.logFileName, 
            "/var/log/fabricmanager.log", sizeof(gFMConfigOptions.logFileName)-1);
    // enable global FM and local FM by default
    gFMConfigOptions.enableGlobalFM = 1;
    gFMConfigOptions.enableLocalFM = 1;
    gFMConfigOptions.fmStartingTcpPort = 16000;
    gFMConfigOptions.fabricMode = FM_MODE_BAREMETAL;
    strncpy(gFMConfigOptions.fmUnixSockPath, "", sizeof(gFMConfigOptions.fmUnixSockPath) - 1);
    strncpy(gFMConfigOptions.fmStateFileName, "/tmp/fabricmanager.state",
            sizeof(gFMConfigOptions.fmStateFileName) - 1);
    gFMConfigOptions.fmDaemonize = 1;
    strncpy(gFMConfigOptions.fmPidFilePath, FM_DEFAULT_PID_FILE_PATH, sizeof(gFMConfigOptions.fmPidFilePath) - 1);
    /* Change FM_DEFAULT_BIND_INTERFACE in FMCommonTypes.h if changing the Bind Interface value below below */
    strncpy(gFMConfigOptions.bindInterfaceIp, "127.0.0.1", sizeof(gFMConfigOptions.bindInterfaceIp) - 1);
    strncpy(gFMConfigOptions.fmLibCmdBindInterface, "127.0.0.1", sizeof(gFMConfigOptions.fmLibCmdBindInterface) - 1);
    strncpy(gFMConfigOptions.fmLibCmdUnixSockPath, "", sizeof(gFMConfigOptions.fmLibCmdUnixSockPath) - 1); 
    /* Change FM_CMD_PORT_NUMBER in lw_fm_types.h if changing the port number below */
    gFMConfigOptions.fmLibPortNumber = 6666;
    gFMConfigOptions.continueWithFailures = 0;
    gFMConfigOptions.accessLinkFailureMode = 0;
    gFMConfigOptions.trunkLinkFailureMode = 0;
    gFMConfigOptions.lwswitchFailureMode = 0;
    gFMConfigOptions.enableTopologyValidation = 0;
    memset(gFMConfigOptions.fabricPartitionDefFile, 0, FM_CONFIG_MAX_STRING_ITEM_LEN);
    gFMConfigOptions.fabricModeRestart = 0;
    gFMConfigOptions.abortLwdaJobsOnFmExit = 1; // by default abort LWCA jobs on FM exit
    gFMConfigOptions.switchHeartbeatTimeout = 9000; //this should be a multiple of FM_SWITCH_HEARTBEAT_FREQ, in milliseconds
    strncpy(gFMConfigOptions.topologyFilePath, "/usr/share/lwpu/lwswitch/", FM_CONFIG_MAX_STRING_ITEM_LEN);
    gFMConfigOptions.disableDegradedMode = 0; // by default degraded mode processing is enabled
    // negative value for gfmWaitTimeout denotes an infinite wait time
    gFMConfigOptions.gfmWaitTimeout = 10;     //gfm waits a default 10 seconds for LFM to come up
    gFMConfigOptions.simMode = 0;             // by default FM is not running in LWSwitch/GPU simulation and emulation environment
    strncpy(gFMConfigOptions.fabricNodeConfigFile, "/usr/share/lwpu/lwswitch/fabric_node_config", FM_CONFIG_MAX_STRING_ITEM_LEN);
    memset(gFMConfigOptions.multiNodeTopology, 0, FM_CONFIG_MAX_STRING_ITEM_LEN);
    gFMConfigOptions.imexReqTimeout = 10;     // in seconds
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    gFMConfigOptions.disableLwlinkAli = 1;    // For now, by default Lwlink ALI is disabled
#endif
    gFMConfigOptions.fmLwLinkRetrainCount= -1; // by default FM LWLink retraining count is set to -1
}

static int
fabricManagerOpenConfigOptionFile(char* configFileName)
{
    gConfigFileHandle = fopen(configFileName, "r");
    if (gConfigFileHandle == NULL) {
        // log detailed error information
        int lastErrNum = errno;
        fprintf(stderr, "fabric manager config file open failed with: %s\n", strerror(lastErrNum));
        return -1;
    }
    return 0;
}

static void
fabricManagerCloseConfigOptionFile()
{
    fclose(gConfigFileHandle);
    gConfigFileHandle = NULL;
}

static int 
fabricManagerRemoveNewLineChar(char *configItemValueBuf) 
{
    size_t len = strlen(configItemValueBuf);
    if (configItemValueBuf[len-1] == '\n') {
        configItemValueBuf[len-1] = '\0';
    }
    return 0;
}

static int 
fabricManagerRemoveWhiteSpaces(char *configItemValueBuf) 
{
    char *startPtr = configItemValueBuf;
    char *endPtr = configItemValueBuf + strlen(configItemValueBuf);

    while (*startPtr == ' ') {
        startPtr++;
    }    
    while (*endPtr == ' ') {
        endPtr--;
    }

    strncpy(configItemValueBuf, startPtr, endPtr - startPtr + 1);
    return 0;
}

static int 
fabricManagerCleanString(char *configItemValueBuf) 
{
    //trim spaces in the string
    if (fabricManagerRemoveWhiteSpaces(configItemValueBuf) < 0) {
        return -1;
    }

    return 0;
}

static bool 
fabricManagerGetIntFromString(char *configItemValueBuf, int *configValue) 
{
    int base = 10;
    long temp; 
    int temp_errno = errno;   //cache the errno as it may change
    char *endPtr;
    temp = strtol(configItemValueBuf, &endPtr, base);

    //check if number is out of bounds
    if (errno == ERANGE && (temp == LONG_MAX || temp == LONG_MIN)) {
        return false;
    }
    else if(errno != 0 && temp == 0) {
        return false;
    }
    //check if there is no number
    if (endPtr == configItemValueBuf || temp > INT_MAX || temp < INT_MIN) {
        return false;
    }
    //number has been parsed successfully
    errno = temp_errno;  //restore errno
    *configValue = (int)temp;
    return true;
}

static 
bool checkRange(int *configValue, int rangeStart, int rangeEnd)
{
    int value = *configValue;
    if (value >= rangeStart && value <=rangeEnd)
        return true;
 
    return false;
}
 
static
bool checkIPAddr(char *configValue)
{
    char *ip, *remainingBuf;
    int dotCount = 0;
    int ipIntVal;
    char lineBuf[FM_CONFIG_MAX_ITEM_LINE_LEN];
    int configStrLen = strlen(configValue) - 1;

    if (configValue[configStrLen] == '.')
        return false;

    memset(lineBuf, 0, sizeof(lineBuf));
    strncpy(lineBuf, configValue, FM_CONFIG_MAX_ITEM_LINE_LEN);
    ip = strtok_r(lineBuf, ".", &remainingBuf);
    while (ip != NULL && (fabricManagerGetIntFromString(ip, &ipIntVal) && ipIntVal <= 255) 
           && (fabricManagerGetIntFromString(ip, &ipIntVal) && ipIntVal >= 0) ) 
    {
        ip = strtok_r(NULL, ".", &remainingBuf);
        dotCount++;
    }

    if ((remainingBuf == NULL || !strncmp(remainingBuf, "\0", 1)) && dotCount == 4) 
        return true;

    return false;
}
 
static
bool checkPathValidity(char *filePath)
{
    //extract directory   
    char *dir, remainingBuf;
    char lineBuf[FM_CONFIG_MAX_ITEM_LINE_LEN];
    memset(lineBuf, 0, sizeof(lineBuf));
    strncpy(lineBuf, filePath, FM_CONFIG_MAX_ITEM_LINE_LEN);
    
    if (strlen(lineBuf) == 0) 
        return true;
 
    dir = dirname(lineBuf);
 
    if (access(dir, F_OK) == 0) {
        return true;
    }
 
    return false;
}
 
static bool 
verifyOption(int index, void *configValue)
{
    fmConfigEnum_t cfg = (fmConfigEnum_t)index;
 
    switch (cfg) {
        case LOG_LEVEL:
            return checkRange((int*) configValue, 0, 5);
 
        case LOG_APPEND_TO_LOG:
        case LOG_USE_SYSLOG:
        case FABRIC_MODE_RESTART:
        case SHARED_FABRIC_MODE_RESTART:
        case DAEMONIZE:
        case DISABLE_DEGRADED_MODE:
            return checkRange((int*) configValue, 0, 1);
 
        case FABRIC_MODE:
        case SHARED_FABRIC_MODE:
            return checkRange((int*) configValue, 0, 2);

        case STARTING_TCP_PORT:
        case FM_CMD_PORT:
            return checkRange((int*) configValue, 1024, 65535);
 
        case BIND_INTERFACE_IP:
        case FM_CMD_BIND_INTERFACE:
            return checkIPAddr((char*)configValue);
 
        case LOG_FILE_NAME:
        case STATE_FILE_NAME:
        case PID_FILE_PATH:
        case UNIX_SOCKET_PATH:
        case FM_CMD_UNIX_SOCKET_PATH:
        case TOPOLOGY_FILE_PATH:
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)            
        case FABRIC_NODE_CONFIG_FILE:
#endif
            return checkPathValidity((char*) configValue);

        default:
            return true;
    }
}
 
 
static void 
fabricmanagerVerifyConfigOptions() 
{
    int idx;
    for (idx = 0; idx < gConfigItemCount; idx ++) {
        if (verifyOption(fmConfigItemList[idx].configEnum, fmConfigItemList[idx].configValue) == false) {
            fprintf(stderr, "fabric manager config file item: %s has an invalid value\n", fmConfigItemList[idx].configName);
            exit(0);
        }
    }
    return;
}

static int
fabricManagerParseNumberConfigValue(int *fmConfigValue, char *configItemValueBuf)
{
    int temp;
    if (fabricManagerGetIntFromString(configItemValueBuf, fmConfigValue) == false) {
        return -1;
    }
    return 0;
}

static int 
fabricManagerParseStringConfigValue(char *fmConfigValue, char *configItemValueBuf)
{
    fabricManagerRemoveNewLineChar(configItemValueBuf);
    if (strlen(configItemValueBuf) > FM_CONFIG_MAX_STRING_ITEM_LEN) {
        return -1;
    }
    memset(fmConfigValue, 0, strlen(fmConfigValue));
    strncpy(fmConfigValue, configItemValueBuf, strlen(configItemValueBuf));
    return 0;
}

static int
fabricManagerParseOneConfigItem(char* configItemNameBuf, char* configItemValueBuf)
{
    int bFound = 0;
    if ((NULL == configItemNameBuf) || (NULL == configItemValueBuf)) {
        return -1;
    }
    for (int idx = 0; idx < gConfigItemCount; idx ++) {
        // compare with each configurable item
        if (strncmp(fmConfigItemList[idx].configName, configItemNameBuf, sizeof(fmConfigItemList[idx].configName)) == 0) {
            // found a matching config item, parse its value
            bFound = 1;
            switch (fmConfigItemList[idx].configValType) {
                case FM_CONFIG_ITEM_TYPE_NUMBER:
                    if (fabricManagerParseNumberConfigValue((int*)fmConfigItemList[idx].configValue, configItemValueBuf) < 0) {
                        fprintf(stderr, "failed to parse fabric manager config file item %s\n", configItemNameBuf);
                        return -1;    
                    }
                    break;
                case FM_CONFIG_ITEM_TYPE_STRING:
                    if (fabricManagerParseStringConfigValue((char*)fmConfigItemList[idx].configValue, configItemValueBuf) < 0) {
                        fprintf(stderr, "failed to parse fabric manager config file item %s\n", configItemNameBuf);
                        return -1;
                    }
                    break;
                default:
                    fprintf(stderr, "invalid config data type for fabric manager config item %s\n", configItemNameBuf);
                    return -1;
            }
        }
    }
    if (0 == bFound) {
        fprintf(stderr, "unsupported config item %s is specified in fabric manager config file\n", configItemNameBuf);
        return -1;        
    }
    // the item is parsed successfully
    return 0;
}

/*****************************************************************************
 Method to read and parse all the config options
*****************************************************************************/

int
fabricManagerLoadConfigOptions(char* configFileName)
{
    if (NULL == configFileName) {
        fprintf(stderr, "fabric manager config file path argument is empty or null");
        return -1;
    }
    // load default config values so that if the option is not explicitly specified
    // we can use the default values.
    fabricManagerSetDefaultConfigOptions();
    // open the config file
    if (fabricManagerOpenConfigOptionFile(configFileName) < 0) {
        // error already logged
        //return 0 as we can still continue with default options
        fprintf(stderr, "%s\n", "failed to open/read fabric manager config file, continuing with default options");
        // Note: logging is not initialized yet, so logging to console instead.
        return 0;
    }
    // parse the config file and read all the values
    ssize_t readLen;
    char lineBuf[FM_CONFIG_MAX_ITEM_LINE_LEN];
    size_t lineBufSize = FM_CONFIG_MAX_ITEM_LINE_LEN;
    char* tempLineBuf=lineBuf;
    while ((readLen = getline(&tempLineBuf, &lineBufSize, gConfigFileHandle)) != -1) {
        // split the line into item and value
        char *configItemNameBuf, *configItemValueBuf, *remainingBuf;
        if (*tempLineBuf == '#') {
            continue;
        }
        configItemNameBuf = strtok_r(lineBuf, "=", &remainingBuf);
        if (NULL == configItemNameBuf) {
            // skip this line as it don't have <parameter=val> format
            continue;
        }
        // get the option value
        configItemValueBuf = strtok_r(NULL, "=", &remainingBuf);
        if (NULL == configItemValueBuf) {
            // skip this line as it don't have <parameter=val> format
            continue;
        }
        fabricManagerCleanString(configItemNameBuf);
        fabricManagerCleanString(configItemValueBuf);
        // now we have configItem and its value. parse the same.
        if (fabricManagerParseOneConfigItem(configItemNameBuf, configItemValueBuf) < 0) {
            // error already logged
            return -1;
        }        
    }

    fabricmanagerVerifyConfigOptions();

    // close the config file
    fabricManagerCloseConfigOptionFile();
    // all looks good
    return 0;
}
