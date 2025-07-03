/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#include "gdm_configurator.h"
#include "lwdiagutils.h"
#include "GlobalFabricManager.h"

namespace
{
    // Different FM operating modes
    typedef enum
    {
          FM_MODE_BAREMETAL = 0x0,         // Baremetal or Full pass through virtualization mode
          FM_MODE_SHARED_LWSWITCH = 0x1,   // Shared LWSwitch multitenancy mode
          FM_MODE_VGPU = 0x2,              // vGPU based multitenancy mode
          FM_MODE_MAX
    } FM_MODE;

    // Default config for a GFM instance
    typedef struct
    {
        UINT32 fabricMode = FM_MODE::FM_MODE_BAREMETAL;
        bool fabricModeRestart = false;
        bool stagedInit = true;
        UINT16 fmStartingTcpPort = 16000;
        char *fmLibCmdBindInterface = strdup("127.0.0.1");
        char *fmBindInterfaceIp = strdup("127.0.0.1");
        char *fmLibCmdSockPath = strdup("");
        UINT16 fmLibPortNumber = 6666;
        char *domainSocketPath = strdup("");
        char *stateFileName = strdup("/tmp/fabricmanager.state");
        bool continueWithFailures = false;
        UINT32 accessLinkFailureMode = 0;
        UINT32 trunkLinkFailureMode = 0;
        UINT32 lwswitchFailureMode = 0;
        bool enableTopologyValidation = false;
        char *fabricPartitionFileName = NULL;
        char *topologyFilePath = strdup("/usr/share/lwpu/lwswitch");
        bool disableDegradedMode = true;
        //bool disablePruning;
        INT32 gfmWaitTimeout = -1;
        bool simMode = false;
        char *fabricNodeConfigFile = strdup("fabric_node_config");
        char *multiNodeTopology = strdup("dgxa100_all_to_all_2node_topology");
    } GfmConfig;

    // TODO: Update the defaults using a config file
    static GfmConfig s_GfmConfig;
}

LwDiagUtils::EC GdmConfig::GdmGetGFMConfig(GlobalFmArgs_t *gfmArgs)
{
    gfmArgs->fabricMode = s_GfmConfig.fabricMode;
    gfmArgs->fabricModeRestart = s_GfmConfig.fabricModeRestart;
    gfmArgs->stagedInit = s_GfmConfig.stagedInit;
    gfmArgs->fmLibCmdBindInterface = s_GfmConfig.fmLibCmdBindInterface;
    gfmArgs->fmStartingTcpPort = s_GfmConfig.fmStartingTcpPort;
    gfmArgs->fmBindInterfaceIp = s_GfmConfig.fmBindInterfaceIp;
    gfmArgs->fmLibCmdSockPath = s_GfmConfig.fmLibCmdSockPath;
    gfmArgs->stateFileName = s_GfmConfig.stateFileName;
    gfmArgs->continueWithFailures = s_GfmConfig.continueWithFailures;
    gfmArgs->accessLinkFailureMode = s_GfmConfig.accessLinkFailureMode;
    gfmArgs->trunkLinkFailureMode = s_GfmConfig.trunkLinkFailureMode;
    gfmArgs->lwswitchFailureMode = s_GfmConfig.lwswitchFailureMode;
    gfmArgs->enableTopologyValidation = s_GfmConfig.enableTopologyValidation;
    gfmArgs->fabricPartitionFileName = s_GfmConfig.fabricPartitionFileName;
    gfmArgs->topologyFilePath = s_GfmConfig.topologyFilePath;
    gfmArgs->disableDegradedMode = s_GfmConfig.disableDegradedMode;
    gfmArgs->gfmWaitTimeout = s_GfmConfig.gfmWaitTimeout;
    gfmArgs->simMode = s_GfmConfig.simMode;
    gfmArgs->fabricNodeConfigFile = s_GfmConfig.fabricNodeConfigFile;
    gfmArgs->multiNodeTopology = s_GfmConfig.multiNodeTopology;

    return LwDiagUtils::OK;
}