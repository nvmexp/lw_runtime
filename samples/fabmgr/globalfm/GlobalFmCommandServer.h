/*
 *  Copyright 2018-2019 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */
#pragma once

#include <iostream>
#include <fstream>
#include <map>
#include <list>
#include <set>

#include "FMCommandServer.h"
#include "GlobalFmFabricParser.h"

class GlobalFabricManager;

/********************************************************************************/
/* Implements command server command handling for global FM                     */
/********************************************************************************/

class GlobalFMCommandServer : public FmCommandServerCmds
{
public:

    GlobalFMCommandServer(GlobalFabricManager *pGlobalFM);
    ~GlobalFMCommandServer();

    // virtual methods in FmCommandServerCmds
    virtual void handleRunCmd(std::string &cmdLine, std::string &cmdResponse);
    virtual void handleQueryCmd(std::string &cmdLine, std::string &cmdResponse);

private:

    void dumpAllGpuInfo(std::string &cmdResponse);
    void dumpAllLWSwitchInfo(std::string &cmdResponse);
    void dumpAllLWLinkConnInfo(std::string &cmdResponse);

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    // Handle Multicast related debug command
    void allocateMulticastGroup(uint32_t partitionId, std::string &cmdResponse);
    void freeMulticastGroup(uint32_t partitionId, uint32_t groupId, std::string &cmdResponse);
    void freeMulticastGroups(uint32_t partitionId, std::string &cmdResponse);
    void setMulticastGroup(uint32_t partitionId, uint32_t groupId, bool reflectiveMode,
                           bool excludeSelf, std::set<GpuKeyType> &gpus, GpuKeyType primaryReplica,
                           std::string &cmdResponse);

    void getAvailableMulticastGroups(uint32_t partitionId, std::string &cmdResponse);
    void getMulticastGroupBaseAddress(uint32_t groupId, std::string &cmdResponse);
    void dumpMulticastGroup( uint32_t partitionId, uint32_t groupId, std::string &cmdResponse);
    void dumpAllMulticastGroup(uint32_t partitionId, std::string &cmdResponse);

    void multicastRunHelpCmd(std::string &cmdResponse);
    void multicastQueryHelpCmd(std::string &cmdResponse);

    void handleMulticastRunCmd(std::vector<std::string> &cmdWords, std::string &cmdResponse);
    void handleMulticastQueryCmd(std::vector<std::string> &cmdWords, std::string &cmdResponse);
#endif

    FmCommandServer *mpCmdServer;
    GlobalFabricManager *mpGlobalFM;
};

