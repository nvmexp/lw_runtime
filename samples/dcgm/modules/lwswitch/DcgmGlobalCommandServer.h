
#pragma once

#include <iostream>
#include <fstream>
#include <map>
#include <list>

#include "DcgmCommandServer.h"

class DcgmGlobalFabricManager;

/********************************************************************************/
/* Implements command server command handling for global FM                     */
/********************************************************************************/

class DcgmGlobalCommandServer : public DcgmCommandServerCmds
{
public:

    DcgmGlobalCommandServer(DcgmGlobalFabricManager *pGlobalFM);
    ~DcgmGlobalCommandServer();

    // virtual methods in DcgmCommandServerCmds
    virtual void handleRunCmd(std::string &cmdLine, std::string &cmdResponse);
    virtual void handleQueryCmd(std::string &cmdLine, std::string &cmdResponse);

private:

    void dumpAllGpuInfo(std::string &cmdResponse);
    void dumpAllLWSwitchInfo(std::string &cmdResponse);
    void dumpAllLWLinkConnInfo(std::string &cmdResponse);

    DcgmCommandServer *mpCmdServer;
    DcgmGlobalFabricManager *mpGlobalFM;
};

