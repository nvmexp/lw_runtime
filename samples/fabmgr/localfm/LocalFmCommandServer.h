
#pragma once

#include <iostream>
#include <fstream>
#include <map>
#include <list>

#include "FMCommandServer.h"

class LocalFabricManagerControl;

/********************************************************************************/
/* Implements command server command handling for global FM                     */
/********************************************************************************/

class LocalFMCommandServer : public FmCommandServerCmds
{
public:

    LocalFMCommandServer(LocalFabricManagerControl *pLocalFM);
    ~LocalFMCommandServer();

    // virtual methods in FmCommandServerCmds
    virtual void handleRunCmd(std::string &cmdLine, std::string &cmdResponse);
    virtual void handleQueryCmd(std::string &cmdLine, std::string &cmdResponse);

private:

    FmCommandServer *mpCmdServer;
    LocalFabricManagerControl *mpLocalFM;

    void parsePortRoutingTableQueryHelpCmd(std::vector<std::string> &cmdWords,
                                           uint32_t &switchIndex, uint32_t &portIndex,
                                           uint32_t &firstIndex,  uint32_t &numEntries);

    void handlePortQueryCmd(std::vector<std::string> &cmdWords, std::string &cmdResponse);
    void portQueryHelpCmd(std::string &cmdResponse);

    void dumpPortRemapTable(std::vector<std::string> &cmdWords, std::string &cmdResponse);
    void dumpPortRidTable(std::vector<std::string> &cmdWords, std::string &cmdResponse);
    void dumpPortRlanTable(std::vector<std::string> &cmdWords, std::string &cmdResponse);
    void dumpPortCounters(std::vector<std::string> &cmdWords, std::string &cmdResponse);
};

