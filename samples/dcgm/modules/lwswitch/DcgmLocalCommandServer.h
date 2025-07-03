
#pragma once

#include <iostream>
#include <fstream>
#include <map>
#include <list>

#include "DcgmCommandServer.h"

class DcgmLocalFabricManagerControl;

/********************************************************************************/
/* Implements command server command handling for global FM                     */
/********************************************************************************/

class DcgmLocalCommandServer : public DcgmCommandServerCmds
{
public:

    DcgmLocalCommandServer(DcgmLocalFabricManagerControl *pLocalFM);
    ~DcgmLocalCommandServer();

    // virtual methods in DcgmCommandServerCmds
    virtual void handleRunCmd(std::string &cmdLine, std::string &cmdResponse);
    virtual void handleQueryCmd(std::string &cmdLine, std::string &cmdResponse);

private:

    DcgmCommandServer *mpCmdServer;
    DcgmLocalFabricManagerControl *mpLocalFM;
};

