
#include <sstream>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <iterator>

#include "logging.h"
#include "DcgmFMError.h"
#include "DcgmFMCommon.h"
#include "DcgmLocalCommandServer.h"
#include "DcgmLocalFabricManager.h"

DcgmLocalCommandServer::DcgmLocalCommandServer(DcgmLocalFabricManagerControl *pLocalFM)
{
    mpLocalFM = pLocalFM;
    mpCmdServer = new DcgmCommandServer(this, LOCAL_FM_CMD_SERVER_PORT, (char*)FM_DEFAULT_BIND_INTERFACE);
    mpCmdServer->Start();
}

DcgmLocalCommandServer::~DcgmLocalCommandServer()
{
    delete mpCmdServer;
    mpCmdServer = NULL;
}

void
DcgmLocalCommandServer::handleRunCmd(std::string &cmdLine, std::string &cmdResponse)
{

}

void
DcgmLocalCommandServer::handleQueryCmd(std::string &cmdLine, std::string &cmdResponse)
{
    cmdResponse = "Unknown Query Command\n";

    // first split the command string into words
    std::istringstream strCmd(cmdLine);
    std::vector<std::string> cmdWords((std::istream_iterator<std::string>(strCmd)),
                                     std::istream_iterator<std::string>());

    // query commands are of the following form. so we expect two words
    // /query lwlink-dev or /query lwlink-conns etc
    if (cmdWords.size() < 2) {
        cmdResponse = "Invalid number of arguments\n";
        return;
    }

    // second word is the alwtal command word
    std::string queryCmd(cmdWords[1]);

    if (queryCmd == "lwlink-dev") {
        std::stringstream outStr;
        mpLocalFM->mLWLinkDevRepo->dumpInfo(&outStr);
        cmdResponse = outStr.str();
    }
}

