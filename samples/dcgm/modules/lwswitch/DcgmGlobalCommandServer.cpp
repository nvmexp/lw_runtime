
#include <sstream>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <iterator>

#include "logging.h"
#include "DcgmFMError.h"
#include "DcgmFMCommon.h"
#include "DcgmGlobalCommandServer.h"
#include "DcgmGlobalFabricManager.h"

DcgmGlobalCommandServer::DcgmGlobalCommandServer(DcgmGlobalFabricManager *pGlobalFM)
{
    mpGlobalFM = pGlobalFM;
    mpCmdServer = new DcgmCommandServer(this, GLOBAL_FM_CMD_SERVER_PORT, (char*)FM_DEFAULT_BIND_INTERFACE);
    mpCmdServer->Start();
}

DcgmGlobalCommandServer::~DcgmGlobalCommandServer()
{
    delete mpCmdServer;
    mpCmdServer = NULL;
}

void
DcgmGlobalCommandServer::handleRunCmd(std::string &cmdLine, std::string &cmdResponse)
{

}

void
DcgmGlobalCommandServer::handleQueryCmd(std::string &cmdLine, std::string &cmdResponse)
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
        mpGlobalFM->mLWLinkDevRepo.dumpInfo(&outStr);
        cmdResponse = outStr.str();
    }

    if (queryCmd == "lwlink-conns") {
        dumpAllLWLinkConnInfo(cmdResponse);
    }

    if (queryCmd == "gpu-dev") {
        dumpAllGpuInfo(cmdResponse);
    }

    if (queryCmd == "lwswitch-dev") {
        dumpAllLWSwitchInfo(cmdResponse);
    }
}

void
DcgmGlobalCommandServer::dumpAllGpuInfo(std::string &cmdResponse)
{
    DcgmFMGpuInfoMap::iterator it;
    DcgmFMGpuInfoMap gpuInfoMap = mpGlobalFM->getGpuInfoMap();
    std::stringstream outStr;

    for ( it = gpuInfoMap.begin(); it != gpuInfoMap.end(); it++ ) {
        outStr << "\t Dumping GPU information for Node Index " << int(it->first) << std::endl;
        DcgmFMGpuInfoList gpuList = it->second;
        DcgmFMGpuInfoList::iterator jit;
        for ( jit = gpuList.begin(); jit != gpuList.end(); jit++ ) {
            DcgmFMGpuInfo gpuInfo = (*jit);
            outStr << "\t gpuIndex: " << int(gpuInfo.gpuIndex) << std::endl;
            outStr << "\t PCI Info:" << std::endl;
            outStr << "\t\t Domain:" << (int)gpuInfo.pciInfo.domain << std::endl;
            outStr << "\t\t Bus:" << (int)gpuInfo.pciInfo.bus << std::endl;
            outStr << "\t\t Device:" << (int)gpuInfo.pciInfo.device << std::endl;
            outStr << "\t\t Function:" << (int)gpuInfo.pciInfo.function << std::endl;
        }
    }

    DcgmFMGpuInfoMap blacklistGpuInfoMap = mpGlobalFM->getBlacklistGpuInfoMap();
    for ( it = blacklistGpuInfoMap.begin(); it != blacklistGpuInfoMap.end(); it++ ) {
        outStr << "\t Dumping blacklisted GPU information for Node Index " << int(it->first) << std::endl;
        DcgmFMGpuInfoList gpuList = it->second;
        DcgmFMGpuInfoList::iterator jit;
        for ( jit = gpuList.begin(); jit != gpuList.end(); jit++ ) {
            DcgmFMGpuInfo gpuInfo = (*jit);
            outStr << "\t gpuIndex: " << int(gpuInfo.gpuIndex) << std::endl;
            outStr << "\t PCI Info:" << std::endl;
            outStr << "\t\t Domain:" << (int)gpuInfo.pciInfo.domain << std::endl;
            outStr << "\t\t Bus:" << (int)gpuInfo.pciInfo.bus << std::endl;
            outStr << "\t\t Device:" << (int)gpuInfo.pciInfo.device << std::endl;
            outStr << "\t\t Function:" << (int)gpuInfo.pciInfo.function << std::endl;
            outStr << "\t\t uuid:" << std::hex << gpuInfo.uuid << std::endl;
        }
    }

    cmdResponse = outStr.str();
    if (cmdResponse.size() == 0) {
        cmdResponse = "No GPU Information is available\n";
    }
}

void
DcgmGlobalCommandServer::dumpAllLWSwitchInfo(std::string &cmdResponse)
{
    DcgmFMLWSwitchInfoMap::iterator it;
    DcgmFMLWSwitchInfoMap switchInfoMap = mpGlobalFM->getLwSwitchInfoMap();
    std::stringstream outStr;

    for ( it = switchInfoMap.begin(); it != switchInfoMap.end(); it++ ) {
        outStr << "\t Dumping LWSwitch information for Node Index " << int(it->first) << std::endl;
        DcgmFMLWSwitchInfoList switchList = it->second;
        DcgmFMLWSwitchInfoList::iterator jit;
        for ( jit = switchList.begin(); jit != switchList.end(); jit++ ) {
            DcgmFMLWSwitchInfo switchInfo = (*jit);
            outStr << "\t switchIndex: " << int(switchInfo.switchIndex) << std::endl;
            outStr << "\t physicalId: " << int(switchInfo.physicalId) << std::endl;
            outStr << "\t PCI Info:" << std::endl;
            outStr << "\t\t Domain:" << (int)switchInfo.pciInfo.domain << std::endl;
            outStr << "\t\t Bus:" << (int)switchInfo.pciInfo.bus << std::endl;
            outStr << "\t\t Device:" << (int)switchInfo.pciInfo.device << std::endl;
            outStr << "\t\t Function:" << (int)switchInfo.pciInfo.function << std::endl;
        }
    }

    cmdResponse = outStr.str();
    if (cmdResponse.size() == 0) {
        cmdResponse = "No LWSwitch Information is available\n";
    }
}

void
DcgmGlobalCommandServer::dumpAllLWLinkConnInfo(std::string &cmdResponse)
{
    std::stringstream outStr;

    outStr << "Dumping all Intra-Node connections" << std::endl;
    LWLinkIntraConnMap::iterator it;
    LWLinkIntraConnMap intraConnMap = mpGlobalFM->mLWLinkConnRepo.getIntraConnections();
    for ( it = intraConnMap.begin(); it != intraConnMap.end(); it++ ) {
        DcgmLWLinkDetailedConnList connList = it->second;
        DcgmLWLinkDetailedConnList::iterator jit;
        outStr << "Intra-Node connections for Node Index:" << it->first << std::endl;
        outStr << "Number of connections:" << connList.size() << std::endl;
        // dump each connection information
        for (jit = connList.begin(); jit != connList.end(); jit++ ) {
            DcgmFMLWLinkDetailedConnInfo *connInfo = (*jit);
            connInfo->dumpConnAndStateInfo(&outStr, mpGlobalFM->mLWLinkDevRepo);
        }
    }

    outStr << "Dumping all Inter-Node connections" << std::endl;
    LWLinkInterNodeConns::iterator jit;
    LWLinkInterNodeConns interConnMap = mpGlobalFM->mLWLinkConnRepo.getInterConnections();
    for (jit = interConnMap.begin(); jit != interConnMap.end(); jit++ ) {
        DcgmFMLWLinkDetailedConnInfo *connInfo = (*jit);
        connInfo->dumpConnAndStateInfo( &outStr, mpGlobalFM->mLWLinkDevRepo);
    }

    cmdResponse = outStr.str();
}
