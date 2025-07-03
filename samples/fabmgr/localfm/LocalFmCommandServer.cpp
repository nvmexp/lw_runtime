
#include <sstream>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <iterator>

#include "fm_log.h"
#include "FMCommonTypes.h"
#include "LocalFmCommandServer.h"
#include "LocalFabricManager.h"

LocalFMCommandServer::LocalFMCommandServer(LocalFabricManagerControl *pLocalFM)
{
    mpLocalFM = pLocalFM;
    mpCmdServer = new FmCommandServer(this, LOCAL_FM_CMD_SERVER_PORT, (char*)FM_DEFAULT_BIND_INTERFACE);
    mpCmdServer->Start();
}

LocalFMCommandServer::~LocalFMCommandServer()
{
    delete mpCmdServer;
    mpCmdServer = NULL;
}

void
LocalFMCommandServer::handleRunCmd(std::string &cmdLine, std::string &cmdResponse)
{

}

void
LocalFMCommandServer::handleQueryCmd(std::string &cmdLine, std::string &cmdResponse)
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

    if (queryCmd == "port") {
        handlePortQueryCmd(cmdWords, cmdResponse);
    }
}

void
LocalFMCommandServer::parsePortRoutingTableQueryHelpCmd(std::vector<std::string> &cmdWords,
                                                        uint32_t &switchIndex, uint32_t &portIndex,
                                                        uint32_t &firstIndex,  uint32_t &numEntries)
{
    switchIndex = stoi(cmdWords[4], nullptr);
    portIndex   = stoi(cmdWords[5], nullptr);
    firstIndex  = stoi(cmdWords[6], nullptr);
    numEntries  = stoi(cmdWords[7], nullptr);
}

void
LocalFMCommandServer::portQueryHelpCmd(std::string &cmdResponse)
{
    std::stringstream outStr;

    outStr << "/query port remap primary   switchIndex portIndex firstIndex numEntries " << std::endl;
    outStr << "/query port remap exta      switchIndex portIndex firstIndex numEntries " << std::endl;
    outStr << "/query port remap extb      switchIndex portIndex firstIndex numEntries " << std::endl;
    outStr << "/query port remap multicast switchIndex portIndex firstIndex numEntries " << std::endl;

    outStr << "/query port rid   unicast   switchIndex portIndex firstIndex numEntries " << std::endl;
    outStr << "/query port rid   multicast switchIndex portIndex firstIndex extended "   << std::endl;
    outStr << "/query port rlan  unicast   switchIndex portIndex firstIndex numEntries " << std::endl;

    outStr << "/query port counter         switchIndex portIndex" << std::endl;

    cmdResponse = outStr.str();
}

void
LocalFMCommandServer::dumpPortRemapTable(std::vector<std::string> &cmdWords, std::string &cmdResponse)
{
    std::stringstream outStr;
    std::string tableSelect(cmdWords[3]);

    uint32_t switchIndex, portIndex, firstIndex, numEntries;
    parsePortRoutingTableQueryHelpCmd(cmdWords, switchIndex, portIndex, firstIndex, numEntries);

    LocalFMSwitchInterface *pSwitchIntf = mpLocalFM->switchInterfaceAtIndex(switchIndex);
    if (pSwitchIntf == NULL) {
        outStr << "Invalid LWSwitch at switchIndex " << switchIndex << std::endl;
        cmdResponse = outStr.str();
        return;
    }

    if (portIndex >= LWSWITCH_MAX_PORTS) {
        outStr << "Invalid portIndex " << portIndex << std::endl;
        cmdResponse = outStr.str();
        return;
    }

    if (numEntries > LWSWITCH_REMAP_POLICY_ENTRIES_MAX) {
        outStr << "numEntries " << numEntries << " is more than max "
                << LWSWITCH_REMAP_POLICY_ENTRIES_MAX << std::endl;
        cmdResponse = outStr.str();
        return;
    }

    switchIoctl_t ioctlStruct;
    LWSWITCH_GET_REMAP_POLICY_PARAMS ioctlParams;

    memset(&ioctlParams, 0, sizeof(ioctlParams));
    memset(&ioctlStruct, 0, sizeof(ioctlStruct));

    ioctlStruct.type = IOCTL_LWSWITCH_GET_REMAP_POLICY;
    ioctlStruct.ioctlParams = &ioctlParams;
    ioctlStruct.paramSize = sizeof(ioctlParams);
    ioctlParams.portNum = portIndex;
    ioctlParams.firstIndex = firstIndex;
    ioctlParams.numEntries = numEntries;

    if (tableSelect == "primary") {
        ioctlParams.tableSelect = LWSWITCH_TABLE_SELECT_REMAP_PRIMARY;

    } else if (tableSelect == "exta") {
        ioctlParams.tableSelect = LWSWITCH_TABLE_SELECT_REMAP_EXTA;

    } else if (tableSelect == "extb") {
        ioctlParams.tableSelect = LWSWITCH_TABLE_SELECT_REMAP_EXTB;

    } else if (tableSelect == "multicast") {
        ioctlParams.tableSelect = LWSWITCH_TABLE_SELECT_REMAP_MULTICAST;

    } else {
        outStr << "Unknown Remap table " << tableSelect << std::endl;
        cmdResponse = outStr.str();
        return;
    }

    if (pSwitchIntf->doIoctl( &ioctlStruct ) != FM_INT_ST_OK) {
        outStr << "IOCTL_LWSWITCH_GET_REMAP_POLICY command failed " << std::endl;
        cmdResponse = outStr.str();
        return;
    }

    for (uint32_t i = 0; i < ioctlParams.numEntries; i++) {
        outStr << "\t index "      << ioctlParams.firstIndex + i
               << "\t entryValid " << (ioctlParams.entry[i].entryValid ? 1 : 0)
               << "\t targetId "   << ioctlParams.entry[i].targetId
               << "\t address "    << std::hex << ioctlParams.entry[i].address
               << "\t flags "      << std::hex << ioctlParams.entry[i].flags
               << "\t reqCtxChk "  << std::hex << ioctlParams.entry[i].reqCtxChk
               << "\t reqCtxRep "  << std::hex << ioctlParams.entry[i].reqCtxRep
               << "\t reqCtxMask " << std::hex << ioctlParams.entry[i].reqCtxMask
               << std::endl;
    }
    cmdResponse = outStr.str();
}

void
LocalFMCommandServer::dumpPortRidTable(std::vector<std::string> &cmdWords, std::string &cmdResponse)
{
    std::stringstream outStr;
    std::string tableSelect(cmdWords[3]);
    switchIoctl_t ioctlStruct;

    uint32_t switchIndex, portIndex, firstIndex, numEntries;
    parsePortRoutingTableQueryHelpCmd(cmdWords, switchIndex, portIndex, firstIndex, numEntries);

    LocalFMSwitchInterface *pSwitchIntf = mpLocalFM->switchInterfaceAtIndex(switchIndex);
    if (pSwitchIntf == NULL) {
        outStr << "Invalid LWSwitch at switchIndex " << switchIndex << std::endl;
        cmdResponse = outStr.str();
        return;
    }

    if (portIndex >= LWSWITCH_MAX_PORTS) {
        outStr << "Invalid portIndex " << portIndex << std::endl;
        cmdResponse = outStr.str();
        return;
    }

    if (numEntries > LWSWITCH_ROUTING_ID_ENTRIES_MAX) {
        outStr << "numEntries " << numEntries << " is more than max "
                << LWSWITCH_ROUTING_ID_ENTRIES_MAX << std::endl;
        cmdResponse = outStr.str();
        return;
    }

    if (tableSelect == "unicast") {

        LWSWITCH_GET_ROUTING_ID_PARAMS ioctlParams;
        memset(&ioctlParams, 0, sizeof(ioctlParams));
        memset(&ioctlStruct, 0, sizeof(ioctlStruct));

        ioctlStruct.type = IOCTL_LWSWITCH_GET_ROUTING_ID;
        ioctlStruct.ioctlParams = &ioctlParams;
        ioctlStruct.paramSize = sizeof(ioctlParams);
        ioctlParams.portNum = portIndex;
        ioctlParams.firstIndex = firstIndex;
        ioctlParams.numEntries = numEntries;

        if (pSwitchIntf->doIoctl( &ioctlStruct ) != FM_INT_ST_OK) {
            outStr << "IOCTL_LWSWITCH_GET_ROUTING_ID command failed " << std::endl;
            cmdResponse = outStr.str();
            return;
        }

        for (uint32_t i = 0; i < ioctlParams.numEntries; i++) {
            outStr << "\t index "         << ioctlParams.entries[i].idx
                   << "\t entryValid "    << (ioctlParams.entries[i].entry.entryValid ? 1 : 0)
                   << "\t useRoutingLan " << ioctlParams.entries[i].entry.useRoutingLan
                   << "\t numEntries "    << ioctlParams.entries[i].entry.numEntries
                   << "\t portList ";

            if (ioctlParams.entries[i].entry.numEntries == LWSWITCH_ROUTING_ID_DEST_PORT_LIST_MAX) {
                ioctlParams.entries[i].entry.numEntries = 0;
            }

            for (uint32_t j = 0; j < ioctlParams.entries[i].entry.numEntries; j++) {
                outStr << "(" << ioctlParams.entries[i].entry.portList[j].vcMap << ","
                       << ioctlParams.entries[i].entry.portList[j].destPortNum << ") ";
            }
            outStr << std::endl;
        }

    } else if (tableSelect == "multicast") {

        LWSWITCH_GET_MC_RID_TABLE_PARAMS ioctlParams;
        memset(&ioctlParams, 0, sizeof(ioctlParams));
        memset(&ioctlStruct, 0, sizeof(ioctlStruct));

        ioctlStruct.type = IOCTL_LWSWITCH_GET_MC_RID_TABLE;
        ioctlStruct.ioctlParams = &ioctlParams;
        ioctlStruct.paramSize = sizeof(ioctlParams);
        ioctlParams.portNum = portIndex;
        ioctlParams.index = firstIndex;

        if (numEntries == 0) {
            ioctlParams.extendedTable = false;
        } else {
            ioctlParams.extendedTable = true;
        };

        if (pSwitchIntf->doIoctl( &ioctlStruct ) != FM_INT_ST_OK) {
            outStr << "IOCTL_LWSWITCH_GET_MC_RID_TABLE command failed " << std::endl;
            cmdResponse = outStr.str();
            return;
        }

        outStr << "\t portNum "        << ioctlParams.portNum
               << "\t index "          << ioctlParams.index
               << "\t entryValid "     << (ioctlParams.entryValid ? 1 : 0)
               << "\t extendedTable "  << ioctlParams.extendedTable
               << "\t mcSize "         << ioctlParams.mcSize
               << "\t numSprayGroups " << ioctlParams.numSprayGroups
               << "\t extendedPtr "    << ioctlParams.extendedPtr
               << "\t extendedValid "  << ioctlParams.extendedValid
               << "\t noDynRsp "       << ioctlParams.noDynRsp
               << std::endl;

        uint32_t portIndex = 0;
        for (uint32_t i = 0; i < ioctlParams.numSprayGroups; i++) {
            outStr << "\t sprayGroup " << i << std::endl
                   << "\t\t portsPerSprayGroup " << ioctlParams.portsPerSprayGroup[i]
                   << "\t\t replicaValid "       << ioctlParams.replicaValid[i]
                   << "\t\t replicaOffset "      << ioctlParams.replicaOffset[i]
                   << std::endl
                   << "\t\t (port,vcHop) ";


            for (uint32_t j = 0; j < ioctlParams.portsPerSprayGroup[i]; j++ ) {
                if (portIndex >= LWSWITCH_MC_MAX_PORTS) {
                    outStr << "Invalid portIndex " << portIndex << std::endl;
                    continue;
                }

                outStr << "(" << ioctlParams.ports[portIndex] << ","
                       << ioctlParams.vcHop[portIndex] << ")" << " ";
                portIndex++;
            }
        }
    } else {
        outStr << "Unknown RID table " << tableSelect << std::endl;
    }

    cmdResponse = outStr.str();
}

void
LocalFMCommandServer::dumpPortRlanTable(std::vector<std::string> &cmdWords, std::string &cmdResponse)
{
    std::stringstream outStr;
    switchIoctl_t ioctlStruct;

    uint32_t switchIndex, portIndex, firstIndex, numEntries;
    parsePortRoutingTableQueryHelpCmd(cmdWords, switchIndex, portIndex, firstIndex, numEntries);

    LocalFMSwitchInterface *pSwitchIntf = mpLocalFM->switchInterfaceAtIndex(switchIndex);
    if (pSwitchIntf == NULL) {
        outStr << "Invalid LWSwitch at switchIndex " << switchIndex << std::endl;
        cmdResponse = outStr.str();
        return;
    }

    if (portIndex >= LWSWITCH_MAX_PORTS) {
        outStr << "Invalid portIndex " << portIndex << std::endl;
        cmdResponse = outStr.str();
        return;
    }

    if (numEntries > LWSWITCH_ROUTING_LAN_ENTRIES_MAX) {
        outStr << "numEntries " << numEntries << " is more than max "
                << LWSWITCH_ROUTING_LAN_ENTRIES_MAX << std::endl;
        cmdResponse = outStr.str();
        return;
    }

    LWSWITCH_GET_ROUTING_LAN_PARAMS ioctlParams;
    memset(&ioctlParams, 0, sizeof(ioctlParams));
    memset(&ioctlStruct, 0, sizeof(ioctlStruct));

    ioctlStruct.type = IOCTL_LWSWITCH_GET_ROUTING_LAN;
    ioctlStruct.ioctlParams = &ioctlParams;
    ioctlStruct.paramSize = sizeof(ioctlParams);
    ioctlParams.portNum = portIndex;
    ioctlParams.firstIndex = firstIndex;
    ioctlParams.numEntries = numEntries;

    if (pSwitchIntf->doIoctl( &ioctlStruct ) != FM_INT_ST_OK) {
        outStr << "IOCTL_LWSWITCH_GET_ROUTING_LAN command failed " << std::endl;
        cmdResponse = outStr.str();
        return;
    }

    for (uint32_t i = 0; i < ioctlParams.numEntries; i++) {
        outStr << "\t index "         << std::dec << ioctlParams.entries[i].idx
               << "\t entryValid "    << std::dec << (ioctlParams.entries[i].entry.entryValid ? 1 : 0)
               << "\t numEntries "    << std::dec << ioctlParams.entries[i].entry.numEntries
               << "\t routingLan ";

        if (ioctlParams.entries[i].entry.numEntries == LWSWITCH_ROUTING_LAN_GROUP_SEL_MAX) {
            ioctlParams.entries[i].entry.numEntries = 0;
        }

        for (uint32_t j = 0; j < ioctlParams.entries[i].entry.numEntries; j++) {
            outStr <<  "(" << ioctlParams.entries[i].entry.portList[j].groupSelect << ","
                   << ioctlParams.entries[i].entry.portList[j].groupSize << ") ";
        }
        outStr << std::endl;
    }

    cmdResponse = outStr.str();
}

void
LocalFMCommandServer::dumpPortCounters(std::vector<std::string> &cmdWords, std::string &cmdResponse)
{
    std::stringstream outStr;
    switchIoctl_t ioctlStruct;

    uint32_t switchIndex = stoi(cmdWords[3], nullptr);
    uint32_t portIndex   = stoi(cmdWords[4], nullptr);

    LocalFMSwitchInterface *pSwitchIntf = mpLocalFM->switchInterfaceAtIndex(switchIndex);
    if (pSwitchIntf == NULL) {
        outStr << "Invalid LWSwitch at switchIndex " << switchIndex << std::endl;
        cmdResponse = outStr.str();
        return;
    }

    if (portIndex >= LWSWITCH_MAX_PORTS) {
        outStr << "Invalid portIndex " << portIndex << std::endl;
        cmdResponse = outStr.str();
        return;
    }

    LWSWITCH_LWLINK_GET_COUNTERS_PARAMS ioctlParams;
    memset(&ioctlParams, 0, sizeof(ioctlParams));
    memset(&ioctlStruct, 0, sizeof(ioctlStruct));

    ioctlStruct.type = IOCTL_LWSWITCH_GET_LWLINK_COUNTERS;
    ioctlStruct.ioctlParams = &ioctlParams;
    ioctlStruct.paramSize = sizeof(ioctlParams);

    ioctlParams.linkId = portIndex;
    ioctlParams.counterMask = 0xFFFF;

    if (pSwitchIntf->doIoctl( &ioctlStruct ) != FM_INT_ST_OK) {
        outStr << "IOCTL_LWSWITCH_GET_LWLINK_COUNTERS command failed " << std::endl;
        cmdResponse = outStr.str();
        return;
    }

    outStr << "\t txCounter0: " << ioctlParams.lwlinkCounters[1]
           << "\t txCounter1: " << ioctlParams.lwlinkCounters[2]
           << "\t rxCounter0: " << ioctlParams.lwlinkCounters[3]
           << "\t rxCounter1: " << ioctlParams.lwlinkCounters[4]
           << std::endl;

    for (int i = 5; i < LWSWITCH_LWLINK_COUNTER_MAX_TYPES; i++) {
        if (ioctlParams.lwlinkCounters[i] != 0) {
            outStr  << "\t counter : " << i << ": " << ioctlParams.lwlinkCounters[i];
        }
    }
    cmdResponse = outStr.str();
}

void
LocalFMCommandServer::handlePortQueryCmd(std::vector<std::string> &cmdWords, std::string &cmdResponse)
{
    std::stringstream outStr;

    if (cmdWords.size() < 5) {
        portQueryHelpCmd(cmdResponse);
        return;
    }

    // 3rd word is the command word
    std::string queryCmd(cmdWords[2]);

    if (queryCmd == "remap") {
        if (cmdWords.size() != 8) {
            portQueryHelpCmd(cmdResponse);
            return;
        }
        dumpPortRemapTable(cmdWords, cmdResponse);

    } else if (queryCmd == "rid") {
        if (cmdWords.size() != 8) {
            portQueryHelpCmd(cmdResponse);
            return;
        }
        dumpPortRidTable(cmdWords, cmdResponse);

    } else if (queryCmd == "rlan") {
        if (cmdWords.size() != 8) {
            portQueryHelpCmd(cmdResponse);
            return;
        }
        dumpPortRlanTable(cmdWords, cmdResponse);

    } else if (queryCmd == "counter") {
        if (cmdWords.size() != 5) {
            portQueryHelpCmd(cmdResponse);
            return;
        }
        dumpPortCounters(cmdWords, cmdResponse);

    } else {
        // unknown command
        portQueryHelpCmd(cmdResponse);
        return;
    }
}

