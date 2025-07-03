/*
 *  Copyright 2018-2021 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */
#include <sstream>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <iterator>
#include <string>

#include "fm_log.h"
#include "FMErrorCodesInternal.h"
#include "FMCommonTypes.h"
#include "GlobalFmCommandServer.h"
#include "GlobalFabricManager.h"
#include "GlobalFmMulticastMgr.h"

GlobalFMCommandServer::GlobalFMCommandServer(GlobalFabricManager *pGlobalFM)
{
    mpGlobalFM = pGlobalFM;
    mpCmdServer = new FmCommandServer(this, GLOBAL_FM_CMD_SERVER_PORT, (char*)FM_DEFAULT_BIND_INTERFACE);
    mpCmdServer->Start();
}

GlobalFMCommandServer::~GlobalFMCommandServer()
{
    delete mpCmdServer;
    mpCmdServer = NULL;
}

void
GlobalFMCommandServer::handleRunCmd(std::string &cmdLine, std::string &cmdResponse)
{
    cmdResponse = "Unknown Run Command\n";

    // first split the command string into words
    std::istringstream strCmd(cmdLine);
    std::vector<std::string> cmdWords((std::istream_iterator<std::string>(strCmd)),
                                      std::istream_iterator<std::string>());

    // run commands are of the following form. so we expect at least two words
    // /run multicast
    if (cmdWords.size() < 2) {
        cmdResponse = "Invalid number of arguments\n";
        return;
    }

    // second word is the command word
    std::string runCmd(cmdWords[1]);

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    if (runCmd == "multicast") {
        handleMulticastRunCmd(cmdWords, cmdResponse);
    }
#endif
}

void
GlobalFMCommandServer::handleQueryCmd(std::string &cmdLine, std::string &cmdResponse)
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

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    if (queryCmd == "multicast") {
        handleMulticastQueryCmd(cmdWords, cmdResponse);
    }
#endif
}

void
GlobalFMCommandServer::dumpAllGpuInfo(std::string &cmdResponse)
{
    FMGpuInfoMap::iterator it;
    FMGpuInfoMap gpuInfoMap = mpGlobalFM->getGpuInfoMap();
    std::stringstream outStr;

    for ( it = gpuInfoMap.begin(); it != gpuInfoMap.end(); it++ ) {
        outStr << "\t Dumping GPU information for Node Index " << int(it->first) << std::endl;
        FMGpuInfoList gpuList = it->second;
        FMGpuInfoList::iterator jit;
        for ( jit = gpuList.begin(); jit != gpuList.end(); jit++ ) {
            FMGpuInfo_t gpuInfo = (*jit);
            outStr << "\t gpuIndex: " << int(gpuInfo.gpuIndex) << std::endl;
            outStr << "\t PCI Info:" << std::endl;
            outStr << "\t\t Domain:" << (int)gpuInfo.pciInfo.domain << std::endl;
            outStr << "\t\t Bus:" << (int)gpuInfo.pciInfo.bus << std::endl;
            outStr << "\t\t Device:" << (int)gpuInfo.pciInfo.device << std::endl;
            outStr << "\t\t Function:" << (int)gpuInfo.pciInfo.function << std::endl;
        }
    }

    FMExcludedGpuInfoMap excludedGpuInfoMap = mpGlobalFM->getExcludedGpuInfoMap();
    FMExcludedGpuInfoMap::iterator blit;
    for ( blit = excludedGpuInfoMap.begin(); blit != excludedGpuInfoMap.end(); blit++ ) {
        outStr << "\t Dumping excluded GPU information for Node Index " << int(blit->first) << std::endl;
        FMExcludedGpuInfoList gpuList = blit->second;
        FMExcludedGpuInfoList::iterator jit;
        for ( jit = gpuList.begin(); jit != gpuList.end(); jit++ ) {
            FMExcludedGpuInfo_t gpuInfo = (*jit);
            //outStr << "\t gpuIndex: " << int(gpuInfo.gpuIndex) << std::endl;
            outStr << "\t PCI Info:" << std::endl;
            outStr << "\t\t Domain:" << std::hex << (int)gpuInfo.pciInfo.domain << std::endl;
            outStr << "\t\t Bus:" << std::hex << (int)gpuInfo.pciInfo.bus << std::endl;
            outStr << "\t\t Device:" << std::hex << (int)gpuInfo.pciInfo.device << std::endl;
            outStr << "\t\t Function:" << std::hex << (int)gpuInfo.pciInfo.function << std::endl;
            outStr << "\t\t uuid:" << std::hex << gpuInfo.uuid.bytes << std::endl;
        }
    }

    cmdResponse = outStr.str();
    if (cmdResponse.size() == 0) {
        cmdResponse = "No GPU Information is available\n";
    }
}

void
GlobalFMCommandServer::dumpAllLWSwitchInfo(std::string &cmdResponse)
{
    FMLWSwitchInfoMap::iterator it;
    FMLWSwitchInfoMap switchInfoMap = mpGlobalFM->getLwSwitchInfoMap();
    std::stringstream outStr;

    for ( it = switchInfoMap.begin(); it != switchInfoMap.end(); it++ ) {
        outStr << "\t Dumping LWSwitch information for Node Index " << int(it->first) << std::endl;
        FMLWSwitchInfoList switchList = it->second;
        FMLWSwitchInfoList::iterator jit;
        for ( jit = switchList.begin(); jit != switchList.end(); jit++ ) {
            FMLWSwitchInfo switchInfo = (*jit);
            outStr << "\t switchIndex: " << int(switchInfo.switchIndex) << std::endl;
            outStr << "\t physicalId: " << std::hex << int(switchInfo.physicalId) << std::endl;
            outStr << "\t Arch Type: " << int(switchInfo.archType) << std::endl;
            outStr << "\t uuid:" << std::hex << switchInfo.uuid.bytes << std::endl;
            outStr << "\t PCI Info:" << std::endl;
            outStr << "\t\t Domain:" << std::hex << (int)switchInfo.pciInfo.domain << std::endl;
            outStr << "\t\t Bus:" << std::hex << (int)switchInfo.pciInfo.bus << std::endl;
            outStr << "\t\t Device:" << std::hex << (int)switchInfo.pciInfo.device << std::endl;
            outStr << "\t\t Function:" << std::hex << (int)switchInfo.pciInfo.function << std::endl;
        }
    }

    FMExcludedLWSwitchInfoMap excludedLwswitchInfoMap = mpGlobalFM->getExcludedLwswitchInfoMap();
    FMExcludedLWSwitchInfoMap::iterator blit;
    for ( blit = excludedLwswitchInfoMap.begin(); blit != excludedLwswitchInfoMap.end(); blit++ ) {
        outStr << "\t Dumping excluded LWSwitch information for Node Index " << int(blit->first) << std::endl;
        FMExcludedLWSwitchInfoList switchList = blit->second;
        FMExcludedLWSwitchInfoList::iterator jit;
        for ( jit = switchList.begin(); jit != switchList.end(); jit++ ) {
            FMExcludedLWSwitchInfo_t switchInfo = (*jit);
            outStr << "\t physicalId: " << std::hex << int(switchInfo.physicalId) << std::endl;
            outStr << "\t uuid:" << std::hex << switchInfo.uuid.bytes << std::endl;
            outStr << "\t PCI Info:" << std::endl;
            outStr << "\t\t Domain:" << std::hex << (int)switchInfo.pciInfo.domain << std::endl;
            outStr << "\t\t Bus:" << std::hex << (int)switchInfo.pciInfo.bus << std::endl;
            outStr << "\t\t Device:" << std::hex << (int)switchInfo.pciInfo.device << std::endl;
            outStr << "\t\t Function:" << std::hex << (int)switchInfo.pciInfo.function << std::endl;
        }
    }

    cmdResponse = outStr.str();
    if (cmdResponse.size() == 0) {
        cmdResponse = "No LWSwitch Information is available\n";
    }
}

void
GlobalFMCommandServer::dumpAllLWLinkConnInfo(std::string &cmdResponse)
{
    std::stringstream outStr;

    outStr << "Dumping all Intra-Node connections" << std::endl;
    LWLinkIntraConnMap::iterator it;
    LWLinkIntraConnMap intraConnMap = mpGlobalFM->mLWLinkConnRepo.getIntraConnections();
    for ( it = intraConnMap.begin(); it != intraConnMap.end(); it++ ) {
        FMLWLinkDetailedConnInfoList connList = it->second;
        FMLWLinkDetailedConnInfoList::iterator jit;
        outStr << "Intra-Node connections for Node Index:" << it->first << std::endl;
        outStr << "Number of connections:" << connList.size() << std::endl;
        // dump each connection information
        for (jit = connList.begin(); jit != connList.end(); jit++ ) {
            FMLWLinkDetailedConnInfo *connInfo = (*jit);
            connInfo->dumpConnAndStateInfo(&outStr, mpGlobalFM, mpGlobalFM->mLWLinkDevRepo);
        }
    }

    outStr << "Dumping all Inter-Node connections" << std::endl;
    LWLinkInterNodeConns::iterator jit;
    LWLinkInterNodeConns interConnMap = mpGlobalFM->mLWLinkConnRepo.getInterConnections();
    for (jit = interConnMap.begin(); jit != interConnMap.end(); jit++ ) {
        FMLWLinkDetailedConnInfo *connInfo = (*jit);
        connInfo->dumpConnAndStateInfo( &outStr, mpGlobalFM, mpGlobalFM->mLWLinkDevRepo);
    }

    cmdResponse = outStr.str();
}

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

void
GlobalFMCommandServer::allocateMulticastGroup(uint32_t partitionId, std::string &cmdResponse)
{
    std::stringstream outStr;
    uint32_t groupId;
    FMIntReturn_t rc = mpGlobalFM->mpMcastMgr->allocateMulticastGroup(partitionId, groupId);

    if (rc == FM_INT_ST_OK) {
        outStr << "Allocated group " << groupId << " on partition " << partitionId << std::endl;
    } else {
        outStr << "Failed to allocate group on partition " << partitionId
               << " with error " << rc << std::endl;
    }

    cmdResponse = outStr.str();
}

void
GlobalFMCommandServer::freeMulticastGroup(uint32_t partitionId, uint32_t groupId, std::string &cmdResponse)
{
    std::stringstream outStr;

    // clear MC FLA from RM
    FMIntReturn_t rc = mpGlobalFM->mpMcastMgr->sendGroupReleaseCompleteReqMsg(0, 0,
            mpGlobalFM->mpMcastMgr->getMulticastGroup(partitionId, groupId), lwswitch::CONFIG_SUCCESS);

    if (rc == FM_INT_ST_OK) {
        outStr << "Failed to send release complete request on partition " << partitionId
               << " groupId " << groupId << " with error " << rc << std::endl;
        cmdResponse = outStr.str();
        return;
    }

    rc = mpGlobalFM->mpMcastMgr->freeMulticastGroup(partitionId, groupId);

    if (rc == FM_INT_ST_OK) {
        outStr << "Freed group " << groupId << " on partition " << partitionId << std::endl;
    } else {
        outStr << "Failed to free group on partition " << partitionId
               << "with error " << rc << std::endl;
    }

    cmdResponse = outStr.str();
}

void
GlobalFMCommandServer::freeMulticastGroups(uint32_t partitionId, std::string &cmdResponse)
{
    std::stringstream outStr;
    mpGlobalFM->mpMcastMgr->freeMulticastGroups(partitionId);

    outStr << "Freed all groups on partition " << partitionId << std::endl;
    cmdResponse = outStr.str();
}

void
GlobalFMCommandServer::getAvailableMulticastGroups(uint32_t partitionId, std::string &cmdResponse)
{
    std::stringstream outStr;
    std::list<uint32_t> groupIds;
    mpGlobalFM->mpMcastMgr->getAvailableMulticastGroups(partitionId, groupIds);

    outStr << "Available groups on partition " << partitionId << std::endl;

    std::list<uint32_t>::iterator it;
    for (it = groupIds.begin(); it != groupIds.end(); it++) {
        uint32_t groupId = *it;
        outStr << " " << groupId;
    }

    outStr << std::endl;
    cmdResponse = outStr.str();
}

void
GlobalFMCommandServer::setMulticastGroup(uint32_t partitionId, uint32_t groupId, bool reflectiveMode,
                                         bool excludeSelf, std::set<GpuKeyType> &gpus, GpuKeyType primaryReplica,
                                         std::string &cmdResponse)
{
    std::stringstream outStr;
    FMIntReturn_t rc = mpGlobalFM->mpMcastMgr->setMulticastGroup(partitionId, groupId, reflectiveMode,
                                                       excludeSelf, gpus, primaryReplica);

    if (rc == FM_INT_ST_OK) {
        outStr << "Set group " << groupId << " on partition " << partitionId << std::endl;

        // send MC FLA to RM
        mpGlobalFM->mpMcastMgr->sendGroupSetupCompleteReqMsg(0, 0,
                mpGlobalFM->mpMcastMgr->getMulticastGroup(partitionId, groupId), lwswitch::CONFIG_SUCCESS);

        if (rc == FM_INT_ST_OK) {
            outStr << "Failed to send setup complete request on partition " << partitionId
                   << " groupId " << groupId << " with error " << rc << std::endl;
            cmdResponse = outStr.str();
            return;
        }

    } else {
        outStr << "Failed to set group on partition " << partitionId
               << "with error " << rc << std::endl;
    }
    cmdResponse = outStr.str();
}

void
GlobalFMCommandServer::getMulticastGroupBaseAddress(uint32_t groupId, std::string &cmdResponse)
{
    std::stringstream outStr;
    uint64_t multicastAddrBase, multicastAddrRange;
    FMIntReturn_t rc = mpGlobalFM->mpMcastMgr->getMulticastGroupBaseAddrAndRange(groupId,
            multicastAddrBase, multicastAddrRange);

    if (rc == FM_INT_ST_OK) {
        outStr << "Group " << groupId << " address base is " << multicastAddrBase
               << std::hex << multicastAddrBase << " address range is "
               << multicastAddrRange << std::endl;
    } else {
        outStr << "Failed to get group " << "groupId" << " address base "
               << "with error " << rc << std::endl;
    }

    cmdResponse = outStr.str();
}

void
GlobalFMCommandServer::dumpMulticastGroup( uint32_t partitionId, uint32_t groupId, std::string &cmdResponse )
{
    std::stringstream outStr;
    mpGlobalFM->mpMcastMgr->dumpMulticastGroup( partitionId, groupId, outStr );
    cmdResponse = outStr.str();
}

void
GlobalFMCommandServer::dumpAllMulticastGroup( uint32_t partitionId, std::string &cmdResponse )
{
    std::stringstream outStr;

    for (uint32_t groupId = 0; groupId < mpGlobalFM->mpMcastMgr->getMaxNumMulitcastGroups(); groupId++ ) {
        mpGlobalFM->mpMcastMgr->dumpMulticastGroup( partitionId, groupId, outStr );
    }
    cmdResponse = outStr.str();
}

void
GlobalFMCommandServer::multicastRunHelpCmd(std::string &cmdResponse)
{
    std::stringstream outStr;

    outStr << "/run multicast allocate partitionId" << std::endl;
    outStr << "/run multicast free     partitionId groupId" << std::endl;
    outStr << "/run multicast freeAll  partitionId" << std::endl;
    outStr << "/run multicast set      partitionId groupId relectiveMode "
           << "excludeSelf gpuPhysicalId1 gpuPhysicalId2 ... gpuPhysicalIdn" << std::endl;

    cmdResponse = outStr.str();
}

void
GlobalFMCommandServer::multicastQueryHelpCmd(std::string &cmdResponse)
{
    std::stringstream outStr;

    outStr << "/query multicast address         groupId" << std::endl;
    outStr << "/query multicast group           partitionId groupId" << std::endl;
    outStr << "/query multicast allGroups       partitionId" << std::endl;
    outStr << "/query multicast availableGroups partitionId" << std::endl;

    cmdResponse = outStr.str();
}

//
// Multicast debug run commands
//
// /run multicast allocate partitionId
//                free     partitionId groupId
//                freeAll  partitionId
//                set      partitionId groupId relectiveMode excludeSelf gpuPhysicalId1 gpuPhysicalId2 ... gpuPhysicalIdn
//
//  note: the last gpuPhysicalId gpuPhysicalIdn is the primaryReplica

void
GlobalFMCommandServer::handleMulticastRunCmd(std::vector<std::string> &cmdWords, std::string &cmdResponse)
{
    if (cmdWords.size() < 4) {
        multicastRunHelpCmd(cmdResponse);
        return;
    }

    // 3rd word is the command word
    std::string runCmd(cmdWords[2]);

    // partitionId is the 4th word
    uint64_t tmp = std::stoull(cmdWords[3]);
    uint32_t partitionId;
    if ( tmp >= FM_MAX_FABRIC_PARTITIONS ) {
        partitionId = ILWALID_FABRIC_PARTITION_ID;
    } else {
        partitionId = tmp;
    }

    uint32 groupId;

    if (runCmd == "allocate") {
        allocateMulticastGroup(partitionId, cmdResponse);

    } else if (runCmd == "free") {
        if (cmdWords.size() < 5) {
            multicastRunHelpCmd(cmdResponse);
            return;
        }

        groupId = stoi(cmdWords[4], nullptr);
        freeMulticastGroup(partitionId, groupId, cmdResponse);

    } else if (runCmd == "freeAll") {
        freeMulticastGroups(partitionId, cmdResponse);

    } else if (runCmd == "set") {
        if (cmdWords.size() < 9) {
            multicastRunHelpCmd(cmdResponse);
            return;
        }

        groupId = stoi(cmdWords[4], nullptr);
        bool reflectiveMode = (stoi(cmdWords[5], nullptr) == 0) ? false : true;
        bool excludeSelf = (stoi(cmdWords[6], nullptr) == 0) ? false : true;

        uint32_t i;
        GpuKeyType gpu;
        uint32_t nodeId = 0;
        gpu.nodeId = nodeId;

        std::set<GpuKeyType> gpus;
        GpuKeyType primaryReplica;
        primaryReplica.nodeId = nodeId;

        // GPUs in the group
        for (i = 7; i < (cmdWords.size() - 1); i++) {
            gpu.physicalId = stoi(cmdWords[i], nullptr);
            gpus.insert(gpu);
        }

        // the last argument is the primaryReplica
        primaryReplica.physicalId = stoi(cmdWords[i], nullptr);

        setMulticastGroup(partitionId, groupId, reflectiveMode,
                          excludeSelf, gpus, primaryReplica, cmdResponse);
    } else {
        // unknown command
        multicastRunHelpCmd(cmdResponse);
        return;
    }
}

//
// Multicast debug query commands
//
// /query multicast address groupId
//                  group partitionId groupId
//                  allGroups partitionId
//                  availableGroups partitionId
//
void
GlobalFMCommandServer::handleMulticastQueryCmd(std::vector<std::string> &cmdWords, std::string &cmdResponse)
{
    std::stringstream outStr;

    // partitionId is the 4th word
    if (cmdWords.size() < 4) {
        multicastQueryHelpCmd(cmdResponse);
        return;
    }

    // 3rd word is the command word
    std::string queryCmd(cmdWords[2]);

    uint32 groupId;
    uint64_t tmp = std::stoull(cmdWords[3]);
    uint32_t partitionId;
    if ( tmp >= FM_MAX_FABRIC_PARTITIONS ) {
        partitionId = ILWALID_FABRIC_PARTITION_ID;
    } else {
        partitionId = tmp;
    }

    if (queryCmd == "address") {
        if (cmdWords.size() != 4) {
            multicastQueryHelpCmd(cmdResponse);
            return;
        }

        groupId = stoi(cmdWords[3], nullptr);
        getMulticastGroupBaseAddress(groupId, cmdResponse);

    } else if (queryCmd == "group") {
        if (cmdWords.size() != 5) {
            multicastQueryHelpCmd(cmdResponse);
            return;
        }

        groupId = stoi(cmdWords[4], nullptr);
        dumpMulticastGroup(partitionId, groupId, cmdResponse);

    } else if (queryCmd == "allGroups") {
        if (cmdWords.size() != 4) {
            multicastQueryHelpCmd(cmdResponse);
            return;
        }

        dumpAllMulticastGroup(partitionId, cmdResponse);

    } else if (queryCmd == "availableGroups") {
        if (cmdWords.size() != 4) {
            multicastQueryHelpCmd(cmdResponse);
            return;
        }

        getAvailableMulticastGroups(partitionId, cmdResponse);

    } else {
        // unknown command
        multicastQueryHelpCmd(cmdResponse);
        return;
    }
}
#endif
