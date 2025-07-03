/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2021 by LWPU Corporation. All rights
 * reserved.  All information contained herein is proprietary and confidential
 * to LWPU Corporation.  Any use, reproduction, or disclosure without the
 * written permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#include "global_fm_mgr.h"
#include "gdm_logger.h"
#include "gdm_server.h"
#include "gdm_configurator.h"
#include "protobuf/pbwriter.h"
#include "message_handler.h"
#include "message_reader.h"
#include "message_writer.h"
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include "GlobalFabricManager.h"

namespace
{
    static unique_ptr<GlobalFabricManager>  m_pGFM = NULL;
    static bool m_bInitializedGFM = false;
}

LwDiagUtils::EC GlobalFmManager::InitGFM()
{
    GdmLogger::Printf(LwDiagUtils::PriNormal, "Attempting to start GFM\n");

    if (m_bInitializedGFM)
    {
        GdmLogger::Printf(LwDiagUtils::PriNormal, "GFM already Initialized \n");
        return LwDiagUtils::OK;
    }

    GlobalFmArgs_t gfm = {};
    GdmConfig::GdmGetGFMConfig(&gfm);

    try
    {
       m_pGFM.reset(new GlobalFabricManager(&gfm));
    }

    catch (const exception& e)
    {
       GdmLogger::Printf(LwDiagUtils::PriError,
                          "Exception in Local Fabric Manager during initialization! : \"%s\"\n",
                           e.what());
       LWDASSERT(!"GFM Initialisation failed");
       return LwDiagUtils::SOFTWARE_ERROR;
    }

    m_bInitializedGFM = true;
    return LwDiagUtils::OK;
}

LwDiagUtils::EC GlobalFmManager::ShutDownGFM()
{
    if (m_bInitializedGFM)
    {
        GdmLogger::Printf(LwDiagUtils::PriNormal, "Shutting Down GFM\n");
        m_pGFM.reset();
    }

    return LwDiagUtils::OK;
}

LwDiagUtils::EC GlobalFmManager::GetNumGpus(void *pvConnection)
{
    // Verified
    InitGFM();
    ByteStream bs;
    auto msg = MessageWriter::Messages::ret_num_gpu(&bs);
    {
        msg
        .header()
        .node_id(0U)
        .app_type(MessageWriter::MessageHeader::AppType::gdm);
    }
    msg.num_gpus(m_pGFM->mpParser->gpuCfg.size());
    msg.Finish();
    GdmServer::SendMessage(bs, pvConnection);
    return LwDiagUtils::OK;
}

LwDiagUtils::EC GlobalFmManager::GetNumLwSwitch(void *pvConnection)
{
    // Verified
    InitGFM();
    ByteStream bs;
    auto msg = MessageWriter::Messages::ret_num_lw_switch(&bs);
    {
        msg
        .header()
        .node_id(0U)
        .app_type(MessageWriter::MessageHeader::AppType::gdm);
    }
    msg.num_lw_switch(m_pGFM->mpParser->lwswitchCfg.size());
    msg.Finish();
    GdmServer::SendMessage(bs, pvConnection);
    return LwDiagUtils::OK;
}

LwDiagUtils::EC GlobalFmManager::GetGpuMaxLwLinks(UINT32 physicalId, void *pvConnection)
{
    // Verified
    InitGFM();
    UINT32 maxLink = 0;
    const TopologyLWLinkConnList& connList = m_pGFM->mpParser->lwLinkConnMap[0];
    for (const TopologyLWLinkConn& conn : connList)
    {
        if (conn.connType != ACCESS_PORT_GPU)
            continue;

        if (conn.farEnd.nodeId == 0 &&
            conn.farEnd.lwswitchOrGpuId == physicalId)
        {
            maxLink = max(maxLink, conn.farEnd.portIndex + 1);
        }
    }

    // Send the max Link as a message to the waiting node
    ByteStream bs;
    auto msg = MessageWriter::Messages::ret_gfm_max_lwlinks(&bs);
    {
        msg
        .header()
        .node_id(0U)
        .app_type(MessageWriter::MessageHeader::AppType::gdm);
    }
    msg.physical_id(physicalId);
    msg.max_lw_links(maxLink);
    msg.Finish();
    GdmServer::SendMessage(bs, pvConnection);
    return LwDiagUtils::OK;
}

LwDiagUtils::EC GlobalFmManager::GetGpuPhysicalId(UINT32 index, void *pvConnection)
{
    // Verified
    InitGFM();
    if (index >= m_pGFM->mpParser->gpuCfg.size())
    {
        GdmLogger::Printf(LwDiagUtils::PriError, "Invalid FM GPU Index = %d\n", index);
        return LwDiagUtils::SOFTWARE_ERROR;
    }

    ByteStream bs;
    auto msg = MessageWriter::Messages::ret_gfm_physical_id(&bs);
    {
        msg
        .header()
        .node_id(0U)
        .app_type(MessageWriter::MessageHeader::AppType::gdm);
    }
    msg.index(index);
    msg.gpu(true);

    UINT32 gpuIdx = 0;
    for (const auto& gpuPair : m_pGFM->mpParser->gpuCfg)
    {
        if (gpuIdx == index)
        {
            msg.physical_id(gpuPair.second->gpuphysicalid());
            msg.Finish();
            GdmServer::SendMessage(bs, pvConnection);
            return LwDiagUtils::OK;
        }
        gpuIdx++;
    }

    return LwDiagUtils::SOFTWARE_ERROR;
}

LwDiagUtils::EC GlobalFmManager::GetLwSwitchPhysicalId(UINT32 index, void *pvConnection)
{
    // Verified
    InitGFM();
    if (index >=m_pGFM->mpParser->lwswitchCfg.size())
    {
        GdmLogger::Printf(LwDiagUtils::PriError, "Invalid FM LwSwitch Index = %d\n", index);
        return LwDiagUtils::SOFTWARE_ERROR;
    }

    ByteStream bs;
    auto msg = MessageWriter::Messages::ret_gfm_physical_id(&bs);
    {
        msg
        .header()
        .node_id(0U)
        .app_type(MessageWriter::MessageHeader::AppType::gdm);
    }
    msg.index(index);
    msg.gpu(false);

    UINT32 switchIdx = 0;
    for (const auto& gpuPair : m_pGFM->mpParser->lwswitchCfg)
    {
        if (switchIdx == index)
        {
            msg.physical_id(gpuPair.second->switchphysicalid());
            msg.Finish();
            GdmServer::SendMessage(bs, pvConnection);
            return LwDiagUtils::OK;
        }
        switchIdx++;
    }

    return LwDiagUtils::SOFTWARE_ERROR;
}

LwDiagUtils::EC GlobalFmManager::GetGpuEnumIndex
(
    UINT32 nodeID,
    UINT32 physicalId,
    void *pvConnection
)
{
    bool found = true;
    uint32_t enumIndex = 0;
    if (!m_pGFM->getGpuEnumIndex(nodeID, physicalId, enumIndex))
        found = false;

    ByteStream bs;
    auto msg = MessageWriter::Messages::ret_gfm_gpu_enum_idx(&bs);
    {
        msg
        .header()
        .node_id(0U)
        .app_type(MessageWriter::MessageHeader::AppType::gdm);
    }
    msg.node_id(nodeID);
    msg.physical_id(physicalId);
    msg.enum_idx(enumIndex);
    msg.found(found);
    msg.gpu(true);
    msg.Finish();
    GdmServer::SendMessage(bs, pvConnection);

    return LwDiagUtils::OK;
}

LwDiagUtils::EC GlobalFmManager::GetLwSwitchEnumIndex
(
    UINT32 nodeID,
    UINT32 physicalId,
    void *pvConnection
)
{
   // TODO - Check if needed
   return LwDiagUtils::OK;
}

LwDiagUtils::EC GlobalFmManager::GetGpuPciBdf
(
    UINT32 nodeID,
    UINT32 enumIndex,
    void *pvConnection
)
{
    FMPciInfo_t pciInfo;
    bool found = true;
    if (!m_pGFM->getGpuPciBdf(nodeID, enumIndex, pciInfo))
        found = false;

    ByteStream bs;
    auto msg = MessageWriter::Messages::ret_gfm_pci_bdf(&bs);
    {
        msg
        .header()
        .node_id(0U)
        .app_type(MessageWriter::MessageHeader::AppType::gdm);
    }
    {
        msg
        .pci_info()
        .domain(pciInfo.domain)
        .bus(pciInfo.bus)
        .device(pciInfo.device)
        .function(pciInfo.function);
    }
    msg.node_id(nodeID);
    msg.enum_idx(enumIndex);
    msg.found(found);
    msg.gpu(true);
    msg.Finish();
    GdmServer::SendMessage(bs, pvConnection);

    return LwDiagUtils::OK;
}

LwDiagUtils::EC GlobalFmManager::GetLwSwitchPciBdf
(
    UINT32 nodeID,
    UINT32 physicalId,
    void *pvConnection
)
{
    FMPciInfo_t pciInfo;
    bool found = true;
    if (!m_pGFM->getLWSwitchPciBdf(nodeID, physicalId, pciInfo))
        found = false;

    ByteStream bs;
    auto msg = MessageWriter::Messages::ret_gfm_pci_bdf(&bs);
    {
        msg
        .header()
        .node_id(0U)
        .app_type(MessageWriter::MessageHeader::AppType::gdm);
    }
    {
        msg
        .pci_info()
        .domain(pciInfo.domain)
        .bus(pciInfo.bus)
        .device(pciInfo.device)
        .function(pciInfo.function);
    }
    msg.node_id(nodeID);
    msg.enum_idx(physicalId);
    msg.found(found);
    msg.gpu(false);
    msg.Finish();
    GdmServer::SendMessage(bs, pvConnection);

    return LwDiagUtils::OK;
}
