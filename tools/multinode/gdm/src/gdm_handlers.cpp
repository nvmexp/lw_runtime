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

#include "lwdiagutils.h"
#include "connection.h"
#include "protobuf/pbwriter.h"
#include "message_handler.h"
#include "message_reader.h"
#include "message_writer.h"
#include "heart_beat_monitor.h"
#include "gdm_logger.h"
#include "gdm_server.h"
#include "global_fm_mgr.h"

//------------------------------------------------------------------------------
LwDiagUtils::EC MessageHandler::HandleVersion
(
    Messages::Version const &msg,
    void *                   pvConnection
)
{
    Connection * pConn = static_cast<Connection *>(pvConnection);
    GdmLogger::Printf(LwDiagUtils::PriNormal,
                      "%s : Node ID %u operating with messages v%u.%u\n",
                      pConn->GetConnectionString().c_str(),
                      msg.header.node_id,
                      msg.major_version,
                      msg.minor_version);
    return LwDiagUtils::OK;
}

//------------------------------------------------------------------------------
LwDiagUtils::EC MessageHandler::HandleShutdown
(
    Messages::Shutdown const &msg,
    void *                   pvConnection
)
{
    Connection * pConn = static_cast<Connection *>(pvConnection);
    GdmLogger::Printf(LwDiagUtils::PriNormal,
                      "%s Node ID %u sent shutdown with status %u\n",
                      pConn->GetConnectionString().c_str(),
                      msg.header.node_id,
                      msg.status);
    ByteStream bs;
    auto sd = MessageWriter::Messages::shutdown_ack(&bs);
    {
        sd
        .header()
        .node_id(0U)
        .app_type(MessageWriter::MessageHeader::AppType::gdm);
    }
    sd.shutdown_success(true);
    sd.Finish();
    return LwDiagUtils::OK;
}

LwDiagUtils::EC MessageHandler::HandleHeartBeat
(
    Messages::HeartBeat const &msg,
    void *                   pvConnection
)
{
    Connection * pConn = static_cast<Connection *>(pvConnection);
    HeartBeatMonitor::SendUpdate(msg.hb_reg_id);

    GdmLogger::Printf(LwDiagUtils::PriNormal,
                      "%s : Node ID %u sent heart beat messages\n",
                      pConn->GetConnectionString().c_str(),
                      msg.header.node_id);
    return LwDiagUtils::OK;
}

LwDiagUtils::EC MessageHandler::HandleRegister
(
    Messages::Register const &msg,
    void *                   pvConnection
)
{
    Connection * pConn = static_cast<Connection *>(pvConnection);
    HeartBeatMonitor::MonitorHandle regId;
    regId = HeartBeatMonitor::RegisterApp (msg.header.node_id,
                                           msg.header.app_type,
                                           msg.heart_beat_period
                                          );
    ByteStream bs;
    auto ri = MessageWriter::Messages::registration_id(&bs);
    {
        ri
        .header()
        .node_id(0U)
        .app_type(MessageWriter::MessageHeader::AppType::gdm);
    }
    ri.registraion_id(regId);
    ri.Finish();
    GdmServer::SendMessage(bs, pvConnection);
    HeartBeatMonitor::RegisterGdm(pvConnection);

    GdmLogger::Printf(LwDiagUtils::PriNormal,
                      "%s : Node ID %u sent Registerd\n",
                      pConn->GetConnectionString().c_str(),
                      msg.header.node_id);
    return LwDiagUtils::OK;
}

LwDiagUtils::EC MessageHandler::HandleUnRegister
(
    Messages::UnRegister const &msg,
    void *                   pvConnection
)
{
    HeartBeatMonitor::UnRegisterApp(msg.hb_reg_id);
    return LwDiagUtils::OK;
}

LwDiagUtils::EC MessageHandler::HandleRegistrationId
(
    Messages::RegistrationId const &msg,
    void *                   pvConnection
)
{
    return LwDiagUtils::OK;
}

LwDiagUtils::EC MessageHandler::HandleMissingHeartBeat
(
    Messages::MissingHeartBeat const &msg,
    void *                   pvConnection
)
{
    return LwDiagUtils::OK;
}

LwDiagUtils::EC MessageHandler::HandleShutdownAck
(
    Messages::ShutdownAck const &msg,
    void *                   pvConnection
)
{
    return LwDiagUtils::OK;
}

LwDiagUtils::EC MessageHandler::HandleGetNumGpu
(
    Messages::GetNumGpu const &msg,
    void *                   pvConnection
)
{    
    LwDiagUtils::EC ec;
    CHECK_EC(GlobalFmManager::GetNumGpus(pvConnection));
    return LwDiagUtils::OK;
}

LwDiagUtils::EC MessageHandler::HandleRetNumGpu
(
    Messages::RetNumGpu const &msg,
    void *                   pvConnection
)
{

    return LwDiagUtils::OK;
}

LwDiagUtils::EC MessageHandler::HandleGetNumLwSwitch
(
    Messages::GetNumLwSwitch const &msg,
    void *                   pvConnection
)
{
    LwDiagUtils::EC ec;
    CHECK_EC(GlobalFmManager::GetNumLwSwitch(pvConnection));
    return LwDiagUtils::OK;
}

LwDiagUtils::EC MessageHandler::HandleRetNumLwSwitch
(
    Messages::RetNumLwSwitch const &msg,
    void *                   pvConnection
)
{
    return LwDiagUtils::OK;
}

LwDiagUtils::EC MessageHandler::HandleGetGfmGpuMaxLwLinks
(
    Messages::GetGfmGpuMaxLwLinks const &msg,
    void *                   pvConnection
)
{
    LwDiagUtils::EC ec;
    CHECK_EC(GlobalFmManager::GetGpuMaxLwLinks(msg.physical_id, pvConnection));
    return LwDiagUtils::OK;
}

LwDiagUtils::EC MessageHandler::HandleRetGfmGpuMaxLwLinks
(
    Messages::RetGfmGpuMaxLwLinks const &msg,
    void *                   pvConnection
)
{

    return LwDiagUtils::OK;
}

LwDiagUtils::EC MessageHandler::HandleGetGfmPhysicalId
(
    Messages::GetGfmPhysicalId const &msg,
    void *                   pvConnection
)
{
    LwDiagUtils::EC ec;
    if (msg.gpu)
    {
        CHECK_EC(GlobalFmManager::GetGpuPhysicalId(msg.index, pvConnection));
    }
    else
    {
        CHECK_EC(GlobalFmManager::GetLwSwitchPhysicalId(msg.index, pvConnection));
    }
    return LwDiagUtils::OK;
}

LwDiagUtils::EC MessageHandler::HandleRetGfmPhysicalId
(
    Messages::RetGfmPhysicalId const &msg,
    void *                   pvConnection
)
{
    return LwDiagUtils::OK;
}


LwDiagUtils::EC MessageHandler::HandleGetGfmGpuEnumIdx
(
    Messages::GetGfmGpuEnumIdx const &msg,
    void *                   pvConnection
)
{
    
    LwDiagUtils::EC ec;
    // TODO : Also call the LwSwitch Enum if needed
    CHECK_EC(GlobalFmManager::GetGpuEnumIndex(msg.node_id,
                                              msg.physical_id,
                                              pvConnection));
    return LwDiagUtils::OK;
}

LwDiagUtils::EC MessageHandler::HandleRetGfmGpuEnumIdx
(
    Messages::RetGfmGpuEnumIdx const &msg,
    void *                   pvConnection
)
{    
    return LwDiagUtils::OK;
}

LwDiagUtils::EC MessageHandler::HandleGetGfmGpuPciBdf
(
    Messages::GetGfmGpuPciBdf const &msg,
    void *                   pvConnection
)
{
    LwDiagUtils::EC ec;
    if (msg.gpu)
    {
        CHECK_EC(GlobalFmManager::GetGpuPciBdf(msg.node_id, msg.enum_idx, pvConnection)); 
    }
    else
    {
        CHECK_EC(GlobalFmManager::GetLwSwitchPciBdf(msg.node_id, msg.enum_idx, pvConnection));
    }
    return LwDiagUtils::OK;
}

LwDiagUtils::EC MessageHandler::HandleRetGfmGpuPciBdf
(
    Messages::RetGfmGpuPciBdf const &msg,
    void *                   pvConnection
)
{    
    return LwDiagUtils::OK;
}



