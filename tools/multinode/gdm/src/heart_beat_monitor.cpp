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

#include "heart_beat_monitor.h"
#include "gdm_logger.h"
#include "gdm_server.h"
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

namespace {
    static constexpr UINT32 HBMONITOR_SLEEP_TIME_MILSEC     = 5000;
    static constexpr UINT32 HBMONITOR_DEFAULT_PERIOD_SEC    = 10;

    struct HeartBeatContext
    {
        UINT64  HeartBeatPeriodMs                      =  0;
        UINT64  HeartBeatPeriodExpirationTime          =  0;
        UINT32  NodeId                                 =  0;
        UINT32  AppId                                  =  0;
    };

    static std::mutex s_HBMutex;
    static std::vector<HeartBeatContext> s_HbContextVec;
    static std::condition_variable s_HbmExitEvent;


    UINT64 GetContextId(HeartBeatMonitor::MonitorHandle handle)
    {
        return static_cast<UINT64>(-handle - 1);
    }

}

static void HeartBeatMonitorThread();

LwDiagUtils::EC HeartBeatMonitor::InitMonitor()
{
    std::thread heartBeatMon(HeartBeatMonitorThread);
    heartBeatMon.detach();
    return LwDiagUtils::OK;
}

LwDiagUtils::EC HeartBeatMonitor::SendUpdate(HeartBeatMonitor::MonitorHandle regId)
{
    std::lock_guard<std::mutex> lock(s_HBMutex);
    const UINT64 contextIndex = GetContextId(regId);
    if (regId == 0 || contextIndex >= s_HbContextVec.size())
    {
        GdmLogger::Printf(LwDiagUtils::PriError, "Heart beat registration Id out of bounds\n");
        LWDASSERT(!"Heart beat registration Id out of bounds");
        return LwDiagUtils::BAD_PARAMETER;
    }
    if (s_HbContextVec[contextIndex].HeartBeatPeriodExpirationTime == 0)
    {
        GdmLogger::Printf(LwDiagUtils::PriError, "Invalid heart beat registration Id\n");
        LWDASSERT(!"Invalid heart beat registration Id");
        return LwDiagUtils::BAD_PARAMETER;

    }
    const UINT64 lwrrentTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    const UINT64 expirationTime = s_HbContextVec[contextIndex].HeartBeatPeriodMs + lwrrentTime;
    s_HbContextVec[contextIndex].HeartBeatPeriodExpirationTime = expirationTime;
    return LwDiagUtils::OK;
}

HeartBeatMonitor::MonitorHandle HeartBeatMonitor::RegisterApp(UINT32 nodeId, UINT32 appId, UINT64 heartBeatPeriodSec)
{
    std::lock_guard<std::mutex> lock(s_HBMutex);
    HeartBeatContext appRegistration;
    appRegistration.NodeId  = nodeId;
    appRegistration.AppId  = appId;
    if (heartBeatPeriodSec == 0)
    {
        GdmLogger::Printf(LwDiagUtils::PriLow, "Hearbeat period of 0 not valid setting it to default value\n");
        heartBeatPeriodSec = HBMONITOR_DEFAULT_PERIOD_SEC;
    }
    appRegistration.HeartBeatPeriodMs = heartBeatPeriodSec * 1000;
    const UINT64 registrationTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    const UINT64 expirationTime = registrationTime + (heartBeatPeriodSec * 1000);
    appRegistration.HeartBeatPeriodExpirationTime = expirationTime;

    const auto unusedSlotIt = find_if(s_HbContextVec.begin(), s_HbContextVec.end(),
            [](const HeartBeatContext& ctx) -> bool { return ctx.HeartBeatPeriodExpirationTime == 0; });
    const INT64 contextIndex = unusedSlotIt - s_HbContextVec.begin();
    if (unusedSlotIt == s_HbContextVec.end())
    {
        s_HbContextVec.push_back(appRegistration);
    }
    else
    {
        s_HbContextVec[contextIndex] = appRegistration;
    }

    return -(1 + contextIndex);
}

LwDiagUtils::EC HeartBeatMonitor::UnRegisterApp(HeartBeatMonitor::MonitorHandle regId)
{
    std::lock_guard<std::mutex> lock(s_HBMutex);
    const UINT64 contextIndex = GetContextId(regId);
    if (regId == 0 || contextIndex >= s_HbContextVec.size())
    {
        GdmLogger::Printf(LwDiagUtils::PriError, "Heart beat registration Id out of bounds\n");
        LWDASSERT(!"Heart beat registration Id out of bounds");
        return LwDiagUtils::BAD_PARAMETER;
    }
    if (s_HbContextVec[contextIndex].HeartBeatPeriodExpirationTime == 0)
    {
        GdmLogger::Printf(LwDiagUtils::PriError, "Invalid registration Id\n");
        LWDASSERT(!"Invalid heart beat registration Id");
        return LwDiagUtils::BAD_PARAMETER;
    }
    s_HbContextVec[contextIndex].HeartBeatPeriodExpirationTime = 0;
    return LwDiagUtils::OK;
}

LwDiagUtils::EC HeartBeatMonitor::RegisterGdm(void *pvConnection)
{
    ByteStream bs;
    auto hb = MessageWriter::Messages::register_app(&bs);
    {
        hb
        .header()
        .node_id(0U)
        .app_type(MessageWriter::MessageHeader::AppType::gdm);
    }
    hb.heart_beat_period(HBMONITOR_DEFAULT_PERIOD_SEC);
    hb.Finish();
    GdmServer::SendMessage(bs, pvConnection);
    return LwDiagUtils::OK;
}

LwDiagUtils::EC HeartBeatMonitor::SendGdmHb()
{
    ByteStream bs;
    auto hb = MessageWriter::Messages::heartbeat(&bs);
    {
        hb
        .header()
        .node_id(0U)
        .app_type(MessageWriter::MessageHeader::AppType::gdm);
    }
    hb.Finish();
    GdmServer::BroadcastMessage(bs);
    return LwDiagUtils::OK;
}

static void HeartBeatMonitorThread()
{
    std::mutex mtx;
    std::unique_lock<std::mutex> lck(mtx);
    while (s_HbmExitEvent.wait_for(lck, std::chrono::milliseconds(HBMONITOR_SLEEP_TIME_MILSEC)) == std::cv_status::timeout)
    {
        std::lock_guard<std::mutex> lock(s_HBMutex);
        for (const auto& appEntry: s_HbContextVec)
        {
            const UINT64 expirationTime = appEntry.HeartBeatPeriodExpirationTime;
            const UINT64 lwrrentTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            if (expirationTime == 0 || lwrrentTime < expirationTime)
                continue;

            GdmLogger::Printf(LwDiagUtils::PriError, 
                    "Missing heartbeat from app id %u on node id %u, last update time %llu"
                    ",current time %llu, expected update interval %llu, exiting\n",
                    appEntry.AppId, appEntry.NodeId,
                    (expirationTime - appEntry.HeartBeatPeriodMs),
                    lwrrentTime, appEntry.HeartBeatPeriodMs
                  );
            ByteStream bs;
            auto hb = MessageWriter::Messages::missing_heartbeat(&bs);
            {
                hb
                .header()
                .node_id(appEntry.NodeId)
                .app_type(appEntry.AppId);
            }
            hb.Finish();
            GdmServer::BroadcastMessage(bs);

        }
    }
    GdmLogger::Printf(LwDiagUtils::PriLow, "Exiting HeartBeatMonitorThread\n");
}
