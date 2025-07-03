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
#include "gdm_server.h"
#include "gdm_logger.h"
#include "mn_msgver.h"
#include "heart_beat_monitor.h"
#include <cstdio>
#include <sstream>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace
{
    struct timespec s_LoopDelayTs;
    LwDiagUtils::EC LoopDelay()
    {
        if (nanosleep(&s_LoopDelayTs, NULL) == -1)
        {
            GdmLogger::Printf(LwDiagUtils::PriError,
                              "nanosleep received signal while waiting\n");
            return LwDiagUtils::SOFTWARE_ERROR;
        }
        return LwDiagUtils::OK;
    }
    class LogHolder
    {
    public:
        LogHolder() { }
        ~LogHolder() { GdmLogger::Close(); }
        LwDiagUtils::EC Open(const string & filename) { return GdmLogger::Open(filename); }
    };
}

//------------------------------------------------------------------------------
int main(int argc, char **argv)
{

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("port,p", po::value<UINT32>()->default_value(0), "set the port to use for the server")
        ("num-connections,c", po::value<UINT32>()->default_value(0), "set the number of expected connections")
        ("loop-delay-ms", po::value<UINT32>()->default_value(1), "set the main loop delay in ms")
        ("connection-timeout-ms", po::value<UINT32>()->default_value(10000), "set the timeout for waiting for all connections")
        ("log-file,l", po::value<string>()->default_value("gdm.mle"), "set the output log file for gdm")
    ;

    po::variables_map vm;
    try
    {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    }
    catch (const exception & e)
    {
        printf("ERROR: %s\n", e.what());
        return -1;
    }
    catch (...)
    {
        printf("ERROR: An unknown error oclwrred during arg parsing\n");
        return -1;
    }

    if (vm.count("help"))
    {
        stringstream s;
        s << desc << "\n";
        GdmLogger::Printf(LwDiagUtils::PriNormal, "%s\n", s.str().c_str());
        return LwDiagUtils::OK;
    }

    LwDiagUtils::EC ec;
    LogHolder l;
    CHECK_EC(l.Open(vm["log-file"].as<string>()));
    LwDiagUtils::Initialize(nullptr, GdmLogger::VAPrintf);

    GdmLogger::Printf(LwDiagUtils::PriNormal, "GDM starting, message version %u.%u!\n",
                      MESSAGE_VERSION_MAJOR, MESSAGE_VERSION_MINOR);

    if ((vm["port"].as<UINT32>() == 0) || (vm["num-connections"].as<UINT32>() == 0))
    {
        GdmLogger::Printf(LwDiagUtils::PriError,
                          "Port and number of connections must be specified!\n");
        return -1;
    }
    CHECK_EC(GdmServer::Start(vm["port"].as<UINT32>(), vm["num-connections"].as<UINT32>()));

    s_LoopDelayTs = { 0, vm["loop-delay-ms"].as<UINT32>() * 1000000 };


    bool bWaitingForConnections = true;
    UINT32 connectionRetries =
        vm["connection-timeout-ms"].as<UINT32>() / vm["loop-delay-ms"].as<UINT32>();
    HeartBeatMonitor::InitMonitor();
    for (;;)
    {
        CHECK_EC(GdmServer::RunOnce());
        HeartBeatMonitor::SendGdmHb();

        if (bWaitingForConnections)
        {
            if (GdmServer::GetNumConnections() == vm["num-connections"].as<UINT32>())
            {
                bWaitingForConnections = false;
            }
            else
            {
                connectionRetries--;

                if ((connectionRetries % 1000) == 0)
                {
                    GdmLogger::Printf(LwDiagUtils::PriError,
                                      "Still waiting on connections\n");
                }
                if (connectionRetries == 0)
                {
                    GdmLogger::Printf(LwDiagUtils::PriError,
                                      "Failed to get the expected connections\n");
                    return -1;
                }
                CHECK_EC(LoopDelay());
                continue;
            }
        }

        CHECK_EC(LoopDelay());
    }

    return 0;
}
