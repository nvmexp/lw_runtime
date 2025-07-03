/*
 * Copyright (c) 2018-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <LwnUtil.h>
#include <CommandProcessor.h>
#include <Communicator.h>
#include <Network.h>
#include <Logger.h>
#include <Profiler.h>
#include <Heartbeat.h>
#include <LwnObjects.h>
#include <nn/os.h>

#if defined(LLGD_REPLAY_ENABLE_PROFILER)
#include <nn/profiler.h>
#endif

#include <ShaderDebuggerNetwork.h>

volatile bool waitForDebugger = false;

extern "C" void nnMain()
{
    while (waitForDebugger) {
        nn::os::SleepThread(nn::TimeSpan::FromMilliSeconds(1));
    }

    // Initialize the logging system. *Make sure to do this first!*
    LlgdReplay::Logger::Instance().Initialize(LlgdReplay::CpuCore{ 2 });

#if defined(LLGD_REPLAY_ENABLE_PROFILER)
    char* pProfilerBuffer = new char[nn::profiler::MinimumBufferSize];
    nn::profiler::Initialize(pProfilerBuffer, nn::profiler::MinimumBufferSize);
#endif

    // Initialize the heartbeat system.
    // *Do this after log initialization, before system initialization*
    auto result = LlgdReplay::Heartbeat::Instance().Initialize(LlgdReplay::CpuCore{ 2 });
    LLGD_USER_LOG_ERR_IF(!result, "REPLAYER ERROR: Failed to initialize heartbeat system.");

    // Initialize the graphics system
    LLGD_LOG_INF("Target Initializing Graphics System");
    result = LlgdReplay::Lwn::Util::System::Instance().Initialize();
    LLGD_USER_LOG_ERR_IF(!result, "REPLAYER ERROR: Failed to initialize display and graphics system.");

    // Initialize the profiler
    LLGD_LOG_INF("Target Initializing Profiler");
    result = LlgdReplay::Profiler::Instance().Initialize();
    LLGD_USER_LOG_ERR_IF(!result, "REPLAYER ERROR: Failed to initialize profiling system.");

    // Initialize the shader debugger networking system.
    // Never receive messages, only send messages from Target to Host.
    LLGD_LOG_INF("Target Initializing Shader Debugger Networking System");
    const bool isSDNetworkInitialized = LlgdReplay::ShaderDebuggerNetwork::Instance().Initialize(LlgdReplay::CpuCore{ 2 });
    result = isSDNetworkInitialized;
    LLGD_USER_LOG_ERR_IF(!result, "REPLAYER ERROR: Failed to initialize shader debugger networking system.");

    // Initialize shader debugger
    bool isSDInitialized = false;
    if (isSDNetworkInitialized) {
        LLGD_LOG_INF("Target Initializing ShaderDebugger");
        isSDInitialized = LlgdReplay::ShaderDebugger::Instance().Initialize();
        result = isSDInitialized;
        LLGD_USER_LOG_ERR_IF(!result, "REPLAYER ERROR: Failed to initialize shader debugger.");
    }

    // Initialize the networking system after all other systems are
    // initialized.  The host can start sending / receiving commands
    // once the listener is available.
    LLGD_LOG_INF("Target Initializing Networking System");
    result = LlgdReplay::Network::Instance().Initialize(LlgdReplay::CpuCore{ 1 });
    LLGD_USER_LOG_ERR_IF(!result, "REPLAYER ERROR: Failed to initialize networking system.");

    // The LLGD replay target is command driven.  The host sends
    // commands, and the target processes them.
    LLGD_LOG_INF("Target Beginning Command Processing");
    LlgdReplay::CommandProcessor::Instance().ProcessCommands();

    // Finalize the shader debugger
    if (isSDInitialized) {
        LLGD_LOG_INF("Target Finalizing ShaderDebugger");
        LlgdReplay::ShaderDebugger::Instance().Finalize();
    }

    // Finalize the shader debugger networking system.
    if (isSDNetworkInitialized) {
        LLGD_LOG_INF("Target Finalizing Shader Debugger Networking System");
        LlgdReplay::ShaderDebuggerNetwork::Instance().Finalize();
    }

    // Finalize the profiler
    LLGD_LOG_INF("Target Finalizing Profiler");
    LlgdReplay::Profiler::Instance().Finalize();

    // Finalize the created objects
    LLGD_LOG_INF("Target Finalizing Created Objects");
    LlgdReplay::Lwn::ObjectFactory::Instance().Finalize();

    // Finalize the networking system
    LLGD_LOG_INF("Target Finalizing Network System");
    LlgdReplay::Network::Instance().Finalize();

    // Finalize the graphics system
    LLGD_LOG_INF("Target Finalizing Graphics System");
    LlgdReplay::Lwn::Util::System::Instance().Finalize();

    // Finalize the heartbeat system
    LlgdReplay::Heartbeat::Instance().Finalize();

    // Finalize the logging system. *Make sure to do this last!*
    LlgdReplay::Logger::Instance().Finalize();
}
