// ----------------------------------------------------------------------------
// LlgdShaderDebuggerPlugin.h
// ----------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <lwn/lwn.h>

// From Llgd/Imports/llgd/shared/inc/
#include <LlgdProgram.h> // Needs to be after lwn.h

#if defined(__GNUC__)
#pragma GCC visibility push(default)
#if !defined(LLGSDApi)
#define LLGSDApi __attribute__((visibility("default")))
#endif
#else
#if !defined(LLGSDApi)
#define LLGSDApi
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/// Major version of the API
const uint32_t LLGDSD_PLUGIN_MAJOR_VERSION = 0;

/// Minor version of the API
const uint32_t LLGDSD_PLUGIN_MINOR_VERSION = 2;

typedef enum LLGDSD_PLUGIN_Result
{
    LLGDSD_PLUGIN_success,
    LLGDSD_PLUGIN_error,
    LLGDSD_PLUGIN_ilwalid_arg,
    LLGDSD_PLUGIN_ilwalid_state,
    LLGDSD_PLUGIN_no_mem,
    LLGDSD_PLUGIN_not_impl,
} LLGDSD_PLUGIN_Result;

struct LLGDSDShaderHandle
{
    uint32_t programId;
    LWNshaderStage stage;
};

// An array of shader instance per LWNshaderStage
typedef struct DTA_ShaderInstance* LLGDSDShaderInstances[LlgdProgramGraphics::NUMBER_OF_PROGRAM_STAGES];

struct LLGDSDProgram
{
    // LLGD data structure transmitted to the llgd-replay in LlgdMsgRpcInitializeProgramPayload
    const LlgdProgram* program;

    // Host generated handles representing unique shader instances
    LLGDSDShaderInstances shaderInstances;
};

// ----------------------------------------------------------------------------
// LLGDSDGetPlugilwersion
//
// Return the plugins' version for versioning check.
//
// ----------------------------------------------------------------------------
LLGSDApi LLGDSD_PLUGIN_Result LLGDSDGetPlugilwersion(uint32_t* pMajorVersion, uint32_t* pMinorVersion);

// ----------------------------------------------------------------------------
// LLGDSDInitializePlugin
//
// Initialize the ShaderDebugger plugin with all its components.
//
// sendTPSMsgFn: callback function that the SassDebugger message server will
// call to transmit TPS messages to the Host using LLGD RPC.
//
// ----------------------------------------------------------------------------
typedef bool (*PFNLLGDSDSENDTPSMSG)(const void* msg, uint32_t msgSize, void* userParam);

LLGSDApi LLGDSD_PLUGIN_Result LLGDSDInitializePlugin(PFNLLGDSDSENDTPSMSG sendTPSMsgFn, void* sendTPSMsgUserParam);

// ----------------------------------------------------------------------------
// LLGDSDFinalizePlugin
// ----------------------------------------------------------------------------
LLGSDApi LLGDSD_PLUGIN_Result LLGDSDFinalizePlugin();

// ----------------------------------------------------------------------------
// LLGDSDReceiveTPSMsg
//
// Passes a new raw TPS message to the SassDebugger plugin message server
// coming from LLGD RPC.
//
// ----------------------------------------------------------------------------
LLGSDApi LLGDSD_PLUGIN_Result LLGDSDReceiveTPSMsg(const void* msg, uint32_t msgSize);

// ----------------------------------------------------------------------------
// LLGDSDBeginDebugSession
//
// Start a debug session on a queue.
//
// ----------------------------------------------------------------------------
LLGSDApi LLGDSD_PLUGIN_Result LLGDSDBeginDebugSession(LWNqueue* queue);

// ----------------------------------------------------------------------------
// LLGDSDEndDebugSession
//
// End a debug session on a queue.
//
// ----------------------------------------------------------------------------
LLGSDApi LLGDSD_PLUGIN_Result LLGDSDEndDebugSession(LWNqueue* queue);

// ----------------------------------------------------------------------------
// LLGDSDSetLwrrentEventIDs
//
// Passes to the plugin the Host generated event IDs so that the sass debugger
// can do the necessary modifications to the constant buffers.
//
// ----------------------------------------------------------------------------
LLGSDApi LLGDSD_PLUGIN_Result LLGDSDSetLwrrentEventIDs(
    LWNqueue* queue,
    uint32_t drawcallDispatchId,
    uint32_t frameId,
    uint32_t eventId);

// ----------------------------------------------------------------------------
// LLGDSDAddProgram
// LLGDSDRemoveProgram
//
// Pass to the debugger state tracker necessary information about the programs
// that are used by the application.
//
// ----------------------------------------------------------------------------
LLGSDApi LLGDSD_PLUGIN_Result LLGDSDAddProgram(LWNdevice* device, const LLGDSDProgram* program);
LLGSDApi LLGDSD_PLUGIN_Result LLGDSDRemoveProgram(LWNdevice* device, const LLGDSDProgram* program);

// ----------------------------------------------------------------------------
// LLGDSDAddAutoProgram
// LLGDSDRemoveAutoProgram
//
// Pass to the debugger state tracker necessary information about programs
// that are created by the replayer for shader replacement purposes.
//
// ----------------------------------------------------------------------------
typedef bool (*PFNLLGDSDSHADERINSTANCENOTIFICATION)(const LLGDSDShaderInstances, void* userParam);
LLGSDApi LLGDSD_PLUGIN_Result LLGDSDAddAutoProgram(
    LWNdevice* device,
    LWNprogram* program,
    PFNLLGDSDSHADERINSTANCENOTIFICATION onRegisterShaderInstancesFn,
    void* onRegisterShaderInstancesFnUserParam);
LLGSDApi LLGDSD_PLUGIN_Result LLGDSDRemoveAutoProgram(
    LWNdevice* device,
    LWNprogram* program,
    PFNLLGDSDSHADERINSTANCENOTIFICATION onDeregisterShaderInstancesFn,
    void* onDeregisterShaderInstancesFnUserParam);

// ----------------------------------------------------------------------------
// LLGDSDDrawBegin
// LLGDSDDrawEnd
//
// Those functions should be called before (resp. after) the draw submit
// commands to update the debugger state tracker.
//
// ----------------------------------------------------------------------------
LLGSDApi LLGDSD_PLUGIN_Result LLGDSDDrawBegin(LWNqueue* queue, const LLGDSDShaderHandle* shaderHandles, uint32_t shadersCount);
LLGSDApi LLGDSD_PLUGIN_Result LLGDSDDrawEnd(LWNqueue* queue);

// ----------------------------------------------------------------------------
// LLGDSDDispatchBegin
// LLGDSDDispatchEnd
//
// Those functions should be called before (resp. after) the dispatch submit
// commands to update the debugger state tracker.
// ----------------------------------------------------------------------------
LLGSDApi LLGDSD_PLUGIN_Result LLGDSDDispatchBegin(LWNqueue* queue, const LLGDSDShaderHandle* computeShaderHandle);
LLGSDApi LLGDSD_PLUGIN_Result LLGDSDDispatchEnd(LWNqueue* queue);

// ----------------------------------------------------------------------------
// LLGDSDGetToolsConstsLocations
//
// Query tools const locations (bank + base offset) for graphics and compute
// contexts.
// ----------------------------------------------------------------------------
LLGSDApi LLGDSD_PLUGIN_Result LLGDSDGetToolsConstsLocations(uint32_t* graphicsBank, uint32_t* graphicsOffset, uint32_t* computeBank, uint32_t* computeOffset);

// ----------------------------------------------------------------------------
// LLGDSDOnDestroyDevice
//
// Notify that a device is going to be destroyed.
// ----------------------------------------------------------------------------
LLGSDApi LLGDSD_PLUGIN_Result LLGDSDOnDestroyDevice(LWNdevice* device);

// ----------------------------------------------------------------------------
// LLGDSDOnDestroyQueue
//
// Notify that a queue is going to be destroyed.
// ----------------------------------------------------------------------------
LLGSDApi LLGDSD_PLUGIN_Result LLGDSDOnDestroyQueue(LWNqueue* queue);

// ----------------------------------------------------------------------------
// LLGDSDOnHostConnectionLost
//
// Should be called when the connection with the host has been lost to shutdown
// the host interface connection on the target side.
// ----------------------------------------------------------------------------
LLGSDApi LLGDSD_PLUGIN_Result LLGDSDOnHostConnectionLost();

// ----------------------------------------------------------------------------
// LLGDSDNotifyHostDetach
//
// Notify the Debug agent that the debug session is going to end.
// Debug agent clears all breakpoints patches and resumes any suspended warps.
// Block breakpoints creation from the host.
// ----------------------------------------------------------------------------
LLGSDApi LLGDSD_PLUGIN_Result LLGDSDNotifyHostDetach();

// ----------------------------------------------------------------------------
// LLGDSDEnableBreakpointMessagesFromHost
//
// Allow back the host to create breakpoints and stop the gpu.
// No-Op if LLGDSDNotifyHostDetach has not been called previously.
// ----------------------------------------------------------------------------
LLGSDApi LLGDSD_PLUGIN_Result LLGDSDEnableBreakpointMessagesFromHost();

#ifdef __cplusplus
} // extern "C"
#endif

#if defined(__GNUC__)
#pragma GCC visibility pop
#endif
