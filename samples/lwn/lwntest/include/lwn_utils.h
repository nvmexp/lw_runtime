/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef __lwn_utils_h__
#define __lwn_utils_h__

#if !defined(__cplusplus)
#error lwn_utils.h only supports C++!
#endif // #if defined(__cplusplus)

#define CPPSHADERS_NON_GL_API
#include "cppshaders.h"                 // C++ shader library with OpenGL support disabled

// Import the generic data type support from the LWN texture packager.
#include "lwnTool/texpkg/lwnTool_DataTypes.h"
namespace lwn {
  namespace dt = lwnTool::texpkg::dt;
}

#include "lwn_utils_noapi.h"

#include <list>
#include <map>
#include <algorithm>
#include <vector>

// Set up variables to decide if we are using C or C++ code.
#include "lwnUtil/lwnUtil_Interface.h"

#include "lwnUtil/lwnUtil_AlignedStorage.h"
#include "lwnUtil/lwnUtil_PoolAllocator.h"
#include "lwnUtil/lwnUtil_GlslcHelper.h"

#include "lwnTest/lwnTest_Framebuffer.h"
#include "lwnTest/lwnTest_VertexState.h"

using namespace lwnUtil;
using namespace lwnTest;

#include "lwnTest/lwnTest_AllocationTracker.h"
#include "lwnTest/lwnTest_Objects.h"

#include "lwnTest/lwnTest_Formats.h"
#include "lwnTest/lwnTest_WindowFramebuffer.h"
#include "lwnTest/lwnTest_Mislwtils.h"
#include "lwnTest/lwnTest_Shader.h"
#include "lwnTest/lwnTest_GlslcHelper.h"

#include "lwnUtil/lwnUtil_CommandMem.h"
#include "lwnUtil/lwnUtil_QueueCmdBuf.h"
#include "lwnUtil/lwnUtil_TexIDPool.h"

#include "lwnTest/lwnTest_DeviceState.h"

//////////////////////////////////////////////////////////////////////////

extern LWNdevice                            *g_lwnDevice;
extern LWNqueue                             *g_lwnQueue;
extern LWNmemoryPool                        *g_lwnScratchMemPool;
extern lwnUtil::GLSLCHelperCache            *g_glslcHelperCache;
extern lwnTest::WindowFramebuffer           g_lwnWindowFramebuffer;
extern lwnUtil::TexIDPool                   *g_lwnTexIDPool;
extern lwnTest::GLSLCHelper                 *g_glslcHelper;
extern lwnUtil::QueueCommandBufferBase      *g_lwnQueueCB;
extern lwnUtil::CommandBufferMemoryManager  g_lwnCommandMem;
extern lwnUtil::CompletionTracker           *g_lwnTracker;
extern lwnTest::DeviceCaps                  g_lwnDeviceCaps;
extern int                                  g_lwnMajorVersion;
extern int                                  g_lwnMinorVersion;


extern "C" PFNLWNGENERICFUNCPTRPROC LWNAPIENTRY lwnBootstrapLoader(const char* name);
extern "C" void LWNAPIENTRY lwnInstallGlobalDebugCallback(PFNLWNDEBUGCALLBACKPROC callback, void* callbackData);

//////////////////////////////////////////////////////////////////////////

// A global default scratch memory size to allocate scratch memory buffers.
// The scratch memory size requirements is computed by taking the number of SM's
// on the device, the number of warps per SM, and the amount of bytes per warp
// required based on information GLSLC returns, and multiplying together.  The granularity
// of this needs to be 128KB.
// On TX1, with 2 SMs and 128 warps per SM, 512 KB allows for 2 KB per warp.
#define DEFAULT_SHADER_SCRATCH_MEMORY_SIZE 524288

// Reload LWN function pointers to get the appropriate version after changing
// devices (e.g., when creating or deleting debug devices).  <device>
// indicates the device the entry points should be loaded for.  <apiDebug>
// indicates whether a debug context is used.  We disable the global debug
// message handler set up by "-lwndebug" when using a debug context in a test,
// since those errors may be intentional.
void ReloadLWNEntryPoints(LWNdevice *device, bool apiDebug);

// Utility functions to load function pointers for the C and C++ interfaces.
void ReloadCInterface(LWNdevice *device, PFNLWNDEVICEGETPROCADDRESSPROC getProcAddress);
void ReloadCppInterface(LWNdevice *device, PFNLWNDEVICEGETPROCADDRESSPROC getProcAddress);

// Global debug callback used when the "-lwndebug" option is set.
extern void LWNAPIENTRY LWNUtilsDebugCallback(LWNdebugCallbackSource source, LWNdebugCallbackType type, int id,
                                              LWNdebugCallbackSeverity severity, LWNstring message, void *userParam);

// Basic sanity check on the LWN debug layer when "-lwnDebug" is specified.
// This code temporarily disables the normal debug callback, installs a new
// callback and then triggers an error to determine if the callback was
// performed.  Returns 1 if the callback worked and 0 otherwise.
extern int SanityCheckLWNDebug(LWNdevice *device);

extern void DebugWarningIgnore(int32_t warningID);
extern void DebugWarningAllow(int32_t warningID);
extern void DebugWarningAllowAll();

#endif // #ifndef __lwn_utils_h__
