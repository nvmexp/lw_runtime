/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

//
// lwnexample.h
//

#if !defined(LWN_USE_C_INTERFACE) && !defined(LWN_USE_CPP_INTERFACE)
#error Must set LWN_USE_C_INTERFACE or LWN_USE_CPP_INTERFACE (or both).
#endif

// Shut up deprecation warnings.
#define LWN_PRE_DEPRECATED
#define LWN_POST_DEPRECATED

#if defined(LWN_USE_C_INTERFACE)
#include <lwn/lwn.h>
#include <lwn/lwn_FuncPtrInline.h>
#endif
#if defined(LWN_USE_CPP_INTERFACE)
#include <lwn/lwn_Cpp.h>
#include <lwn/lwn_CppMethods.h>
#endif

#include "lwnutil.h"
#include "lwnUtil/lwnUtil_PoolAllocator.h"
#include "lwnUtil/lwnUtil_GlslcHelper.h"

using namespace lwnUtil;

#define NUM_PRESENT_TEXTURES 2

#if defined(LWN_USE_C_INTERFACE)
struct LWNBasicWindowC {
    LWNwindow   win;
    LWNtexture  *presentTexture[NUM_PRESENT_TEXTURES];
};

// "Global" LWN objects created at initialization time, to be used by C code.
struct LWNSampleTestCInterface {
    LWNdevice *device;
    LWNqueue *queue;
    LWNcommandBuffer *queueCB;
    LWNcommandBufferMemoryManager *cmdMemMgr;
    CompletionTracker *completionTracker;
    lwnUtil::GLSLCLibraryHelper *glslcLibraryHelper;
    LWNBasicWindowC  *window;
};

struct LWNBasicWindowCPP;
#endif

#if defined(LWN_USE_CPP_INTERFACE)
struct LWNBasicWindowCPP {
    lwn::objects::Window   win;
    lwn::Texture  *presentTexture[NUM_PRESENT_TEXTURES];
};

// "Global" LWN objects created at initialization time, to be used by C++ code.
struct LWNSampleTestCPPInterface {
    lwn::Device *device;
    lwn::Queue *queue;
    lwn::CommandBuffer *queueCB;
    LWNcommandBufferMemoryManager *cmdMemMgr;
    CompletionTracker *completionTracker;
    lwnUtil::GLSLCLibraryHelper *glslcLibraryHelper;
    LWNBasicWindowCPP *window;
};

struct LWNBasicWindowC;
#endif

struct LWNSampleTestConfig
{
    enum SubmitMode {
        QUEUE,                          // submit commands to the queue
        COMMAND,                        // submit commands to a non-transient command buffer
        COMMAND_TRANSIENT,              // submit commands to a transient command buffer
    };

    SubmitMode  m_submitMode;           // how to submit commands to LWN
    bool m_benchmark;                   // run in benchmark mode?
    bool m_multisample;                 // run with a multisample render target?
    bool m_geometryShader;              // run with a geometry shader?
    bool m_tessControlShader;           // run with a tessellation control shader?
    bool m_tessEvalShader;              // run with a tessellation evaluation shader?
    bool m_wireframe;                   // run in wireframe mode
    bool m_bindless;                    // run using bindless textures?
    bool m_debug;                       // run extra tests of the debug layer
    bool m_cpp;                         // use C++ interface

    int m_windowWidth;
    int m_windowHeight;

    static struct LWNSampleTestCInterface       *m_c_interface;
    static struct LWNSampleTestCPPInterface     *m_cpp_interface;

    LWNSampleTestConfig() :
        m_submitMode(QUEUE), m_benchmark(false), m_multisample(false),
        m_geometryShader(false), m_bindless(false),
        m_debug(false), m_cpp(false),
        m_windowWidth(0), m_windowHeight(0)
    { }

    LWNBasicWindowC* cCreateWindow(LWNnativeWindow nativeWindow, int w, int h);
    LWNBasicWindowCPP* cppCreateWindow(LWNnativeWindow nativeWindow, int w, int h);

    void cDeleteWindow(void);
    void cppDeleteWindow(void);

    void cDisplay(void);                // render using C interface
    void cppDisplay(void);              // render using C++ interface
    void generateDebugMessages(void);   // trigger debug messages
};

extern MemoryPoolAllocator *g_texAllocator;
extern MemoryPoolAllocator *g_bufferAllocator;

#if defined(LWN_USE_CPP_INTERFACE)
// C++ interface to retrieve the registered texture/sampler IDs that our
// C-based allocation code saved away next to the LWN objects.
extern int lwnGetRegisteredTextureID(lwn::Texture *texture);
extern int lwnGetRegisteredSamplerID(lwn::Sampler *sampler);

#endif
