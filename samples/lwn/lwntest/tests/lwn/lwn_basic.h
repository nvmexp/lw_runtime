/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef __lwn_basic_h__
#define __lwn_basic_h__
//
// lwn_basic.h:  Shared header from the LWN sample application, used by the
// code in lwn_basic.cpp (C interface sample) and lwn_basic_cpp.cpp (C++
// interface sampler).
//

#include "lwnUtil/lwnUtil_Interface.h"

//////////////////////////////////////////////////////////////////////////

//
// lwnexample.h
//

// "Global" LWN objects created at initialization time, to be used by C code.
struct LWNSampleTestCInterface {
    LWNdevice *device;
    LWNqueue *queue;
    LWNcommandBuffer *queueCB;
    lwnUtil::CommandBufferMemoryManager *cmdMemMgr;
};

#if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_CPP
// "Global" LWN objects created at initialization time, to be used by C++ code.
struct LWNSampleTestCPPInterface {
    lwn::Device *device;
    lwn::Queue *queue;
    lwn::CommandBuffer *queueCB;
    lwnUtil::CommandBufferMemoryManager *cmdMemMgr;
};
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

    static struct LWNSampleTestCInterface       *m_c_interface;
#if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_CPP
    static struct LWNSampleTestCPPInterface     *m_cpp_interface;
#endif
    LWNSampleTestConfig() :
        m_submitMode(QUEUE), m_benchmark(false), m_multisample(false),
        m_geometryShader(false), m_bindless(false), m_debug(false), m_cpp(false)
    { }

    void cDisplay(void);                // render using C interface
    void cppDisplay(void);              // render using C++ interface
    void generateDebugMessages(void);   // trigger debug messages
};

#endif // #ifndef __lwn_basic_h__
