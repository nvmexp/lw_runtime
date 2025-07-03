/*
* Copyright (c) 2015, Lwpu Corporation.  All rights reserved.
*
* THE INFORMATION CONTAINED HEREIN IS PROPRIETARY AND CONFIDENTIAL TO
* LWPU, CORPORATION.  USE, REPRODUCTION OR DISCLOSURE TO ANY THIRD PARTY
* IS SUBJECT TO WRITTEN PRE-APPROVAL BY LWPU, CORPORATION.
*
*
*/
#ifndef __lwnTest_DeviceState_h__
#define __lwnTest_DeviceState_h__

#include "lwnUtil/lwnUtil_Interface.h"

#include "lwn/lwn.h"
#if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_CPP
#include "lwn/lwn_Cpp.h"
#endif

#include "lwnUtil/lwnUtil_QueueCmdBuf.h"
#include "lwnUtil/lwnUtil_TexIDPool.h"

#include "lwnTest_GlslcHelper.h"

namespace lwnTest {

// Structure identifying capabilities of the LWN device.
struct DeviceCaps {
    void init(LWNdevice *device);
    LWNint supportsMinMaxReduction;
    LWNint supportsStencil8;
    LWNint supportsASTC;
    LWNint supportsConservativeRaster;
    LWNint supportsZeroFromUndefinedMappedPoolPages;
    LWNint supportsPassthroughGeometryShaders;
    LWNint supportsViewportSwizzle;
    LWNint supportsMaxwellSparsePackagedTextures;
    LWNint supportsAdvancedBlendModes;
    LWNint supportsDrawTexture;
    LWNint supportsTargetIndependentRasterization;
    LWNint supportsFragmentCoverageToColor;
    LWNint supportsPostDepthCoverage;
    LWNint supportsImagesUsingTextureHandles;
    LWNint supportsSampleLocations;
    LWNint supportsFragmentShaderInterlock;
    LWNint supportsDebugLayer;
    LWNint supportsShadingRateImage;
    LWNint supportsScissorExclusive;
    LWNint supportsMaxwell2Features;
    LWNint supportsShaderSubgroup;
};


// Class encapsulating a variety of objects used for a single device,
// including the LWN device, and LWN queue, a QueueCommandBuffer, and other
// helper classes for managing device state.  We also include a structure to
// query/check various device capabilities of the device.
class DeviceState {
private:
    // Set up a default program pool size of 1MB.
    static const size_t DeviceProgramPoolSize = 0x100000;

    // Bitfield indicating the portions of the device that have been
    // initialized.  INITIALIZED_ALL is set if the device is fully
    // initialized.
    enum InitializedBits {
        INITIALIZED_DEVICE              = 0x0001,
        INITIALIZED_QUEUE               = 0x0002,
        INITIALIZED_QUEUECB             = 0x0004,
        INITIALIZED_TEXIDPOOL           = 0x0008,
        INITIALIZED_COMPLETION_TRACKER  = 0x0010,
        INITIALIZED_CMDBUF_MANAGER      = 0x0020,
        INITIALIZED_GLSLC_HELPER        = 0x0040,
        INITIALIZED_SCRATCH_POOL        = 0x0080,
        INITIALIZED_DEBUG_CALLBACK      = 0x0100,
        INITIALIZED_ALL                 = 0x8000,
    };
    unsigned int                        m_initBits;

    // Did the caller request that we create a debug device? We may force it
    // on via "-lwnDebug".
    bool                                m_requestedDebug;

    // Objects created as part of the device state.
    LWNdevice                           m_device;
    LWNqueue                            m_queue;
    lwnUtil::QueueCommandBufferBase     m_queueCB;
    void                                *m_queueMemory;
    lwnUtil::TexIDPool                  *m_texIDPool;
    lwnUtil::CompletionTracker          *m_completionTracker;
    lwnUtil::CommandBufferMemoryManager m_cmdMemManager;
    lwnTest::GLSLCHelper                *m_glslcHelper;
    LWNmemoryPool                       m_shaderScratchPool;
    void                                *m_shaderScratchPoolMemory;

    // Capabilities queried as part of the device state.
    DeviceCaps                          m_caps;
    LWNint                              m_lwnMajorVersion;
    LWNint                              m_lwnMinorVersion;

    // Query the flags set up when creating the device.
    LWNdeviceFlagBits                   m_deviceFlags;

public:
    void init(LWNdeviceFlagBits   deviceFlags,
              LWNwindowOriginMode windowOriginMode,
              LWNdepthMode        depthMode,
              LWNqueueFlags       queueFlags);
    DeviceState(LWNdeviceFlagBits   deviceFlags = LWNdeviceFlagBits(0),
                LWNwindowOriginMode windowOriginMode = LWN_WINDOW_ORIGIN_MODE_LOWER_LEFT,
                LWNdepthMode        depthMode = LWN_DEPTH_MODE_NEAR_IS_MINUS_W,
                LWNqueueFlags       queueFlags = LWNqueueFlags(0))
    {
        init(deviceFlags, windowOriginMode, depthMode, queueFlags);
    }
#if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_CPP
    DeviceState(lwn::DeviceFlagBits deviceFlags,
                lwn::WindowOriginMode windowOriginMode = lwn::WindowOriginMode::LOWER_LEFT,
                lwn::DepthMode depthMode = lwn::DepthMode::NEAR_IS_MINUS_W,
                lwn::QueueFlags queueFlags = lwn::QueueFlags::NONE)
    {
        LWNdeviceFlagBits cflags = LWNdeviceFlagBits(int(deviceFlags));
        LWNwindowOriginMode corigin = LWNwindowOriginMode(int(windowOriginMode));
        LWNdepthMode cdepth = LWNdepthMode(int(depthMode));
        LWNqueueFlags qflags = LWNqueueFlags(int(queueFlags));
        init(cflags, corigin, cdepth, qflags);
    }
#endif

    ~DeviceState();

    // Check if a new DeviceState object is valid and can be used by tests.
    bool isValid()  { return 0 != (m_initBits & INITIALIZED_ALL); }

    // lwntest has one default device created during initialization.  We store
    // this in a singleton in the class.
    static DeviceState              *g_defaultDevice;
    void                            SetDefault();
    static DeviceState              *GetDefault();

    // lwntest supports only a single active device at a time.  In particular,
    // we will use function pointers queried for that device.  We store the
    // active device in a singleton in the class.
    static DeviceState              *g_activeDevice;
    void                            SetActive();
    static void                     SetDefaultActive();
    static DeviceState              *GetActive();

    // Query various LWN objects and helpers set up by the DeviceState object.
    lwnUtil::QueueCommandBuffer &
        getQueueCB()                    { return m_queueCB; }
    lwnUtil::TexIDPool *getTexIDPool()  { return m_texIDPool; }
    lwnUtil::CompletionTracker *
        getCompletionTracker()          { return m_completionTracker; }
    lwnUtil::CommandBufferMemoryManager *
        getCmdBufMemoryManager()        { return &m_cmdMemManager; }
    lwnTest::GLSLCHelper *
        getGLSLCHelper()                { return m_glslcHelper; }

    // lwntest tears down and rebuilds the GLSLC helper for every test run.
    void destroyGLSLCHelper();
    void rebuildGLSLCHelper();

    // Use the DeviceState's completion tracker to insert a fence in the queue
    // that we can wait on to reuse command memory.
    void insertFence()
    {
        m_completionTracker->insertFence(&m_queue);
    }

    // Hack to finalize the DeviceState's device without finalizing the queue
    // for bug 1802719.
    void skipQueueFinalization()
    {
        m_initBits &= ~INITIALIZED_QUEUE;
    }

    //
    // Methods to query properties of the device state have separate C and C++
    // implementations.  They can't coexist as overloads because they are
    // different only in return type.  The C++ methods use reinterpret_cast to
    // colwert to/from native C types.
    //
#if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_CPP
    lwn::Device *getDevice()
    {
        return reinterpret_cast<lwn::Device *>(&m_device);
    }
    lwn::Queue *getQueue()
    {
        return reinterpret_cast<lwn::Queue *>(&m_queue);
    }
    lwn::MemoryPool *getShaderScratchPool()
    {
        return reinterpret_cast<lwn::MemoryPool *>(&m_shaderScratchPool);
    }
    lwn::DeviceFlagBits getDeviceFlags()
    {
        return (lwn::DeviceFlagBits)m_deviceFlags;
    }
#else
    LWNdevice *getDevice()
    {
        return &m_device;
    }
    LWNqueue *getQueue()
    {
        return &m_queue;
    }
    LWNmemoryPool *getShaderScratchPool()
    {
        return &m_shaderScratchPool;
    }
    LWNdeviceFlagBits getDeviceFlags()
    {
        return m_deviceFlags;
    }
#endif

};

} // namespace lwnTest

#endif // #ifndef __lwnTest_DeviceState_h__
