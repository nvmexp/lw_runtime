#pragma once

/*
** Copyright (c) 2020 LWPU CORPORATION.  All rights reserved.
**
** LWPU CORPORATION and its licensors retain all intellectual property
** and proprietary rights in and to this software, related documentation
** and any modifications thereto.  Any use, reproduction, disclosure or
** distribution of this software and related documentation without an express
** license agreement from LWPU CORPORATION is strictly prohibited.
*/

// Note: When using debug layer on Win32, the debug layer will warn about
// the creation of a memory pool without backing system memory.

#include <algorithm>
#include <assert.h>
#include <iostream>
#include <vector>
#include <memory>

#ifdef _WIN32
  #define NOMINMAX
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>

  // Exporting this special variable causes the LWPU drivers to render with our
  // GPU on Optimus systems without any special application profiles.
  extern "C" {
     _declspec(dllexport) unsigned long LwOptimusEnablement = 0x00000001;
  }
#endif

#include "lwn/lwn.h"
#include "lwn/lwn_Cpp.h"
#include "lwn/lwn_CppFuncPtr.h"
#include "lwn/lwn_FuncPtrImpl.h"
#include "lwn/lwn_CppFuncPtrImpl.h"
#include "lwn/lwn_CppMethods.h"
#include "lwn/lwn_FuncPtrGlobal.h"

template <typename T1, typename T2>
T1 Align(T1 value, T2 alignSize)
{
    alignSize -= 1;
    return (value + alignSize) & ~(alignSize);
}

static void *AlignPtr(void *ptr, size_t alignSize)
{
    return (void *) Align((uintptr_t) ptr, alignSize);
}

class LwnMemoryPool
{
public:
    LwnMemoryPool(lwn::Device *device, size_t size, lwn::MemoryPoolFlags flags)
    {
        lwn::MemoryPoolBuilder memPoolBuilder;
        memPoolBuilder.SetDefaults();
        memPoolBuilder.SetDevice(device);
        memPoolBuilder.SetFlags(flags);

#ifdef _WIN32
        // Windows does not require system memory to back memory pool.
        memPoolBuilder.SetStorage(nullptr, size);
#else
        m_allocPtr = malloc(size + 2 * pageSize);   // Two alignments: ptr and size
        memPoolBuilder.SetStorage(AlignPtr(m_allocPtr, pageSize), Align(size, pageSize));
#endif
        m_flags = flags;
        m_poolSize = size;
        m_usedSize = 0;

        m_apiPool.Initialize(&memPoolBuilder);
    }

    size_t Allocate(size_t size, size_t alignment = 0)        // Returns offset
    {
        if (alignment != 0)
        {
            m_usedSize = Align(m_usedSize, alignment);
        }

        if (size + m_usedSize > m_poolSize)
        {
            Error("Allocation request exceeds pool size");
        }
        size_t offset = m_usedSize;

        m_usedSize += size;
        return offset;
    }

    ~LwnMemoryPool()
    {
        m_apiPool.Finalize();
        free(m_allocPtr);
    }

    lwn::MemoryPool m_apiPool;
    lwn::MemoryPoolFlags m_flags;

private:
#ifndef _WIN32
    static constexpr int pageSize = 4096;
    void   *m_allocPtr;
#endif
    size_t m_poolSize;
    size_t m_usedSize;
};

class LwnBuffer
{
public:
    static const int BufferBaseAlignment = 256;     // Choose the worst alignment - this way
                                                    // everything is covered.

    LwnBuffer(lwn::Device *device, LwnMemoryPool *pool, size_t size)
    {
        m_size = Align(size, BufferBaseAlignment);
        size_t offset = pool->Allocate(m_size);

        lwn::BufferBuilder bufferBuilder;
        bufferBuilder.SetDefaults();
        bufferBuilder.SetDevice(device);
        bufferBuilder.SetStorage(&pool->m_apiPool, offset, m_size);

        m_apiBuffer.Initialize(&bufferBuilder);
        m_gpuAddr = m_apiBuffer.GetAddress();

        // In theory, we can map the memory pool instead of mapping individual objects.
        if (static_cast<uint32_t> (pool->m_flags & lwn::MemoryPoolFlags::CPU_NO_ACCESS) == 0)
        {
            m_cpuAddr = m_apiBuffer.Map();
        }
    }

    ~LwnBuffer()
    {
        m_apiBuffer.Finalize();
    }

    lwn::Buffer        m_apiBuffer;
    void              *m_cpuAddr;
    lwn::BufferAddress m_gpuAddr;
    size_t             m_size;          // Needed for programming vertex buffer
};

#ifndef _WIN32
extern "C" PFNLWNGENERICFUNCPTRPROC LWNAPIENTRY lwnBootstrapLoader(const char *name);
#endif

class LwnGraphics
{
public:
    void InitLwnGraphics(void *nativeWindow, int width, int height, lwn::Format::Enum format)
    {
        using namespace lwn;

        // NULL native window is not supported.
        if (!nativeWindow)
        {
            Error("Null native window");
        }

        // LWN setup, including bootstrapping, window swap chain, and queue/command buffer
#ifdef _WIN32
        auto lwnBootstrapLoader = (BootstrapLoaderFunc)wglGetProcAddress("rq34nd2ffz");
        if (lwnBootstrapLoader == NULL)
        {
            Error("Failed proc address for lwn bootstrap function");
        }
#endif
        auto pGetProcAddress =
            (DeviceGetProcAddressFunc) lwnBootstrapLoader("lwnDeviceGetProcAddress");

        if (pGetProcAddress == NULL)
        {
            Error("Failed proc address for lwn GetProcAddress function");
        }

        // Load C procs in addition to CPP procs since Glslc depends upon the C interface
        lwnLoadCProcs(nullptr, (PFNLWNDEVICEGETPROCADDRESSPROC) pGetProcAddress);
        lwnLoadCPPProcs(nullptr, pGetProcAddress);

        DeviceBuilder deviceBuilder;
        deviceBuilder.SetDefaults();

        if (!g_args.m_debug) {
            mDevice.Initialize(&deviceBuilder);
        } else {
            deviceBuilder.SetFlags(DeviceFlagBits::DEBUG_ENABLE_LEVEL_4);
            mDevice.Initialize(&deviceBuilder);
            mDevice.InstallDebugCallback(DebugCallback, this, true);
        }

        lwnLoadCPPProcs(&mDevice, pGetProcAddress);

        QueueBuilder queueBuilder;
        queueBuilder.SetDefaults().SetDevice(&mDevice);
        mQueue.Initialize(&queueBuilder);

        // Allocate window textures
        mNumDisplayBuffers = g_args.m_tripleBuffer ? 3 : 2;
        if (g_args.m_debug) {
            printf("Number of buffers in swapchain: %d\n", mNumDisplayBuffers);
        }

        TextureBuilder textureBuilder;
        textureBuilder.SetDefaults()
                      .SetDevice(&mDevice)
                      .SetFormat(format) // Win32: Match the format of the donor OpenGL context.
                      .SetSize2D(width, height);
        if (g_args.m_compress) {
            textureBuilder.SetFlags(TextureFlags::DISPLAY | TextureFlags::COMPRESSIBLE);
            if (g_args.m_debug) {
                printf("Enabling compression for display buffers\n");
            }
        } else {
            textureBuilder.SetFlags(TextureFlags::DISPLAY);
            if (g_args.m_debug) {
                printf("Disabling compression for display buffers\n");
            }
        }

        size_t texSize   = textureBuilder.GetStorageSize();
        size_t alignment = textureBuilder.GetStorageAlignment();

        mWindowTexturePool = new LwnMemoryPool(&mDevice,
                             mNumDisplayBuffers * (texSize + alignment),
                             MemoryPoolFlags::GPU_CACHED | MemoryPoolFlags::CPU_NO_ACCESS |
                             MemoryPoolFlags::COMPRESSIBLE);

        for (int i = 0; i < mNumDisplayBuffers; i++) {
            textureBuilder.SetStorage(&mWindowTexturePool->m_apiPool,
                                       mWindowTexturePool->Allocate(texSize, alignment));
            mWindowTextures[i].Initialize(&textureBuilder);
        }

        // Create Window & sync
        WindowBuilder windowBuilder;

        Texture* windowTexturePtrs[MAX_DISPLAY_BUFFERS];
        for (int i = 0; i < mNumDisplayBuffers; i++)
        {
            windowTexturePtrs[i] = &mWindowTextures[i];
        }
        windowBuilder.SetDefaults()
                     .SetDevice(&mDevice)
                     .SetTextures(mNumDisplayBuffers, windowTexturePtrs);
        windowBuilder.SetNativeWindow(nativeWindow);

        mWindow.Initialize(&windowBuilder);
        mWindowTextureAvailableSync.Initialize(&mDevice);

        // Create command buffer and control memory
        mCommandBuffer.Initialize(&mDevice);
        mCommandBuffer.SetMemoryCallback(OutOfMemoryCallback);
        mCommandBuffer.SetMemoryCallbackData(this);

        // Create control buffer and command buffer with half the allocated size. When
        // out of memory, switch to the other half. See out of memory functions that
        // toggle the memory offsets.
        mControlMemory.resize(2 * CC_SIZE);
        mControlMemoryOffset = 0;
        mCommandBuffer.AddControlMemory(mControlMemory.data() + mControlMemoryOffset, CC_SIZE);

        mCommandMemoryPool = new LwnMemoryPool(&mDevice, 2 * CB_SIZE,
                                MemoryPoolFlags::CPU_UNCACHED | MemoryPoolFlags::GPU_UNCACHED);
        mCommandMemoryOffset = 0;
        mCommandBuffer.AddCommandMemory(&mCommandMemoryPool->m_apiPool, mCommandMemoryOffset, CB_SIZE);
    }

    ~LwnGraphics()
    {
        mQueue.Finish();
        mCommandBuffer.Finalize();
        mWindow.Finalize();

        for (int i = 0; i < mNumDisplayBuffers; i++)
        {
            mWindowTextures[i].Finalize();
        }
        delete mWindowTexturePool;
        delete mCommandMemoryPool;

        mQueue.Finalize();
        mDevice.Finalize();
    }

private:
    static constexpr int MAX_DISPLAY_BUFFERS = 4;

    // A size large enough to fully satisfy command or control memory demands at any given point.
    static constexpr size_t CB_SIZE = 65536;
    static constexpr size_t CC_SIZE =  4096;    // Requires a much smaller buffer

    static void LWNAPIENTRY OutOfMemoryCallback(lwn::CommandBuffer* cmdBuf,
                                                lwn::CommandBufferMemoryEvent::Enum event,
                                                size_t minSize, void* callbackData)
    {
        LwnGraphics* self = static_cast<LwnGraphics*>(callbackData);
        self->HandleOutOfMemory(cmdBuf, event, minSize);
    }

    void HandleOutOfMemory(lwn::CommandBuffer* cmdBuf, lwn::CommandBufferMemoryEvent::Enum event,
                           size_t minSize)
    {
        // A chunk of memory that is double CB_SIZE is allocated for each type of command buffer
        // memory. To satisfy the allocation requests, ping-pong between each half of each chunk
        // of memory.
        if (event == lwn::CommandBufferMemoryEvent::OUT_OF_COMMAND_MEMORY) {
            mCommandMemoryOffset = CB_SIZE - mCommandMemoryOffset;
            cmdBuf->AddCommandMemory(&mCommandMemoryPool->m_apiPool, mCommandMemoryOffset, CB_SIZE);
        } else {
            mControlMemoryOffset = CC_SIZE - mControlMemoryOffset;
            cmdBuf->AddControlMemory(mControlMemory.data() + mControlMemoryOffset, CC_SIZE);
        }
    }

    static void LWNAPIENTRY DebugCallback(lwn::DebugCallbackSource::Enum source,
                                          lwn::DebugCallbackType::Enum type, int id,
                                          lwn::DebugCallbackSeverity::Enum severity,
                                          const char* message, void* userParam)
    {
        std::cout << "LWN debug: " << message << std::endl;
    }

protected:
    int mNumDisplayBuffers;
    lwn::Device mDevice;
    lwn::Queue mQueue;
    lwn::Texture mWindowTextures[MAX_DISPLAY_BUFFERS];
    lwn::Window mWindow;
    lwn::Sync mWindowTextureAvailableSync;
    lwn::CommandBuffer mCommandBuffer;
    lwn::CommandHandle mCommandHandle = 0;
    std::vector<uint8_t> mControlMemory;
    int mControlMemoryOffset = 0;

    LwnMemoryPool *mWindowTexturePool;
    LwnMemoryPool *mCommandMemoryPool;

    int mCommandMemoryOffset = 0;
};
