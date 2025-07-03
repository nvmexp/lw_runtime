#pragma once

/*
 * Copyright (c) 2018, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <algorithm>
#include <assert.h>
#include <iostream>
#include <vector>

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

// Exporting this special variable causes the LWPU drivers to render with our
// GPU on Optimus systems without any special application profiles.
extern "C" {
   _declspec(dllexport) unsigned long LwOptimusEnablement = 0x00000001;
}

#include "lwn/lwn.h"
#include "lwn/lwn_Cpp.h"
#include "lwn/lwn_CppFuncPtr.h"
#include "lwn/lwn_FuncPtrImpl.h"
#include "lwn/lwn_CppFuncPtrImpl.h"
#include "lwn/lwn_CppMethods.h"
#include "lwn/lwn_FuncPtrGlobal.h"

#include "lwnGlslc.h"

// The debug layer will warn about the creation of a memory pool without backing system memory,
// which is only required on the NX device.
#define USE_DEBUG_LAYER 0

template <typename T1, typename T2>
T1 Align(T1 value, T2 alignSize)
{
        alignSize -= 1;
            return (value + alignSize) & ~(alignSize);
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
        memPoolBuilder.SetStorage(nullptr, size);

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
    }

    lwn::MemoryPool m_apiPool;
    lwn::MemoryPoolFlags m_flags;

private:
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

class LwnTexture
{
public:
    LwnTexture(lwn::Device *device, LwnMemoryPool *pool, lwn::Format texFormat, unsigned int width,
                                            unsigned int height, void *data, lwn::TextureFlags flags = 0)
    {
        lwn::TextureBuilder textureBuilder;
        textureBuilder.SetDefaults();
        textureBuilder.SetDevice(device);
        textureBuilder.SetTarget(lwn::TextureTarget::TARGET_2D);
        textureBuilder.SetSize2D(width, height);
        textureBuilder.SetLevels(1);
        textureBuilder.SetFormat(texFormat);
        textureBuilder.SetFlags(flags);

        size_t alignment = textureBuilder.GetStorageAlignment();
        size_t size = textureBuilder.GetStorageSize();

        size_t offset = pool->Allocate(size, alignment);
        textureBuilder.SetStorage(&pool->m_apiPool, offset);

        m_apiTexture.Initialize(&textureBuilder);
        m_gpuAddr = m_apiTexture.GetTextureAddress();

        if (data != NULL)
        {
            // Write texels will fail for COMPRESSIBLE textures.
            assert ((static_cast<uint32_t>(flags) & lwn::TextureFlags::COMPRESSIBLE) == 0);

            lwn::CopyRegion copyRegion = {
                0, 0, 0,                        // xoffset, yoffset, zoffset
                (int) width, (int) height, 1,   // width, height, depth
            };
            // Don't need TextureView when incoming data is fully compatible
            m_apiTexture.WriteTexels(NULL, &copyRegion, data);
        }
    }

    lwn::Texture       m_apiTexture;
    lwn::BufferAddress m_gpuAddr;
};

enum LwnDescriptorType {SAMPLER, TEXTURE};

template <class T>
class LwnDescriptorPool
{
public:
    LwnDescriptorPool(lwn::Device *device, LwnDescriptorType descType, int allocCount)
    {
        int reservedCount, descriptorSize;

        device->GetInteger(descType == SAMPLER ? lwn::DeviceInfo::SAMPLER_DESCRIPTOR_SIZE :
                                                 lwn::DeviceInfo::TEXTURE_DESCRIPTOR_SIZE,
                                                 &descriptorSize);

        device->GetInteger(descType == SAMPLER ? lwn::DeviceInfo::RESERVED_SAMPLER_DESCRIPTORS :
                                                 lwn::DeviceInfo::RESERVED_TEXTURE_DESCRIPTORS,
                                                 &reservedCount);

        allocCount += reservedCount;
        int allocSize = Align(allocCount * descriptorSize, 32);     // Bytes

        m_descMemPool = new LwnMemoryPool(device, allocSize,  MemoryPoolFlags::CPU_UNCACHED |
                                                              MemoryPoolFlags::GPU_CACHED);
        m_nextId = reservedCount;
        m_maxId  = allocCount;

        m_apiDescPool.Initialize(&m_descMemPool->m_apiPool, 0, allocCount);
    }

    int Alloc()
    {
        assert(m_nextId < m_maxId);
        return m_nextId++;
    }

    T m_apiDescPool;

private:
    int m_nextId;
    int m_maxId;

    LwnMemoryPool   *m_descMemPool;
};

// Code below is copied from: High dynamic range / wide color gamut on LWN demo application.

class LwnGraphics
{
public:
    void InitLwnGraphics(int width, int height, lwn::Format::Enum format)
    {
        // LWN setup, including bootstrapping, window swap chain, and queue/command buffer
        using namespace lwn;
        auto pBootstrapLoader = (BootstrapLoaderFunc)wglGetProcAddress("rq34nd2ffz");
        if (pBootstrapLoader == NULL)
        {
            Error("Failed proc address for lwn bootstrap function");
        }

        auto pGetProcAddress =
            (DeviceGetProcAddressFunc)pBootstrapLoader("lwnDeviceGetProcAddress");
        if (pGetProcAddress == NULL)
        {
            Error("Failed proc address for lwn GetProcAddress function");
        }

        // Load C procs in addition to CPP procs since Glslc depends upon the C interface
        lwnLoadCProcs(nullptr, (PFNLWNDEVICEGETPROCADDRESSPROC) pGetProcAddress);
        lwnLoadCPPProcs(nullptr, pGetProcAddress);

        DeviceBuilder deviceBuilder;
        deviceBuilder.SetDefaults();
#if USE_DEBUG_LAYER
        deviceBuilder.SetFlags(DeviceFlagBits::DEBUG_ENABLE_LEVEL_4);
#endif
        mDevice.Initialize(&deviceBuilder);
#if USE_DEBUG_LAYER
        mDevice.InstallDebugCallback(DebugCallback, this, true);
#endif
        lwnLoadCPPProcs(&mDevice, pGetProcAddress);

        QueueBuilder queueBuilder;
        queueBuilder.SetDefaults().SetDevice(&mDevice);
        mQueue.Initialize(&queueBuilder);
        TextureBuilder textureBuilder;
        textureBuilder.SetDefaults()
                      .SetDevice(&mDevice)
                      .SetFormat(format) // Match the format of the donor OpenGL context.
                      .SetSize2D(width, height)
                      .SetFlags(TextureFlags::DISPLAY);
        size_t texSize = textureBuilder.GetStorageSize();
        MemoryPoolBuilder memPoolBuilder;
        static constexpr int NUM_BUFFERS = 2;
        // NON-PORTABLE: Windows doesn't require system memory to back memory pools, but NX would
        // need it.
        memPoolBuilder.SetDefaults()
                      .SetDevice(&mDevice)
                      .SetFlags(MemoryPoolFlags::GPU_CACHED | MemoryPoolFlags::CPU_NO_ACCESS)
                      .SetStorage(nullptr, texSize * NUM_BUFFERS);
        mWindowTexturePool.Initialize(&memPoolBuilder);
        for (int i = 0; i < NUM_BUFFERS; ++i) {
            textureBuilder.SetStorage(&mWindowTexturePool, i * texSize);
            mWindowTextures[i].Initialize(&textureBuilder);
        }
        WindowBuilder windowBuilder;
        Texture* const windowTexturePtrs[2] = {mWindowTextures, mWindowTextures + 1};
        // SetNativeWindow with a non-NULL HWND is not lwrrently supported. Instead, rely on the
        // window associated with the current OpenGL context.
        windowBuilder.SetDefaults()
                     .SetDevice(&mDevice)
                     .SetTextures(2, windowTexturePtrs);

        windowBuilder.SetNativeWindow(WindowFromDC(wglGetLwrrentDC()));

        mWindow.Initialize(&windowBuilder);
        mWindowTextureAvailableSync.Initialize(&mDevice);
        mControlMemory.resize(2 * CB_SIZE);
        memPoolBuilder.SetDefaults()
                      .SetDevice(&mDevice)
                      .SetFlags(MemoryPoolFlags::CPU_UNCACHED | MemoryPoolFlags::GPU_UNCACHED)
                      // See CB_SIZE and OOM callback definitions.
                      .SetStorage(nullptr, 2 * CB_SIZE);
        mCommandMemory.Initialize(&memPoolBuilder);
        mCommandBuffer.Initialize(&mDevice);
        mCommandBuffer.SetMemoryCallback(OutOfMemoryCallback);
        mCommandBuffer.SetMemoryCallbackData(this);
    }

    ~LwnGraphics()
    {
        mQueue.Finish();
        mCommandBuffer.Finalize();
        mCommandMemory.Finalize();
        mWindow.Finalize();
        mWindowTextures[1].Finalize();
        mWindowTextures[0].Finalize();
        mWindowTexturePool.Finalize();
        mQueue.Finalize();
        mDevice.Finalize();
    }

private:
    // A size large enough to fully satisfy command or control memory demands at any given point.
    static constexpr size_t CB_SIZE = 65536;

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
            cmdBuf->AddCommandMemory(&mCommandMemory, mCommandMemoryOffset, CB_SIZE);
            mCommandMemoryOffset = CB_SIZE - mCommandMemoryOffset;
        } else {
            cmdBuf->AddControlMemory(mControlMemory.data() + mControlMemoryOffset, CB_SIZE);
            mControlMemoryOffset = CB_SIZE - mControlMemoryOffset;
        }
    }

#if USE_DEBUG_LAYER
    static void LWNAPIENTRY DebugCallback(lwn::DebugCallbackSource::Enum source,
                                          lwn::DebugCallbackType::Enum type, int id,
                                          lwn::DebugCallbackSeverity::Enum severity,
                                          const char* message, void* userParam)
    {
        std::cout << "LWN debug: " << message << std::endl;
    }
#endif

protected:
    lwn::Device mDevice;
    lwn::Queue mQueue;
    lwn::MemoryPool mWindowTexturePool;
    lwn::Texture mWindowTextures[2];
    lwn::Window mWindow;
    lwn::Sync mWindowTextureAvailableSync;
    lwn::CommandBuffer mCommandBuffer;
    lwn::CommandHandle mCommandHandle = 0;
    std::vector<uint8_t> mControlMemory;
    int mControlMemoryOffset = 0;
    lwn::MemoryPool mCommandMemory;
    int mCommandMemoryOffset = 0;
};
