/*
 * Copyright (c) 2016 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

//
// lwn_interface_cpp.cpp
//
// This module includes lwntest utility code specific to the native C++
// interface for LWN.
//
#include "lwntest_cpp.h"
#include "lwn_utils.h"
#include "lwn/lwn_CppFuncPtr.h"
#include "lwn/lwn_CppFuncPtrImpl.h"

#include "lwnTest/lwnTest_Mislwtils.h"

// Utility code to reload the native C++ interface for a new device or when
// switching back to an old one.
void ReloadCppInterface(LWNdevice *device, PFNLWNDEVICEGETPROCADDRESSPROC getProcAddress)
{
    lwn::objects::Device *cppdev = reinterpret_cast<lwn::objects::Device *>(device);
    lwn::objects::DeviceGetProcAddressFunc cppGetProcAddress =
        reinterpret_cast<lwn::objects::DeviceGetProcAddressFunc>(getProcAddress);
    lwnLoadCPPProcs(cppdev, cppGetProcAddress);
}

// "Create" and "Free" methods for the native C++ interface that use our C
// interface "create" functions via reinterpret_cast.
namespace lwn {

Queue *Device::CreateQueue()
{
    LWNdevice *cthis = reinterpret_cast<LWNdevice *>(this);
    LWNqueue *cobj = lwnDeviceCreateQueue(cthis);
    return reinterpret_cast<Queue *>(cobj);
}

CommandBuffer *Device::CreateCommandBuffer()
{
    LWNdevice *cthis = reinterpret_cast<LWNdevice *>(this);
    LWNcommandBuffer *cobj = lwnDeviceCreateCommandBuffer(cthis);
    return reinterpret_cast<CommandBuffer *>(cobj);
}

BlendState *Device::CreateBlendState()
{
    LWNdevice *cthis = reinterpret_cast<LWNdevice *>(this);
    LWNblendState *cobj = lwnDeviceCreateBlendState(cthis);
    return reinterpret_cast<BlendState *>(cobj);
}

ChannelMaskState *Device::CreateChannelMaskState()
{
    LWNdevice *cthis = reinterpret_cast<LWNdevice *>(this);
    LWNchannelMaskState *cobj = lwnDeviceCreateChannelMaskState(cthis);
    return reinterpret_cast<ChannelMaskState *>(cobj);
}

ColorState *Device::CreateColorState()
{
    LWNdevice *cthis = reinterpret_cast<LWNdevice *>(this);
    LWNcolorState *cobj = lwnDeviceCreateColorState(cthis);
    return reinterpret_cast<ColorState *>(cobj);
}

DepthStencilState *Device::CreateDepthStencilState()
{
    LWNdevice *cthis = reinterpret_cast<LWNdevice *>(this);
    LWNdepthStencilState *cobj = lwnDeviceCreateDepthStencilState(cthis);
    return reinterpret_cast<DepthStencilState *>(cobj);
}

MultisampleState *Device::CreateMultisampleState()
{
    LWNdevice *cthis = reinterpret_cast<LWNdevice *>(this);
    LWNmultisampleState *cobj = lwnDeviceCreateMultisampleState(cthis);
    return reinterpret_cast<MultisampleState *>(cobj);
}

PolygonState *Device::CreatePolygonState()
{
    LWNdevice *cthis = reinterpret_cast<LWNdevice *>(this);
    LWNpolygonState *cobj = lwnDeviceCreatePolygonState(cthis);
    return reinterpret_cast<PolygonState *>(cobj);
}

VertexAttribState *Device::CreateVertexAttribState()
{
    LWNdevice *cthis = reinterpret_cast<LWNdevice *>(this);
    LWLwertexAttribState *cobj = lwnDeviceCreateVertexAttribState(cthis);
    return reinterpret_cast<VertexAttribState *>(cobj);
}

VertexStreamState *Device::CreateVertexStreamState()
{
    LWNdevice *cthis = reinterpret_cast<LWNdevice *>(this);
    LWLwertexStreamState *cobj = lwnDeviceCreateVertexStreamState(cthis);
    return reinterpret_cast<VertexStreamState *>(cobj);
}

Program *Device::CreateProgram()
{
    LWNdevice *cthis = reinterpret_cast<LWNdevice *>(this);
    LWNprogram *cobj = lwnDeviceCreateProgram(cthis);
    return reinterpret_cast<Program *>(cobj);
}

MemoryPool *Device::CreateMemoryPool(void *memory, size_t size, MemoryPoolType poolType)
{
    LWNdevice *cthis = reinterpret_cast<LWNdevice *>(this);
    LWNmemoryPool *cobj = lwnDeviceCreateMemoryPool(cthis, memory, size, poolType);
    return reinterpret_cast<MemoryPool *>(cobj);
}

MemoryPool *Device::CreateMemoryPoolWithFlags(void *memory, size_t size, MemoryPoolFlags poolFlags)
{
    LWNdevice *cthis = reinterpret_cast<LWNdevice *>(this);
    LWNmemoryPool *cobj = lwnDeviceCreateMemoryPool(cthis, memory, size, LWNmemoryPoolFlags(LWNbitfield(poolFlags)));
    return reinterpret_cast<MemoryPool *>(cobj);
}

BufferBuilder *Device::CreateBufferBuilder()
{
    LWNdevice *cthis = reinterpret_cast<LWNdevice *>(this);
    LWNbufferBuilder *cobj = lwnDeviceCreateBufferBuilder(cthis);
    return reinterpret_cast<BufferBuilder *>(cobj);
}

TextureBuilder *Device::CreateTextureBuilder()
{
    LWNdevice *cthis = reinterpret_cast<LWNdevice *>(this);
    LWNtextureBuilder *cobj = lwnDeviceCreateTextureBuilder(cthis);
    return reinterpret_cast<TextureBuilder *>(cobj);
}

WindowBuilder *Device::CreateWindowBuilder()
{
    LWNdevice *cthis = reinterpret_cast<LWNdevice *>(this);
    LWNwindowBuilder *cobj = lwnDeviceCreateWindowBuilder(cthis);
    return reinterpret_cast<WindowBuilder *>(cobj);
}

SamplerBuilder *Device::CreateSamplerBuilder()
{
    LWNdevice *cthis = reinterpret_cast<LWNdevice *>(this);
    LWNsamplerBuilder *cobj = lwnDeviceCreateSamplerBuilder(cthis);
    return reinterpret_cast<SamplerBuilder *>(cobj);
}

Sync *Device::CreateSync()
{
    LWNdevice *cthis = reinterpret_cast<LWNdevice *>(this);
    LWNsync *cobj = lwnDeviceCreateSync(cthis);
    return reinterpret_cast<Sync *>(cobj);
}

Sampler *SamplerBuilder::CreateSampler()
{
    LWNsamplerBuilder *cthis = reinterpret_cast<LWNsamplerBuilder *>(this);
    LWNsampler *cobj = lwnSamplerBuilderCreateSampler(cthis);
    return reinterpret_cast<Sampler *>(cobj);
}

Buffer *BufferBuilder::CreateBufferFromPool(MemoryPool *storage, LWNuintptr offset, LWNsizeiptr size)
{
    LWNbufferBuilder *cthis = reinterpret_cast<LWNbufferBuilder *>(this);
    LWNmemoryPool *cpool = reinterpret_cast<LWNmemoryPool *>(storage);
    LWNbuffer *cobj = lwnBufferBuilderCreateBufferFromPool(cthis, cpool, offset, size);
    return reinterpret_cast<Buffer *>(cobj);
}

Texture *TextureBuilder::CreateTextureFromPool(MemoryPool *storage, LWNuintptr offset)
{
    LWNtextureBuilder *cthis = reinterpret_cast<LWNtextureBuilder *>(this);
    LWNmemoryPool *cpool = reinterpret_cast<LWNmemoryPool *>(storage);
    LWNtexture *cobj = lwnTextureBuilderCreateTextureFromPool(cthis, cpool, offset);
    return reinterpret_cast<Texture *>(cobj);
}

Window *WindowBuilder::CreateWin()
{
    LWNwindowBuilder *cthis = reinterpret_cast<LWNwindowBuilder *>(this);
    LWNwindow *cobj = lwnWindowBuilderCreateWindow(cthis);
    return reinterpret_cast<Window *>(cobj);
}

void Device::Free()
{
    LWNdevice *cthis = reinterpret_cast<LWNdevice *>(this);
    lwnDeviceFree(cthis);
}
void Queue::Free()
{
    LWNqueue *cthis = reinterpret_cast<LWNqueue *>(this);
    lwnQueueFree(cthis);
}
void CommandBuffer::Free()
{
    LWNcommandBuffer *cthis = reinterpret_cast<LWNcommandBuffer *>(this);
    lwnCommandBufferFree(cthis);
}
void BlendState::Free()
{
    LWNblendState *cthis = reinterpret_cast<LWNblendState *>(this);
    lwnBlendStateFree(cthis);
}
void ChannelMaskState::Free()
{
    LWNchannelMaskState *cthis = reinterpret_cast<LWNchannelMaskState *>(this);
    lwnChannelMaskStateFree(cthis);
}
void ColorState::Free()
{
    LWNcolorState *cthis = reinterpret_cast<LWNcolorState *>(this);
    lwnColorStateFree(cthis);
}
void DepthStencilState::Free()
{
    LWNdepthStencilState *cthis = reinterpret_cast<LWNdepthStencilState *>(this);
    lwnDepthStencilStateFree(cthis);
}
void MultisampleState::Free()
{
    LWNmultisampleState *cthis = reinterpret_cast<LWNmultisampleState *>(this);
    lwnMultisampleStateFree(cthis);
}
void PolygonState::Free()
{
    LWNpolygonState *cthis = reinterpret_cast<LWNpolygonState *>(this);
    lwnPolygonStateFree(cthis);
}
void Program::Free()
{
    LWNprogram *cthis = reinterpret_cast<LWNprogram *>(this);
    lwnProgramFree(cthis);
}
void MemoryPool::Free()
{
    LWNmemoryPool *cthis = reinterpret_cast<LWNmemoryPool *>(this);
    lwnMemoryPoolFree(cthis);
}
void BufferBuilder::Free()
{
    LWNbufferBuilder *cthis = reinterpret_cast<LWNbufferBuilder *>(this);
    lwnBufferBuilderFree(cthis);
}
void Buffer::Free()
{
    LWNbuffer *cthis = reinterpret_cast<LWNbuffer *>(this);
    lwnBufferFree(cthis);
}
void Texture::Free()
{
    LWNtexture *cthis = reinterpret_cast<LWNtexture *>(this);
    lwnTextureFree(cthis);
}
void TextureBuilder::Free()
{
    LWNtextureBuilder *cthis = reinterpret_cast<LWNtextureBuilder *>(this);
    lwnTextureBuilderFree(cthis);
}
void SamplerBuilder::Free()
{
    LWNsamplerBuilder *cthis = reinterpret_cast<LWNsamplerBuilder *>(this);
    lwnSamplerBuilderFree(cthis);
}
void Sampler::Free()
{
    LWNsampler *cthis = reinterpret_cast<LWNsampler *>(this);
    lwnSamplerFree(cthis);
}
void WindowBuilder::Free()
{
    LWNwindowBuilder *cthis = reinterpret_cast<LWNwindowBuilder *>(this);
    lwnWindowBuilderFree(cthis);
}
void Window::Free()
{
    LWNwindow *cthis = reinterpret_cast<LWNwindow *>(this);
    lwnWindowFree(cthis);
}
void Sync::Free()
{
    LWNsync *cthis = reinterpret_cast<LWNsync *>(this);
    lwnSyncFree(cthis);
}
void VertexAttribState::Free()
{
    LWLwertexAttribState *cthis = reinterpret_cast<LWLwertexAttribState *>(this);
    lwlwertexAttribStateFree(cthis);
}
void VertexStreamState::Free()
{
    LWLwertexStreamState *cthis = reinterpret_cast<LWLwertexStreamState *>(this);
    lwlwertexStreamStateFree(cthis);
}
Event* EventBuilder::CreateEvent()
{
    LWNeventBuilder *cthis = reinterpret_cast<LWNeventBuilder*>(this);
    LWNevent *cobj = new LWNevent;
    lwnEventInitialize(cobj, cthis);
    return reinterpret_cast<Event *>(cobj);
}
void Event::Free()
{
    LWNevent *cthis = reinterpret_cast<LWNevent *>(this);
    lwnEventFinalize(cthis);
    delete cthis;
}

} // namespace lwn

//////////////////////////////////////////////////////////////////////////

// Implementation of the pure C++ interfaces for miscellaneous utility code in
// lwnTest_Mislwtils.h.  We don't use the implementation file because it is
// built in code using the C interface only.
namespace lwnTest {

lwn::Buffer *AllocAndFillBuffer(lwn::Device *device, lwn::Queue *queue,
                                lwn::CommandBuffer *cmdBuf,
                                lwnUtil::MemoryPoolAllocator& allocator,
                                const void *data, int sizeofdata,
                                lwnUtil::BufferAlignBits alignBits, bool useCopy)
{
    LWNdevice *cdevice = reinterpret_cast<LWNdevice *>(device);
    LWNqueue *cqueue = reinterpret_cast<LWNqueue *>(queue);
    LWNcommandBuffer *ccmdbuf = reinterpret_cast<LWNcommandBuffer *>(cmdBuf);
    LWNbuffer *buffer = AllocAndFillBuffer(cdevice, cqueue, ccmdbuf, allocator, data, sizeofdata,
                                           alignBits, useCopy);
    return reinterpret_cast<lwn::Buffer *>(buffer);
}

// Utility method to allocate and fill a <format> 2D texture object
// belonging to device <device>, copying <width> * <height> elements of
// size <sizeofelement> bytes from <data> on <queue>.
lwn::Texture *AllocAndFillTexture2D(lwn::Device *device, lwn::Queue *queue,
                                    lwn::CommandBuffer *cmdBuf,
                                    lwnUtil::MemoryPoolAllocator& tex_allocator,
                                    lwnUtil::MemoryPoolAllocator& coherent_allocator,
                                    const void *data, int sizeoftexel,
                                    int width, int height, lwn::Format format)
{
    LWNdevice *cdevice = reinterpret_cast<LWNdevice *>(device);
    LWNqueue *cqueue = reinterpret_cast<LWNqueue *>(queue);
    LWNcommandBuffer *ccmdbuf = reinterpret_cast<LWNcommandBuffer *>(cmdBuf);
    LWNformat cformat = LWNformat(int(format));
    LWNtexture *texture = AllocAndFillTexture2D(cdevice, cqueue, ccmdbuf, tex_allocator,
                                                coherent_allocator, data, sizeoftexel,
                                                width, height, cformat);
    return reinterpret_cast<lwn::Texture *>(texture);
}

// Utility method to set both viewport and scissor to the region defined by
// <cell_x>, <cell_y> with <padding> pixels of padding. Commands are issued
// on <cmd>.
void SetCellViewportScissorPadded(lwn::CommandBuffer *cmd, int cell_x, int cell_y,
                                  int padding)
{
    LWNcommandBuffer *ccmdbuf = reinterpret_cast<LWNcommandBuffer *>(cmd);
    SetCellViewportScissorPadded(ccmdbuf, cell_x, cell_y, padding);
}

// Utility method to read a non-compressed texture back from <tex> to system 
// memory <texData>.
void ReadTextureData(lwn::Device *device, lwn::Queue *queue, lwn::CommandBuffer *cmd,
                     lwn::Texture *tex, int width, int height, int depth, int bpp,
                     void *texData)
{
    LWNdevice *cdevice = reinterpret_cast<LWNdevice *>(device);
    LWNqueue *cqueue = reinterpret_cast<LWNqueue *>(queue);
    LWNcommandBuffer *ccmdbuf = reinterpret_cast<LWNcommandBuffer *>(cmd);
    LWNtexture *ctexture = reinterpret_cast<LWNtexture *>(tex);
    ReadTextureData(cdevice, cqueue, ccmdbuf, ctexture, width, height, depth, bpp, texData);
}


// Utility method to read an RGBA8 texture back from <tex> to system memory
// <texData>.
void ReadTextureDataRGBA8(lwn::Device *device, lwn::Queue *queue, lwn::CommandBuffer *cmd,
                          lwn::Texture *tex, int width, int height, void *texData)
{
    ReadTextureData(device, queue, cmd, tex, width, height, 1, 4, texData);
}

} // namespace lwnTest
