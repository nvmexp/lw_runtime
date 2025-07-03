/*
* Copyright (c) 2015, Lwpu Corporation.  All rights reserved.
*
* THE INFORMATION CONTAINED HEREIN IS PROPRIETARY AND CONFIDENTIAL TO
* LWPU, CORPORATION.  USE, REPRODUCTION OR DISCLOSURE TO ANY THIRD PARTY
* IS SUBJECT TO WRITTEN PRE-APPROVAL BY LWPU, CORPORATION.
*
*
*/
#ifndef __lwnTest_Objects_h__
#define __lwnTest_Objects_h__

#include "lwnUtil/lwnUtil_Interface.h"

//////////////////////////////////////////////////////////////////////////
//
//                   LWN OBJECT ALLOCATION FUNCTIONS
//
// We provide Create/Free functions to allocate LWN API objects, emulating
// driver-side object creation and deletion.  These APIs also track creation
// and deletion for automatic end-of-test resource cleanup.  These functions
// are also used to provide auto-registration support for textures and
// samplers (below).
//
LWNdevice *lwnCreateDevice();
LWNqueue *lwnDeviceCreateQueue(LWNdevice *device);
LWNcommandBuffer *lwnDeviceCreateCommandBuffer(LWNdevice *device);
LWNblendState *lwnDeviceCreateBlendState(LWNdevice *device);
LWNchannelMaskState *lwnDeviceCreateChannelMaskState(LWNdevice *device);
LWNcolorState *lwnDeviceCreateColorState(LWNdevice *device);
LWNdepthStencilState *lwnDeviceCreateDepthStencilState(LWNdevice *device);
LWNmultisampleState *lwnDeviceCreateMultisampleState(LWNdevice *device);
LWNpolygonState *lwnDeviceCreatePolygonState(LWNdevice *device);
LWLwertexAttribState *lwnDeviceCreateVertexAttribState(LWNdevice *device);
LWLwertexStreamState *lwnDeviceCreateVertexStreamState(LWNdevice *device);
LWNprogram *lwnDeviceCreateProgram(LWNdevice *device);
LWNmemoryPool *lwnDeviceCreateMemoryPool(LWNdevice *device, void *memory, size_t size, LWNmemoryPoolFlags poolFlags);
LWNbufferBuilder *lwnDeviceCreateBufferBuilder(LWNdevice *device);
LWNtextureBuilder *lwnDeviceCreateTextureBuilder(LWNdevice *device);
LWNsamplerBuilder *lwnDeviceCreateSamplerBuilder(LWNdevice *device);
LWNsync *lwnDeviceCreateSync(LWNdevice *device);
LWNsampler *lwnSamplerBuilderCreateSampler(LWNsamplerBuilder *builder);
LWNbuffer *lwnBufferBuilderCreateBufferFromPool(LWNbufferBuilder *builder, LWNmemoryPool *storage, ptrdiff_t offset, size_t size);
LWNtexture *lwnTextureBuilderCreateTextureFromPool(LWNtextureBuilder *builder, LWNmemoryPool *storage, ptrdiff_t offset);
LWNwindowBuilder *lwnDeviceCreateWindowBuilder(LWNdevice *device);
LWNwindow *lwnWindowBuilderCreateWindow(LWNwindowBuilder *builder);

void lwnDeviceFree(LWNdevice *object);
void lwnQueueFree(LWNqueue *object);
void lwnCommandBufferFree(LWNcommandBuffer *object);
void lwnBlendStateFree(LWNblendState *object);
void lwnChannelMaskStateFree(LWNchannelMaskState *object);
void lwnColorStateFree(LWNcolorState *object);
void lwnDepthStencilStateFree(LWNdepthStencilState *object);
void lwnMultisampleStateFree(LWNmultisampleState *object);
void lwnPolygonStateFree(LWNpolygonState *object);
void lwlwertexAttribStateFree(LWLwertexAttribState *object);
void lwlwertexStreamStateFree(LWLwertexStreamState *object);
void lwnProgramFree(LWNprogram *object);
void lwnMemoryPoolFree(LWNmemoryPool *object);
void lwnBufferBuilderFree(LWNbufferBuilder *object);
void lwnBufferFree(LWNbuffer *object);
void lwnTextureFree(LWNtexture *object);
void lwnTextureBuilderFree(LWNtextureBuilder *object);
void lwnSamplerBuilderFree(LWNsamplerBuilder *object);
void lwnSamplerFree(LWNsampler *object);
void lwnSyncFree(LWNsync *object);
void lwnWindowFree(LWNwindow *object);
void lwnWindowBuilderFree(LWNwindowBuilder *object);

//////////////////////////////////////////////////////////////////////////
//
//          LWN TEXTURE AND SAMPLER AUTO-REGISTRATION
//
// The LWN API requires manual registration of textures and samplers in
// texture and sampler pools.  Our overloaded object definitions automatically
// register new allocations in the appropriate pool.  We provide extended C
// types that wrap the LWNsampler and LWNtexture objects and also save the
// registered IDs.  We also provide C APIs to query the registered IDs.
struct __LWNsamplerInternal {
    LWNsampler  sampler;
    LWNuint     lastRegisteredID;
};
struct __LWNtextureInternal {
    LWNtexture  texture;
    LWNuint     lastRegisteredTextureID;
};
LWNuint lwnTextureGetRegisteredTextureID(const LWNtexture *texture);
LWNuint lwnSamplerGetRegisteredID(const LWNsampler *sampler);

//////////////////////////////////////////////////////////////////////////
//
//       LWN MEMORY POOL BACKWARDS COMPATIBILITY HACKS
//

// Backwards-compatibility #defines to map old LWN_MEMORY_POOL_TYPE_*
// definitions to their corresponding flags (from API version 36.0).
#define LWN_MEMORY_POOL_TYPE_GPU_ONLY                               \
    LWNmemoryPoolFlags(LWN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS_BIT |    \
                       LWN_MEMORY_POOL_FLAGS_GPU_CACHED_BIT    |    \
                       LWN_MEMORY_POOL_FLAGS_COMPRESSIBLE_BIT)

#define LWN_MEMORY_POOL_TYPE_CPU_COHERENT                           \
    LWNmemoryPoolFlags(LWN_MEMORY_POOL_FLAGS_CPU_UNCACHED_BIT |     \
                       LWN_MEMORY_POOL_FLAGS_GPU_CACHED_BIT    |    \
                       LWN_MEMORY_POOL_FLAGS_COMPRESSIBLE_BIT)

#define LWN_MEMORY_POOL_TYPE_CPU_NON_COHERENT                       \
    LWNmemoryPoolFlags(LWN_MEMORY_POOL_FLAGS_CPU_CACHED_BIT |       \
                       LWN_MEMORY_POOL_FLAGS_GPU_CACHED_BIT    |    \
                       LWN_MEMORY_POOL_FLAGS_COMPRESSIBLE_BIT)

// Backwards-compatibility C++ enum class to support old MemoryPoolType::XXX
// definitions and map them to C flags (from API version 36.0).
namespace lwn {
struct MemoryPoolType {
public:
    enum Enum {
        GPU_ONLY,
        CPU_COHERENT,
        CPU_NON_COHERENT,
        LWN_ENUM_32BIT(MEMORY_POOL_TYPE),
    };
private:
    Enum m_value;
public:
    MemoryPoolType(Enum value) : m_value(value) {}
    operator LWNmemoryPoolFlags() const
    {
        switch (m_value) {
        default:
            assert(0);
        case GPU_ONLY:
            return LWN_MEMORY_POOL_TYPE_GPU_ONLY;
        case CPU_COHERENT:
            return LWN_MEMORY_POOL_TYPE_CPU_COHERENT;
        case CPU_NON_COHERENT:
            return LWN_MEMORY_POOL_TYPE_CPU_NON_COHERENT;
        }
    }
};
} // namespace lwn

//////////////////////////////////////////////////////////////////////////
//
//                    LWN WRAPPED C++ API OBJECTS
//
// We use the LWN_OVERLOAD_CPP_OBJECTS feature of the C++ headers to inject
// the standard C++ API object classes into the namespace "lwn::objects" and
// allow the application code to define its own extended object definition in
// the namespace "lwn".
//
// All of our wrapped object classes inherit from the base class in
// "lwn::objects".
//
// In addition to tracking support, we provide various additional colwenience
// methods for certain object types.
//

#if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_CPP

namespace lwn {

// Declare a (lwrrently empty) macro LWN_OVERLOAD_CPP_CLASS_COLWERTERS, which
// could be used to colwert between native C and C++ types.
#define LWN_OVERLOAD_CPP_CLASS_COLWERTERS(cpptype, ctype)

// We create forward declarations of API classes, since methods in one class
// reference other classes.
class DeviceBuilder;
class Device;
class QueueBuilder;
class Queue;
class CommandBuffer;
class BlendState;
class ChannelMaskState;
class ColorState;
class DepthStencilState;
class MultisampleState;
class PolygonState;
class VertexAttribState;
class VertexStreamState;
class Program;
class MemoryPool;
class BufferBuilder;
class Buffer;
class Texture;
class TextureBuilder;
class TextureView;
class SamplerBuilder;
class Sampler;
class Window;
class WindowBuilder;
class Sync;

class QueueBuilder : public objects::QueueBuilder
{
public:
    LWN_OVERLOAD_CPP_CLASS_COLWERTERS(QueueBuilder, LWNqueueBuilder);
};

class Queue : public objects::Queue
{
public:
    LWN_OVERLOAD_CPP_CLASS_COLWERTERS(Queue, LWNqueue);
    void Free();
};

class VertexAttribState : public objects::VertexAttribState
{
public:
    LWN_OVERLOAD_CPP_CLASS_COLWERTERS(VertexAttribState, LWLwertexAttribState);
    void Free();
};

class VertexStreamState : public objects::VertexStreamState
{
public:
    LWN_OVERLOAD_CPP_CLASS_COLWERTERS(VertexStreamState, LWLwertexStreamState);
    void Free();
};

class CommandBuffer : public objects::CommandBuffer
{
public:
    LWN_OVERLOAD_CPP_CLASS_COLWERTERS(CommandBuffer, LWNcommandBuffer);
    void Free();

    void ClearColor(LWNuint index, const LWNfloat *color, ClearColorMask mask = ClearColorMask::RGBA)
    {
        objects::CommandBuffer::ClearColor(index, color, mask);
    }
    void ClearColor(LWNuint index = 0, LWNfloat r = 0.0, LWNfloat g = 0.0, LWNfloat b = 0.0, LWNfloat a = 1.0,
                    ClearColorMask mask = ClearColorMask::RGBA)
    {
        LWNfloat color[4] = { r, g, b, a };
        objects::CommandBuffer::ClearColor(index, color, mask);
    }
    void SetViewportScissor(int x, int y, int w, int h)
    {
        SetViewport(x, y, w, h);
        SetScissor(x, y, w, h);
    }

#if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_CPP
    // The C++ interface for SetRenderTargets takes an array of pointers to
    // lwn::Texture and lwn::TextureView, but the SetRenderTargets interface
    // from lwn_Cpp.h wants arrays of lwn::object::{Texture,TextureView}
    // pointers.  We need to inject an extra layer here because C++ doesn't
    // automatically colwert these arrays of pointers.  For complex
    // inheritance, a pointer to the derived class -- Texture -- may not point
    // at the base class object.  But for our setup, it will.
    void SetRenderTargets(int numColors,
                          const Texture * const *colors,
                          const TextureView * const *colorViews,
                          const Texture *depthStencil,
                          const TextureView *depthStencilView)
    {
        const objects::Texture * const * rc_colors =
            reinterpret_cast<const objects::Texture * const *>(colors);
        const objects::TextureView * const * rc_colorViews =
            reinterpret_cast<const objects::TextureView * const *>(colorViews);
        const objects::Texture * rc_depthStencil =
            reinterpret_cast<const objects::Texture *>(depthStencil);
        const objects::TextureView * rc_depthStencilView =
            reinterpret_cast<const objects::TextureView *>(depthStencilView);
        objects::CommandBuffer::SetRenderTargets(numColors, rc_colors, rc_colorViews,
                                                 rc_depthStencil, rc_depthStencilView);
    }
#endif

    // CommandBuffer method to send commands from the VertexArrayState utility
    // class to the command buffer.
    inline void BindVertexArrayState(const VertexArrayState &arrays)
    {
        arrays.bind(reinterpret_cast<LWNcommandBuffer *>(this));
    }
};

class BlendState : public objects::BlendState
{
public:
    LWN_OVERLOAD_CPP_CLASS_COLWERTERS(BlendState, LWNblendState);
    void Free();
};

class ChannelMaskState : public objects::ChannelMaskState
{
public:
    LWN_OVERLOAD_CPP_CLASS_COLWERTERS(ChannelMaskState, LWNchannelMaskState);
    void Free();
};

class ColorState : public objects::ColorState
{
public:
    LWN_OVERLOAD_CPP_CLASS_COLWERTERS(ColorState, LWNcolorState);
    void Free();
};

class DepthStencilState : public objects::DepthStencilState
{
public:
    LWN_OVERLOAD_CPP_CLASS_COLWERTERS(DepthStencilState, LWNdepthStencilState);
    void Free();
};

class MultisampleState : public objects::MultisampleState
{
public:
    LWN_OVERLOAD_CPP_CLASS_COLWERTERS(MultisampleState, LWNmultisampleState);
    void Free();
};

class PolygonState : public objects::PolygonState
{
public:
    LWN_OVERLOAD_CPP_CLASS_COLWERTERS(PolygonState, LWNpolygonState);
    void Free();
};

class Program : public objects::Program
{
public:
    LWN_OVERLOAD_CPP_CLASS_COLWERTERS(Program, LWNprogram);
    void Free();
};

class MemoryPoolBuilder : public objects::MemoryPoolBuilder
{
public:
    LWN_OVERLOAD_CPP_CLASS_COLWERTERS(MemoryPoolBuilder, LWNmemoryPoolBuilder);
};

class MemoryPool : public objects::MemoryPool
{
public:
    LWN_OVERLOAD_CPP_CLASS_COLWERTERS(MemoryPool, LWNmemoryPool);
    void Free();
};

class BufferBuilder : public objects::BufferBuilder
{
public:
    LWN_OVERLOAD_CPP_CLASS_COLWERTERS(BufferBuilder, LWNbufferBuilder);
    Buffer *CreateBufferFromPool(MemoryPool *storage, LWNuintptr offset, LWNsizeiptr size);
    void Free();
};

class Buffer : public objects::Buffer
{
public:
    LWN_OVERLOAD_CPP_CLASS_COLWERTERS(Buffer, LWNbuffer);
    void Free();
};

class TextureBuilder : public objects::TextureBuilder
{
public:
    LWN_OVERLOAD_CPP_CLASS_COLWERTERS(TextureBuilder, LWNtextureBuilder);
    Texture *CreateTextureFromPool(MemoryPool *storage, LWNuintptr offset);
    void Free();
    size_t GetPaddedStorageSize()
    {
        size_t size = GetStorageSize();
        size_t alignment = GetStorageAlignment();
        return lwnUtil::AlignSize(size, alignment);
    }
};

class Texture : public objects::Texture
{
public:
    LWN_OVERLOAD_CPP_CLASS_COLWERTERS(Texture, LWNtexture);
    void Free();
    LWNuint GetRegisteredTextureID() const      { return lwnTextureGetRegisteredTextureID(reinterpret_cast<const LWNtexture *>(this)); }
};

class TextureView : public objects::TextureView
{
public:
    LWN_OVERLOAD_CPP_CLASS_COLWERTERS(TextureView, LWNtextureView);
    static TextureView *Create()    { return new TextureView; }
    void Free()                     { delete this; }
};

class TexturePool : public objects::TexturePool
{
public:
    LWN_OVERLOAD_CPP_CLASS_COLWERTERS(TexturePool, LWNtexturePool);
};

class SamplerBuilder : public objects::SamplerBuilder
{
public:
    LWN_OVERLOAD_CPP_CLASS_COLWERTERS(SamplerBuilder, LWNsamplerBuilder);
    Sampler *CreateSampler();
    void Free();
};

class Sampler : public objects::Sampler
{
public:
    LWN_OVERLOAD_CPP_CLASS_COLWERTERS(Sampler, LWNsampler);
    void Free();
    LWNuint GetRegisteredID() const      { return lwnSamplerGetRegisteredID(reinterpret_cast<const LWNsampler *>(this)); }
};

class SamplerPool : public objects::SamplerPool
{
public:
    LWN_OVERLOAD_CPP_CLASS_COLWERTERS(SamplerPool, LWNsamplerPool);
};

class Sync : public objects::Sync
{
public:
    LWN_OVERLOAD_CPP_CLASS_COLWERTERS(Sync, LWNsync);
    void Free();
};

class WindowBuilder : public objects::WindowBuilder
{
public:
    LWN_OVERLOAD_CPP_CLASS_COLWERTERS(WindowBuilder, LWNwindowBuilder);
    Window *CreateWin();
    void Free();

#if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_CPP
    // Create an overload for SetTextures, since the base function takes an
    // array of pointers to the base class (objects::Texture) while we take an
    // array of pointers to the derived class (Texture).
    WindowBuilder & SetTextures(int numTextures, Texture * const *textures)
    {
        objects::Texture * const *otextures = reinterpret_cast<objects::Texture * const *>(textures);
        objects::WindowBuilder::SetTextures(numTextures, otextures);
        return *this;
    }
#endif
};

class Window : public objects::Window
{
public:
    LWN_OVERLOAD_CPP_CLASS_COLWERTERS(Window, LWNwindow);
    void Free();
};

class DeviceBuilder : public objects::DeviceBuilder
{
public:
    LWN_OVERLOAD_CPP_CLASS_COLWERTERS(DeviceBuilder, LWNdeviceBuilder);
};

class Device : public objects::Device
{
public:
    LWN_OVERLOAD_CPP_CLASS_COLWERTERS(Device, LWNdevice);
    Queue *CreateQueue();
    CommandBuffer *CreateCommandBuffer();
    BlendState *CreateBlendState();
    ChannelMaskState *CreateChannelMaskState();
    ColorState *CreateColorState();
    DepthStencilState *CreateDepthStencilState();
    MultisampleState *CreateMultisampleState();
    PolygonState *CreatePolygonState();
    VertexAttribState *CreateVertexAttribState();
    VertexStreamState *CreateVertexStreamState();
    Program *CreateProgram();
    MemoryPool *CreateMemoryPool(void *memory, size_t size, MemoryPoolType poolType);
    MemoryPool *CreateMemoryPoolWithFlags(void *memory, size_t size, MemoryPoolFlags poolFlags);
    BufferBuilder *CreateBufferBuilder();
    TextureBuilder *CreateTextureBuilder();
    SamplerBuilder *CreateSamplerBuilder();
    WindowBuilder *CreateWindowBuilder();
    Sync *CreateSync();
    void Free();

    int GetInfo(DeviceInfo info) const
    {
        int result;
        GetInteger(info, &result);
        return result;
    }
};


class EventBuilder : public objects::EventBuilder
{
public:
    LWN_OVERLOAD_CPP_CLASS_COLWERTERS(EventBuilder, LWNeventBuilder);
    Event* CreateEvent();
};

class Event : public objects::Event
{
public:
    LWN_OVERLOAD_CPP_CLASS_COLWERTERS(Event, LWNevent);
    void Free();
};

} // namespace lwn

#endif // #if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_CPP

#endif // #ifndef __lwnTest_Objects_h__
