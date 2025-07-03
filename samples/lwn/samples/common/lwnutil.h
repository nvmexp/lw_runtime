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
// lwnutil.h
//
// Common utilities for LWN API usage.
//

#ifndef __LWNUTIL_H__
#define __LWNUTIL_H__

#include "lwn/lwn.h"
#include <assert.h>

#include <map>

// Backwards-compatibility #defines to map old LWN_MEMORY_POOL_TYPE_*
// definitions to their corresponding flags (from API version 36.0).  We set
// SHADER_CODE and COMPRESSIBLE on all pools to be conservative and allow
// pools using these #defines for for resources of those types.
#define LWN_MEMORY_POOL_TYPE_GPU_ONLY                               \
    LWNmemoryPoolFlags(LWN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS_BIT |    \
                       LWN_MEMORY_POOL_FLAGS_GPU_CACHED_BIT |       \
                       LWN_MEMORY_POOL_FLAGS_SHADER_CODE_BIT |      \
                       LWN_MEMORY_POOL_FLAGS_COMPRESSIBLE_BIT)

#define LWN_MEMORY_POOL_TYPE_CPU_COHERENT                       \
    LWNmemoryPoolFlags(LWN_MEMORY_POOL_FLAGS_CPU_UNCACHED_BIT | \
                       LWN_MEMORY_POOL_FLAGS_GPU_CACHED_BIT |   \
                       LWN_MEMORY_POOL_FLAGS_SHADER_CODE_BIT |  \
                       LWN_MEMORY_POOL_FLAGS_COMPRESSIBLE_BIT)

#define LWN_MEMORY_POOL_TYPE_CPU_NON_COHERENT                   \
    LWNmemoryPoolFlags(LWN_MEMORY_POOL_FLAGS_CPU_CACHED_BIT |   \
                       LWN_MEMORY_POOL_FLAGS_GPU_CACHED_BIT |   \
                       LWN_MEMORY_POOL_FLAGS_SHADER_CODE_BIT |  \
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
}

// Prototypes for backwards-compatibility lwn*Create* and lwn*Free APIs.
// These were removed in LWN API version 15.0, but we emulate driver-managed 
// objects in lwogtest to reduce test churn.
LWNdevice *lwnCreateDevice();
LWNqueue *lwnDeviceCreateQueue(LWNdevice *device);
LWNcommandBuffer *lwnDeviceCreateCommandBuffer(LWNdevice *device);
LWNblendState *lwnDeviceCreateBlendState(LWNdevice *device);
LWNchannelMaskState *lwnDeviceCreateChannelMaskState(LWNdevice *device);
LWNcolorState *lwnDeviceCreateColorState(LWNdevice *device);
LWNdepthStencilState *lwnDeviceCreateDepthStencilState(LWNdevice *device);
LWNmultisampleState *lwnDeviceCreateMultisampleState(LWNdevice *device);
LWNpolygonState *lwnDeviceCreatePolygonState(LWNdevice *device);
LWNprogram *lwnDeviceCreateProgram(LWNdevice *device);
LWNmemoryPool *lwnDeviceCreateMemoryPool(LWNdevice *device, void *memory, size_t size, LWNmemoryPoolFlags poolFlags);
LWNbufferBuilder *lwnDeviceCreateBufferBuilder(LWNdevice *device);
LWNtextureBuilder *lwnDeviceCreateTextureBuilder(LWNdevice *device);
LWNsamplerBuilder *lwnDeviceCreateSamplerBuilder(LWNdevice *device);
LWNsync *lwnDeviceCreateSync(LWNdevice *device);
LWNsampler *lwnSamplerBuilderCreateSampler(LWNsamplerBuilder *builder);
LWNbuffer *lwnBufferBuilderCreateBufferFromPool(LWNbufferBuilder *builder, LWNmemoryPool *storage, ptrdiff_t offset, size_t size);
LWNtexture *lwnTextureBuilderCreateTextureFromPool(LWNtextureBuilder *builder, LWNmemoryPool *storage, ptrdiff_t offset);

// API version 33.0 removes Texture::GetRegistered{Texture,Sampler}ID and
// Sampler::GetRegisteredID.
struct __LWNsamplerInternal {
    LWNsampler  sampler;
    int         lastRegisteredID;
};
struct __LWNtextureInternal {
    LWNtexture  texture;
    int         lastRegisteredTextureID;
};
int lwnTextureGetRegisteredTextureID(const LWNtexture *texture);
int lwnSamplerGetRegisteredID(const LWNsampler *sampler);

void lwnDeviceFree(LWNdevice *object);
void lwnQueueFree(LWNqueue *object);
void lwnCommandBufferFree(LWNcommandBuffer *object);
void lwnBlendStateFree(LWNblendState *object);
void lwnChannelMaskStateFree(LWNchannelMaskState *object);
void lwnColorStateFree(LWNcolorState *object);
void lwnDepthStencilStateFree(LWNdepthStencilState *object);
void lwnMultisampleStateFree(LWNmultisampleState *object);
void lwnPolygonStateFree(LWNpolygonState *object);
void lwnProgramFree(LWNprogram *object);
void lwnMemoryPoolFree(LWNmemoryPool *object);
void lwnBufferBuilderFree(LWNbufferBuilder *object);
void lwnBufferFree(LWNbuffer *object);
void lwnTextureFree(LWNtexture *object);
void lwnTextureBuilderFree(LWNtextureBuilder *object);
void lwnSamplerBuilderFree(LWNsamplerBuilder *object);
void lwnSamplerFree(LWNsampler *object);
void lwnSyncFree(LWNsync *object);

//////////////////////////////////////////////////////////////////////////
//
//                      LWN WRAPPED API OBJECTS
//
// We use the LWN_OVERLOAD_CPP_OBJECTS feature of the C++ headers to inject
// the standard C++ API object classes into the namespace "lwn::objects" and
// allow the application code to define its own extended object definition
// in the namespace "lwn".
//
// All of our wrapped object classes inherit from the base class in
// "lwn::objects".  We also provide Create/Free functions to emulate driver-side
// object creation and deletion to minimize changes to code written when LWN
// API objects were allocated by the driver.
//
#if defined(__cplusplus) && defined(LWN_OVERLOAD_CPP_OBJECTS)

#include "lwn/lwn_Cpp.h"

namespace lwn {

    // Forward declarations of API classes, since methods in one class reference
    // other classes.
    class Device;
    class Queue;
    class CommandBuffer;
    class BlendState;
    class ChannelMaskState;
    class ColorState;
    class DepthStencilState;
    class MultisampleState;
    class PolygonState;
    class VertexStateBuilder;
    class VertexState;
    class Program;
    class MemoryPool;
    class BufferBuilder;
    class Buffer;
    class Texture;
    class TextureBuilder;
    class SamplerBuilder;
    class Sampler;
    class Sync;

    class Queue : public objects::Queue
    {
    public:
        void Free();
    };

    class CommandBuffer : public objects::CommandBuffer
    {
    public:
        void Free();
    };

    class BlendState : public objects::BlendState
    {
    public:
        void Free();
    };

    class ChannelMaskState : public objects::ChannelMaskState
    {
    public:
        void Free();
    };

    class ColorState : public objects::ColorState
    {
    public:
        void Free();
    };

    class DepthStencilState : public objects::DepthStencilState
    {
    public:
        void Free();
    };

    class MultisampleState : public objects::MultisampleState
    {
    public:
        void Free();
    };

    class PolygonState : public objects::PolygonState
    {
    public:
        void Free();
    };

    class Program : public objects::Program
    {
    public:
        void Free();
    };

    class MemoryPool : public objects::MemoryPool
    {
    public:
        void Free();
    };

    class BufferBuilder : public objects::BufferBuilder
    {
    public:
        Buffer *CreateBufferFromPool(MemoryPool *storage, ptrdiff_t offset, size_t size);
        void Free();
    };

    class Buffer : public objects::Buffer
    {
    public:
        void Free();
    };

    class TextureBuilder : public objects::TextureBuilder
    {
    public:
        Texture *CreateTextureFromPool(MemoryPool *storage, ptrdiff_t offset);
        void Free();
    };

    class Texture : public objects::Texture
    {
    public:
        void Free();
    };

    class SamplerBuilder : public objects::SamplerBuilder
    {
    public:
        Sampler *CreateSampler();
        void Free();
    };

    class Sampler : public objects::Sampler
    {
    public:
        void Free();
    };

    class Sync : public objects::Sync
    {
    public:
        void Free();
    };

    class Device : public objects::Device
    {
    public:
        Queue *CreateQueue();
        CommandBuffer *CreateCommandBuffer();
        BlendState *CreateBlendState();
        ChannelMaskState *CreateChannelMaskState();
        ColorState *CreateColorState();
        DepthStencilState *CreateDepthStencilState();
        MultisampleState *CreateMultisampleState();
        PolygonState *CreatePolygonState();
        VertexStateBuilder *CreateVertexStateBuilder();
        Program *CreateProgram();
        MemoryPool *CreateMemoryPool(void *memory, size_t size, MemoryPoolType poolType);
        BufferBuilder *CreateBufferBuilder();
        TextureBuilder *CreateTextureBuilder();
        SamplerBuilder *CreateSamplerBuilder();
        Sync *CreateSync();
        void Free();
    };
};
#endif // #if defined(__cplusplus) && defined(LWN_OVERLOAD_CPP_OBJECTS)


class LWNsystemTexIDPool
{
public:
    explicit LWNsystemTexIDPool(LWNdevice* device, LWNcommandBuffer *queueCB);
    ~LWNsystemTexIDPool();

    // Simple allocation and deallocation of IDs
    int AllocTextureID();
    int AllocSamplerID();
    void FreeTextureID(int id);
    void FreeSamplerID(int id);

    // Allocation/deallocation of IDs, plus registration. LWN doesn't actually
    // deregister IDs, but we need a hook for maintaining the mapping between
    // objects and IDs, so let's just pretend.
    int Register(LWNtexture* texture);
    int Register(LWNsampler* sampler);
    void Deregister(LWNtexture* texture);
    void Deregister(LWNsampler* sampler);

    static int GetNumPublicTextures()   { return NUM_PUBLIC_TEXTURES; }
    static int GetNumPublicSamplers()   { return NUM_PUBLIC_SAMPLERS; }

private:
    int AllocID(uint32_t * handlePool, int numPublicIDs, int numReservedIDs, int* lastWord);
    void FreeID(int id, uint32_t * handlePool, int numPublicIDs, int numReservedIDs);

    // Both must be multiples of 32.
    enum {
        NUM_PUBLIC_TEXTURES = 8192,
        NUM_PUBLIC_SAMPLERS = 2048
    };

    typedef std::map<intptr_t, int> ObjectIDMap;

    LWNdevice* mDevice;
    int mNumReservedTextures;
    int mNumReservedSamplers;
    void* mPoolMemory;
    LWNmemoryPool mDescriptorPool;
    // ID pools are represented as bit streams. 1 == oclwpied.
    uint32_t mTextureIDs[NUM_PUBLIC_TEXTURES / 32];
    uint32_t mSamplerIDs[NUM_PUBLIC_SAMPLERS / 32];
    ObjectIDMap mTextureIDMap;
    ObjectIDMap mSamplerIDMap;
    int mLastTextureWord;
    int mLastSamplerWord;
    LWNsamplerPool mSamplerPool;
    LWNtexturePool mTexturePool;
};

class LWNutility
{
public:
    LWNsystemTexIDPool      *m_texIDPool;            // class managing texture pools and IDs
};

class CompletionTracker;
CompletionTracker *initCompletionTracker(LWNdevice *device, int size);
void insertCompletionTrackerFence(CompletionTracker *tracker, LWNqueue *queue);

template <typename T> class TrackedChunkRingBuffer;


//
// QueueCommandBuffer utility class
//
// Utility class derived setting up a command buffer object to be associated
// with a queue.
//
class QueueCommandBuffer : public LWNcommandBuffer
{
    // Device and queue owning the command buffer.
    LWNdevice                           *m_device;
    LWNqueue                            *m_queue;

    // Completion tracker associated with the queue/command buffer.
    class CompletionTracker             *m_tracker;

    // On HOS, this is the memory used for internal storage in m_commandPool.
    // If NULL, the storage is managed by the driver.
    void                                *m_commandPoolMemory;

    // Tracked ring buffer objects managing the command and control memory
    // usage of the command buffer.
    class TrackedCommandMemRingBuffer   *m_commandMem;
    LWNmemoryPool                       *m_commandPool;

    class TrackedControlMemRingBuffer   *m_controlMem;
    char                                *m_controlPool;

    // Usage counters tracking the write pointers at the last submit; used to
    // assert that there are no unsubmitted commands at the end of a frame.
    struct Counters {
        size_t                          commandMemUsage;
        size_t                          controlMemUsage;
    }                                   *m_lastSubmitCounters;

    size_t                              m_minSupportedCommandMemSize;
    size_t                              m_minSupportedControlMemSize;

    // Constants specifying the size and alignment of command buffer memory,
    // including the sizes of chunks we dole out to the command buffer object.
    // To stress out-of-memory conditions, set Max*ChunkSize to low values.
    static const size_t     MinCommandChunkSize = 0x1000;
    static const size_t     MaxCommandChunkSize = 0x40000000;
    static const size_t     CommandChunkAlignment = 4;
    static const size_t     CommandPoolAllocSize = 16 * 1024 * 1024;

    static const size_t     MinControlChunkSize = 0x400;
    static const size_t     MaxControlChunkSize = 0x40000000;
    static const size_t     ControlChunkAlignment = 8;
    static const size_t     ControlPoolAllocSize = 1024 * 1024;

public:
    QueueCommandBuffer() :
        m_device(NULL), m_queue(NULL), m_tracker(NULL), m_commandPoolMemory(NULL),
        m_commandMem(NULL), m_commandPool(NULL),
        m_controlMem(NULL), m_controlPool(NULL),
        m_lastSubmitCounters(NULL),
        m_minSupportedCommandMemSize(1),
        m_minSupportedControlMemSize(1)
    {}

    // Initialize the command buffer, including allocating any resources
    // required for submissions.
    bool init(LWNdevice *device, LWNqueue *queue, CompletionTracker *tracker);
    bool initCommand();
    bool initControl();

    // Destroy the command buffer, including any resources it needed.
    void destroy();

    // Out-of-memory callback function installed in the command buffer object.
    static void LWNAPIENTRY outOfMemory(LWNcommandBuffer *cmdBuf, LWNcommandBufferMemoryEvent event,
                                        size_t minSize, void *callbackData);

    // Read the command and control memory write counters.
    void getCounters(Counters *counters);

    // Check the command buffer usage tracking counters recorded in the last
    // submit against the current write pointers and assert if they don't
    // match.
    void checkUnflushedCommands();

    // Reset the queue command buffer usage tracking counters to the current
    // write pointers to avoid assertions for tests that use the embedded
    // command buffer outside the queue command buffer object.
    void resetCounters();

    // Submit any commands queued up in the command buffer to the queue.
    void submit();
};

//
// LWNcommandBufferMemoryManager utility class
//
// Utility class holding memory that can be used to easily back API command
// buffer objects.  Holds command memory from both coherent and non-coherent
// pools, plus malloc control memory.
//
class LWNcommandBufferMemoryManager
{
private:
    typedef TrackedChunkRingBuffer<uintptr_t>   CommandMemory;
    typedef TrackedChunkRingBuffer<char *>      ControlMemory;

    // Device owning the command buffer memory.
    LWNdevice           *m_device;

    // Memory pool/malloc memory objects used to back the command buffers.
    void                *m_coherentPoolMemory;
    LWNmemoryPool       *m_coherentPool;
    void                *m_nonCoherentPoolMemory;
    LWNmemoryPool       *m_nonCoherentPool;
    char                *m_controlPool;

    // Ring buffer managers tracking the different types of command buffer
    // memory.
    CommandMemory       *m_coherentMem;
    CommandMemory       *m_nonCoherentMem;
    ControlMemory       *m_controlMem;

    size_t              m_minSupportedCommandMemSize;
    size_t              m_minSupportedControlMemSize;

    // Default sizes for command and control memory allocations.
    static const int    coherentPoolSize = 1024 * 1024;
    static const int    nonCoherentPoolSize = 1024 * 1024;
    static const int    controlPoolSize = 256 * 1024;

    // Default chunk sizes for command and control memory; one chunk of each
    // type will be transferred to the command buffer as needed.
    static const int    coherentChunkSize = 16 * 1024;
    static const int    nonCoherentChunkSize = 16 * 1024;
    static const int    controlChunkSize = 1024;

public:
    // Enum indicating the type of command memory is requested for the pool.
    // IMPORTANT:  Non-coherent memory needs to be flushed before submitting
    // if submitting directly to a queue.
    enum CommandMemType { Coherent, NonCoherent };

    // Set up command buffer memory.
    bool init(LWNdevice *device, CompletionTracker *tracker);

    // Tear down command buffer memory and internal allocations.
    void destroy();

    // Populate <cmdBuf> with memory chunks from the ring buffer.
    bool populateCommandBuffer(LWNcommandBuffer *cmdBuf, CommandMemType commandType);

    // Add command memory from the pool to <cmdBuf>.
    void addCommandMem(LWNcommandBuffer *cmdBuf, CommandMemory *cmdMem, LWNmemoryPool *cmdPool, size_t minRequiredSize);

    // Add control memory from the pool to <cmdBuf>.
    void addControlMem(LWNcommandBuffer *cmdBuf, size_t minRequiredSize);

private:
    // Callback functions used when a managed command buffer runs out of
    // memory.
    static void LWNAPIENTRY coherentCallback(LWNcommandBuffer *cmdBuf, LWNcommandBufferMemoryEvent event,
                                             size_t minSize, void *data);
    static void LWNAPIENTRY nonCoherentCallback(LWNcommandBuffer *cmdBuf, LWNcommandBufferMemoryEvent event,
                                                size_t minSize, void *data);

};

extern LWNutility g_lwn;

#ifndef POOL_ALIGN_DEFINED
#define POOL_ALIGN_DEFINED
// Function template to round a value up to the lowest multiple of
// LWN_MEMORY_POOL_STORAGE_ALIGNMENT.
template <typename T> T PoolAlign(T value)
{
    T lowBits = LWN_MEMORY_POOL_STORAGE_ALIGNMENT - 1;
    return (value + lowBits) & ~lowBits;
}
#endif
#endif // __LWNUTIL_H__
