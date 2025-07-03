/*
 * Copyright (c) 2015-2020 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

//
// lwnutil.cpp
//
// Common utilities for LWN API usage.
//

#include <stdlib.h>
#include <list>
#include <assert.h>
#include <string.h>

#define LWN_OVERLOAD_CPP_OBJECTS
#include "lwn/lwn.h"
#include "lwnutil.h"
#include "lwn/lwn_Cpp.h"
#include "lwn/lwn_CppMethods.h"
#include "lwn/lwn_FuncPtrInline.h"

#include "lwnUtil/lwnUtil_AlignedStorage.h"

// Backwards-compatibility lwn*Create* and lwn*Free APIs.  These were removed
// in LWN API version 15.0, but we emulate in client code to reduce churn.
#define LWN_FREE_EMULATE_FINALIZE(CType, TypeName)  \
    void lwn##TypeName##Free(CType *object)         \
        {                                           \
        lwn##TypeName##Finalize(object);            \
        delete object;                              \
        }


#define LWN_FREE_EMULATE_DEREGISTER(CType, TypeName)    \
    void lwn##TypeName##Free(CType *object)             \
        {                                               \
        g_lwn.m_texIDPool->Deregister(object);          \
        lwn##TypeName##Finalize(object);                \
        delete object;                                  \
        }

#define LWN_FREE_EMULATE_NO_FINALIZE(CType, TypeName)   \
    void lwn##TypeName##Free(CType *object)             \
        {                                               \
        delete object;                                  \
        }

LWN_FREE_EMULATE_FINALIZE(LWNdevice, Device);
LWNdevice *lwnCreateDevice()
{
    LWNdevice *object = new LWNdevice;
    if (!object) {
        return NULL;
    }
    LWNdeviceBuilder db;
    lwnDeviceBuilderSetDefaults(&db);
    lwnDeviceBuilderSetFlags(&db, 0);
    if (!lwnDeviceInitialize(object, &db)) {
        delete object;
        return NULL;
    }
    return object;
}

LWN_FREE_EMULATE_FINALIZE(LWNqueue, Queue);
LWNqueue *lwnDeviceCreateQueue(LWNdevice *device)
{
    LWNqueue *object = new LWNqueue;
    if (!object) {
        return NULL;
    }
    LWNqueueBuilder qb;
    lwnQueueBuilderSetDevice(&qb, device);
    lwnQueueBuilderSetDefaults(&qb);
    if (!lwnQueueInitialize(object, &qb)) {
        delete object;
        return NULL;
    }
    return object;
}

LWN_FREE_EMULATE_FINALIZE(LWNcommandBuffer, CommandBuffer);
LWNcommandBuffer *lwnDeviceCreateCommandBuffer(LWNdevice *device)
{
    LWNcommandBuffer *object = new LWNcommandBuffer;
    if (!object) {
        return NULL;
    }
    if (!lwnCommandBufferInitialize(object, device)) {
        delete object;
        return NULL;
    }
    return object;
}

LWN_FREE_EMULATE_NO_FINALIZE(LWNblendState, BlendState);
LWNblendState *lwnDeviceCreateBlendState(LWNdevice *device)
{
    LWNblendState *object = new LWNblendState;
    if (!object) {
        return NULL;
    }
    lwnBlendStateSetDefaults(object);
    return object;
}

LWN_FREE_EMULATE_NO_FINALIZE(LWNchannelMaskState, ChannelMaskState);
LWNchannelMaskState *lwnDeviceCreateChannelMaskState(LWNdevice *device)
{
    LWNchannelMaskState * object = new LWNchannelMaskState;
    if (!object) {
        return NULL;
    }
    lwnChannelMaskStateSetDefaults(object);
    return object;
}

LWN_FREE_EMULATE_NO_FINALIZE(LWNcolorState, ColorState);
LWNcolorState *lwnDeviceCreateColorState(LWNdevice *device)
{
    LWNcolorState * object = new LWNcolorState;
    if (!object) {
        return NULL;
    }
    lwnColorStateSetDefaults(object);
    return object;
}

LWN_FREE_EMULATE_NO_FINALIZE(LWNdepthStencilState, DepthStencilState);
LWNdepthStencilState *lwnDeviceCreateDepthStencilState(LWNdevice *device)
{
    LWNdepthStencilState * object = new LWNdepthStencilState;
    if (!object) {
        return NULL;
    }
    lwnDepthStencilStateSetDefaults(object);
    return object;
}

LWN_FREE_EMULATE_NO_FINALIZE(LWNmultisampleState, MultisampleState);
LWNmultisampleState *lwnDeviceCreateMultisampleState(LWNdevice *device)
{
    LWNmultisampleState * object = new LWNmultisampleState;
    if (!object) {
        return NULL;
    }
    lwnMultisampleStateSetDefaults(object);
    return object;
}

LWN_FREE_EMULATE_NO_FINALIZE(LWNpolygonState, PolygonState);
LWNpolygonState *lwnDeviceCreatePolygonState(LWNdevice *device)
{
    LWNpolygonState * object = new LWNpolygonState;
    if (!object) {
        return NULL;
    }
    lwnPolygonStateSetDefaults(object);
    return object;
}

LWN_FREE_EMULATE_FINALIZE(LWNprogram, Program);
LWNprogram *lwnDeviceCreateProgram(LWNdevice *device)
{
    LWNprogram *object = new LWNprogram;
    if (!object) {
        return NULL;
    }
    if (!lwnProgramInitialize(object, device)) {
        delete object;
        return NULL;
    }
    return object;
}

void lwnMemoryPoolFree(LWNmemoryPool *object)
{
    lwnMemoryPoolFinalize(object);
    free(object);
}

LWNmemoryPool * lwnDeviceCreateMemoryPool(LWNdevice *device, void *memory, size_t size, LWNmemoryPoolFlags poolFlags)
{
    LWNmemoryPool *object;

    if (!memory && !(poolFlags & LWN_MEMORY_POOL_FLAGS_VIRTUAL_BIT)) {
        // If the caller passes NULL pointer, then this helper function is responsible for allocating storage.

        // Align up size to minimum granularity.
        size = (size + LWN_MEMORY_POOL_STORAGE_GRANULARITY - 1) & ~(LWN_MEMORY_POOL_STORAGE_GRANULARITY - 1);

        // Reserve space for the memory pool structure and storage.
        size_t storageSize = sizeof(LWNmemoryPool) + size + LWN_MEMORY_POOL_STORAGE_ALIGNMENT;
        void *storage = malloc(storageSize);
        if (!storage) {
            return NULL;
        }

        object = (LWNmemoryPool *)storage;
        memory = lwnUtil::PoolStorageAlign(object + 1);
        assert((char *)object + storageSize >= (char *)memory + size);
    } else {
        assert(size % LWN_MEMORY_POOL_STORAGE_GRANULARITY == 0);
        assert((uintptr_t)memory % LWN_MEMORY_POOL_STORAGE_ALIGNMENT == 0);
        object = (LWNmemoryPool *)malloc(sizeof(LWNmemoryPool));
    }

    LWNmemoryPoolBuilder builder;
    lwnMemoryPoolBuilderSetDefaults(&builder);
    lwnMemoryPoolBuilderSetDevice(&builder, device);
    lwnMemoryPoolBuilderSetStorage(&builder, memory, size);
    lwnMemoryPoolBuilderSetFlags(&builder, poolFlags);
    if (!lwnMemoryPoolInitialize(object, &builder)) {
        free(object);
        return NULL;
    }
    return object;
}

LWN_FREE_EMULATE_NO_FINALIZE(LWNbufferBuilder, BufferBuilder);
LWNbufferBuilder *lwnDeviceCreateBufferBuilder(LWNdevice *device)
{
    LWNbufferBuilder *object = new LWNbufferBuilder;
    if (!object) {
        return NULL;
    }
    lwnBufferBuilderSetDevice(object, device);
    lwnBufferBuilderSetDefaults(object);
    return object;
}

LWN_FREE_EMULATE_NO_FINALIZE(LWNtextureBuilder, TextureBuilder);
LWNtextureBuilder *lwnDeviceCreateTextureBuilder(LWNdevice *device)
{
    LWNtextureBuilder *object = new LWNtextureBuilder;
    if (!object) {
        return NULL;
    }
    lwnTextureBuilderSetDevice(object, device);
    lwnTextureBuilderSetDefaults(object);
    return object;
}

LWN_FREE_EMULATE_NO_FINALIZE(LWNsamplerBuilder, SamplerBuilder);
LWNsamplerBuilder *lwnDeviceCreateSamplerBuilder(LWNdevice *device)
{
    LWNsamplerBuilder *object = new LWNsamplerBuilder;
    if (!object) {
        return NULL;
    }
    lwnSamplerBuilderSetDevice(object, device);
    lwnSamplerBuilderSetDefaults(object);
    return object;
}

LWN_FREE_EMULATE_FINALIZE(LWNsync, Sync);
LWNsync *lwnDeviceCreateSync(LWNdevice *device)
{
    LWNsync *object = new LWNsync;
    if (!object) {
        return NULL;
    }
    if (!lwnSyncInitialize(object, device)) {
        delete object;
        return NULL;
    }
    return object;
}


LWN_FREE_EMULATE_DEREGISTER(LWNsampler, Sampler)
LWNsampler *lwnSamplerBuilderCreateSampler(LWNsamplerBuilder *builder)
{
    __LWNsamplerInternal *alloc = new __LWNsamplerInternal;
    LWNsampler *object = &alloc->sampler;
    if (!lwnSamplerInitialize(object, builder)) {
        delete object;
        return NULL;
    }
    alloc->lastRegisteredID = g_lwn.m_texIDPool->Register(object);
    return object;
}

LWN_FREE_EMULATE_FINALIZE(LWNbuffer, Buffer);
LWNbuffer *lwnBufferBuilderCreateBufferFromPool(LWNbufferBuilder *builder, LWNmemoryPool *storage, ptrdiff_t offset, size_t size)
{
    LWNbuffer *object = new LWNbuffer;
    if (!object) {
        return NULL;
    }
    lwnBufferBuilderSetStorage(builder, storage, offset, size);
    if (!lwnBufferInitialize(object, builder)) {
        delete object;
        return NULL;
    }
    return object;
}

LWN_FREE_EMULATE_DEREGISTER(LWNtexture, Texture)
LWNtexture *lwnTextureBuilderCreateTextureFromPool(LWNtextureBuilder *builder, LWNmemoryPool *storage, ptrdiff_t offset)
{
    __LWNtextureInternal *alloc = new __LWNtextureInternal;
    LWNtexture *object = &alloc->texture;
    lwnTextureBuilderSetStorage(builder, storage, offset);
    if (!lwnTextureInitialize(object, builder)) {
        delete object;
        return NULL;
    }
    alloc->lastRegisteredTextureID = g_lwn.m_texIDPool->Register(object);
    return object;
}

int lwnTextureGetRegisteredTextureID(const LWNtexture *texture)
{
    const __LWNtextureInternal *alloc = reinterpret_cast<const __LWNtextureInternal *>(texture);
    return alloc->lastRegisteredTextureID;
}

int lwnSamplerGetRegisteredID(const LWNsampler *sampler)
{
    const __LWNsamplerInternal *alloc = reinterpret_cast<const __LWNsamplerInternal *>(sampler);
    return alloc->lastRegisteredID;
}

//
// Create C++ Create/Free methods for backward compatibility with older versions
// of the LWN C++ API.
//
namespace lwn {

    Queue *Device::CreateQueue()
    { return reinterpret_cast<Queue *>(lwnDeviceCreateQueue(reinterpret_cast<LWNdevice *>(this))); }

    CommandBuffer *Device::CreateCommandBuffer()
    { return reinterpret_cast<CommandBuffer *>(lwnDeviceCreateCommandBuffer(reinterpret_cast<LWNdevice *>(this))); }

    BlendState *Device::CreateBlendState()
    { return reinterpret_cast<BlendState *>(lwnDeviceCreateBlendState(reinterpret_cast<LWNdevice *>(this))); }

    ChannelMaskState *Device::CreateChannelMaskState()
    { return reinterpret_cast<ChannelMaskState *>(lwnDeviceCreateChannelMaskState(reinterpret_cast<LWNdevice *>(this))); }

    ColorState *Device::CreateColorState()
    { return reinterpret_cast<ColorState *>(lwnDeviceCreateColorState(reinterpret_cast<LWNdevice *>(this))); }

    DepthStencilState *Device::CreateDepthStencilState()
    { return reinterpret_cast<DepthStencilState *>(lwnDeviceCreateDepthStencilState(reinterpret_cast<LWNdevice *>(this))); }

    MultisampleState *Device::CreateMultisampleState()
    { return reinterpret_cast<MultisampleState *>(lwnDeviceCreateMultisampleState(reinterpret_cast<LWNdevice *>(this))); }

    PolygonState *Device::CreatePolygonState()
    { return reinterpret_cast<PolygonState *>(lwnDeviceCreatePolygonState(reinterpret_cast<LWNdevice *>(this))); }

    Program *Device::CreateProgram()
    { return reinterpret_cast<Program *>(lwnDeviceCreateProgram(reinterpret_cast<LWNdevice *>(this))); }

    MemoryPool *Device::CreateMemoryPool(void *memory, size_t size, MemoryPoolType poolType)
    { return reinterpret_cast<MemoryPool *>(lwnDeviceCreateMemoryPool(reinterpret_cast<LWNdevice *>(this), memory, size, 
                                                                      LWNmemoryPoolFlags(poolType))); }

    BufferBuilder *Device::CreateBufferBuilder()
    { return reinterpret_cast<BufferBuilder *>(lwnDeviceCreateBufferBuilder(reinterpret_cast<LWNdevice *>(this))); }

    TextureBuilder *Device::CreateTextureBuilder()
    { return reinterpret_cast<TextureBuilder *>(lwnDeviceCreateTextureBuilder(reinterpret_cast<LWNdevice *>(this))); }

    SamplerBuilder *Device::CreateSamplerBuilder()
    { return reinterpret_cast<SamplerBuilder *>(lwnDeviceCreateSamplerBuilder(reinterpret_cast<LWNdevice *>(this))); }

    Sync *Device::CreateSync()
    { return reinterpret_cast<Sync *>(lwnDeviceCreateSync(reinterpret_cast<LWNdevice *>(this))); }

    Sampler *SamplerBuilder::CreateSampler()
    { return reinterpret_cast<Sampler *>(lwnSamplerBuilderCreateSampler(reinterpret_cast<LWNsamplerBuilder *>(this))); }

    Buffer *BufferBuilder::CreateBufferFromPool(MemoryPool *storage, ptrdiff_t offset, size_t size)
    { return reinterpret_cast<Buffer *>(lwnBufferBuilderCreateBufferFromPool(reinterpret_cast<LWNbufferBuilder *>(this), reinterpret_cast<LWNmemoryPool *>(storage), offset, size)); }

    Texture *TextureBuilder::CreateTextureFromPool(MemoryPool *storage, ptrdiff_t offset)
    { return reinterpret_cast<Texture *>(lwnTextureBuilderCreateTextureFromPool(reinterpret_cast<LWNtextureBuilder *>(this), reinterpret_cast<LWNmemoryPool *>(storage), offset)); }

    void Device::Free()                 { lwnDeviceFree(reinterpret_cast<LWNdevice *>(this)); }
    void Queue::Free()                  { lwnQueueFree(reinterpret_cast<LWNqueue *>(this)); }
    void CommandBuffer::Free()          { lwnCommandBufferFree(reinterpret_cast<LWNcommandBuffer *>(this)); }
    void BlendState::Free()             { lwnBlendStateFree(reinterpret_cast<LWNblendState *>(this)); }
    void ChannelMaskState::Free()       { lwnChannelMaskStateFree(reinterpret_cast<LWNchannelMaskState *>(this)); }
    void ColorState::Free()             { lwnColorStateFree(reinterpret_cast<LWNcolorState *>(this)); }
    void DepthStencilState::Free()      { lwnDepthStencilStateFree(reinterpret_cast<LWNdepthStencilState *>(this)); }
    void MultisampleState::Free()       { lwnMultisampleStateFree(reinterpret_cast<LWNmultisampleState *>(this)); }
    void PolygonState::Free()           { lwnPolygonStateFree(reinterpret_cast<LWNpolygonState *>(this)); }
    void Program::Free()                { lwnProgramFree(reinterpret_cast<LWNprogram *>(this)); }
    void MemoryPool::Free()             { lwnMemoryPoolFree(reinterpret_cast<LWNmemoryPool *>(this)); }
    void BufferBuilder::Free()          { lwnBufferBuilderFree(reinterpret_cast<LWNbufferBuilder *>(this)); }
    void Buffer::Free()                 { lwnBufferFree(reinterpret_cast<LWNbuffer *>(this)); }
    void Texture::Free()                { lwnTextureFree(reinterpret_cast<LWNtexture *>(this)); }
    void TextureBuilder::Free()         { lwnTextureBuilderFree(reinterpret_cast<LWNtextureBuilder *>(this)); }
    void SamplerBuilder::Free()         { lwnSamplerBuilderFree(reinterpret_cast<LWNsamplerBuilder *>(this)); }
    void Sampler::Free()                { lwnSamplerFree(reinterpret_cast<LWNsampler *>(this)); }
    void Sync::Free()                   { lwnSyncFree(reinterpret_cast<LWNsync *>(this)); }

}


//////////////////////////////////////////////////////////////////////////

//
// RingBufferManager utility template class
//
// This class manages a ring buffer of <size> entries.  If <T> is an integer
// type, the entries are numbered <start> through <start>+<size>-1.  If <T> is
// a pointer to type X, type entries are maintained as pointers, with the
// first entry at <start> and the last at <start>+(<size>-1)*sizeof(X).
//
// We maintain read and write pointers in m_read and m_write.  When m_read ==
// m_write, the ring buffer is empty.  Both pointers wrap at the end of the
// ring buffer back to the beginning.
//
// To write to the ring buffer, applications request one or multiple entries
// of contiguous memory via getWriteSpace() and update the write pointer via
// syncWrite().  If a request for more than one entry would cause us to run
// off the end of the buffer, we leave blank space at the end and wrap back to
// the beginning.
//
// Reads from the ring buffer are assumed to happen externally, and the read
// pointer is updated via the setRead() method.
//
// This class doesn't provide the storage of the ring buffer, it only manages
// storage provided from other sources.
// 
template <typename T> class RingBufferManager {
private:
    int         m_size;             // number of entries in the ring buffer

    T           m_start;            // first entry in the ring buffer
    T           m_read;             // read pointer (next entry to be consumed)
    T           m_write;            // write pointer (next entry to be added)
    T           m_end;              // limit of the ring buffer (one past the last entry)

    bool        m_writing;          // have we reserved space via getWriteSpace?
    T           m_writeLimit;       // limit of the reserved write space

    enum WriteSpaceRequest {
        GetExactSize,               // allocate only the requested size
        GetMaximumSize              // allocate the maximum available contiguous size
    };

public:
    RingBufferManager(int size, T start) :
        m_size(size),
        m_start(start),
        m_read(start),
        m_write(start),
        m_end(start + size),
        m_writing(false),
        m_writeLimit(start)
    {}

    int size() const                    { return m_size; }
    T getStart() const                  { return m_start; }
    T getRead() const                   { return m_read; }
    T getWrite() const                  { return m_write; }
    T getEnd() const                    { return m_end; }

    bool isEmpty() const                { return m_write == m_read; }
    bool isFull() const                 { return getAvailableSpace() == 0; }

    void setRead(T read)                { m_read = read; }

    // Check for available space in the ring buffer.  Note that we don't allow
    // the ring buffer to get completely full -- m_read == m_write implies an
    // empty buffer, not a full one.  If m_read is ahead of m_write, the free
    // space in the ring buffer is between the two pointers:
    //
    //     m_write    m_read
    //    -----+#########+-----
    //
    // Leaving one entry free, that gives us:
    //
    //     (m_read - m_write) - 1
    //
    // entries available.  If m_read is behind (or equal to) m_write, the free
    // space in the ring buffer is everything not between the two pointers:
    //
    //      m_read    m_write
    //    #####+---------+######
    //
    // That has(m_write - m_read) entries oclwpied, which leaves:
    //
    //     (m_size - 1) - (m_write - m_read)
    //   = (m_size - 1) + (m_read - m_write)
    //   = m_size + ((m_read - m_write) - 1)
    //
    // entries available.
    inline size_t getAvailableSpace() const
    {
        int space = (int) (m_read - m_write) - 1;
        if (space < 0) {
            space += m_size;
        }
        return (size_t) space;
    }

    // Reserve <space> contiguous entries in the ring buffer.  If
    // <requestType> is GetExactSize, that exact number of entries are
    // reserved.  If <requestType> is GetMaximum size, the maximum size
    // allocation starting from the write pointer is reserved.
    //
    // The number of entries allocated is returned, or 0 if no space is
    // available.  The write pointer is returned in <current>.
    inline size_t getWriteSpace(T &current, size_t space = 1,
                             WriteSpaceRequest requestType = GetExactSize)
    {
        assert(space < (size_t) m_size);
        assert(!m_writing);

        // If we're near the end of the buffer and don't have enough
        // contiguous space, try to reserve padding to the end of the buffer
        // and wrap around to the start.
        if ((T) (m_write + space) > (T) m_end) {
            if (!getWriteSpace(current, m_end - m_write)) {
                return 0;
            }
            m_writing = false;  // this isn't a real reservation
            m_write = m_start;
        }

        assert((T) (m_write + space) <= (T) m_end);
        if (space > getAvailableSpace()) {
            return 0;
        }

        // If this is a request for the maximum size, compute the amount of
        // available contiguous and update the size request accordingly.
        if (requestType == GetMaximumSize) {
            long maxSpace;
            if (m_read > m_write) {
                // Compute the space between the write and read pointers,
                // leaving the last one blank.
                maxSpace = (long) ((m_read - m_write) - 1);
            } else {
                // Compute the space between the write pointer and the end of
                // the buffer, leaving the last entry unused if the read
                // pointer is at the start.
                maxSpace = (long) (m_end - m_write);
                if (m_read == m_start) {
                    maxSpace--;
                }
            }
            assert(maxSpace >= (long) space);
            space = maxSpace;
        }

        m_writing = true;
        m_writeLimit = (T) (m_write + space);
        current = m_write;
        return space;
    }

    // Request the maximum number of contiguous entries available, as long as
    // it's at least <minSpace>.
    inline size_t getWriteSpaceMax(T &current, size_t minSpace)
    {
        return getWriteSpace(current, minSpace, GetMaximumSize);
    }

    // Update the write pointer to <current> after writing in the reserved
    // write space.
    inline void syncWrite(T current)
    {
        assert(m_writing);
        assert(current <= m_writeLimit);
        m_write = wrapEntry(current);
        m_writing = false;
        m_writeLimit = m_write;
    }

    // Apply wrapping (if needed) to a ring buffer entry pointer, wrapping
    // back to the beginning when it has run off the end.
    inline T wrapEntry(T entry)
    {
        if (entry >= m_end) {
            entry -= m_size;
        }
        assert(entry >= m_start);
        assert(entry < m_end);
        return entry;
    }
};


//
// CompletionTracker utility class
//
// Uses a ring buffer of <size> LWNsync objects to track the completion of
// commands sent to queues.
//
// The completion tracker manages a list of tracked allocators and sends
// notifications to each allocator when a new fence is inserted or when an old
// sync object has been waited on successfully.  Both notifications include a
// <fenceid> value indicating the location of the sync object in the ring
// buffer.
//
class CompletionTracker {
private:
    typedef class TrackedAllocator *Allocator;
    typedef std::list<Allocator> AllocatorList;
    RingBufferManager<int>          m_ring;
    AllocatorList                   m_allocators;
    LWNsync                         *m_objects;

public:
    CompletionTracker(LWNdevice *device, int size) :
        m_ring(size, 0), m_allocators(), m_objects(NULL)
    {
        m_objects = new LWNsync[size];
        for (int i = 0; i < size; i++) {
            lwnSyncInitialize(&m_objects[i], device);
        }
    };

    ~CompletionTracker()
    {
        m_allocators.clear();
        for (int i = 0; i < m_ring.size(); i++) {
            lwnSyncFinalize(&m_objects[i]);
        }
        delete[] m_objects;
    }

    int size() const        { return m_ring.size(); }
    bool isEmpty() const    { return m_ring.isEmpty(); }

    // Register and unregister tracked allocators.
    bool addAllocator(Allocator allocator);
    bool removeAllocator(Allocator allocator);

    // Send notifications to tracked allocators when a new sync object is
    // inserted (FenceSync) or removed (SyncWait) from the queue.
    void notifyFenceInserted(int fenceid);
    void notifySyncCompleted(int fenceid);

    // Insert a sync object into the ring buffer (at the write pointer) and
    // notify tracked allocators.
    void insertFence(LWNqueue *queue);

    // Check the completion of one or more sync objects (starting at the read
    // pointer).  If <wait> is true, wait for at least one sync object to
    // complete.  Returns true if and only if any sync object was detected to
    // be completed.
    bool updateGet(bool wait = false);
};


//
// TrackedAllocator utility class
//
// Abstract base class used to track and free allocations once dependent LWN
// commands have completed exelwtion.
//
class TrackedAllocator {
private:
    CompletionTracker   *m_tracker;

public:
    TrackedAllocator(CompletionTracker *tracker = NULL) : m_tracker(tracker) {}
    virtual ~TrackedAllocator() {}
    void setTracker(CompletionTracker *tracker)     { m_tracker = tracker; }
    CompletionTracker *getTracker() const           { return m_tracker; }
    virtual void notifyFenceInserted(int fenceid) = 0;
    virtual void notifySyncCompleted(int fenceid) = 0;
};


//
// TrackedRingBuffer utility class
//
// Utility class that manages a ring buffer of transient memory allocations
// where all allocations performed before a fence notification are assumed to
// be completed when the corresponding sync object has landed.
//
// This class maintains an array of fences (m_fences) that records the current
// write pointer each time a fence is inserted.  The read pointer is updated
// to the fence when the corresponding sync object has landed.
//
template <typename T>
class TrackedRingBuffer :
    public RingBufferManager<T>,
    public TrackedAllocator
{
private:
    T  *m_fences;
    size_t m_alignmentMask;        // alignment required for each allocation

public:
    TrackedRingBuffer(CompletionTracker *tracker, T start, size_t size, size_t alignment) :
        RingBufferManager<T>((int) size, start),
        TrackedAllocator(tracker),
        m_alignmentMask(~(alignment - 1))
    {
        m_fences = new T[size];
        for (size_t i = 0; i < size; i++) {
            m_fences[i] = start;
        }
        tracker->addAllocator(this);
    }

    virtual ~TrackedRingBuffer()
    {
        CompletionTracker *tracker = getTracker();
        tracker->removeAllocator(this);
        delete[] m_fences;
    }

    void setAlignment(size_t alignment)     { m_alignmentMask = ~(alignment - 1); }

    // Record the current write pointer as a fence when a sync object is
    // inserted.
    void setFence(int fenceId, T fence)
    {
        m_fences[fenceId] = fence;
    }

    // Update fences when a sync object is inserted.  This function is virtual
    // so that derived command buffer memory classes can override; we don't
    // continuously track the write pointer on the client side.
    virtual void notifyFenceInserted(int fenceid)
    {
        setFence(fenceid, RingBufferManager<T>::getWrite());
    }

    // Update the read pointer from a previously stored fence when a sync
    // object has completed.
    void notifySyncCompleted(int fenceid)
    {
        RingBufferManager<T>::setRead(m_fences[fenceid]);
    }

    // Request <minSpace> bytes of write space in the ring buffer.  The
    // resulting amount of space allocated is clamped to <maxSpace> if
    // specified.  Returns the number of bytes allocated and stores the write
    // pointer in <current>.
    size_t getWriteSpace(T &current, size_t minSpace, size_t maxSpace)
    {
        CompletionTracker *tracker = getTracker();
        size_t reservedSize = RingBufferManager<T>::getWriteSpaceMax(current, (int) minSpace);
        bool forceWait = false;
        while (reservedSize == 0) {
            assert(!tracker->isEmpty());
            tracker->updateGet(forceWait);
            reservedSize = RingBufferManager<T>::getWriteSpaceMax(current, (int) minSpace);
            forceWait = true;
        }
        if (reservedSize > maxSpace) {
            reservedSize = maxSpace;
        }
        reservedSize &= m_alignmentMask;
        return reservedSize;
    }
};

//
// TrackedChunkRingBuffer utility class
//
// Utility class derived from TrackedRingBuffer that doles out memory in
// chunks with fixed minimum and maximum sizes.  The amount of size returned
// is variable if the minimum and maximum chunk sizes don't match.
//
template <typename T>
class TrackedChunkRingBuffer : public TrackedRingBuffer < T >
{
private:
    size_t m_minChunkSize;          // minimum chunk size required for an allocation
    size_t m_maxChunkSize;          // maximum chunk size allowed for an allocation
public:
    TrackedChunkRingBuffer(CompletionTracker *tracker, T start, size_t size,
                           size_t minChunkSize, size_t maxChunkSize,
                           size_t alignment) :
                           TrackedRingBuffer<T>(tracker, start, size, alignment),
                           m_minChunkSize(minChunkSize),
                           m_maxChunkSize(maxChunkSize)
    {
    }

    void setMaxChunkSize(size_t size)      { m_maxChunkSize = size; }
    void setMinChunkSize(size_t size)      { m_minChunkSize = size; }

    size_t getWriteSpace(T &current, size_t minRequiredChunkSize)
    {
        size_t minChunkSize = m_minChunkSize;
        size_t maxChunkSize = m_maxChunkSize;

        if (minRequiredChunkSize > minChunkSize) {
            minChunkSize = minRequiredChunkSize;

            if (minChunkSize > maxChunkSize) {
                maxChunkSize = minChunkSize;
            }
        }

        size_t space = TrackedRingBuffer<T>::getWriteSpace(current, minChunkSize, maxChunkSize);
        return space;
    }
};


//
// TrackedCommandMemRingBuffer utility class
//
// Utility class derived from TrackedRingBuffer that plugs ring buffer memory
// into the command memory of the specified command buffer.
//
class TrackedCommandMemRingBuffer : public TrackedChunkRingBuffer < uintptr_t >
{
    LWNcommandBuffer *m_cmdBuf;         // command buffer owning the ring buffer
    LWNmemoryPool *m_pool;              // memory pool providing storage
    uintptr_t  m_lastChunk;             // offset of last chunk given to m_cmdBuf

public:
    TrackedCommandMemRingBuffer(LWNcommandBuffer *cmdBuf, LWNmemoryPool *pool,
                                CompletionTracker *tracker, size_t size, uintptr_t start,
                                size_t minChunkSize, size_t maxChunkSize,
                                size_t alignment) :
                                TrackedChunkRingBuffer<uintptr_t>(tracker, start, size, minChunkSize, maxChunkSize, alignment),
                                m_cmdBuf(cmdBuf),
                                m_pool(pool),
                                m_lastChunk(start)
    {
    }

    // Allocate a new chunk of memory from the ring buffer and plug it into
    // the command buffer.
    bool setupNewChunk(size_t minRequiredChunkSize)
    {
        size_t reservedSize = getWriteSpace(m_lastChunk, minRequiredChunkSize);
        assert(reservedSize);
        lwnCommandBufferAddCommandMemory(m_cmdBuf, m_pool, m_lastChunk, reservedSize);
        return true;
    }

    // Update fences when a sync object is inserted.  We need to query the
    // write pointer from the command buffer since we're not tracking
    // continuously.
    void notifyFenceInserted(int fenceid)
    {
        size_t used = lwnCommandBufferGetCommandMemoryUsed(m_cmdBuf);
        setFence(fenceid, m_lastChunk + used);
    }

    // Handle an out-of-memory notification by grabbing and inserting a new
    // chunk of memory.
    void notifyOutOfMemory(size_t minSize)
    {
        size_t used = lwnCommandBufferGetCommandMemoryUsed(m_cmdBuf);
        syncWrite(m_lastChunk + used);
        setupNewChunk(minSize);
    }
};

//
// TrackedCommandMemRingBuffer utility class
//
// Utility class derived from TrackedRingBuffer that plugs ring buffer memory
// into the control memory of the specified command buffer.
//
class TrackedControlMemRingBuffer : public TrackedChunkRingBuffer < char * >
{
    LWNcommandBuffer *m_cmdBuf;         // command buffer owning the ring buffer
    char *m_lastChunk;                  // pointer to last chunk given to m_cmdBuf

public:
    TrackedControlMemRingBuffer(LWNcommandBuffer *cmdBuf, CompletionTracker *tracker,
                                int size, char *start,
                                size_t minChunkSize, size_t maxChunkSize,
                                size_t alignment) :
                                TrackedChunkRingBuffer<char *>(tracker, start, size, minChunkSize, maxChunkSize, alignment),
                                m_cmdBuf(cmdBuf),
                                m_lastChunk(start)
    {
    }

    // Allocate a new chunk of memory from the ring buffer and plug it into
    // the command buffer.
    bool setupNewChunk(size_t minRequiredChunkSize)
    {
        size_t reservedSize = getWriteSpace(m_lastChunk, minRequiredChunkSize);
        assert(reservedSize);
        lwnCommandBufferAddControlMemory(m_cmdBuf, m_lastChunk, reservedSize);
        return true;
    }

    // Update fences when a sync object is inserted.  We need to query the
    // write pointer from the command buffer since we're not tracking
    // continuously.
    void notifyFenceInserted(int fenceid)
    {
        size_t used = lwnCommandBufferGetControlMemoryUsed(m_cmdBuf);
        setFence(fenceid, m_lastChunk + used);
    }

    // Handle an out-of-memory notification by grabbing and inserting a new
    // chunk of memory.
    void notifyOutOfMemory(size_t minSize)
    {
        size_t used = lwnCommandBufferGetControlMemoryUsed(m_cmdBuf);
        syncWrite(m_lastChunk + used);
        setupNewChunk(minSize);
    }
};


bool CompletionTracker::addAllocator(Allocator allocator)
{
    AllocatorList::iterator it;
    for (it = m_allocators.begin(); it != m_allocators.end(); it++) {
        if (*it == allocator) {
            return false;
        }
    }
    m_allocators.push_back(allocator);
    return true;
}

bool CompletionTracker::removeAllocator(Allocator allocator)
{
    AllocatorList::iterator it;
    for (it = m_allocators.begin(); it != m_allocators.end(); it++) {
        if (*it == allocator) {
            m_allocators.erase(it);
            return true;
        }
    }
    return false;
}

void CompletionTracker::notifyFenceInserted(int fenceid)
{
    AllocatorList::iterator it;
    for (it = m_allocators.begin(); it != m_allocators.end(); it++) {
        Allocator allocator = *it;
        allocator->notifyFenceInserted(fenceid);
    }
}

void CompletionTracker::notifySyncCompleted(int fenceid)
{
    AllocatorList::iterator it;
    for (it = m_allocators.begin(); it != m_allocators.end(); it++) {
        Allocator allocator = *it;
        allocator->notifySyncCompleted(fenceid);
    }
}

void CompletionTracker::insertFence(LWNqueue *queue)
{
    // Before writing a new fence, wait on a previous fence if the ring buffer
    // is full.
    if (m_ring.isFull()) {
        updateGet(true);
    }

    int put = 0;
    size_t reserved = m_ring.getWriteSpace(put);
    assert(reserved);
    (void) reserved;

    lwnQueueFenceSync(queue, &m_objects[put], LWN_SYNC_CONDITION_ALL_GPU_COMMANDS_COMPLETE,
                      LWN_SYNC_FLAG_FLUSH_FOR_CPU_BIT);
    lwnQueueFlush(queue);
    notifyFenceInserted(put);


    put++;
    m_ring.syncWrite(put);
}

bool CompletionTracker::updateGet(bool wait /*= false*/)
{
    bool updated = false;
    uint64_t timeout = wait ? LWN_WAIT_TIMEOUT_MAXIMUM : LWN_WAIT_TIMEOUT_NONE;

    while (!m_ring.isEmpty()) {
        int get = m_ring.getRead();
        LWNsyncWaitResult condition = lwnSyncWait(&m_objects[get], timeout);
        if (condition == LWN_SYNC_WAIT_RESULT_TIMEOUT_EXPIRED) {
            break;
        }
        notifySyncCompleted(get);
        get = m_ring.wrapEntry(get + 1);
        m_ring.setRead(get);
        updated = true;
        timeout = LWN_WAIT_TIMEOUT_NONE;
    }
    return updated;
}


bool QueueCommandBuffer::init(LWNdevice *device, LWNqueue *queue, CompletionTracker *tracker)
{
    m_device = device;
    m_queue = queue;
    m_tracker = tracker;

    int minCommandMemSize = 0, minControlMemSize = 0;
    lwnDeviceGetInteger(m_device, LWN_DEVICE_INFO_COMMAND_BUFFER_MIN_COMMAND_SIZE, &minCommandMemSize);
    lwnDeviceGetInteger(m_device, LWN_DEVICE_INFO_COMMAND_BUFFER_MIN_CONTROL_SIZE, &minControlMemSize);
    m_minSupportedCommandMemSize = minCommandMemSize;
    m_minSupportedControlMemSize = minControlMemSize;

    // Set up the command buffer object for the queue.
    if (!lwnCommandBufferInitialize(this, device)) {
        return false;
    }

    // Initialize command and control memory trackers.
    if (!initCommand()) {
        return false;
    }
    if (!initControl()) {
        return false;
    }

    // Initialize command buffer usage counters.
    m_lastSubmitCounters = new Counters;
    if (!m_lastSubmitCounters) {
        return false;
    }
    m_lastSubmitCounters->commandMemUsage = 0;
    m_lastSubmitCounters->controlMemUsage = 0;

    // Set up out-of-memory callbacks for the command buffer object.
    lwnCommandBufferSetMemoryCallback(this, outOfMemory);
    lwnCommandBufferSetMemoryCallbackData(this, this);

    // Start recording; we keep the command buffer "open" for recording
    // continuously.
    lwnCommandBufferBeginRecording(this);
    return true;
}

bool QueueCommandBuffer::initCommand()
{
    assert(m_device);
    assert(m_tracker);
    assert(!m_commandMem);
    assert(!m_commandPool);
    assert(!m_commandPoolMemory);

    m_commandPoolMemory = lwnUtil::PoolStorageAlloc(CommandPoolAllocSize);

    // Set up the memory pool for command memory.
    m_commandPool = new LWNmemoryPool;
    if (!m_commandPool) {
        return false;
    }
    LWNmemoryPoolBuilder builder;
    lwnMemoryPoolBuilderSetDefaults(&builder);
    lwnMemoryPoolBuilderSetDevice(&builder, m_device);
    lwnMemoryPoolBuilderSetStorage(&builder, m_commandPoolMemory, CommandPoolAllocSize);
    lwnMemoryPoolBuilderSetFlags(&builder, (LWN_MEMORY_POOL_FLAGS_CPU_UNCACHED_BIT |
                                            LWN_MEMORY_POOL_FLAGS_GPU_UNCACHED_BIT));
    if (!lwnMemoryPoolInitialize(m_commandPool, &builder)) {
        delete m_commandPool;
        m_commandPool = NULL;
        return false;
    }


    // Set up the ring buffer tracking object for the command memory.
    m_commandMem = new TrackedCommandMemRingBuffer(this, m_commandPool, m_tracker, CommandPoolAllocSize, 0,
                                                   MinCommandChunkSize, MaxCommandChunkSize, CommandChunkAlignment);
    if (!m_commandMem) {
        return false;
    }
    m_commandMem->setupNewChunk(m_minSupportedCommandMemSize);

    return true;
}

bool QueueCommandBuffer::initControl()
{
    assert(m_tracker);
    assert(!m_controlMem);
    assert(!m_controlPool);

    // Set up the memory pool for command memory.
    m_controlPool = new char[ControlPoolAllocSize];
    if (!m_controlPool) {
        return false;
    }

    // Set up the ring buffer tracking object for the control memory.
    m_controlMem = new TrackedControlMemRingBuffer(this, m_tracker, ControlPoolAllocSize, m_controlPool,
                                                   MinControlChunkSize, MaxControlChunkSize, ControlChunkAlignment);
    if (!m_controlMem) {
        return false;
    }
    m_controlMem->setupNewChunk(m_minSupportedControlMemSize);

    return true;
}

void QueueCommandBuffer::destroy()
{
    lwnCommandBufferEndRecording(this);
    lwnCommandBufferFinalize(this);
    if (m_queue) {
        lwnQueueFinish(m_queue);
    }

    delete m_commandMem;
    if (m_commandPool) {
        lwnMemoryPoolFinalize(m_commandPool);
        delete m_commandPool;
    }
    delete m_controlMem;
    delete[] m_controlPool;
    delete m_lastSubmitCounters;
    lwnUtil::PoolStorageFree(m_commandPoolMemory);
}

void LWNAPIENTRY QueueCommandBuffer::outOfMemory(LWNcommandBuffer *cmdBuf, LWNcommandBufferMemoryEvent event,
                                                 size_t minSize, void *callbackData)
{
    QueueCommandBuffer *cb = (QueueCommandBuffer *) callbackData;
    if (event == LWN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_COMMAND_MEMORY) {
        cb->m_commandMem->notifyOutOfMemory(minSize);
    } else if (event == LWN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_CONTROL_MEMORY) {
        cb->m_controlMem->notifyOutOfMemory(minSize);
    } else {
        assert(!"Unknown command buffer event.");
    }

    // Slam the counters for the last submit to invalid values just to be sure
    // we won't match old counters if we have an out-of-memory event and then
    // fail to submit what we wrote.
    cb->m_lastSubmitCounters->commandMemUsage = -1;
    cb->m_lastSubmitCounters->controlMemUsage = -1;
}

void QueueCommandBuffer::getCounters(Counters *counters)
{
    counters->commandMemUsage = lwnCommandBufferGetCommandMemoryUsed(this);
    counters->controlMemUsage = lwnCommandBufferGetControlMemoryUsed(this);
}

void QueueCommandBuffer::checkUnflushedCommands()
{
    Counters counters;
    getCounters(&counters);
    assert(counters.commandMemUsage == m_lastSubmitCounters->commandMemUsage);
    assert(counters.controlMemUsage == m_lastSubmitCounters->controlMemUsage);
}

void QueueCommandBuffer::resetCounters()
{
    getCounters(m_lastSubmitCounters);
}

void QueueCommandBuffer::submit()
{
    LWNcommandHandle handle = lwnCommandBufferEndRecording(this);
    lwnQueueSubmitCommands(m_queue, 1, &handle);
    getCounters(m_lastSubmitCounters);
    lwnCommandBufferBeginRecording(this);
}

bool LWNcommandBufferMemoryManager::init(LWNdevice *device, CompletionTracker *tracker)
{
    m_device = device;

    int minCommandMemSize = 0, minControlMemSize = 0;
    lwnDeviceGetInteger(m_device, LWN_DEVICE_INFO_COMMAND_BUFFER_MIN_COMMAND_SIZE, &minCommandMemSize);
    lwnDeviceGetInteger(m_device, LWN_DEVICE_INFO_COMMAND_BUFFER_MIN_CONTROL_SIZE, &minControlMemSize);
    m_minSupportedCommandMemSize = minCommandMemSize;
    m_minSupportedControlMemSize = minControlMemSize;

    m_coherentPoolMemory = lwnUtil::PoolStorageAlloc(coherentPoolSize);

    m_coherentPool = new LWNmemoryPool;
    if (!m_coherentPool) {
        return false;
    }

    LWNmemoryPoolBuilder builder;
    lwnMemoryPoolBuilderSetDefaults(&builder);
    lwnMemoryPoolBuilderSetDevice(&builder, device);
    lwnMemoryPoolBuilderSetStorage(&builder, m_coherentPoolMemory, coherentPoolSize);
    lwnMemoryPoolBuilderSetFlags(&builder, (LWN_MEMORY_POOL_FLAGS_CPU_UNCACHED_BIT |
                                            LWN_MEMORY_POOL_FLAGS_GPU_UNCACHED_BIT));
    if (!lwnMemoryPoolInitialize(m_coherentPool, &builder)) {
        return false;
    }


    m_nonCoherentPoolMemory = lwnUtil::PoolStorageAlloc(nonCoherentPoolSize);

    m_nonCoherentPool = new LWNmemoryPool;
    if (!m_coherentPool) {
        return false;
    }

    lwnMemoryPoolBuilderSetStorage(&builder, m_nonCoherentPoolMemory, nonCoherentPoolSize);
    lwnMemoryPoolBuilderSetFlags(&builder, (LWN_MEMORY_POOL_FLAGS_CPU_CACHED_BIT |
                                            LWN_MEMORY_POOL_FLAGS_GPU_UNCACHED_BIT));
    if (!lwnMemoryPoolInitialize(m_nonCoherentPool, &builder)) {
        return false;
    }

    m_controlPool = new char[controlPoolSize];
    if (!m_controlPool) {
        return false;
    }

    m_coherentMem = new CommandMemory(tracker, 0, coherentPoolSize, coherentChunkSize, coherentChunkSize, 4);
    if (!m_coherentMem) {
        return false;
    }
    m_nonCoherentMem = new CommandMemory(tracker, 0, nonCoherentPoolSize, nonCoherentChunkSize, nonCoherentChunkSize, 4);
    if (!m_nonCoherentMem) {
        return false;
    }
    m_controlMem = new ControlMemory(tracker, m_controlPool, controlPoolSize, controlChunkSize, controlChunkSize, 8);
    if (!m_controlMem) {
        return false;
    }

    // Register callbacks so that the ring buffer managers can track the
    // insertion and completion of sync objects.
    tracker->addAllocator(m_coherentMem);
    tracker->addAllocator(m_nonCoherentMem);
    tracker->addAllocator(m_controlMem);

    return true;
}

void LWNcommandBufferMemoryManager::destroy()
{
    if (m_coherentPool) {
        lwnMemoryPoolFinalize(m_coherentPool);
    }
    if (m_nonCoherentPool) {
        lwnMemoryPoolFinalize(m_nonCoherentPool);
    }

    delete m_coherentPool;
    delete m_nonCoherentPool;
    delete m_controlPool;

    delete m_coherentMem;
    delete m_nonCoherentMem;
    delete m_controlMem;

    lwnUtil::PoolStorageFree(m_coherentPoolMemory);
    lwnUtil::PoolStorageFree(m_nonCoherentPoolMemory);
}

bool LWNcommandBufferMemoryManager::populateCommandBuffer(LWNcommandBuffer *cmdBuf, CommandMemType commandType)
{
    if (commandType == Coherent) {
        lwnCommandBufferSetMemoryCallback(cmdBuf, coherentCallback);
        addCommandMem(cmdBuf, m_coherentMem, m_coherentPool, m_minSupportedCommandMemSize);
    } else {
        lwnCommandBufferSetMemoryCallback(cmdBuf, nonCoherentCallback);
        addCommandMem(cmdBuf, m_nonCoherentMem, m_nonCoherentPool, m_minSupportedCommandMemSize);
    }
    addControlMem(cmdBuf, m_minSupportedControlMemSize);
    lwnCommandBufferSetMemoryCallbackData(cmdBuf, this);
    return true;
}

void LWNcommandBufferMemoryManager::addCommandMem(LWNcommandBuffer *cmdBuf, CommandMemory *cmdMem, LWNmemoryPool *cmdPool, size_t minRequiredSize)
{
    size_t cmdWrite;
    size_t reservedSize;
    reservedSize = cmdMem->getWriteSpace(cmdWrite, minRequiredSize);
    assert(reservedSize >= minRequiredSize);
    lwnCommandBufferAddCommandMemory(cmdBuf, cmdPool, cmdWrite, reservedSize);
    cmdMem->syncWrite(cmdWrite + reservedSize);
}

void LWNcommandBufferMemoryManager::addControlMem(LWNcommandBuffer *cmdBuf, size_t minRequiredSize)
{
    char *ctrlWrite;
    size_t reservedSize;
    reservedSize = m_controlMem->getWriteSpace(ctrlWrite, minRequiredSize);
    assert(reservedSize >= minRequiredSize);
    lwnCommandBufferAddControlMemory(cmdBuf, ctrlWrite, reservedSize);
    m_controlMem->syncWrite(ctrlWrite + reservedSize);
}

void LWNAPIENTRY LWNcommandBufferMemoryManager::coherentCallback(LWNcommandBuffer *cmdBuf, LWNcommandBufferMemoryEvent event, size_t minSize, void *data)
{
    LWNcommandBufferMemoryManager *mm = (LWNcommandBufferMemoryManager *) data;
    if (event == LWN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_COMMAND_MEMORY) {
        mm->addCommandMem(cmdBuf, mm->m_coherentMem, mm->m_coherentPool, minSize);
    } else if (event == LWN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_CONTROL_MEMORY) {
        mm->addControlMem(cmdBuf, minSize);
    } else {
        assert(!"Unknown command buffer event.");
    }
}

void LWNAPIENTRY LWNcommandBufferMemoryManager::nonCoherentCallback(LWNcommandBuffer *cmdBuf, LWNcommandBufferMemoryEvent event, size_t minSize, void *data)
{
    LWNcommandBufferMemoryManager *mm = (LWNcommandBufferMemoryManager *) data;
    if (event == LWN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_COMMAND_MEMORY) {
        mm->addCommandMem(cmdBuf, mm->m_nonCoherentMem, mm->m_nonCoherentPool, minSize);
    } else if (event == LWN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_CONTROL_MEMORY) {
        mm->addControlMem(cmdBuf, minSize);
    } else {
        assert(!"Unknown command buffer event.");
    }
}

//////////////////////////////////////////////////////////////////////////

// TexIDPool implementation

LWNsystemTexIDPool::LWNsystemTexIDPool(LWNdevice* device, LWNcommandBuffer *queueCB)
{
    mDevice = device;
    int textureSize, samplerSize;
    lwnDeviceGetInteger(device, LWN_DEVICE_INFO_TEXTURE_DESCRIPTOR_SIZE, &textureSize);
    lwnDeviceGetInteger(device, LWN_DEVICE_INFO_SAMPLER_DESCRIPTOR_SIZE, &samplerSize);
    lwnDeviceGetInteger(device, LWN_DEVICE_INFO_RESERVED_TEXTURE_DESCRIPTORS, &mNumReservedTextures);
    lwnDeviceGetInteger(device, LWN_DEVICE_INFO_RESERVED_SAMPLER_DESCRIPTORS, &mNumReservedSamplers);
    int targetSize = (mNumReservedSamplers + NUM_PUBLIC_SAMPLERS) * samplerSize +
                     (mNumReservedTextures + NUM_PUBLIC_TEXTURES) * textureSize;

    size_t poolSize = lwnUtil::PoolStorageSize(targetSize);
    mPoolMemory = lwnUtil::PoolStorageAlloc(poolSize);

    LWNmemoryPoolBuilder builder;
    lwnMemoryPoolBuilderSetDefaults(&builder);
    lwnMemoryPoolBuilderSetDevice(&builder, device);
    lwnMemoryPoolBuilderSetStorage(&builder, mPoolMemory, poolSize);
#if defined(_WIN32)
    lwnMemoryPoolBuilderSetFlags(&builder, LWN_MEMORY_POOL_TYPE_GPU_ONLY);
#else
    lwnMemoryPoolBuilderSetFlags(&builder, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
#endif
    lwnMemoryPoolInitialize(&mDescriptorPool, &builder);

    lwnSamplerPoolInitialize(&mSamplerPool, &mDescriptorPool, 0,
                                 mNumReservedSamplers + NUM_PUBLIC_SAMPLERS);
    lwnTexturePoolInitialize(&mTexturePool, &mDescriptorPool,
                                 (mNumReservedSamplers + NUM_PUBLIC_SAMPLERS) * samplerSize,
                                 mNumReservedTextures + NUM_PUBLIC_TEXTURES);
    memset(mTextureIDs, 0, sizeof(mTextureIDs));
    memset(mSamplerIDs, 0, sizeof(mSamplerIDs));
    lwnCommandBufferSetSamplerPool(queueCB, &mSamplerPool);
    lwnCommandBufferSetTexturePool(queueCB, &mTexturePool);
    mLastTextureWord = 0;
    mLastSamplerWord = 0;
}

LWNsystemTexIDPool::~LWNsystemTexIDPool()
{
    lwnSamplerPoolFinalize(&mSamplerPool);
    lwnTexturePoolFinalize(&mTexturePool);
    lwnMemoryPoolFinalize(&mDescriptorPool);
    lwnUtil::PoolStorageFree(mPoolMemory);
}

int LWNsystemTexIDPool::AllocTextureID()
{
    return AllocID(mTextureIDs, NUM_PUBLIC_TEXTURES, mNumReservedTextures, &mLastTextureWord);
}

int LWNsystemTexIDPool::AllocSamplerID()
{
    return AllocID(mSamplerIDs, NUM_PUBLIC_SAMPLERS, mNumReservedSamplers, &mLastSamplerWord);
}

void LWNsystemTexIDPool::FreeTextureID(int id)
{
    FreeID(id, mTextureIDs, NUM_PUBLIC_TEXTURES, mNumReservedTextures);
}

void LWNsystemTexIDPool::FreeSamplerID(int id)
{
    FreeID(id, mSamplerIDs, NUM_PUBLIC_SAMPLERS, mNumReservedSamplers);
}

int LWNsystemTexIDPool::Register(LWNtexture* texture)
{
    assert(texture);
    int id = AllocTextureID();
    lwnTexturePoolRegisterTexture(&mTexturePool, id, texture, NULL);
    mTextureIDMap[intptr_t(texture)] = id;
    return id;
}

int LWNsystemTexIDPool::Register(LWNsampler* sampler)
{
    assert(sampler);
    int id = AllocSamplerID();
    lwnSamplerPoolRegisterSampler(&mSamplerPool, id, sampler);
    mSamplerIDMap[intptr_t(sampler)] = id;
    return id;
}

void LWNsystemTexIDPool::Deregister(LWNtexture* texture)
{
    if (!texture) {
        return;
    }
    ObjectIDMap::iterator it = mTextureIDMap.find(intptr_t(texture));
    assert(it != mTextureIDMap.end());
    FreeTextureID(it->second);
    mTextureIDMap.erase(it);
}

void LWNsystemTexIDPool::Deregister(LWNsampler* sampler)
{
    if (!sampler) {
        return;
    }
    ObjectIDMap::iterator it = mSamplerIDMap.find(intptr_t(sampler));
    assert(it != mSamplerIDMap.end());
    FreeSamplerID(it->second);
    mSamplerIDMap.erase(it);
}

int LWNsystemTexIDPool::AllocID(uint32_t* idPool, int numPublicIDs, int numReservedIDs, int* lastWord)
{
    // As a heuristic, try to allocate conselwtive IDs in different words. This spreads the allocations
    // throughout the pool and hopefully makes finding a free slot faster.
    int numWords = numPublicIDs / 32;
    int wordIndex;
    for (wordIndex = (*lastWord + 1) % numWords; wordIndex != *lastWord;
         wordIndex = (wordIndex + 1) % numWords) {
        if (idPool[wordIndex] != 0xffffffff) {
            break;
        }
    }
    if (wordIndex == *lastWord) {
        // You're gonna need a bigger boat.
        assert(!"TexIDPool capacity is too small for this test.");
        return 0;
    }
    uint32_t word = idPool[wordIndex];
    int bitIndex = 0;
    if ((word & 0xffff) == 0xffff) {
        bitIndex += 16;
        word >>= 16;
    }
    if ((word & 0xff) == 0xff) {
        bitIndex += 8;
        word >>= 8;
    }
    if ((word & 0xf) == 0xf) {
        bitIndex += 4;
        word >>= 4;
    }
    if ((word & 0x3) == 0x3) {
        bitIndex += 2;
        word >>= 2;
    }
    if ((word & 0x1) == 0x1) {
        bitIndex += 1;
    }
    idPool[wordIndex] |= (0x1 << bitIndex);
    *lastWord = wordIndex;
    return wordIndex * 32 + bitIndex + numReservedIDs;
}

void LWNsystemTexIDPool::FreeID(int id, uint32_t* idPool, int numPublicIDs, int numReservedIDs)
{
    // Assume that an assigned ID of 0 means that the allocation never actually happened,
    // a la free(NULL).
    if (id == 0) {
        return;
    }
    id -= numReservedIDs;
    assert(id < numPublicIDs);
    int wordIndex = id / 32;
    int bitIndex = id % 32;
    idPool[wordIndex] &= ~(1 << bitIndex);
}

////////////////////////////////////////////////////////////////////////////

LWNutility g_lwn;

CompletionTracker *initCompletionTracker(LWNdevice *device, int size)
{
    CompletionTracker *tracker = new CompletionTracker(device, size);
    return tracker;
}

void insertCompletionTrackerFence(CompletionTracker *tracker, LWNqueue *queue)
{
    tracker->insertFence(queue);
}
