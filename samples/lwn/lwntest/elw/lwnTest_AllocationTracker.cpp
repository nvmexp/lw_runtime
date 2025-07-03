/*
* Copyright (c) 2015, Lwpu Corporation.  All rights reserved.
*
* THE INFORMATION CONTAINED HEREIN IS PROPRIETARY AND CONFIDENTIAL TO
* LWPU, CORPORATION.  USE, REPRODUCTION OR DISCLOSURE TO ANY THIRD PARTY
* IS SUBJECT TO WRITTEN PRE-APPROVAL BY LWPU, CORPORATION.
*
*
*/

#include "lwntest_c.h"
#include "lwnTest/lwnTest_AllocationTracker.h"

#include "cmdline.h"
#include "lwn_utils.h"

namespace lwnTest {

// Global enable for automatically tracking new object allocations.  Can be
// tweaked to temporarily disable tracking to allow allocations that live
// longer than one test run.
static bool g_trackLWNObjects = false;

// Global enable for automatically registering and tracking texture and
// sampler handles when using Create functions or a memory pool allocator.
// lwnTextureInitialize and lwnSamplerInitialize are not affected.
static bool g_autoRegisterTexIDs = true;

} // namespace lwnTest

//
// Create C entry points to emulate allocation and freeing API objects as
// though it were done by the driver.  Also includes support for tracking
// unfreed allocations so they can be cleaned up at the end of test runs.
//

// Macros to stamp out emulation functions that free various application-side
// API objects, either explicitly via emulated lwn*Free() functions or in a
// "cleanup" phase at the end of a test run via lwnTest::AllocationTracker
// template classes.  Some objects don't have Initialize/Finalize methods, so
// don't need allocation trackers.  Texture and sampler objects also need a
// special variant that removes ("deregisters") them from the texture ID pool.
#define LWN_FREE_EMULATE_FINALIZE(CType, TypeName)                              \
    static lwnTest::AllocationTracker<CType *> lwnAllocationTracker_##CType;    \
    namespace lwnTest {                                                         \
        template <> void                                                        \
        lwnTest::AllocationTracker<CType *>::CleanupAPIObject(CType *&object)   \
        {                                                                       \
            lwn##TypeName##Finalize(object);                                    \
            delete object;                                                      \
        }                                                                       \
    }                                                                           \
    void lwn##TypeName##Free(CType *object)                                     \
    {                                                                           \
        if (lwnTest::g_trackLWNObjects) {                                       \
            lwnAllocationTracker_##CType.TrackFree(object);                     \
        }                                                                       \
        lwn##TypeName##Finalize(object);                                        \
        delete object;                                                          \
    }

#define LWN_FREE_EMULATE_NO_FINALIZE(CType, TypeName)                           \
    static lwnTest::AllocationTracker<CType *> lwnAllocationTracker_##CType;    \
    namespace lwnTest {                                                         \
        template <> void                                                        \
        lwnTest::AllocationTracker<CType *>::CleanupAPIObject(CType *&object)   \
        {                                                                       \
            delete object;                                                      \
        }                                                                       \
    }                                                                           \
    void lwn##TypeName##Free(CType *object)                                     \
    {                                                                           \
        if (lwnTest::g_trackLWNObjects) {                                       \
            lwnAllocationTracker_##CType.TrackFree(object);                     \
        }                                                                       \
        delete object;                                                          \
    }

LWN_FREE_EMULATE_FINALIZE(LWNdevice, Device);
LWNdevice *lwnCreateDevice()
{
    LWNdeviceBuilder builder;
    lwnDeviceBuilderSetDefaults(&builder);

    if (lwnDebugEnabled) {
        lwnDeviceBuilderSetFlags(&builder, LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_2_BIT | LWN_DEVICE_FLAG_DEBUG_SKIP_CALLS_ON_ERROR_BIT);
    } else {
        lwnDeviceBuilderSetFlags(&builder, 0);
    }
    
    LWNdevice *object = new LWNdevice;
    if (!lwnDeviceInitialize(object, &builder)) {
        delete object;
        return NULL;
    }
    if (lwnTest::g_trackLWNObjects) {
        lwnAllocationTracker_LWNdevice.TrackCreate(object);
    }
    return object;
}

LWN_FREE_EMULATE_FINALIZE(LWNqueue, Queue);
LWNqueue *lwnDeviceCreateQueue(LWNdevice *device)
{
    LWNqueue *object = new LWNqueue;
    LWNqueueBuilder qb;
    lwnQueueBuilderSetDevice(&qb, device);
    lwnQueueBuilderSetDefaults(&qb);
    if (!lwnQueueInitialize(object, &qb)) {
        delete object;
        return NULL;
    }
    if (lwnTest::g_trackLWNObjects) {
        lwnAllocationTracker_LWNqueue.TrackCreate(object);
    }
    return object;
}

LWN_FREE_EMULATE_FINALIZE(LWNcommandBuffer, CommandBuffer);
LWNcommandBuffer *lwnDeviceCreateCommandBuffer(LWNdevice *device)
{
    LWNcommandBuffer *object = new LWNcommandBuffer;
    if (!lwnCommandBufferInitialize(object, device)) {
        delete object;
        return NULL;
    }
    if (lwnTest::g_trackLWNObjects) {
        lwnAllocationTracker_LWNcommandBuffer.TrackCreate(object);
    }
    return object;
}

LWN_FREE_EMULATE_NO_FINALIZE(LWNblendState, BlendState);
LWNblendState *lwnDeviceCreateBlendState(LWNdevice *device)
{
    LWNblendState *object = new LWNblendState;
    lwnBlendStateSetDefaults(object);
    if (lwnTest::g_trackLWNObjects) {
        lwnAllocationTracker_LWNblendState.TrackCreate(object);
    }
    return object;
}

LWN_FREE_EMULATE_NO_FINALIZE(LWNchannelMaskState, ChannelMaskState);
LWNchannelMaskState *lwnDeviceCreateChannelMaskState(LWNdevice *device)
{
    LWNchannelMaskState * object = new LWNchannelMaskState;
    lwnChannelMaskStateSetDefaults(object);
    if (lwnTest::g_trackLWNObjects) {
        lwnAllocationTracker_LWNchannelMaskState.TrackCreate(object);
    }
    return object;
}

LWN_FREE_EMULATE_NO_FINALIZE(LWNcolorState, ColorState);
LWNcolorState *lwnDeviceCreateColorState(LWNdevice *device)
{
    LWNcolorState * object = new LWNcolorState;
    lwnColorStateSetDefaults(object);
    if (lwnTest::g_trackLWNObjects) {
        lwnAllocationTracker_LWNcolorState.TrackCreate(object);
    }
    return object;
}

LWN_FREE_EMULATE_NO_FINALIZE(LWNdepthStencilState, DepthStencilState);
LWNdepthStencilState *lwnDeviceCreateDepthStencilState(LWNdevice *device)
{
    LWNdepthStencilState * object = new LWNdepthStencilState;
    lwnDepthStencilStateSetDefaults(object);
    if (lwnTest::g_trackLWNObjects) {
        lwnAllocationTracker_LWNdepthStencilState.TrackCreate(object);
    }
    return object;
}

LWN_FREE_EMULATE_NO_FINALIZE(LWNmultisampleState, MultisampleState);
LWNmultisampleState *lwnDeviceCreateMultisampleState(LWNdevice *device)
{
    LWNmultisampleState * object = new LWNmultisampleState;
    lwnMultisampleStateSetDefaults(object);
    if (lwnTest::g_trackLWNObjects) {
        lwnAllocationTracker_LWNmultisampleState.TrackCreate(object);
    }
    return object;
}

LWN_FREE_EMULATE_NO_FINALIZE(LWNpolygonState, PolygonState);
LWNpolygonState *lwnDeviceCreatePolygonState(LWNdevice *device)
{
    LWNpolygonState * object = new LWNpolygonState;
    lwnPolygonStateSetDefaults(object);
    if (lwnTest::g_trackLWNObjects) {
        lwnAllocationTracker_LWNpolygonState.TrackCreate(object);
    }
    return object;
}

LWN_FREE_EMULATE_NO_FINALIZE(LWLwertexAttribState, VertexAttribState);
LWLwertexAttribState *lwnDeviceCreateVertexAttribState(LWNdevice *device)
{
    LWLwertexAttribState * object = new LWLwertexAttribState;
    lwlwertexAttribStateSetDefaults(object);
    if (lwnTest::g_trackLWNObjects) {
        lwnAllocationTracker_LWLwertexAttribState.TrackCreate(object);
    }
    return object;
}

LWN_FREE_EMULATE_NO_FINALIZE(LWLwertexStreamState, VertexStreamState);
LWLwertexStreamState *lwnDeviceCreateVertexStreamState(LWNdevice *device)
{
    LWLwertexStreamState * object = new LWLwertexStreamState;
    lwlwertexStreamStateSetDefaults(object);
    if (lwnTest::g_trackLWNObjects) {
        lwnAllocationTracker_LWLwertexStreamState.TrackCreate(object);
    }
    return object;
}

LWN_FREE_EMULATE_FINALIZE(LWNprogram, Program);
LWNprogram *lwnDeviceCreateProgram(LWNdevice *device)
{
    LWNprogram *object = new LWNprogram;
    if (!lwnProgramInitialize(object, device)) {
        delete object;
        return NULL;
    }
    if (lwnTest::g_trackLWNObjects) {
        lwnAllocationTracker_LWNprogram.TrackCreate(object);
    }
    return object;
}

static lwnTest::AllocationTracker<LWNmemoryPool *> lwnAllocationTracker_LWNmemoryPool;
namespace lwnTest {
    template <> void
    lwnTest::AllocationTracker<LWNmemoryPool *>::CleanupAPIObject(LWNmemoryPool *&object)
    {
        lwnMemoryPoolFinalize(object);
        free(object);
    }
}
void lwnMemoryPoolFree(LWNmemoryPool *object)
{
    lwnMemoryPoolFinalize(object);
    if (lwnTest::g_trackLWNObjects) {
        lwnAllocationTracker_LWNmemoryPool.TrackFree(object);
    }
    free(object);
}


LWNmemoryPool * lwnDeviceCreateMemoryPool(LWNdevice *device, void *memory, size_t size, LWNmemoryPoolFlags poolFlags)
{
    LWNmemoryPool *object;

    if (!memory && !(poolFlags & LWN_MEMORY_POOL_FLAGS_VIRTUAL_BIT)) {
        // If the caller passes NULL pointer, then this helper function is responsible for allocating storage.

        // Align up size to minimum granularity.
        size = PoolStorageSize(size);

        // Reserve space for the memory pool structure and storage.
        size_t storageSize = sizeof(LWNmemoryPool) + size + LWN_MEMORY_POOL_STORAGE_ALIGNMENT;
        void *storage = malloc(storageSize);
        if (!storage) {
            return NULL;
        }

        object = (LWNmemoryPool *)storage;
        memory = AlignPointer(object + 1, LWN_MEMORY_POOL_STORAGE_ALIGNMENT);
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
    if (lwnTest::g_trackLWNObjects) {
        lwnAllocationTracker_LWNmemoryPool.TrackCreate(object);
    }
    return object;
}

LWN_FREE_EMULATE_NO_FINALIZE(LWNbufferBuilder, BufferBuilder);
LWNbufferBuilder *lwnDeviceCreateBufferBuilder(LWNdevice *device)
{
    LWNbufferBuilder *object = new LWNbufferBuilder;
    lwnBufferBuilderSetDevice(object, device);
    lwnBufferBuilderSetDefaults(object);
    if (lwnTest::g_trackLWNObjects) {
        lwnAllocationTracker_LWNbufferBuilder.TrackCreate(object);
    }
    return object;
}

LWN_FREE_EMULATE_NO_FINALIZE(LWNtextureBuilder, TextureBuilder);
LWNtextureBuilder *lwnDeviceCreateTextureBuilder(LWNdevice *device)
{
    LWNtextureBuilder *object = new LWNtextureBuilder;
    lwnTextureBuilderSetDevice(object, device);
    lwnTextureBuilderSetDefaults(object);
    if (lwnTest::g_trackLWNObjects) {
        lwnAllocationTracker_LWNtextureBuilder.TrackCreate(object);
    }
    return object;
}

LWN_FREE_EMULATE_NO_FINALIZE(LWNsamplerBuilder, SamplerBuilder);
LWNsamplerBuilder *lwnDeviceCreateSamplerBuilder(LWNdevice *device)
{
    LWNsamplerBuilder *object = new LWNsamplerBuilder;
    lwnSamplerBuilderSetDevice(object, device);
    lwnSamplerBuilderSetDefaults(object);
    if (lwnTest::g_trackLWNObjects) {
        lwnAllocationTracker_LWNsamplerBuilder.TrackCreate(object);
    }
    return object;
}

LWN_FREE_EMULATE_NO_FINALIZE(LWNwindowBuilder, WindowBuilder);
LWNwindowBuilder *lwnDeviceCreateWindowBuilder(LWNdevice *device)
{
    LWNwindowBuilder *object = new LWNwindowBuilder;
    lwnWindowBuilderSetDevice(object, device);
    lwnWindowBuilderSetDefaults(object);
    if (lwnTest::g_trackLWNObjects) {
        lwnAllocationTracker_LWNwindowBuilder.TrackCreate(object);
    }
    return object;
}

LWN_FREE_EMULATE_FINALIZE(LWNsync, Sync);
LWNsync *lwnDeviceCreateSync(LWNdevice *device)
{
    LWNsync *object = new LWNsync;
    if (!lwnSyncInitialize(object, device)) {
        delete object;
        return NULL;
    }
    if (lwnTest::g_trackLWNObjects) {
        lwnAllocationTracker_LWNsync.TrackCreate(object);
    }
    return object;
}

static lwnTest::AllocationTracker<LWNsampler *> lwnAllocationTracker_LWNsampler;
namespace lwnTest {
    template <> void
    lwnTest::AllocationTracker<LWNsampler *>::CleanupAPIObject(LWNsampler *&object)
    {
        if (g_autoRegisterTexIDs) {
            g_lwnTexIDPool->Deregister(object);
        }
        lwnSamplerFinalize(object);
        delete object;
    }
}
void lwnSamplerFree(LWNsampler *object)
{
    if (g_autoRegisterTexIDs) {
        g_lwnTexIDPool->Deregister(object);
    }
    if (lwnTest::g_trackLWNObjects) {
        lwnAllocationTracker_LWNsampler.TrackFree(object);
    }
    lwnSamplerFinalize(object);
    delete object;
}

LWNsampler *lwnSamplerBuilderCreateSampler(LWNsamplerBuilder *builder)
{
    __LWNsamplerInternal *alloc = new __LWNsamplerInternal;
    LWNsampler *object = &alloc->sampler;
    if (!lwnSamplerInitialize(object, builder)) {
        delete object;
        return NULL;
    }
    if (g_autoRegisterTexIDs) {
        alloc->lastRegisteredID = g_lwnTexIDPool->Register(object);
    } else {
        alloc->lastRegisteredID = 0;
    }
    if (lwnTest::g_trackLWNObjects) {
        lwnAllocationTracker_LWNsampler.TrackCreate(object);
    }
    return object;
}

LWN_FREE_EMULATE_FINALIZE(LWNbuffer, Buffer);
LWNbuffer *lwnBufferBuilderCreateBufferFromPool(LWNbufferBuilder *builder, LWNmemoryPool *storage, ptrdiff_t offset, size_t size)
{
    LWNbuffer *object = new LWNbuffer;
    lwnBufferBuilderSetStorage(builder, storage, offset, size);
    if (!lwnBufferInitialize(object, builder)) {
        delete object;
        return NULL;
    }
    if (lwnTest::g_trackLWNObjects) {
        lwnAllocationTracker_LWNbuffer.TrackCreate(object);
    }
    return object;
}

static lwnTest::AllocationTracker<LWNtexture *> lwnAllocationTracker_LWNtexture;
namespace lwnTest {
    template <> void
    lwnTest::AllocationTracker<LWNtexture *>::CleanupAPIObject(LWNtexture *&object)
    {
        if (g_autoRegisterTexIDs) {
            g_lwnTexIDPool->Deregister(object);
            g_lwnTexIDPool->DeregisterImage(object);
        }
        lwnTextureFinalize(object);
        delete object;
    }
}

void lwnTextureFree(LWNtexture *object)
{
    if (g_autoRegisterTexIDs) {
        g_lwnTexIDPool->Deregister(object);
        g_lwnTexIDPool->DeregisterImage(object);
    }
    if (lwnTest::g_trackLWNObjects) {
        lwnAllocationTracker_LWNtexture.TrackFree(object);
    }
    lwnTextureFinalize(object);
    delete object;
}

LWN_FREE_EMULATE_FINALIZE(LWNwindow, Window);
LWNwindow *lwnWindowBuilderCreateWindow(LWNwindowBuilder *builder)
{
    LWNwindow *object = new LWNwindow;
    if (!lwnWindowInitialize(object, builder)) {
        delete object;
        return NULL;
    }
    if (lwnTest::g_trackLWNObjects) {
        lwnAllocationTracker_LWNwindow.TrackCreate(object);
    }
    return object;
}

LWNtexture *lwnTextureBuilderCreateTextureFromPool(LWNtextureBuilder *builder, LWNmemoryPool *storage, ptrdiff_t offset)
{
    __LWNtextureInternal *alloc = new __LWNtextureInternal;
    LWNtexture *object = &alloc->texture;
    lwnTextureBuilderSetStorage(builder, storage, offset);
    if (!lwnTextureInitialize(object, builder)) {
        delete object;
        return NULL;
    }
    if (g_autoRegisterTexIDs) {
        alloc->lastRegisteredTextureID = g_lwnTexIDPool->Register(object);
    } else {
        alloc->lastRegisteredTextureID = 0;
    }
    if (lwnTest::g_trackLWNObjects) {
        lwnAllocationTracker_LWNtexture.TrackCreate(object);
    }
    return object;
}

namespace lwnTest {

// The global allocation cleanup function simply calls into the
// TrackedObjectCleanup method of each object type's allocation tracker to
// delete any objects of that type. Note that the order of cleanup is chosen
// to avoid dangling references. E.g., textures must be destroyed before
// any memory pools they use, and devices should be destroyed last.
extern void allocationCleanup()
{
    lwnAllocationTracker_LWNblendState.TrackedObjectCleanup();
    lwnAllocationTracker_LWNchannelMaskState.TrackedObjectCleanup();
    lwnAllocationTracker_LWNcolorState.TrackedObjectCleanup();
    lwnAllocationTracker_LWNdepthStencilState.TrackedObjectCleanup();
    lwnAllocationTracker_LWNmultisampleState.TrackedObjectCleanup();
    lwnAllocationTracker_LWNpolygonState.TrackedObjectCleanup();
    lwnAllocationTracker_LWLwertexAttribState.TrackedObjectCleanup();
    lwnAllocationTracker_LWLwertexStreamState.TrackedObjectCleanup();
    lwnAllocationTracker_LWNbufferBuilder.TrackedObjectCleanup();
    lwnAllocationTracker_LWNtextureBuilder.TrackedObjectCleanup();
    lwnAllocationTracker_LWNsamplerBuilder.TrackedObjectCleanup();
    lwnAllocationTracker_LWNwindowBuilder.TrackedObjectCleanup();

    lwnAllocationTracker_LWNqueue.TrackedObjectCleanup();
    lwnAllocationTracker_LWNcommandBuffer.TrackedObjectCleanup();
    lwnAllocationTracker_LWNprogram.TrackedObjectCleanup();
    lwnAllocationTracker_LWNbuffer.TrackedObjectCleanup();
    lwnAllocationTracker_LWNtexture.TrackedObjectCleanup();
    lwnAllocationTracker_LWNsampler.TrackedObjectCleanup();
    lwnAllocationTracker_LWNsync.TrackedObjectCleanup();
    lwnAllocationTracker_LWNwindow.TrackedObjectCleanup();
    lwnAllocationTracker_LWNmemoryPool.TrackedObjectCleanup();
    lwnAllocationTracker_LWNdevice.TrackedObjectCleanup();
}

static bool g_LWNObjectTrackingStateIsPushed = false;
static bool g_SavedLWNObjectTrackingState = false;

void EnableLWNObjectTracking()
{
    g_trackLWNObjects = true;
}

void DisableLWNObjectTracking()
{
    g_trackLWNObjects = false;
}

extern void PushLWNObjectTracking()
{
    assert(!g_LWNObjectTrackingStateIsPushed);
    g_LWNObjectTrackingStateIsPushed = true;
    g_SavedLWNObjectTrackingState = g_trackLWNObjects;
}

extern void PopLWNObjectTracking()
{
    assert(g_LWNObjectTrackingStateIsPushed);
    g_LWNObjectTrackingStateIsPushed = false;
    g_trackLWNObjects = g_SavedLWNObjectTrackingState;
}

bool IsLWNObjectTrackingEnabled()
{
    return g_trackLWNObjects;
}

} // namespace lwnTest
