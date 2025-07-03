/*
 * Copyright (c) 2017 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwntest_cpp.h"
#include "lwn_utils.h"
#include "lwn_basic.h"
#if defined(LW_TEGRA)
#include <lwnExt/lwnExt_Internal.h>
#include <lwrm_memmgr.h>
#endif

using namespace lwn;

namespace lwnTest
{
    void fail(...);
    bool failed();
}

class MemoryPoolExternalTest
{
public:
    LWNTEST_CppMethods();
};


lwString MemoryPoolExternalTest::getDescription() const
{
    lwStringBuf sb;
    sb <<   "Test creating memory pool from external memory handle.\n"
            "Output should be solid green background.";
    return sb.str();
}

#if !defined(LW_TEGRA)
int MemoryPoolExternalTest::isSupported() const
{
    return false;
}
void MemoryPoolExternalTest::doGraphics() const
{
}

#else
int MemoryPoolExternalTest::isSupported() const
{
    return true; // lwogCheckLWNAPIVersion(,);
}

LwRmDeviceHandle g_hMemoryManager;
PFNLWNMEMORYPOOLGETNATIVEHANDLELWXPROC lwnMemoryPoolGetNativeHandleLWX;
PFNLWNMEMORYPOOLBUILDERSETNATIVEHANDLELWXPROC lwnMemoryPoolBuilderSetNativeHandleLWX;
PFLWNMEMORYPOOLBUILDERGETNATIVEHANDLELWXPROC  lwnMemoryPoolBuilderGetNativeHandleLWX;

struct Memory
{
    bool allocated;
    void *storage;
    LwRmMemHandle hMem;
    size_t size;

    bool allocate()
    {
        storage = aligned_alloc(64 * 1024, size);
        if (!storage) {
            return false;
        }

        const bool CPU_CACHED = true;
        const int SMMU_ADDRESS_ALIGNMENT = 64 << 10;
        const LwOsMemAttribute coherency = CPU_CACHED ? LwOsMemAttribute_WriteBack : LwOsMemAttribute_WriteCombined;
        LWRM_DEFINE_MEM_HANDLE_ATTR(attr);
        LWRM_MEM_HANDLE_SET_ATTR(attr, SMMU_ADDRESS_ALIGNMENT, coherency,
                                 size, LwRmMemTags_None);
        LWRM_MEM_HANDLE_SET_VA_ATTR(attr, (uint64_t)storage);
        if (LwRmMemHandleAllocAttr(g_hMemoryManager, &attr, &hMem)) {
            free(storage);
            storage = NULL;
            return false;
        }

        allocated = true;
        return true;
    }

    Memory(size_t size) : allocated(false), storage(NULL), hMem(0), size(size)
    {
    }

    ~Memory()
    {
        if (allocated) {
            free(storage);
            LwRmMemHandleFree(hMem);
        }
    }
};

static void testExternalPool(bool compression, bool physical)
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    MemoryPool importedPool;
    MemoryPoolBuilder poolBuilder;
    const int size = 64 * 1024;
    Memory memory(size);

    if (!memory.allocate()) {
        lwnTest::fail("Failed to allocate %d bytes or create memory handle", size);
        return;
    }

    // Create a memory pool from imported handle.
    MemoryPoolFlags poolFlags = MemoryPoolFlags::CPU_CACHED;
    if (compression) {
        poolFlags |= MemoryPoolFlags::COMPRESSIBLE;
    }
    if (physical) {
        poolFlags |= MemoryPoolFlags::PHYSICAL | MemoryPoolFlags::GPU_NO_ACCESS;
    } else {
        poolFlags |= MemoryPoolFlags::GPU_CACHED;
    }
    poolBuilder.SetDevice(device)
               .SetDefaults()
               .SetFlags(poolFlags);
    lwnMemoryPoolBuilderSetNativeHandleLWX((LWNmemoryPoolBuilder *)&poolBuilder, uint64_t(memory.hMem));

    if (uint64_t(memory.hMem) != lwnMemoryPoolBuilderGetNativeHandleLWX((LWNmemoryPoolBuilder *)&poolBuilder)) {
        lwnTest::fail("Queried native handle does not match the one in MemoryPoolBuilder");
        return;
    }
    if (!importedPool.Initialize(&poolBuilder)) {
        lwnTest::fail("Failed to initialize %s pool from imported mem handle", physical ? "physical" : "normal");
        return;
    }

    if (memory.hMem != (LwRmMemHandle)lwnMemoryPoolGetNativeHandleLWX((LWNmemoryPool *)&importedPool)) {
        lwnTest::fail("Queried native handle does not match the one provided in initialization");
    }

    if (importedPool.Map() != memory.storage) {
        lwnTest::fail("Pointer returned from Map (%p) does not match the one provided in initialization (%p)", importedPool.Map(), memory.storage);
    }

    // For physical imported pool, also verify that we can successfully map it to GPU.
    if (physical) {
        MemoryPool virtualPool;
        poolBuilder.SetFlags(MemoryPoolFlags::CPU_NO_ACCESS | MemoryPoolFlags::GPU_CACHED | MemoryPoolFlags::VIRTUAL);
        poolBuilder.SetStorage(NULL, size);
        if (!virtualPool.Initialize(&poolBuilder)) {
            lwnTest::fail("Failed to create virtual pool");
        } else {
            MappingRequest req = {
                .physicalPool = &importedPool,
                .physicalOffset = 0,
                .virtualOffset = 0,
                .size = uint64_t(size),
            };

            if (compression) {
                // Map first as compressible to trigger lazy allocation of compbits.
                req.storageClass = 0x1db;
                if (!virtualPool.MapVirtual(1, &req)) {
                    lwnTest::fail("MapVirtual with compressible storage class failed");
                }
            }

            // Map as linear.
            req.storageClass = LWN_STORAGE_CLASS_BUFFER;
            if (!virtualPool.MapVirtual(1, &req)) {
                lwnTest::fail("MapVirtual with buffer storage class failed");
            }

            virtualPool.Finalize();
        }
    }

    importedPool.Finalize();
}


void MemoryPoolExternalTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    LwRmOpenNew(&g_hMemoryManager);

    lwnMemoryPoolGetNativeHandleLWX =
        (PFNLWNMEMORYPOOLGETNATIVEHANDLELWXPROC)device->GetProcAddress("lwnMemoryPoolGetNativeHandleLWX");
    if (!lwnMemoryPoolGetNativeHandleLWX) {
        lwnTest::fail("Failed to query function pointer for lwnMemoryPoolGetNativeHandleLWX");
    }
    lwnMemoryPoolBuilderSetNativeHandleLWX =
        (PFNLWNMEMORYPOOLBUILDERSETNATIVEHANDLELWXPROC)device->GetProcAddress("lwnMemoryPoolBuilderSetNativeHandleLWX");
    if (!lwnMemoryPoolBuilderSetNativeHandleLWX) {
        lwnTest::fail("Failed to query function pointer for lwnMemoryPoolBuilderSetNativeHandleLWX");
    }
    lwnMemoryPoolBuilderGetNativeHandleLWX =
        (PFLWNMEMORYPOOLBUILDERGETNATIVEHANDLELWXPROC)device->GetProcAddress("lwnMemoryPoolBuilderGetNativeHandleLWX");
    if (!lwnMemoryPoolBuilderGetNativeHandleLWX) {
        lwnTest::fail("Failed to query function pointer for lwnMemoryPoolBuilderGetNativeHandleLWX");
    }

    if (!lwnTest::failed()) {
        for (int compression = 0; compression < 2; compression++)
        for (int physical = 0; physical < 2; physical++) {
            testExternalPool(compression, physical);
        }
    }

    if (lwnTest::failed()) {
        queueCB.ClearColor(0, 1.0, 0.0, 0.0, 1.0);
    } else {
        queueCB.ClearColor(0, 0.0, 1.0, 0.0, 1.0);
    }

    queueCB.submit();
    queue->Finish();

    LwRmClose(g_hMemoryManager);
}
#endif

OGTEST_CppTest(MemoryPoolExternalTest, lwn_mempool_external, );
