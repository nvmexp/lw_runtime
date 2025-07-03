/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef __lwnUtil_CommandMemImpl_h__
#define __lwnUtil_CommandMemImpl_h__

#define LOG_COMPLETION_INFO 0
#if LOG_COMPLETION_INFO
#define LOG_COMPLETION(x)           printf x
#else
#define LOG_COMPLETION(x)
#endif

namespace lwnUtil {

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

    int put;
    int reserved = m_ring.getWriteSpace(put);
    assert(reserved);
    (void)reserved;

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


bool CommandBufferMemoryManager::init(LWNdevice *device, CompletionTracker *tracker)
{
    m_device = device;

    lwnDeviceGetInteger(m_device, LWN_DEVICE_INFO_COMMAND_BUFFER_MIN_COMMAND_SIZE, &m_minSupportedCommandMemSize);
    lwnDeviceGetInteger(m_device, LWN_DEVICE_INFO_COMMAND_BUFFER_MIN_CONTROL_SIZE, &m_minSupportedControlMemSize);

    m_coherentPoolMemory = PoolStorageAlloc(coherentPoolSize);

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

    m_nonCoherentPoolMemory = PoolStorageAlloc(nonCoherentPoolSize);

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

void CommandBufferMemoryManager::destroy()
{
    if (m_coherentPool) {
        lwnMemoryPoolFinalize(m_coherentPool);
    }
    if (m_nonCoherentPool) {
        lwnMemoryPoolFinalize(m_nonCoherentPool);
    }

    delete m_coherentPool;
    delete m_nonCoherentPool;
    delete[] m_controlPool;

    delete m_coherentMem;
    delete m_nonCoherentMem;
    delete m_controlMem;

    PoolStorageFree(m_coherentPoolMemory);
    PoolStorageFree(m_nonCoherentPoolMemory);
}

bool CommandBufferMemoryManager::populateCommandBuffer(LWNcommandBuffer *cmdBuf, CommandMemType commandType)
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

void CommandBufferMemoryManager::addCommandMem(LWNcommandBuffer *cmdBuf, CommandMemory *cmdMem, LWNmemoryPool *cmdPool, LWNsizeiptr minRequiredSize)
{
    LWNuintptr cmdWrite;
    int reservedSize;
    reservedSize = cmdMem->getWriteSpace(cmdWrite, minRequiredSize);
    assert(reservedSize >= minRequiredSize);
    lwnCommandBufferAddCommandMemory(cmdBuf, cmdPool, cmdWrite, reservedSize);
    cmdMem->syncWrite(cmdWrite + reservedSize);
}

void CommandBufferMemoryManager::addControlMem(LWNcommandBuffer *cmdBuf, LWNsizeiptr minRequiredSize)
{
    char *ctrlWrite;
    int reservedSize;
    reservedSize = m_controlMem->getWriteSpace(ctrlWrite, minRequiredSize);
    assert(reservedSize >= minRequiredSize);
    lwnCommandBufferAddControlMemory(cmdBuf, ctrlWrite, reservedSize);
    m_controlMem->syncWrite(ctrlWrite + reservedSize);
}

void LWNAPIENTRY CommandBufferMemoryManager::coherentCallback(LWNcommandBuffer *cmdBuf, LWNcommandBufferMemoryEvent event, 
                                                              size_t minSize, void *data)
{
    CommandBufferMemoryManager *mm = (CommandBufferMemoryManager *) data;
    if (event == LWN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_COMMAND_MEMORY) {
        mm->addCommandMem(cmdBuf, mm->m_coherentMem, mm->m_coherentPool, minSize);
    } else if (event == LWN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_CONTROL_MEMORY) {
        mm->addControlMem(cmdBuf, minSize);
    } else {
        assert(!"Unknown command buffer event.");
    }
}

void LWNAPIENTRY CommandBufferMemoryManager::nonCoherentCallback(LWNcommandBuffer *cmdBuf, LWNcommandBufferMemoryEvent event, 
                                                                 size_t minSize, void *data)
{
    CommandBufferMemoryManager *mm = (CommandBufferMemoryManager *) data;
    if (event == LWN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_COMMAND_MEMORY) {
        mm->addCommandMem(cmdBuf, mm->m_nonCoherentMem, mm->m_nonCoherentPool, minSize);
    } else if (event == LWN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_CONTROL_MEMORY) {
        mm->addControlMem(cmdBuf, minSize);
    } else {
        assert(!"Unknown command buffer event.");
    }
}

} // namespace lwnUtil

#endif // #ifndef __lwnUtil_CommandMemImpl_h__
