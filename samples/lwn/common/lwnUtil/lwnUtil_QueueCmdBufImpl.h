/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef __lwnUtil_QueueCmdBufImpl_h__
#define __lwnUtil_QueueCmdBufImpl_h__

#include "lwnUtil_QueueCmdBuf.h"

namespace lwnUtil {

bool QueueCommandBufferBase::init(LWNdevice *device, LWNqueue *queue, CompletionTracker *tracker)
{
    LWNcommandBuffer *cmdBuf = this;
    m_device = device;
    m_queue = queue;
    m_tracker = tracker;

    lwnDeviceGetInteger(m_device, LWN_DEVICE_INFO_COMMAND_BUFFER_MIN_COMMAND_SIZE, &m_minSupportedCommandMemSize);
    lwnDeviceGetInteger(m_device, LWN_DEVICE_INFO_COMMAND_BUFFER_MIN_CONTROL_SIZE, &m_minSupportedControlMemSize);

    // Set up the command buffer object for the queue.
    if (!lwnCommandBufferInitialize(cmdBuf, device)) {
        return false;
    }
    m_initialized = true;

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
    lwnCommandBufferSetMemoryCallback(cmdBuf, outOfMemory);
    lwnCommandBufferSetMemoryCallbackData(cmdBuf, this);

    // Start recording; we keep the command buffer "open" for recording
    // continuously.
    lwnCommandBufferBeginRecording(cmdBuf);
    m_recordingStarted = true;

    return true;
}

bool QueueCommandBufferBase::initCommand()
{
    LWNcommandBuffer *cmdBuf = this;
    assert(cmdBuf);
    assert(m_device);
    assert(m_tracker);
    assert(!m_commandMem);
    assert(!m_commandPool);
    assert(!m_commandPoolMemory);

    m_commandPoolMemory = PoolStorageAlloc(CommandPoolAllocSize);

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
    m_commandMem = new TrackedCommandMemRingBuffer(cmdBuf, m_commandPool, m_tracker, CommandPoolAllocSize, 0,
                                                     MinCommandChunkSize, MaxCommandChunkSize, CommandChunkAlignment);
    if (!m_commandMem) {
        return false;
    }
    m_commandMem->setupNewChunk(m_minSupportedCommandMemSize);

    return true;
}

bool QueueCommandBufferBase::initControl()
{
    LWNcommandBuffer *cmdBuf = this;
    assert(cmdBuf);
    assert(m_tracker);
    assert(!m_controlMem);
    assert(!m_controlPool);

    // Set up the memory pool for command memory.
    m_controlPool = new char[ControlPoolAllocSize];
    if (!m_controlPool) {
        return false;
    }

    // Set up the ring buffer tracking object for the control memory.
    m_controlMem = new TrackedControlMemRingBuffer(cmdBuf, m_tracker, ControlPoolAllocSize, m_controlPool,
                                                   MinControlChunkSize, MaxControlChunkSize, ControlChunkAlignment);
    if (!m_controlMem) {
        return false;
    }
    m_controlMem->setupNewChunk(m_minSupportedControlMemSize);

    return true;
}

void QueueCommandBufferBase::destroy()
{
    LWNcommandBuffer *cmdBuf = this;
    if (m_recordingStarted) {
        lwnCommandBufferEndRecording(cmdBuf);
    }
    if (m_initialized) {
        lwnCommandBufferFinalize(cmdBuf);
    }
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
    PoolStorageFree(m_commandPoolMemory);
}

void LWNAPIENTRY QueueCommandBufferBase::outOfMemory(LWNcommandBuffer *cmdBuf, LWNcommandBufferMemoryEvent event,
                                                     size_t minSize, void *callbackData)
{
    QueueCommandBufferBase *cb = (QueueCommandBufferBase *) callbackData;
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

void QueueCommandBufferBase::getCounters(Counters *counters)
{
    LWNcommandBuffer *cmdBuf = this;
    counters->commandMemUsage = lwnCommandBufferGetCommandMemoryUsed(cmdBuf);
    counters->controlMemUsage = lwnCommandBufferGetControlMemoryUsed(cmdBuf);
}

void QueueCommandBufferBase::checkUnflushedCommands()
{
    Counters counters;
    getCounters(&counters);
    assert(counters.commandMemUsage == m_lastSubmitCounters->commandMemUsage);
    assert(counters.controlMemUsage == m_lastSubmitCounters->controlMemUsage);
}

void QueueCommandBufferBase::resetCounters()
{
    getCounters(m_lastSubmitCounters);
}

void QueueCommandBufferBase::submit()
{
    LWNcommandBuffer *cmdBuf = this;
    LWNcommandHandle handle = lwnCommandBufferEndRecording(cmdBuf);
    lwnQueueSubmitCommands(m_queue, 1, &handle);

    lwnCommandBufferBeginRecording(cmdBuf);

    // Record the current command and control memory usage immediately after beginning
    // recording, so we can check that the caller didn't leave behind any unfinished
    // command buffer work. We check after beginning recording because the debug layer
    // or other code might insert commands at the beginning of the command set.
    getCounters(m_lastSubmitCounters);
}

} // namespace lwnUtil;

#endif // #ifndef __lwnUtil_QueueCmdBufImpl_h__
