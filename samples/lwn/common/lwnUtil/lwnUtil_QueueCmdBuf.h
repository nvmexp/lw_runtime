/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef __lwnUtil_QueueCmdBuf_h__
#define __lwnUtil_QueueCmdBuf_h__

#include "lwnUtil_Interface.h"

#include "lwn/lwn.h"
#if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_CPP
#include "lwn/lwn_Cpp.h"
#endif

#include "lwnUtil/lwnUtil_CommandMem.h"

namespace lwnUtil {

//
// The QueueCommandBuffer classes are used to manage a command buffer directly
// connected to an LWN queue object.
//
// The base class (QueueCommandBufferBase) derives from the LWNcommandBuffer
// object and provides methods to initialize and use the command buffer with
// the provided queue, including support for setting up command buffer memory
// and submitting commands.
//
// The QueueCommandBufferTemplate template class is provided to allow tests
// using the C++ command buffer classes to reuse the QueueCommandBufferBase
// object via reinterpret_cast.  These classes derive from C++ command buffer
// classes, which use the same storage as the C LWNcommandBuffer object that
// QueueCommandBufferBase derives from.
//
class QueueCommandBufferBase;
template <typename T> class QueueCommandBufferTemplate;


// Base queue command buffer class, using the LWN C interface command buffer
// object (LWNcommandBuffer) as the base class.
class QueueCommandBufferBase : public LWNcommandBuffer
{
    // Disable automatic copy constructor and assignment operators for this
    // object (by creating unimplemented methods) because LWN doesn't permit
    // copying base LWNcommandBuffer objects.
    QueueCommandBufferBase(const QueueCommandBufferBase &other);
    QueueCommandBufferBase & operator = (const QueueCommandBufferBase &other);

    // Device and queue owning the command buffer.
    LWNdevice                           *m_device;
    LWNqueue                            *m_queue;

    // Completion tracker associated with the queue/command buffer.
    CompletionTracker                   *m_tracker;

    // On HOS, this is the memory used for internal storage in m_commandPool.
    // If NULL, the storage is managed by the driver.
    void                                *m_commandPoolMemory;

    // Tracked ring buffer objects managing the command and control memory
    // usage of the command buffer.
    TrackedCommandMemRingBuffer         *m_commandMem;
    LWNmemoryPool                       *m_commandPool;

    TrackedControlMemRingBuffer         *m_controlMem;
    char                                *m_controlPool;

    // Usage counters tracking the write pointers at the last submit; used to
    // assert that there are no unsubmitted commands at the end of a frame.
    struct Counters {
        LWNsizeiptr                     commandMemUsage;
        LWNsizeiptr                     controlMemUsage;
    }                                   *m_lastSubmitCounters;

    LWNint                              m_minSupportedCommandMemSize;
    LWNint                              m_minSupportedControlMemSize;

    // Flags to track initialization and recording state of the embedded
    // command buffer object.
    bool                                m_initialized;
    bool                                m_recordingStarted;

    // Constants specifying the size and alignment of command buffer memory,
    // including the sizes of chunks we dole out to the command buffer object.
    // To stress out-of-memory conditions, set Max*ChunkSize to low values.
    static const LWNsizeiptr MinCommandChunkSize = 0x1000;
    static const LWNsizeiptr MaxCommandChunkSize = 0x40000000;
    static const LWNuintptr  CommandChunkAlignment = 4;
    static const int         CommandPoolAllocSize = 1024 * 1024;

    static const LWNsizeiptr MinControlChunkSize = 0x400;
    static const LWNsizeiptr MaxControlChunkSize = 0x40000000;
    static const LWNuintptr  ControlChunkAlignment = 8;
    static const int         ControlPoolAllocSize = 256 * 1024;

public:
    QueueCommandBufferBase() :
        m_device(NULL), m_queue(NULL), m_tracker(NULL),
        m_commandPoolMemory(NULL), m_commandMem(NULL),
        m_commandPool(NULL), m_controlMem(NULL),
        m_controlPool(NULL), m_lastSubmitCounters(NULL),
        m_minSupportedCommandMemSize(1),
        m_minSupportedControlMemSize(1),
        m_initialized(false),
        m_recordingStarted(false)
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

    // Template methods to automatically colwert the base queue command buffer
    // class to one of the queue command buffer template classes (with C++
    // command buffer interfaces).
    template <typename T>
    operator QueueCommandBufferTemplate<T> &()
    {
        return *(reinterpret_cast<QueueCommandBufferTemplate <T> *>(this));
    }
};

template <typename T>
class QueueCommandBufferTemplate : public T
{
    // Disable the default constructor and copy/assignment operators for the
    // template class because it doesn't provide the storage for any of the
    // members of QueueCommandBufferBase.  We should never create unique
    // template class instances; these should only be reinterpreted from the
    // base queue command buffer class.
    QueueCommandBufferTemplate();
    QueueCommandBufferTemplate(QueueCommandBufferTemplate &);
    QueueCommandBufferTemplate & operator = (const QueueCommandBufferTemplate &other);
    QueueCommandBufferTemplate(QueueCommandBufferBase &);
    QueueCommandBufferTemplate & operator = (const QueueCommandBufferBase &other);

    // Method to colwert a pointer to the template class back to the base
    // queue command buffer class.
    class QueueCommandBufferBase *base()
    {
        return reinterpret_cast<class QueueCommandBufferBase *>(this);
    }

public:
    // Template class methods providing the same interface.
    void submit()                   { base()->submit(); }
    void resetCounters()            { base()->resetCounters(); }
    void checkUnflushedCommands()   { base()->checkUnflushedCommands(); }

#if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_C
    // For the native C interface, allow colwersion to an LWNcommandBuffer.
    operator LWNcommandBuffer *() {
        return base();
    }
#else
    // For the native C++ interface, allow colwersion to an
    // lwn::CommandBuffer.
    operator lwn::CommandBuffer *() {
        LWNcommandBuffer *ccb = base();
        return reinterpret_cast<lwn::CommandBuffer *>(ccb);
    }
#endif
};


#if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_CPP
typedef QueueCommandBufferTemplate<lwn::CommandBuffer> QueueCommandBuffer;
#else
typedef QueueCommandBufferTemplate<LWNcommandBuffer> QueueCommandBuffer;
#endif

} // namespace lwnUtil

#endif // #ifndef __lwnUtil_QueueCmdBuf_h__
