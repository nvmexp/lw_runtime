/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef __lwnUtil_CommandMem_h__
#define __lwnUtil_CommandMem_h__

#include "lwnUtil_Interface.h"

#include <list>

#include "lwnUtil_Interface.h"

namespace lwnUtil {

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
    inline int getWriteSpace(T &current, int space = 1,
                             WriteSpaceRequest requestType = GetExactSize)
    {
        assert(space < m_size);
        assert(!m_writing);

        // If we're near the end of the buffer and don't have enough
        // contiguous space, try to reserve padding to the end of the buffer
        // and wrap around to the start.
        if (m_write + space > m_end) {
            if (!getWriteSpace(current, (int) (m_end - m_write))) {
                return 0;
            }
            m_writing = false;  // this isn't a real reservation
            m_write = m_start;
        }

        assert(m_write + space <= m_end);
        if (space > (int) getAvailableSpace()) {
            return 0;
        }

        // If this is a request for the maximum size, compute the amount of
        // available contiguous and update the size request accordingly.
        if (requestType == GetMaximumSize) {
            int maxSpace;
            if (m_read > m_write) {
                // Compute the space between the write and read pointers,
                // leaving the last one blank.
                maxSpace = (int) ((m_read - m_write) - 1);
            } else {
                // Compute the space between the write pointer and the end of
                // the buffer, leaving the last entry unused if the read
                // pointer is at the start.
                maxSpace = (int) (m_end - m_write);
                if (m_read == m_start) {
                    maxSpace--;
                }
            }
            assert(maxSpace >= space);
            space = maxSpace;
        }

        m_writing = true;
        m_writeLimit = m_write + space;
        current = m_write;
        return space;
    }

    // Request the maximum number of contiguous entries available, as long as
    // it's at least <minSpace>.
    inline int getWriteSpaceMax(T &current, int minSpace)
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

#if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_CPP
    //
    // Methods to provide a native C++ interface to the core CompletionTracker
    // class, using reinterpret_cast to colwert between C and C++ object
    // types.
    //
    void insertFence(lwn::Queue *queue)
    {
        insertFence(reinterpret_cast<LWNqueue *>(queue));
    }
#endif
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
    LWNuintptr  m_alignmentMask;        // alignment required for each allocation

public:
    TrackedRingBuffer(CompletionTracker *tracker, T start, int size, LWNuintptr alignment) :
        RingBufferManager<T>(size, start),
        TrackedAllocator(tracker),
        m_alignmentMask(~(alignment - 1))
    {
        m_fences = new T[size];
        for (int i = 0; i < size; i++) {
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

    void setAlignment(LWNuintptr alignment)     { m_alignmentMask = ~(alignment - 1); }

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
    int getWriteSpace(T &current, LWNsizeiptr minSpace, LWNsizeiptr maxSpace)
    {
        CompletionTracker *tracker = getTracker();
        LWNsizeiptr reservedSize = RingBufferManager<T>::getWriteSpaceMax(current, (int) minSpace);
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
        return (int) reservedSize;
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
class TrackedChunkRingBuffer : public TrackedRingBuffer<T>
{
private:
    LWNsizeiptr m_minChunkSize;         // minimum chunk size required for an allocation
    LWNsizeiptr m_maxChunkSize;         // maximum chunk size allowed for an allocation
public:
    TrackedChunkRingBuffer(CompletionTracker *tracker, T start, int size,
                           LWNsizeiptr minChunkSize, LWNsizeiptr maxChunkSize,
                           LWNuintptr alignment) :
        TrackedRingBuffer<T>(tracker, start, size, alignment),
        m_minChunkSize(minChunkSize),
        m_maxChunkSize(maxChunkSize)
    {
    }

    void setMaxChunkSize(LWNsizeiptr size)      { m_maxChunkSize = size; }
    void setMinChunkSize(LWNsizeiptr size)      { m_minChunkSize = size; }

    int getWriteSpace(T &current, LWNsizeiptr minRequiredChunkSize)
    {
        LWNsizeiptr minChunkSize = m_minChunkSize;
        LWNsizeiptr maxChunkSize = m_maxChunkSize;

        if (minRequiredChunkSize > minChunkSize) {
            minChunkSize = minRequiredChunkSize;

            if (minChunkSize > maxChunkSize) {
                maxChunkSize = minChunkSize;
            }
        }
        
        int space = TrackedRingBuffer<T>::getWriteSpace(current, minChunkSize, maxChunkSize);
        return space;
    }
};


//
// TrackedCommandMemRingBuffer utility class
//
// Utility class derived from TrackedRingBuffer that plugs ring buffer memory
// into the command memory of the specified command buffer.
//
class TrackedCommandMemRingBuffer : public TrackedChunkRingBuffer < LWNuintptr >
{
    LWNcommandBuffer *m_cmdBuf;         // command buffer owning the ring buffer
    LWNmemoryPool *m_pool;              // memory pool providing storage
    LWNuintptr m_lastChunk;             // offset of last chunk given to m_cmdBuf

public:
    TrackedCommandMemRingBuffer(LWNcommandBuffer *cmdBuf, LWNmemoryPool *pool,
                                CompletionTracker *tracker, int size, LWNuintptr start,
                                LWNsizeiptr minChunkSize, LWNsizeiptr maxChunkSize,
                                LWNuintptr alignment) :
        TrackedChunkRingBuffer<LWNuintptr>(tracker, start, size, minChunkSize, maxChunkSize, alignment), 
        m_cmdBuf(cmdBuf),
        m_pool(pool),
        m_lastChunk(start)
    {
    }

    // Allocate a new chunk of memory from the ring buffer and plug it into
    // the command buffer.
    bool setupNewChunk(LWNsizeiptr minRequiredChunkSize)
    {
        int reservedSize = getWriteSpace(m_lastChunk, minRequiredChunkSize);
        assert(reservedSize);
        lwnCommandBufferAddCommandMemory(m_cmdBuf, m_pool, m_lastChunk, reservedSize);
        return true;
    }

    // Update fences when a sync object is inserted.  We need to query the
    // write pointer from the command buffer since we're not tracking
    // continuously.
    void notifyFenceInserted(int fenceid)
    {
        LWNsizeiptr used = lwnCommandBufferGetCommandMemoryUsed(m_cmdBuf);
        setFence(fenceid, m_lastChunk + used);
    }

    // Handle an out-of-memory notification by grabbing and inserting a new
    // chunk of memory.
    void notifyOutOfMemory(LWNsizeiptr minSize)
    {
        LWNsizeiptr used = lwnCommandBufferGetCommandMemoryUsed(m_cmdBuf);
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
                                LWNsizeiptr minChunkSize, LWNsizeiptr maxChunkSize,
                                LWNuintptr alignment) :
        TrackedChunkRingBuffer<char *>(tracker, start, size, minChunkSize, maxChunkSize, alignment),
        m_cmdBuf(cmdBuf),
        m_lastChunk(start)
    {
    }

    // Allocate a new chunk of memory from the ring buffer and plug it into
    // the command buffer.
    bool setupNewChunk(LWNsizeiptr minRequiredChunkSize)
    {
        int reservedSize = getWriteSpace(m_lastChunk, minRequiredChunkSize);
        assert(reservedSize);
        lwnCommandBufferAddControlMemory(m_cmdBuf, m_lastChunk, reservedSize);
        return true;
    }

    // Update fences when a sync object is inserted.  We need to query the
    // write pointer from the command buffer since we're not tracking
    // continuously.
    void notifyFenceInserted(int fenceid)
    {
        LWNsizeiptr used = lwnCommandBufferGetControlMemoryUsed(m_cmdBuf);
        setFence(fenceid, m_lastChunk + used);
    }

    // Handle an out-of-memory notification by grabbing and inserting a new
    // chunk of memory.
    void notifyOutOfMemory(LWNsizeiptr minSize)
    {
        LWNsizeiptr used = lwnCommandBufferGetControlMemoryUsed(m_cmdBuf);
        syncWrite(m_lastChunk + used);
        setupNewChunk(minSize);
    }
};


//
// CommandBufferMemoryManager utility class
//
// Utility class holding memory that can be used to easily back API command
// buffer objects.  Holds command memory from both coherent and non-coherent
// pools, plus malloc control memory.
//
class CommandBufferMemoryManager
{
private:
    typedef TrackedChunkRingBuffer<LWNuintptr>  CommandMemory;
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

    LWNint              m_minSupportedCommandMemSize;
    LWNint              m_minSupportedControlMemSize;

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
    CommandBufferMemoryManager() : 
        m_device(NULL),
        m_coherentPool(NULL), m_nonCoherentPool(NULL), m_controlPool(NULL),
        m_coherentMem(NULL), m_nonCoherentMem(NULL), m_controlMem(NULL),
        m_minSupportedCommandMemSize(1), m_minSupportedControlMemSize(1)
    {}

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
    void addCommandMem(LWNcommandBuffer *cmdBuf, CommandMemory *cmdMem, LWNmemoryPool *cmdPool, LWNsizeiptr minRequiredSize);

    // Add control memory from the pool to <cmdBuf>.
    void addControlMem(LWNcommandBuffer *cmdBuf, LWNsizeiptr minRequiredSize);

#if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_CPP
    //
    // Methods to provide a native C++ interface to the core
    // CommandBufferMemoryManager classes, using reinterpret_cast to colwert
    // between C and C++ object types.
    //
    bool populateCommandBuffer(lwn::CommandBuffer *cmdBuf, CommandMemType commandType)
    {
        LWNcommandBuffer *ccb = reinterpret_cast<LWNcommandBuffer *>(cmdBuf);
        return populateCommandBuffer(ccb, commandType);
    }
    void addCommandMem(lwn::CommandBuffer *cmdBuf, CommandMemory *cmdMem, lwn::MemoryPool *cmdPool, LWNsizeiptr minRequiredSize)
    {
        LWNcommandBuffer *ccb = reinterpret_cast<LWNcommandBuffer *>(cmdBuf);
        LWNmemoryPool *cpool = reinterpret_cast<LWNmemoryPool *>(cmdPool);
        addCommandMem(ccb, cmdMem, cpool, minRequiredSize);
    }
    void addControlMem(lwn::CommandBuffer *cmdBuf, LWNsizeiptr minRequiredSize)
    {
        LWNcommandBuffer *ccb = reinterpret_cast<LWNcommandBuffer *>(cmdBuf);
        addControlMem(ccb, minRequiredSize);
    }
#endif

private:
    // Callback functions used when a managed command buffer runs out of
    // memory.
    static void LWNAPIENTRY coherentCallback(LWNcommandBuffer *cmdBuf, LWNcommandBufferMemoryEvent event,
                                             size_t minSize, void *data);
    static void LWNAPIENTRY nonCoherentCallback(LWNcommandBuffer *cmdBuf, LWNcommandBufferMemoryEvent event,
                                                size_t minSize, void *data);

};

}   // namespace lwnUtil

#endif // #ifndef __lwnUtil_CommandMem_h__
