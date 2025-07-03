// Copyright LWPU Corporation 2015
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include "BufferStorage.hpp"
#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/IlwalidOperation.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/exceptions/MemoryAllocationFailed.h>
#include <prodlib/system/Knobs.h>
#include <corelib/misc/String.h>
#include <algorithm>
#include <memory.h>

using namespace prodlib::bvhtools;
using namespace prodlib;

//------------------------------------------------------------------------

namespace
{
Knob<bool>  k_memScribbleOn(          RT_DSTRING("bvhtools.memScribble"),            false, RT_DSTRING("Turn on memory scribbling") );
Knob<int>   k_memScribbleValueHost(   RT_DSTRING("bvhtools.memScribbleValueHost"),   0xcd,  RT_DSTRING("Byte value to scribbled into host allocations") );
}

//------------------------------------------------------------------------

BufferStorage::BufferStorage(void)
:   m_refCount          (1),
    m_numBytes          (0),

    m_isExternal        (false),
    m_extPtr            (NULL),
    m_extNumBytes       (0),
    m_extMemSpace       (MemorySpace_None),

    m_childType         (ChildType_None),
    m_parent            (NULL),
    m_topmost           (NULL),

    m_isMaterialized    (false),
    m_lwdaUtils         (NULL),
    m_allocExtra        (0.0f),
    m_ofsInTopmost      (0),

    m_initStatus        (InitStatus_JustMaterialized),
    m_memSpacesInSync   (MemorySpace_None),
    m_asyncEvent        (NULL),

    m_lwrPtr            (NULL),
    m_lwrAccType        (AccessType_None),
    m_lwrMemSpace       (MemorySpace_None)
{
    m_topmost = this;

    for (int i = 0; i < (int)MemSpaceIdx_Max; i++)
    {
        m_allocSize[i] = 0;
        m_allocPtr[i]  = NULL;
        m_rawPtr[i]    = NULL;
    }
}

//------------------------------------------------------------------------

BufferStorage::~BufferStorage(void)
{
    // TODO: Don't throw in destructor. Do something else in the driver.
    //RT_ASSERT(!m_children.size());

    // Wait for async memcopies to finish.
    // Note: We must use silent=true here, since destructors should never throw exceptions.

    waitEvent(m_asyncEvent, m_lwdaUtils, true);

    // Unmaterialize.

    markAsMaterialized(false, m_lwdaUtils);

    // Free memory allocations.

    for (int i = 1; (i & MemorySpace_Any) != 0; i <<= 1)
        freeMemSpace((MemorySpace)i);

    // Detach from the parent.

    if (m_parent)
    {
        std::vector<BufferStorage*>& c = m_parent->m_children;
        c.erase(std::remove(c.begin(), c.end(), this), c.end());
        m_parent->decRefCount();
    }
}

//------------------------------------------------------------------------

void BufferStorage::initExternal(unsigned char* ptr, size_t numBytes, MemorySpace memSpace)
{
    RT_ASSERT(!m_isExternal);
    RT_ASSERT(!m_isMaterialized);
    RT_ASSERT(!m_numBytes && m_childType == ChildType_None);

    if (!ptr && numBytes)
        throw IlwalidValue(RT_EXCEPTION_INFO, "NULL pointer specified for a non-empty external buffer!");

    if (getMemSpaceIdx(memSpace) == MemSpaceIdx_Max)
        throw IlwalidValue(RT_EXCEPTION_INFO, "Invalid MemorySpace specified for an external buffer!", memSpace);

    m_isExternal    = true;
    m_extPtr        = ptr;
    m_extNumBytes   = numBytes;
    m_extMemSpace   = memSpace;
    m_numBytes      = numBytes;
    m_initStatus    = InitStatus_Initialized;
}

//------------------------------------------------------------------------

size_t BufferStorage::getNumBytes(void) const
{
    if (m_childType != ChildType_None && !m_isMaterialized)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "The size of an aggregate/overlay buffer is undefined until it has been materialized!");

    return m_numBytes;
}

//------------------------------------------------------------------------

size_t BufferStorage::getOffsetInTopmost(void) const
{
    if (!m_numBytes)
        return 0;

    if (!m_isMaterialized)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "The offset of a buffer is undefined until it has been materialized!");

    return m_ofsInTopmost;
}

//------------------------------------------------------------------------

size_t BufferStorage::getAllocSize(void) const
{
    if (!m_isMaterialized)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "Allocation size of a buffer is undefined until it has been materialized!");

    if (m_parent)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "getAllocSize() is not allowed on a child buffer!");

    size_t allocSize = m_numBytes;
    if (!m_isExternal)
        allocSize += std::max((size_t)((float)allocSize * m_allocExtra), (size_t)0);
    return allocSize;
}

//------------------------------------------------------------------------

void BufferStorage::setNumBytes(size_t numBytes)
{
    if (m_childType != ChildType_None && !m_isMaterialized)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "Resizing an aggregate/overlay buffer is not allowed!");

    if (numBytes == m_numBytes)
        return;

    if (m_isMaterialized)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "Resizing a buffer is not allowed after it has been materialized!");

    if (m_isExternal)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "Resizing an external buffer is not allowed!");

    if (m_childType != ChildType_None)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "Resizing an aggregate/overlay buffer is not allowed!");

    m_numBytes = numBytes;
}

//------------------------------------------------------------------------

void BufferStorage::aggregate(BufferStorage* child)
{
    if (m_childType == ChildType_Overlay)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "Tried to turn an overlay buffer into an aggregate!");

    attachChild(child);

    if (m_childType == ChildType_None)
    {
        m_childType = ChildType_Aggregate;
        if (!m_isExternal)
            m_numBytes = 0; // filled in when the buffer is materialized
    }
}

//------------------------------------------------------------------------

void BufferStorage::overlay(BufferStorage* child)
{
    if (m_childType == ChildType_Aggregate)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "Tried to turn an aggregate buffer into an overlay!");

    attachChild(child);

    if (m_childType == ChildType_None)
    {
        m_childType = ChildType_Overlay;
        if (!m_isExternal)
            m_numBytes = 0; // filled in when the buffer is materialized
    }
}

//------------------------------------------------------------------------

void BufferStorage::detachChildren(void)
{
    if (m_isMaterialized)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "Modifying the layout of a buffer is not allowed after it has been materialized!");

    if (m_childType == ChildType_None)
        return;

    // The children are not going to remember any pending async ops; wait for them to finish first.

    waitEvent(getTopmost()->m_asyncEvent, m_lwdaUtils);

    // Mark children as topmost.

    for (int i = 0; i < (int)m_children.size(); i++)
    {
        BufferStorage* child = m_children[i];
        child->m_parent = NULL;
        child->m_topmost = child;
    }

    // Clear child array.

    m_refCount -= (int)m_children.size();
    m_childType = ChildType_None;
    m_children.clear();

    // Reset size.

    if (!m_isExternal)
        m_numBytes = 0;

    // No more references => delete.

    if (!m_refCount)
        delete this;
}

//------------------------------------------------------------------------

void BufferStorage::materialize(LwdaUtils* lwdaUtils)
{
    if (m_parent)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "Materializing a child buffer is not allowed!");

    if (m_isMaterialized && m_lwdaUtils != lwdaUtils)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "Tried to materialize a buffer that had already been materialized with a different LwdaUtils!");

    if (m_isMaterialized)
        return;

    // Materialize.

    markAsMaterialized(true, lwdaUtils);
    resolveLayout();

    // Non-external buffer => mark as uninitialized.
    // External buffer => mark as up-to-date unless explicitly marked as uninitialized.

    if (!m_isExternal)
        markAsUninitialized(InitStatus_JustMaterialized);

    else if (m_initStatus != InitStatus_Initialized)
        markAsUninitialized(m_initStatus);

    else
    {
        access(AccessType_MarkAsInitialized, m_extMemSpace);
        access(AccessType_SyncIfInitialized, m_extMemSpace);
    }

    // Make sure there are no leftover accesses.

    cancelAccess(true);
}

//------------------------------------------------------------------------

void BufferStorage::unmaterialize(void)
{
    if (m_parent)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "Unmaterializing a child buffer is not allowed!");

    cancelAccess(true);
    markAsMaterialized(false, m_lwdaUtils);
}

//------------------------------------------------------------------------

void BufferStorage::markAsUninitialized(InitStatus reason, bool ignoreIfEqual)
{
    RT_ASSERT(reason >= 0 && reason < InitStatus_Max);
    RT_ASSERT(reason != InitStatus_Initialized);

    if (ignoreIfEqual && m_initStatus == reason)
        return;

    m_initStatus = reason;
    m_memSpacesInSync = MemorySpace_None;
    cancelAccess();

    for (int i = 0; i < (int)m_children.size(); i++)
        m_children[i]->markAsUninitialized(reason, ignoreIfEqual);
}

//------------------------------------------------------------------------

void BufferStorage::setAllocExtra(float allocExtra)
{
    if (!(allocExtra >= 0.0f))
        throw IlwalidValue(RT_EXCEPTION_INFO, "The value of allocExtra must be non-negative!", allocExtra);

    if (m_parent)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "setAllocExtra() is not allowed on a child buffer!");

    m_allocExtra = allocExtra;
}

//------------------------------------------------------------------------

void BufferStorage::freeMemExcept(MemorySpace memSpace)
{
    if (memSpace != MemorySpace_None && getMemSpaceIdx(memSpace) == MemSpaceIdx_Max)
        throw IlwalidValue(RT_EXCEPTION_INFO, "Invalid MemorySpace specified for freeMemExcept()!", memSpace);

    if (m_parent)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "freeMemExcept() is not allowed on a child buffer!");

    // Materialized => make sure the specified memory space is up-to-date.

    if (m_isMaterialized && memSpace != MemorySpace_None)
        access(AccessType_Prefetch, memSpace);

    // Free all memory allocations except the specified one.

    for (int i = 1; (i & MemorySpace_Any) != 0; i <<= 1)
        if (i != memSpace)
            freeMemSpace((MemorySpace)i);

    // Cancel current accesses and mark as uninitialized.

    cancelAccess(true);

    if (memSpace == MemorySpace_None && !m_isExternal)
        markAsUninitialized(InitStatus_MemoryFreedByUser);
}

//------------------------------------------------------------------------

void BufferStorage::access(AccessType accType, MemorySpace memSpace)
{
    if (getMemSpaceIdx(memSpace) == MemSpaceIdx_Max)
        throw IlwalidValue(RT_EXCEPTION_INFO, "Invalid MemorySpace specified for buffer access!", memSpace);

    if (!m_isMaterialized && (m_childType != ChildType_None || m_numBytes))
        throw IlwalidOperation(RT_EXCEPTION_INFO, "Accessing a buffer is not allowed before it has been materialized!");

    // Empty buffer => fail silently.

    if (!m_numBytes)
        return;

    // Already matches the current access => done.

    if (m_lwrAccType == accType && m_lwrMemSpace == memSpace)
        return;

    // Cancel previous access.

    cancelAccess();

    // Handle AccessType_Allocate.

    unsigned char* ptr = obtainMemSpacePtr(memSpace, ((accType & AccessType_Allocate) != 0));
    if (!ptr)
        return;

    // Handle AccessType_RequireInitialized.

    if ((accType & AccessType_RequireInitialized) != 0 && m_initStatus != InitStatus_Initialized)
    {
        switch (m_initStatus)
        {
        case InitStatus_JustMaterialized:
            throw IlwalidOperation(RT_EXCEPTION_INFO, "Tried to read from a buffer that had not been written after materialization!");

        case InitStatus_IlwalidatedByUser:
            throw IlwalidOperation(RT_EXCEPTION_INFO, "Tried to read from a buffer that had been explicitly marked as uninitialized!");

        case InitStatus_IlwalidatedByOverlay:
            throw IlwalidOperation(RT_EXCEPTION_INFO, "Tried to read from a buffer that had been ilwalidated as a result of accessing another overlaid buffer!");

        case InitStatus_MemoryFreedByUser:
            throw IlwalidOperation(RT_EXCEPTION_INFO, "Tried to read from a buffer that had been explicitly freed!");

        default:
            RT_ASSERT(false);
            break;
        }
    }

    // Handle AccessType_SyncIfInitialized, MarkAsInitialized, MarkAsNeedingSync, and IlwalidateOverlays.

    accessDescendants(accType, memSpace);
    accessAncestors(accType, memSpace);

    // Handle AccessType_WaitForSync.
    // Note: We do not need to do anything for MemorySpace_LWDA, because all device-side ops are implicitly synchronous with each other.

    if ((accType & AccessType_WaitForSync) != 0 && memSpace == MemorySpace_Host)
        waitEvent(getTopmost()->m_asyncEvent, m_lwdaUtils);

    // Accept as the current access.

    m_lwrPtr      = ptr;
    m_lwrAccType  = accType;
    m_lwrMemSpace = memSpace;
}

//------------------------------------------------------------------------

void BufferStorage::clear(size_t ofs, MemorySpace memSpace, unsigned char value, size_t numBytes)
{
    if (numBytes > getNumBytes() || ofs > getNumBytes() - numBytes)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "Tried to clear a memory range that extends beyond the buffer!");

    if (!numBytes)
        return;

    AccessType accType = (numBytes == m_numBytes) ? AccessType_WriteDiscard : AccessType_Write;
    access(accType, memSpace);

    RT_ASSERT(getLwrMemorySpace() == memSpace);
    clearMemory(getLwrPtr() + ofs, memSpace, value, numBytes, m_lwdaUtils);

    cancelAccess();
}

//------------------------------------------------------------------------

void BufferStorage::copy(size_t dstOfs, MemorySpace dstMemSpace, BufferStorage& src, size_t srcOfs, MemorySpace srcMemSpace, size_t numBytes)
{
    if (numBytes > getNumBytes() || dstOfs > getNumBytes() - numBytes || srcOfs > src.getNumBytes() - numBytes)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "Tried to copy a memory range that extends beyond the buffer!");

    if (!numBytes)
        return;

    // Decide whether to wait for pending async ops before the copy.

    bool syncDst = (dstMemSpace == MemorySpace_Host && srcMemSpace == MemorySpace_Host);
    bool syncSrc = (syncDst || src.m_lwdaUtils != m_lwdaUtils);

    // Access the buffers.

    AccessType dstAccType = (numBytes == m_numBytes) ? AccessType_WriteDiscard : AccessType_Write;
    if (!syncDst)
        dstAccType = (AccessType)(dstAccType & ~AccessType_WaitForSync);
    access(dstAccType, dstMemSpace);

    if (&src == this)
        srcMemSpace = dstMemSpace;
    else
    {
        AccessType srcAccType = AccessType_Read;
        if (!syncSrc)
            srcAccType = (AccessType)(srcAccType & ~AccessType_WaitForSync);
        src.access(srcAccType, srcMemSpace);
    }

    RT_ASSERT(getLwrMemorySpace() == dstMemSpace);
    RT_ASSERT(src.getLwrMemorySpace() == srcMemSpace);

    // Perform the copy.

    copyMemoryAsync(getLwrPtr() + dstOfs, dstMemSpace, src.getLwrPtr() + srcOfs, srcMemSpace, numBytes, m_lwdaUtils);

    // Host<=>device copy => record async events.

    if (dstMemSpace != srcMemSpace)
    {
        recordEvent(getTopmost()->m_asyncEvent, m_lwdaUtils);
        recordEvent(src.getTopmost()->m_asyncEvent, src.m_lwdaUtils);
    }

    // Clean up.

    cancelAccess();
    src.cancelAccess();
}

//------------------------------------------------------------------------

MemorySpace BufferStorage::chooseMemSpace(void) const
{
    if (m_lwrMemSpace != MemorySpace_None)
        return m_lwrMemSpace;

    if (m_memSpacesInSync != MemorySpace_None)
        return (MemorySpace)(m_memSpacesInSync & ~(m_memSpacesInSync - 1)); // find lowest set bit

    if (m_isExternal)
        return m_extMemSpace;

    if (m_lwdaUtils)
        return MemorySpace_LWDA;

    return MemorySpace_Host;
}

//------------------------------------------------------------------------

void BufferStorage::attachChild(BufferStorage* child)
{
    if (!child)
        throw IlwalidValue(RT_EXCEPTION_INFO, "NULL child buffer specified!");

    if (child == this)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "Tried to attach a buffer as its own child!");

    if (getTopmost() == child)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "Tried to attach an ancestor of a buffer as its child!");

    if (child->m_isExternal)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "Tried to attach an external buffer as a child!");

    if (child->m_parent)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "Tried to attach a non-topmost buffer as a child!");

    if (m_isMaterialized || child->m_isMaterialized)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "Modifying buffer layout is not allowed after it has been materialized!");

    // Free any dedicated memory that may have been allocated for the child.

    for (int i = 1; (i & MemorySpace_Any) != 0; i <<= 1)
        child->freeMemSpace((MemorySpace)i);

    // Attach child.

    m_children.push_back(child);
    child->m_parent = this;
    child->m_topmost = m_topmost;
    incRefCount();
}

//------------------------------------------------------------------------

BufferStorage* BufferStorage::getTopmost(void) const
{
    if (m_topmost->m_parent)
        m_topmost = m_topmost->getTopmost();
    return m_topmost;
}

//------------------------------------------------------------------------

void BufferStorage::cancelAccess(bool relwrsive)
{
    m_lwrPtr      = NULL;
    m_lwrAccType  = AccessType_None;
    m_lwrMemSpace = MemorySpace_None;

    if (relwrsive)
        for (int i = 0; i < (int)m_children.size(); i++)
            m_children[i]->cancelAccess(true);
}

//------------------------------------------------------------------------

void BufferStorage::markAsMaterialized(bool setMaterialized, LwdaUtils* lwdaUtils)
{
    if (setMaterialized == m_isMaterialized)
        return;

    // Unmaterializing an external buffer => make sure that the data is up-to-date.

    if (m_isExternal && !setMaterialized)
        access((AccessType)(AccessType_SyncIfInitialized | AccessType_WaitForSync), m_extMemSpace);

    // LwdaUtils or getDefaultStream() changed => wait for async ops and free old buffers.

    if (m_lwdaUtils != lwdaUtils)
    {
        waitEvent(m_asyncEvent, m_lwdaUtils);
        for (int i = 1; (i & MemorySpace_Any) != 0; i <<= 1)
            freeMemSpace((MemorySpace)i);
    }
    else if (m_lwdaUtils && m_lwdaUtils->getDefaultStream() != lwdaUtils->getDefaultStream())
    {
        waitEvent(m_asyncEvent, m_lwdaUtils);
    }

    // Update state.

    m_isMaterialized = setMaterialized;
    m_lwdaUtils      = lwdaUtils;

    // Process children.

    for (int i = 0; i < (int)m_children.size(); i++)
        m_children[i]->markAsMaterialized(setMaterialized, lwdaUtils);
}

//------------------------------------------------------------------------

size_t BufferStorage::getAlign(void) const
{
    size_t align = 128;
    if (m_lwdaUtils)
        align = std::max(align, (size_t)m_lwdaUtils->getTextureAlign());
    return align;
}

//------------------------------------------------------------------------

void BufferStorage::resolveLayout(void)
{
    // This is the topmost buffer => initialize offset.

    if (!m_parent)
        m_ofsInTopmost = 0;

    // Layout children.

    size_t numBytes = 0;

    switch (m_childType)
    {
    case ChildType_None:
        RT_ASSERT(!m_children.size());
        numBytes = m_numBytes;
        break;

    case ChildType_Aggregate:
        for (int i = 0; i < (int)m_children.size(); i++)
        {
            BufferStorage* child = m_children[i];
            numBytes = (numBytes + child->getAlign() - 1) & ~(child->getAlign() - 1);
            child->m_ofsInTopmost = m_ofsInTopmost + numBytes;
            child->resolveLayout();
            numBytes += child->m_numBytes;
        }
        break;

    case ChildType_Overlay:
        for (int i = 0; i < (int)m_children.size(); i++)
        {
            BufferStorage* child = m_children[i];
            child->m_ofsInTopmost = m_ofsInTopmost;
            child->resolveLayout();
            numBytes = std::max(numBytes, child->m_numBytes);
        }
        break;

    default:
        RT_ASSERT(false);
        break;
    }

    // Update size.
    // External buffer => check size.

    m_numBytes = numBytes;
    if (m_isExternal && m_numBytes > m_extNumBytes)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "An external buffer used as an aggregate/overlay requires more space than provided!");

    // Does not fit in previous memory allocations => free.

    for (int i = 1; (i & MemorySpace_Any) != 0; i <<= 1)
        if (m_numBytes > m_allocSize[getMemSpaceIdx((MemorySpace)i)])
            freeMemSpace((MemorySpace)i);
}

//------------------------------------------------------------------------

unsigned char* BufferStorage::obtainMemSpacePtr(MemorySpace memSpace, bool allowAlloc)
{
    RT_ASSERT(m_isMaterialized);
    RT_ASSERT(m_numBytes);

    unsigned char* ptr = getTopmost()->obtainMemSpacePtrTop(memSpace, allowAlloc);
    if (!ptr)
        return NULL;

    return ptr + m_ofsInTopmost;
}

//------------------------------------------------------------------------

unsigned char* BufferStorage::obtainMemSpacePtrTop(MemorySpace memSpace, bool allowAlloc)
{
    MemSpaceIdx msIdx = getMemSpaceIdx(memSpace);
    RT_ASSERT(msIdx != MemSpaceIdx_Max);
    RT_ASSERT(m_isMaterialized);
    RT_ASSERT(!m_parent);
    RT_ASSERT(m_ofsInTopmost == 0);
    RT_ASSERT(m_numBytes);

    // Pointer already available?

    if (m_isExternal && memSpace == m_extMemSpace)
        return m_extPtr;

    if (m_allocPtr[msIdx])
    {
        RT_ASSERT(m_allocSize[msIdx] >= m_numBytes);
        return m_allocPtr[msIdx];
    }

    // Memory allocation disallowed => fail.

    if (!allowAlloc)
        return NULL;

    // Allocate raw buffer.

    size_t align = getAlign();
    size_t allocSize = getAllocSize();
    m_rawPtr[msIdx] = allocateMemory(allocSize + align - 1, memSpace, m_lwdaUtils);
    RT_ASSERT(m_rawPtr[msIdx]);

    // Update state.

    m_allocSize[msIdx] = allocSize;
    m_allocPtr[msIdx] = (unsigned char*)(((uintptr_t)m_rawPtr[msIdx] + align - 1) & ~(align - 1));
    return m_allocPtr[msIdx];
}

//------------------------------------------------------------------------

void BufferStorage::freeMemSpace(MemorySpace memSpace)
{
    MemSpaceIdx msIdx = getMemSpaceIdx(memSpace);
    RT_ASSERT(msIdx != MemSpaceIdx_Max);

    waitEvent(getTopmost()->m_asyncEvent, m_lwdaUtils);
    freeMemory(m_rawPtr[msIdx], memSpace, m_lwdaUtils);

    m_allocSize[msIdx] = 0;
    m_allocPtr[msIdx]  = NULL;
    m_rawPtr[msIdx]    = NULL;
}

//------------------------------------------------------------------------

void BufferStorage::accessDescendants(AccessType accType, MemorySpace memSpace)
{
    RT_ASSERT(getMemSpaceIdx(memSpace) != MemSpaceIdx_Max);
    RT_ASSERT(m_isMaterialized);

    // Cancel previous access.

    cancelAccess();

    // Handle AccessType_SyncIfInitialized.

    if ((accType & AccessType_SyncIfInitialized) != 0 && m_initStatus == InitStatus_Initialized && (m_memSpacesInSync & memSpace) == 0)
    {
        if (m_childType == ChildType_None && m_memSpacesInSync != MemorySpace_None && m_numBytes)
        {
            MemorySpace srcMemSpace = (MemorySpace)(m_memSpacesInSync & ~(m_memSpacesInSync - 1)); // find lowest set bit
            copyMemoryAsync(obtainMemSpacePtr(memSpace, false), memSpace, obtainMemSpacePtr(srcMemSpace, false), srcMemSpace, m_numBytes, m_lwdaUtils);
            recordEvent(getTopmost()->m_asyncEvent, m_lwdaUtils);
        }
        m_memSpacesInSync = (MemorySpace)(m_memSpacesInSync | memSpace); // set bit
    }

    // Handle AccessType_MarkAsInitialized.

    if ((accType & AccessType_MarkAsInitialized) != 0)
        m_initStatus = InitStatus_Initialized;

    // Handle AccessType_MarkAsNeedingSync.

    if ((accType & AccessType_MarkAsNeedingSync) != 0 && m_initStatus == InitStatus_Initialized)
        m_memSpacesInSync = memSpace;

    // Overlay => cannot propagate AccessType_MarkAsInitialized to children.

    AccessType childAccType = accType;
    if (m_childType == ChildType_Overlay)
        childAccType = (AccessType)(childAccType & ~AccessType_MarkAsInitialized);

    // Process children.

    for (int i = 0; i < (int)m_children.size(); i++)
        m_children[i]->accessDescendants(childAccType, memSpace);
}

//------------------------------------------------------------------------

void BufferStorage::accessAncestors(AccessType accType, MemorySpace memSpace)
{
    RT_ASSERT(m_isMaterialized);

    // Cancel previous access.

    cancelAccess();

    // Handle AccessType_MarkAsInitialized.

    if ((accType & AccessType_MarkAsInitialized) != 0)
        m_initStatus = InitStatus_Initialized;

    // Handle AccessType_MarkAsNeedingSync.

    if ((accType & AccessType_MarkAsNeedingSync) != 0)
        m_memSpacesInSync = (MemorySpace)(m_memSpacesInSync & memSpace); // clear bits

    // Handle AccessType_IlwalidateOverlays.

    if ((accType & AccessType_IlwalidateOverlays) != 0 && m_parent && m_parent->m_childType == ChildType_Overlay)
    {
        for (int i = 0; i < (int)m_parent->m_children.size(); i++)
        {
            BufferStorage* child = m_parent->m_children[i];
            if (child != this)
                child->markAsUninitialized(InitStatus_IlwalidatedByOverlay, true);
        }
    }

    // Process parent.

    if (m_parent)
        m_parent->accessAncestors(accType, memSpace);
}

//------------------------------------------------------------------------

BufferStorage::MemSpaceIdx BufferStorage::getMemSpaceIdx(MemorySpace memSpace)
{
    switch (memSpace)
    {
    case MemorySpace_Host:  return MemSpaceIdx_Host;
    case MemorySpace_LWDA:  return MemSpaceIdx_LWDA;
    default:                return MemSpaceIdx_Max;
    }
}

//------------------------------------------------------------------------

void* BufferStorage::allocateMemory(size_t numBytes, MemorySpace memSpace, LwdaUtils* lwdaUtils)
{
    RT_ASSERT(getMemSpaceIdx(memSpace) != MemSpaceIdx_Max);

    switch (memSpace)
    {
    case MemorySpace_Host:
        if (lwdaUtils)
            return lwdaUtils->hostAlloc(numBytes);
        else
        {
            void* ptr = malloc(numBytes);
            if (!ptr)
                throw MemoryAllocationFailed(RT_EXCEPTION_INFO, "Out of host memory!");

            if (k_memScribbleOn.get())
                memset(ptr, k_memScribbleValueHost.get(), numBytes);
            return ptr;
        }

    case MemorySpace_LWDA:
        if (!lwdaUtils)
            throw IlwalidOperation(RT_EXCEPTION_INFO, "Tried to allocate LWCA memory with no LwdaUtils specified!");
        return lwdaUtils->deviceAlloc(numBytes);

    default:
        RT_ASSERT(false);
        return NULL;
    }
}

//------------------------------------------------------------------------

void BufferStorage::freeMemory(void* ptr, MemorySpace memSpace, LwdaUtils* lwdaUtils)
{
    RT_ASSERT(getMemSpaceIdx(memSpace) != MemSpaceIdx_Max);

    if (!ptr)
        return;

    if (memSpace == MemorySpace_Host)
    {
        if (lwdaUtils)
            lwdaUtils->hostFree(ptr);
        else
            free(ptr);
    }
    else
    {
        RT_ASSERT(memSpace == MemorySpace_LWDA);
        if (!lwdaUtils)
            throw IlwalidOperation(RT_EXCEPTION_INFO, "Tried to free LWCA memory with no LwdaUtils specified!");
        lwdaUtils->deviceFree(ptr);
    }
}

//------------------------------------------------------------------------

void BufferStorage::clearMemory(void* ptr, MemorySpace memSpace, unsigned char value, size_t numBytes, LwdaUtils* lwdaUtils)
{
    RT_ASSERT(ptr || !numBytes);
    RT_ASSERT(getMemSpaceIdx(memSpace) != MemSpaceIdx_Max);

    if (!numBytes)
        return;

    if (memSpace == MemorySpace_Host)
        memset(ptr, value, numBytes);
    else
    {
        RT_ASSERT(memSpace == MemorySpace_LWDA);
        if (!lwdaUtils)
            throw IlwalidOperation(RT_EXCEPTION_INFO, "Tried to clear LWCA memory with no LwdaUtils specified!");
        lwdaUtils->clearDeviceBuffer(ptr, value, numBytes);
    }
}

//------------------------------------------------------------------------

void BufferStorage::copyMemoryAsync(void* dstPtr, MemorySpace dstMemSpace, const void* srcPtr, MemorySpace srcMemSpace, size_t numBytes, LwdaUtils* lwdaUtils)
{
    RT_ASSERT(dstPtr || !numBytes);
    RT_ASSERT(srcPtr || !numBytes);
    RT_ASSERT(getMemSpaceIdx(dstMemSpace) != MemSpaceIdx_Max);
    RT_ASSERT(getMemSpaceIdx(srcMemSpace) != MemSpaceIdx_Max);

    if (!numBytes)
        return;

    if (dstMemSpace == srcMemSpace && (uintptr_t)dstPtr + numBytes > (uintptr_t)srcPtr && (uintptr_t)dstPtr < (uintptr_t)srcPtr + numBytes)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "Tried to copy data between overlapping memory ranges!");

    // Host-to-host => call memcpy().

    if (srcMemSpace == MemorySpace_Host && dstMemSpace == MemorySpace_Host)
    {
        memcpy(dstPtr, srcPtr, numBytes);
        return;
    }

    // Check that we have a LwdaUtils.

    if (!lwdaUtils)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "Tried to copy LWCA memory with no LwdaUtils specified!");

    // Call LwdaUtils.

    if (srcMemSpace == MemorySpace_Host && dstMemSpace == MemorySpace_LWDA)
        lwdaUtils->memcpyHtoDAsync(dstPtr, srcPtr, numBytes);
    else if (srcMemSpace == MemorySpace_LWDA && dstMemSpace == MemorySpace_Host)
        lwdaUtils->memcpyDtoHAsync(dstPtr, srcPtr, numBytes);
    else
    {
        RT_ASSERT(srcMemSpace == MemorySpace_LWDA && dstMemSpace == MemorySpace_LWDA);
        lwdaUtils->memcpyDtoDAsync(dstPtr, srcPtr, numBytes);
    }
}

//------------------------------------------------------------------------

void BufferStorage::recordEvent(lwdaEvent_t& ev, LwdaUtils* lwdaUtils)
{
    RT_ASSERT(lwdaUtils);
    if (!ev)
        ev = lwdaUtils->createEvent();
    lwdaUtils->recordEvent(ev);
}

//------------------------------------------------------------------------

void BufferStorage::waitEvent(lwdaEvent_t& ev, LwdaUtils* lwdaUtils, bool silent)
{
    if (!ev)
        return;

    RT_ASSERT(lwdaUtils);
    lwdaUtils->eventSynchronize(ev, silent);
    lwdaUtils->destroyEvent(ev);
    ev = NULL;
}

//------------------------------------------------------------------------
