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

#pragma once
#include "LwdaUtils.hpp"
#include <vector>

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------

enum MemorySpace                    // How to access a given memory pointer?
{
    MemorySpace_Host    = 1 << 0,   // Using host CPU.
    MemorySpace_LWDA    = 1 << 1,   // Using LWCA device.

    MemorySpace_None    = 0,
    MemorySpace_Any     = MemorySpace_Host | MemorySpace_LWDA,
};

//------------------------------------------------------------------------

enum AccessType                                 // Which operations to perform when accessing a buffer?
{
    AccessType_Allocate             = 1 << 0,   // Allocate data for this memory space if not already allocated. If not specified, access() may fail silently.
    AccessType_RequireInitialized   = 1 << 1,   // Give an error if the data has not been marked as initialized.
    AccessType_SyncIfInitialized    = 1 << 2,   // If the data has been marked as initialized, fire async memcopies to bring this memory space up-to-date.
    AccessType_MarkAsInitialized    = 1 << 3,   // Mark the data as initialized.
    AccessType_MarkAsNeedingSync    = 1 << 4,   // Mark the data in other memory spaces as out-of-date, requiring sync when accessed.
    AccessType_IlwalidateOverlays   = 1 << 5,   // For all buffers overlaid with this buffer, mark the data as uninitialized.
    AccessType_WaitForSync          = 1 << 6,   // Wait for any async memcopies that might conflict with this access.

    AccessType_None                 = 0,
    AccessType_Read                 = AccessType_Allocate | AccessType_RequireInitialized | AccessType_SyncIfInitialized | AccessType_WaitForSync,
    AccessType_Write                = AccessType_Allocate | AccessType_SyncIfInitialized  | AccessType_MarkAsInitialized | AccessType_MarkAsNeedingSync  | AccessType_IlwalidateOverlays | AccessType_WaitForSync,
    AccessType_WriteDiscard         = AccessType_Allocate | AccessType_MarkAsInitialized  | AccessType_MarkAsNeedingSync | AccessType_IlwalidateOverlays | AccessType_WaitForSync,
    AccessType_ReadWrite            = AccessType_Allocate | AccessType_RequireInitialized | AccessType_SyncIfInitialized | AccessType_MarkAsNeedingSync  | AccessType_IlwalidateOverlays | AccessType_WaitForSync,
    AccessType_Prefetch             = AccessType_Allocate | AccessType_SyncIfInitialized,
};

//------------------------------------------------------------------------
// BufferStorage is only intended to be used through BufferRef.
// Members accessed by BufferRef are tagged as 'protected'.
// Private members are not intended to be accessed from the outside.

class BufferStorage
{
    template <class T> friend class BufferRef;

protected:
    enum MemSpaceIdx
    {
        MemSpaceIdx_Host = 0,
        MemSpaceIdx_LWDA,

        MemSpaceIdx_Max
    };

    enum ChildType                          // How to interpret the children of a BufferStorage?
    {
        ChildType_None = 0,                 // No children.
        ChildType_Aggregate,                // Children are placed one after another within the parent.
        ChildType_Overlay,                  // Children are overlaid on top of each other within the parent.

        ChildType_Max
    };

    enum InitStatus                         // Is the buffer initialized, and if not, why not?
    {
        InitStatus_Initialized = 0,         // Initialized.
        InitStatus_JustMaterialized,        // Uninitialized because the buffer was just materialized.
        InitStatus_IlwalidatedByUser,       // Uninitialized because the user explicitly called markAsUninitialized().
        InitStatus_IlwalidatedByOverlay,    // Uninitialized because of accessing an overlaid buffer.
        InitStatus_MemoryFreedByUser,       // Uninitialized because the user explicitly freed the memory.

        InitStatus_Max
    };

protected:
                                BufferStorage       (void);                 // Initially empty, unallocated, and has refCount of 1.
                                ~BufferStorage      (void);                 // Typically called through decRefCount().

    void                        initExternal        (unsigned char* ptr, size_t numBytes, MemorySpace memSpace); // Call right after the constructor.

    inline void                 incRefCount         (void)                  { m_refCount++; }
    inline void                 decRefCount         (void)                  { m_refCount--; if (!m_refCount) delete this; }

    // Getters.

    size_t                      getNumBytes         (void) const;
    ChildType                   getChildType        (void) const            { return m_childType; }
    BufferStorage*              getParent           (void) const            { return m_parent; }
    size_t                      getOffsetInTopmost  (void) const;
    size_t                      getAllocSize        (void) const;

    bool                        isExternal          (void) const            { return m_isExternal; }
    bool                        isMaterialized      (void) const            { return m_isMaterialized; }

    // Layout.

    void                        setNumBytes         (size_t numBytes);
    void                        aggregate           (BufferStorage* child);
    void                        overlay             (BufferStorage* child);
    void                        detachChildren      (void);

    // Materialization.

    void                        materialize         (LwdaUtils* lwdaUtils);
    void                        unmaterialize       (void);
    void                        markAsUninitialized (InitStatus reason, bool ignoreIfEqual = false);

    void                        setAllocExtra       (float allocExtra);
    void                        freeMemExcept       (MemorySpace memSpace);

    // Data access.

    void                        access              (AccessType accType, MemorySpace memSpace);
    unsigned char*              getLwrPtr           (void) const            { return m_lwrPtr; }
    AccessType                  getLwrAccessType    (void) const            { return m_lwrAccType; }
    MemorySpace                 getLwrMemorySpace   (void) const            { return m_lwrMemSpace; }

    void                        clear               (size_t ofs, MemorySpace memSpace, unsigned char value, size_t numBytes);
    void                        copy                (size_t dstOfs, MemorySpace dstMemSpace, BufferStorage& src, size_t srcOfs, MemorySpace srcMemSpace, size_t numBytes);
    MemorySpace                 chooseMemSpace      (void) const;

private:
    void                        attachChild         (BufferStorage* child);
    BufferStorage*              getTopmost          (void) const;
    void                        cancelAccess        (bool relwrsive = false);

    void                        markAsMaterialized  (bool setMaterialized, LwdaUtils* lwdaUtils);
    size_t                      getAlign            (void) const;
    void                        resolveLayout       (void);

    unsigned char*              obtainMemSpacePtr   (MemorySpace memSpace, bool allowAlloc); // Any materialized buffer.
    unsigned char*              obtainMemSpacePtrTop(MemorySpace memSpace, bool allowAlloc); // Topmost materialized buffer.
    void                        freeMemSpace        (MemorySpace memSpace);
    void                        accessDescendants   (AccessType accType, MemorySpace memSpace);
    void                        accessAncestors     (AccessType accType, MemorySpace memSpace);

    static MemSpaceIdx          getMemSpaceIdx      (MemorySpace memSpace);
    static void*                allocateMemory      (size_t numBytes, MemorySpace memSpace, LwdaUtils* lwdaUtils);
    static void                 freeMemory          (void* ptr, MemorySpace memSpace, LwdaUtils* lwdaUtils);
    static void                 clearMemory         (void* ptr, MemorySpace memSpace, unsigned char value, size_t numBytes, LwdaUtils* lwdaUtils);
    static void                 copyMemoryAsync     (void* dstPtr, MemorySpace dstMemSpace, const void* srcPtr, MemorySpace srcMemSpace, size_t numBytes, LwdaUtils* lwdaUtils);
    static void                 recordEvent         (lwdaEvent_t& ev, LwdaUtils* lwdaUtils);
    static void                 waitEvent           (lwdaEvent_t& ev, LwdaUtils* lwdaUtils, bool silent = false);

private:
                                BufferStorage       (const BufferStorage&); // forbidden
    BufferStorage&              operator=           (const BufferStorage&); // forbidden

private:
    int                         m_refCount;
    size_t                      m_numBytes;

    bool                        m_isExternal;
    unsigned char*              m_extPtr;
    size_t                      m_extNumBytes;
    MemorySpace                 m_extMemSpace;

    ChildType                   m_childType;
    std::vector<BufferStorage*> m_children;         // Children are silently removed from the list when their refCount reaches zero.
    BufferStorage*              m_parent;           // Each child holds a reference to its parent.
    mutable BufferStorage*      m_topmost;          // Cached pointer to the topmost buffer. Updated lazily by getTopmost().

    bool                        m_isMaterialized;
    LwdaUtils*                  m_lwdaUtils;
    float                       m_allocExtra;
    size_t                      m_ofsInTopmost;     // Offset of this buffer, relative to the topmost buffer.

    size_t                      m_allocSize [MemSpaceIdx_Max];  // Size of the lwrrently allocated memory buffer.
    unsigned char*              m_allocPtr  [MemSpaceIdx_Max];  // Pointer to the allocated memory buffer.
    void*                       m_rawPtr    [MemSpaceIdx_Max];  // Actual pointer returned by malloc, possibly unaligned.

    InitStatus                  m_initStatus;
    MemorySpace                 m_memSpacesInSync;  // Which memory spaces are in sync with the most up-to-date data?
    lwdaEvent_t                 m_asyncEvent;       // Topmost buffer only: Event indicating when async host<=>device memcopies are done. NULL if there are no memcopies in flight.

    unsigned char*              m_lwrPtr;
    AccessType                  m_lwrAccType;
    MemorySpace                 m_lwrMemSpace;
};

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
