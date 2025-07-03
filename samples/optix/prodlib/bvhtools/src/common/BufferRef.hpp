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
#include "BufferStorage.hpp"
#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/IlwalidOperation.h>
#include <prodlib/exceptions/IlwalidValue.h>

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------
// EmptyBuf token. Can be used as a short-hand for BufferRef<MyType>(0).

static const struct EmptyBufType {} EmptyBuf = {};

//------------------------------------------------------------------------
// BufferRef<T> = reference to a memory buffer storing elements of type T
// 
// Features:
// - Memory spaces: The data is accessible on both the CPU and the GPU. Memory allocation and data transfers are automatically handled under the hood.
// - Garbage collection: BufferRefs employ automatic refcounting. The memory is freed as soon as all BufferRefs referencing it go out of scope.
// - Colwenience: BufferRef behaves like a pointer in many ways. It is a lightweight type that can be embedded in structs and passed as function
//   arguments. In many situations, the data can also be accessed using the same syntax as with pointers.
// - Access semantics: The validity of the data is tracked automatically by the implementation. Errors, such as reading uninitialized data,
//   will automatically trigger an error.
// - Aggregation: Buffers can be nested within each other to form a hierarchy. It is also possible to overlay several buffers to reuse the same
//   region of memory for different purposes.
// 
// Reference semantics:
// - Methods beginning with "assign" modify the reference itself. All other methods operate on the underlying buffer.
// - Example: If two BufferRefs reference the same buffer, calling setNumElems() on one will affect the result of getNumElems() on the other.
// - "const BufferRef<T>" means that the reference itself is const, but the buffer and its contents are non-const.
// - "BufferRef<const T>" means that the reference and the buffer are non-const, but the contents of the buffer are const.
// - BufferRef<> is a synonym for BufferRef<unsigned char>. It can be used in cases where the actual contents of the buffer are not relevant.
// - "EmptyBuf" is a special token that can be used in place of BufferRef<T> to indicate an empty buffer (i.e. NULL).
// 
// Subranges and reinterpret casts:
// - "myBuf.getSubrange(10, 20)" returns another BufferRef that represents the 20-element subrange of myBuf starting at index 10.
// - The subrange can be used like any other buffer, except for specific restrictions (see below).
// - "myBuf.reinterpret<AnotherType>()" returns another BufferRef that interprets the same data array elements as AnotherType.
// - BufferRef<T> can be colwerted to BufferRef<const T> without an explicit reinterpret cast.
// 
// Materialization:
// - A newly created BufferRef is initially in a "non-materialized" state. It can be transitioned into a "materialized" state
//   by calling "myBuf.materialize(&lwdaUtils)".
// - Non-materialized: The implementation will not perform any memory allocations or data transfers.
//   The buffer can be passed around, resized, and aggregated, but it cannot be accessed.
// - Materialized: The implementation is allowed to allocate memory and transfer data, but the data layout is set in stone.
//   The buffer can be accessed, but it cannot be resized or aggregated.
// - If no LwdaUtils is specified, the buffer can only be accessed on the CPU.
//   If it is, all memory allocation and data transfers are done using the corresponding LwdaUtils methods.
// - All device-side operations (user and BufferRef alike) are assumed to be synchronous wrt. lwdaUtils->getDefaultStream().
// - A materialized buffer can be unmaterialized and then re-materialized as many times as needed.
// - Subranges and child buffers cannot be materialized explicitly; they materialize automatically alongside the corresponding "topmost" buffer.
// 
// Memory allocation:
// - Materialization does not imply allocation. Memory allocations are done lazily by the implementation when the data is accessed.
// - Unmaterialization does not imply deallocation. Memory that has already been allocated is kept around for subsequent re-materialization.
// - If a buffer is re-materialized several times with a different size, it may be beneficial to call "myBuf.setAllocExtra()".
//   This instructs the implementation to allocate slightly more memory that needed to reduce the number of unnecessary reallocations.
// - The memory can be freed explicitly by calling "myBuf.freeMem()" and "myBuf.freeMemExcept(MemorySpace_XXX)".
// - All buffers are automatically aligned to either 128 bytes or LWCA texture alignment, whichever is greater.
// 
// Access semantics:
// - To access the data after materialization, one has to specify an AccessType and a MemorySpace.
// - AccessType distinguishes between different types of read/write/modify semantics.
// - MemorySpace specifies whether data is being accessed on the CPU or the GPU.
// - Each buffer keeps track of the "current access", along with the associated data pointer.
//   The current access becomes invalid when another access is made to the same memory.
// - AccessType_Read: Read the data. Gives an error if the data is uninitialized.
// - AccessType_Write: Write some parts of the data. Old data is retained in the parts that were not overwritten.
// - AccessType_WriteDiscard: Overwrite the entire buffer with new data. Old data is of no relevance, and the implementation may replace it with garbage.
// - AccessType_ReadWrite: Read the data and then modify some parts of it. Gives an error if the data is uninitialized.
// - AccessType_Prefetch: Fire async memcopies to bring the given memory space up-to-date. Access the buffer again with a different AccessType to use the data.
// - "myBuf.markAsUninitialized()" can be used to explicitly mark a buffer as uninitialized. This is useful for preventing accidental reads.
// - A buffer is always considered uninitialized after materialization, unless it is an external buffer (see below).
// 
// Access syntax:
// - BufferRef provides several helper methods for accessing the data colweniently.
//   For example: "myBuf.readLWDA()" returns a const pointer that can be passed directly into a LWCA kernel.
// - If the current access is for MemorySpace_Host, the buffer can be used like a pointer with automatic out-of-bounds checks.
//   For example: "myBuf.modifyHost(); myBuf[i] += j; myBuf->field = value;"
// - To clear a buffer, use "myBuf.clearHost(byteVal)" or "myBuf.clearLWDA(byteVal)".
// - To copy a buffer into another, use "memcpyXtoX(dst, src)" where X is H/D for host/device.
// - Subranges are useful in buffer clear/copy: "memcpyHtoD(dst.getSubrange(i, n), src.getSubrange(j, n))".
// - The memory space used in clear/copy makes no semantical difference, but it may have significant performance implications.
// - There are also variants of clear/copy that try to guess the memory space based on the last access.
// 
// External buffers:
// - BufferRef can be used to represent an externally allocated data pointer using the syntax "BufferRef<int>(dataPtr, numElems, memSpace)".
// - For the most part, an external buffer behaves exactly like any other buffer:
//   it can be accessed in any memory space, and it needs to be materialized before access.
// - By default, an external buffer is considered as initialized after materialization, unlike non-external buffers.
// - The implementation will never allocate or free any memory for the MemorySpace where the original data lives.
// - When an external buffer is unmaterialized, the implementation automatically copies the most up-to-date data to the original memory space.
// - The size of an external buffer is set in stone. The buffer cannot be resized even when unmaterialized.
// - The caller must ensure that the external data pointer remains valid as long as the buffer is accessed.
// 
// Aggregates/overlays
// - Multiple buffers can be nested to form a hierarchy. There are three main benefits:
//   saving memory, reducing the cost of memory allocations, and controlling a large group of buffers as one.
// - "myBuf.aggregate(anotherBuf)" colwerts "myBuf" into an "aggregate buffer" and adds "anotherBuf" as is its child.
// - Several calls to "aggregate()" can be chained together: "myBuf.aggregate(childA).aggregate(childB).aggregate(childC)".
// - "aggregate(bufA, bufB)" creates a new BufferRef<> that represents the aggregate of bufA and bufB.
// - Similarly, "myBuf.overlay(anotherBuf)" and "overlay(bufA, bufB)" can be used to create an "overlay buffer".
// - An aggregate buffer represents a group of child buffers placed at conselwtive (aligned) memory addresses.
//   It has no data of its own, and its size is dictated by its children.
// - An overlay buffer represents a group of child buffers that share the same memory space with each other.
//   Accessing a child buffer will automatically ilwalidate the other child buffers.
// - The child buffers can be accessed normally, and accessing the aggregate/overlay buffer is also possible.
// - Aggregate/overlay buffers can be further be added as the children of other aggregates/overlays.
//   Materializing the "topmost" aggregate/overlay automatically materializes the entire hierarchy.
// - It is possible to turn an external buffer to an aggregate/overlay as well.
//   In this case, the total size of the hierarchy must not exceed the size of the original memory allocation.
//

template <typename T = unsigned char>
class BufferRef
{
    template <typename O> friend class BufferRef; // Needed by assignReinterpret().

public:
                        BufferRef           (void)                                                          { init(); }
                        BufferRef           (EmptyBufType)                                                  { init(); }
    explicit            BufferRef           (size_t numElems)                                               { init(); assignNew(numElems); }
                        BufferRef           (const BufferRef& other)                                        { init(); assignReference(other); }
                        BufferRef           (const BufferRef& other, size_t firstElem, size_t numElems)     { init(); assignSubrange(other, firstElem, numElems); }
                        BufferRef           (T* ptr, size_t numElems, MemorySpace memSpace)                 { init(); assignExternal(ptr, numElems, memSpace); }

    template <typename O>
    explicit            BufferRef           (const BufferRef<O>& other)                                     { init(); assignReinterpret(other); }

                        ~BufferRef          (void)                                                          { deinit(); }

    // Assignment.
    // - These methods respecify the storage referenced by this BufferRef; the rest of the methods operate on the storage itself.

    BufferRef&          assignNew           (size_t numElems = 0);                                          // Creates new storage, initially unallocated.
    BufferRef&          assignReference     (const BufferRef& other);                                       // References the same storage as the other BufferRef.
    BufferRef&          assignSubrange      (const BufferRef& other, size_t firstElem, size_t numElems);    // References a subrange of the storage.
    BufferRef&          assignExternal      (T* ptr, size_t numElems, MemorySpace memSpace);                // Creates new storage to represent an externally allocated buffer.

    template <typename O>
    BufferRef&          assignReinterpret   (const BufferRef<O>& other);                                    // Reinterprets the storage using a different data type.

    // Getters.

    size_t              getNumElems         (void) const                                { return getNumBytes() / sizeof(T); }
    size_t              getNumBytes         (void) const;                               // May be affected by other buffers through e.g. reference sharing and aggregation.
    size_t              getBytesPerElem     (void) const                                { return sizeof(T); }
    size_t              getElemAlign        (void) const;
    size_t              getOffsetInTopmost  (void) const;                               // Byte offset in the topmost buffer. Must be materialized.
    size_t              getAllocSize        (void) const;                               // True size in memory. Must be materialized and topmost.

    BufferRef           getSubrange         (size_t firstElem, size_t numElems) const   { return BufferRef(*this, firstElem, numElems); }
    BufferRef           getSubrange         (size_t firstElem) const                    { return BufferRef(*this, firstElem, getNumElems() - firstElem); }

    template <typename O>
    BufferRef<O>        reinterpret         (void) const                                { return BufferRef<O>(*this); }
    BufferRef<>         reinterpretRaw      (void) const                                { return BufferRef<>(*this); }

    bool                isSubrange          (void) const;
    bool                isExternal          (void) const;
    bool                isAggregate         (void) const;
    bool                isOverlay           (void) const;
    bool                isTopmost           (void) const;
    bool                isMaterialized      (void) const;

    // Layout.
    // - These methods are forbidden once the buffer has been materialized.

    const BufferRef&    setNumElems         (size_t numElems) const                     { return setNumBytes(numElems * sizeof(T)); }
    const BufferRef&    setNumBytes         (size_t numBytes) const;                    // Forbidden on subranges, externals, aggregates, and overlays.

    template <typename O>
    const BufferRef&    aggregate           (const BufferRef<O>& child) const;          // Turns this buffer into an aggregate buffer, and adds the given buffer as a child.
                                                                                        // Aggregate buffers represent memory regions where the child buffers are placed one after another.

    template <typename O>
    const BufferRef&    overlay             (const BufferRef<O>& child) const;          // Turns this buffer into an overlay buffer, and adds the given buffer as a child.
                                                                                        // Overlay buffers represent memory regions where the child buffers are overlaid on top of each other.

    const BufferRef&    detachChildren      (void) const;                               // Reverts the effect of aggregate()/overlay() by turning this buffer back to normal (initially zero-sized).

    // Materialization.
    // - Materializing a buffer finalizes the memory layout and enables data access. Actual memory allocation happens lazily under the hood.

    const BufferRef&    materialize         (LwdaUtils* lwdaUtils = NULL) const;        // Transitions the buffer to materialized state. Can only be called on a topmost buffer.
    const BufferRef&    unmaterialize       (void) const;                               // Transitions the buffer to non-materialized state.
    const BufferRef&    markAsUninitialized (void) const;                               // Marks all underlying data as uninitialized.

    const BufferRef&    setAllocExtra       (float allocExtra) const;                   // Allocate slightly more memory than needed (0.1f => 10% more) to reduce unnecessary re-allocations.
    const BufferRef&    freeMemExcept       (MemorySpace memSpace) const;               // Frees all memory allocations except the specified one.
    const BufferRef&    freeMem             (void) const                                { freeMemExcept(MemorySpace_None); return *this; }

    // Data access.
    // - These methods are forbidden until the buffer has been materialized.

    T*                  access              (AccessType accType, MemorySpace memSpace) const; // Pointer remains valid until next access(), ilwalidateData(), or write to an overlaid buffer.
    T*                  getLwrPtr           (void) const;                               // NULL iff empty (number of elements is zero).
    AccessType          getLwrAccessType    (void) const;
    MemorySpace         getLwrMemorySpace   (void) const;

    const T*            read                (MemorySpace memSpace) const                { return access(AccessType_Read, memSpace); }
    T*                  write               (MemorySpace memSpace) const                { return access(AccessType_Write, memSpace); }
    T*                  writeDiscard        (MemorySpace memSpace) const                { return access(AccessType_WriteDiscard, memSpace); }
    T*                  readWrite           (MemorySpace memSpace) const                { return access(AccessType_ReadWrite, memSpace); }
    void                prefetch            (MemorySpace memSpace) const                { access(AccessType_Prefetch, memSpace); }

    const BufferRef&    clear               (MemorySpace memSpace, unsigned char value = 0) const; // MemorySpace specifies whether to use memset() or lwdaMemset(). MemorySpace_Any guesses based on last access.
    const BufferRef&    copy                (MemorySpace dstMemSpace, const BufferRef<const T>& src, MemorySpace srcMemSpace) const;

    // Data access helpers.

    const T*            readHost            (void) const                                { return read(MemorySpace_Host); }
    const T*            readLWDA            (void) const                                { return read(MemorySpace_LWDA); }
    T*                  writeHost           (void) const                                { return write(MemorySpace_Host); }
    T*                  writeLWDA           (void) const                                { return write(MemorySpace_LWDA); }
    T*                  writeDiscardHost    (void) const                                { return writeDiscard(MemorySpace_Host); }
    T*                  writeDiscardLWDA    (void) const                                { return writeDiscard(MemorySpace_LWDA); }
    T*                  readWriteHost       (void) const                                { return readWrite(MemorySpace_Host); }
    T*                  readWriteLWDA       (void) const                                { return readWrite(MemorySpace_LWDA); }
    void                prefetchHost        (void) const                                { prefetch(MemorySpace_Host); }
    void                prefetchLWDA        (void) const                                { prefetch(MemorySpace_LWDA); }

    const BufferRef&    clearHost           (unsigned char value = 0) const             { return clear(MemorySpace_Host, value); }
    const BufferRef&    clearLWDA           (unsigned char value = 0) const             { return clear(MemorySpace_LWDA, value); }
    const BufferRef&    clear               (unsigned char value = 0) const             { return clear(MemorySpace_Any, value); }
    const BufferRef&    copy                (const BufferRef<const T>& src) const       { return copy(MemorySpace_Any, src, MemorySpace_Any); }

    // Operators.

    BufferRef&          operator=           (const BufferRef& other)                    { assignReference(other); return *this; }
    operator BufferRef<const T>&            (void)                                      { return *(BufferRef<const T>*)this; }
    operator const BufferRef<const T>&      (void) const                                { return *(const BufferRef<const T>*)this; }

    T&                  operator[]          (size_t elemIdx) const;
    T&                  operator*           (void) const                                { return operator[](0); }
    T*                  operator->          (void) const                                { return &operator[](0); }

private:
    void                init                (void);
    void                deinit              (void);
    BufferStorage&      getStorage          (void) const;

private:
    mutable BufferStorage* m_storage;       // Storage referenced by this BufferRef. Created lazily by getStorage().
    bool                m_isSubrange;       // True iff this BufferRef represents a subrange of the storage.
    size_t              m_subrangeByteOfs;  // Byte offset of the first element in the subrange. 0 if not a subrange.
    size_t              m_subrangeNumElems; // Number of elements in the subrange. 0 if not a subrange
};

//------------------------------------------------------------------------
// Colwenience functions for copying data between buffers.
//------------------------------------------------------------------------

template <typename T> void memcpyHtoH(BufferRef<T> dst, BufferRef<const T> src) { dst.copy(MemorySpace_Host, src, MemorySpace_Host); }
template <typename T> void memcpyHtoD(BufferRef<T> dst, BufferRef<const T> src) { dst.copy(MemorySpace_LWDA, src, MemorySpace_Host); }
template <typename T> void memcpyDtoH(BufferRef<T> dst, BufferRef<const T> src) { dst.copy(MemorySpace_Host, src, MemorySpace_LWDA); }
template <typename T> void memcpyDtoD(BufferRef<T> dst, BufferRef<const T> src) { dst.copy(MemorySpace_LWDA, src, MemorySpace_LWDA); }

template <typename T> void memcpyHtoH(BufferRef<T> dst, BufferRef<T> src)       { dst.copy(MemorySpace_Host, src, MemorySpace_Host); }
template <typename T> void memcpyHtoD(BufferRef<T> dst, BufferRef<T> src)       { dst.copy(MemorySpace_LWDA, src, MemorySpace_Host); }
template <typename T> void memcpyDtoH(BufferRef<T> dst, BufferRef<T> src)       { dst.copy(MemorySpace_Host, src, MemorySpace_LWDA); }
template <typename T> void memcpyDtoD(BufferRef<T> dst, BufferRef<T> src)       { dst.copy(MemorySpace_LWDA, src, MemorySpace_LWDA); }

//------------------------------------------------------------------------
// Colwenience functions for creating aggregates/overlays of 1-4 buffers.
//------------------------------------------------------------------------

template <typename T0>
BufferRef<> aggregate(BufferRef<T0> b0)
{ return BufferRef<>().aggregate(b0); }

template <typename T0, typename T1>
BufferRef<> aggregate(BufferRef<T0> b0, BufferRef<T1> b1)
{ return BufferRef<>().aggregate(b0).aggregate(b1); }

template <typename T0, typename T1, typename T2>
BufferRef<> aggregate(BufferRef<T0> b0, BufferRef<T1> b1, BufferRef<T2> b2)
{ return BufferRef<>().aggregate(b0).aggregate(b1).aggregate(b2); }

template <typename T0, typename T1, typename T2, typename T3>
BufferRef<> aggregate(BufferRef<T0> b0, BufferRef<T1> b1, BufferRef<T2> b2, BufferRef<T3> b3)
{ return BufferRef<>().aggregate(b0).aggregate(b1).aggregate(b2).aggregate(b3); }

//------------------------------------------------------------------------

template <typename T0>
BufferRef<> overlay(BufferRef<T0> b0)
{ return BufferRef<>().overlay(b0); }

template <typename T0, typename T1>
BufferRef<> overlay(BufferRef<T0> b0, BufferRef<T1> b1)
{ return BufferRef<>().overlay(b0).overlay(b1); }

template <typename T0, typename T1, typename T2>
BufferRef<> overlay(BufferRef<T0> b0, BufferRef<T1> b1, BufferRef<T2> b2)
{ return BufferRef<>().overlay(b0).overlay(b1).overlay(b2); }

template <typename T0, typename T1, typename T2, typename T3>
BufferRef<> overlay(BufferRef<T0> b0, BufferRef<T1> b1, BufferRef<T2> b2, BufferRef<T3> b3)
{ return BufferRef<>().overlay(b0).overlay(b1).overlay(b2).overlay(b3); }

//------------------------------------------------------------------------
// Implementation.
//------------------------------------------------------------------------

template <typename T>
BufferRef<T>& BufferRef<T>::assignNew(size_t numElems)
{
    deinit();
    init();
    setNumElems(numElems);
    return *this;
}

//------------------------------------------------------------------------

template <typename T>
BufferRef<T>& BufferRef<T>::assignReference(const BufferRef& other)
{
    if (&other == this)
        return *this;

    deinit();
    init();

    m_storage = &other.getStorage();
    m_storage->incRefCount();

    if (other.m_isSubrange)
    {
        m_isSubrange = true;
        m_subrangeByteOfs = other.m_subrangeByteOfs;
        m_subrangeNumElems = other.m_subrangeNumElems;
    }
    return *this;
}

//------------------------------------------------------------------------

template <typename T>
BufferRef<T>& BufferRef<T>::assignSubrange(const BufferRef& other, size_t firstElem, size_t numElems)
{
    if (numElems > other.getNumElems() || firstElem > other.getNumElems() - numElems)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "Tried to create a subrange that extends beyond the original buffer!");

    size_t ofs = firstElem * sizeof(T);
    if (other.m_isSubrange)
        ofs += other.m_subrangeByteOfs;

    assignReference(other);

    m_isSubrange        = true;
    m_subrangeByteOfs   = ofs;
    m_subrangeNumElems  = numElems;
    return *this;
}

//------------------------------------------------------------------------

template <typename T>
BufferRef<T>& BufferRef<T>::assignExternal(T* ptr, size_t numElems, MemorySpace memSpace)
{
    if ((uintptr_t)ptr % getElemAlign() != 0)
        throw IlwalidValue(RT_EXCEPTION_INFO, "External buffer pointer is not aligned correctly for the datatype!");

    deinit();
    init();
    getStorage().initExternal((unsigned char*)ptr, numElems * sizeof(T), memSpace);
    return *this;
}

//------------------------------------------------------------------------

template <typename T> template <typename O>
BufferRef<T>& BufferRef<T>::assignReinterpret(const BufferRef<O>& other)
{
    if ((void*)&other == (void*)this)
        return *this;

    deinit();
    init();

    m_storage = &other.getStorage();
    m_storage->incRefCount();

    if (other.m_isSubrange)
    {
        m_isSubrange        = true;
        m_subrangeByteOfs   = other.m_subrangeByteOfs;
        m_subrangeNumElems  = (other.m_subrangeNumElems * sizeof(O)) / sizeof(T);
    }
    return *this;
}

//------------------------------------------------------------------------

template <typename T>
size_t BufferRef<T>::getNumBytes(void) const
{
    size_t numBytes = (m_storage) ? m_storage->getNumBytes() : 0;
    if (m_isSubrange)
    {
        if (numBytes < m_subrangeByteOfs + m_subrangeNumElems * sizeof(T))
            throw IlwalidOperation(RT_EXCEPTION_INFO, "Buffer has become smaller than the subrange after resize!");
        return m_subrangeNumElems * sizeof(T);
    }
    return numBytes;
}

//------------------------------------------------------------------------

template <typename T>
size_t BufferRef<T>::getElemAlign(void) const
{
#if defined(_MSC_VER) && _MSC_VER < 1900 // alignof() is defined by VS2015
    return _alignof(T);
#else
    return alignof(T);
#endif
}

//------------------------------------------------------------------------

template <typename T>
size_t BufferRef<T>::getOffsetInTopmost(void) const
{
    size_t ofs = getStorage().getOffsetInTopmost();
    RT_ASSERT(ofs % getElemAlign() == 0);

    if (m_isSubrange)
    {
        RT_ASSERT(m_subrangeByteOfs % getElemAlign() == 0);
        ofs += m_subrangeByteOfs;
    }
    return ofs;
}

//------------------------------------------------------------------------

template <typename T>
size_t BufferRef<T>::getAllocSize(void) const
{
    if (m_isSubrange)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "getAllocSize() is not allowed on a subrange!");

    return getStorage().getAllocSize();
}

//------------------------------------------------------------------------

template <typename T>
bool BufferRef<T>::isSubrange(void) const
{
    return m_isSubrange;
}

//------------------------------------------------------------------------

template <typename T>
bool BufferRef<T>::isExternal(void) const
{
    return (m_storage && m_storage->isExternal());
}

//------------------------------------------------------------------------

template <typename T>
bool BufferRef<T>::isAggregate(void) const
{
    return (m_storage && m_storage->getChildType() == BufferStorage::ChildType_Aggregate && !m_isSubrange);
}

//------------------------------------------------------------------------

template <typename T>
bool BufferRef<T>::isOverlay(void) const
{
    return (m_storage && m_storage->getChildType() == BufferStorage::ChildType_Overlay && !m_isSubrange);
}

//------------------------------------------------------------------------

template <typename T>
bool BufferRef<T>::isTopmost(void) const
{
    return ((!m_storage || !m_storage->getParent()) && !m_isSubrange);
}

//------------------------------------------------------------------------

template <typename T>
bool BufferRef<T>::isMaterialized(void) const
{
    return (m_storage && m_storage->isMaterialized());
}

//------------------------------------------------------------------------

template <typename T>
const BufferRef<T>& BufferRef<T>::setNumBytes(size_t numBytes) const
{
    if (m_isSubrange)
    {
        if (numBytes == m_subrangeNumElems * sizeof(T))
            return *this;
        throw IlwalidOperation(RT_EXCEPTION_INFO, "Resizing a subrange is not allowed!");
    }

    if (m_storage || numBytes)
        getStorage().setNumBytes(numBytes);
    return *this;
}

//------------------------------------------------------------------------

template <typename T> template <typename O>
const BufferRef<T>& BufferRef<T>::aggregate(const BufferRef<O>& child) const
{
    if (m_isSubrange || child.m_isSubrange)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "Aggregating a subrange is not allowed!");

    getStorage().aggregate(&child.getStorage());
    return *this;
}

//------------------------------------------------------------------------

template <typename T> template <typename O>
const BufferRef<T>& BufferRef<T>::overlay(const BufferRef<O>& child) const
{
    if (m_isSubrange || child.m_isSubrange)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "Overlaying a subrange is not allowed!");

    getStorage().overlay(&child.getStorage());
    return *this;
}

//------------------------------------------------------------------------

template <typename T>
const BufferRef<T>& BufferRef<T>::detachChildren(void) const
{
    if (m_isSubrange)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "Detaching the children of a subrange is not allowed!");

    if (m_storage)
        m_storage->detachChildren();
    return *this;
}

//------------------------------------------------------------------------

template <typename T>
const BufferRef<T>& BufferRef<T>::materialize(LwdaUtils* lwdaUtils) const
{
    if (m_isSubrange)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "Materializing a subrange is not allowed!");

    getStorage().materialize(lwdaUtils);
    return *this;
}

//------------------------------------------------------------------------

template <typename T>
const BufferRef<T>& BufferRef<T>::unmaterialize(void) const
{
    if (m_isSubrange)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "Unmaterializing a subrange is not allowed!");

    if (m_storage)
        m_storage->unmaterialize();
    return *this;
}

//------------------------------------------------------------------------

template <typename T>
const BufferRef<T>& BufferRef<T>::markAsUninitialized(void) const
{
    if (m_storage)
        m_storage->markAsUninitialized(BufferStorage::InitStatus_IlwalidatedByUser);
    return *this;
}

//------------------------------------------------------------------------

template <typename T>
const BufferRef<T>& BufferRef<T>::setAllocExtra(float allocExtra) const
{
    if (m_isSubrange)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "setAllocExtra() is not allowed on a subrange!");

    getStorage().setAllocExtra(allocExtra);
    return *this;
}

//------------------------------------------------------------------------

template <typename T>
const BufferRef<T>& BufferRef<T>::freeMemExcept(MemorySpace memSpace) const
{
    if (m_isSubrange)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "freeMemExcept() is not allowed on a subrange!");

    if (m_storage)
        m_storage->freeMemExcept(memSpace);
    return *this;
}

//------------------------------------------------------------------------

template <typename T>
T* BufferRef<T>::access(AccessType accType, MemorySpace memSpace) const
{
    if (m_isSubrange && getNumBytes() != getStorage().getNumBytes())
        accType = (AccessType)(accType | AccessType_SyncIfInitialized); // AccessType_WriteDiscard => AccessType_Write

    getStorage().access(accType, memSpace);
    return getLwrPtr();
}

//------------------------------------------------------------------------

template <typename T>
T* BufferRef<T>::getLwrPtr(void) const
{
    unsigned char* ptr = (m_storage) ? m_storage->getLwrPtr() : NULL;
    RT_ASSERT((uintptr_t)ptr % getElemAlign() == 0);

    if (m_isSubrange && ptr)
    {
        RT_ASSERT(m_subrangeByteOfs % getElemAlign() == 0);
        ptr += m_subrangeByteOfs;
    }
    return (T*)ptr;
}

//------------------------------------------------------------------------

template <typename T>
AccessType BufferRef<T>::getLwrAccessType(void) const
{
    return (m_storage) ? m_storage->getLwrAccessType() : AccessType_None;
}

//------------------------------------------------------------------------

template <typename T>
MemorySpace BufferRef<T>::getLwrMemorySpace(void) const
{
    return (m_storage) ? m_storage->getLwrMemorySpace() : MemorySpace_None;
}

//------------------------------------------------------------------------

template <typename T>
const BufferRef<T>& BufferRef<T>::clear(MemorySpace memSpace, unsigned char value) const
{
    if (memSpace == MemorySpace_Any)
        memSpace = getStorage().chooseMemSpace();

    size_t ofs = (m_isSubrange) ? m_subrangeByteOfs : 0;
    getStorage().clear(ofs, memSpace, value, getNumBytes());
    return *this;
}

//------------------------------------------------------------------------

template <typename T>
const BufferRef<T>& BufferRef<T>::copy(MemorySpace dstMemSpace, const BufferRef<const T>& src, MemorySpace srcMemSpace) const
{
    if ((void*)&src == (void*)this)
        throw IlwalidOperation(RT_EXCEPTION_INFO, "Tried to copy a buffer onto itself!");

    if (src.getNumBytes() != getNumBytes())
        throw IlwalidOperation(RT_EXCEPTION_INFO, "Tried to copy data from a buffer of different size!");

    size_t dstOfs = (m_isSubrange) ? m_subrangeByteOfs : 0;
    if (dstMemSpace == MemorySpace_Any)
        dstMemSpace = getStorage().chooseMemSpace();

    size_t srcOfs = (src.m_isSubrange) ? src.m_subrangeByteOfs : 0;
    if (srcMemSpace == MemorySpace_Any)
        srcMemSpace = src.getStorage().chooseMemSpace();

    getStorage().copy(dstOfs, dstMemSpace, src.getStorage(), srcOfs, srcMemSpace, getNumBytes());
    return *this;
}

//------------------------------------------------------------------------

template <typename T>
T& BufferRef<T>::operator[](size_t elemIdx) const
{
    if (elemIdx >= getNumElems())
        throw IlwalidValue(RT_EXCEPTION_INFO, "Element index out of bounds!", elemIdx);

    if (getLwrMemorySpace() != MemorySpace_Host)
        throw IlwalidOperation(RT_EXCEPTION_INFO,
            "Tried to dereference BufferRef elements before announcing the access! Did you forget to call readHost(), writeHost(), or modifyHost()?");

    return getLwrPtr()[elemIdx];
}

//------------------------------------------------------------------------

template <typename T>
void BufferRef<T>::init(void)
{
    m_storage           = NULL;
    m_isSubrange        = false;
    m_subrangeByteOfs   = 0;
    m_subrangeNumElems  = 0;
}

//------------------------------------------------------------------------

template <typename T>
void BufferRef<T>::deinit(void)
{
    if (m_storage)
        m_storage->decRefCount();
}

//------------------------------------------------------------------------

template <typename T>
BufferStorage& BufferRef<T>::getStorage(void) const
{
    if (!m_storage)
        m_storage = new BufferStorage;
    return *m_storage;
}

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
