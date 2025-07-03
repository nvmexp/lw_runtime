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

#include "BufferRef.hpp"

using namespace prodlib::bvhtools;

//------------------------------------------------------------------------
// Reference each BufferRef method to ensure that they compile correctly.
// Note: This function is not intended to actually do anything meaningful.

void checkBufferRefCompilation(void)
{
    // Template types.

    BufferRef<>();
    BufferRef<int>();
    BufferRef<const int>();
    BufferRef<int2>();
    BufferRef<MemorySpace>();
    BufferRef<int*>();

    // Constructors.

    BufferRef<>();
    BufferRef<>(EmptyBuf);
    BufferRef<> a(10);
    BufferRef<> b(a);
    BufferRef<> c(b, 1, 8);
    BufferRef<int2> d(NULL, 0, MemorySpace_Host);
    BufferRef<> e(d);

    // Assignment.

    a.assignNew(10);
    b.assignReference(a);
    c.assignSubrange(b, 1, 8);
    d.assignExternal(NULL, 0, MemorySpace_Host);
    e.assignReinterpret(d);

    // Getters.

    a.getNumElems();
    a.getNumBytes();
    a.getBytesPerElem();
    a.getElemAlign();
    a.getOffsetInTopmost();
    a.getAllocSize();

    a.getSubrange(1, 8);
    a.getSubrange(1);
    a.reinterpret<int2>();
    a.reinterpretRaw();

    a.isSubrange();
    a.isExternal();
    a.isAggregate();
    a.isOverlay();
    a.isTopmost();
    a.isMaterialized();

    // Layout.

    a.setNumElems(10);
    e.aggregate(d);
    e.overlay(d);

    // Materialization.

    a.materialize(NULL);
    a.unmaterialize();
    a.markAsUninitialized();

    a.setAllocExtra(0.3f);
    a.freeMemExcept(MemorySpace_Host);
    a.freeMem();

    // Data access.

    a.access(AccessType_WriteDiscard, MemorySpace_Host);
    a.getLwrPtr();
    a.getLwrAccessType();
    a.getLwrMemorySpace();

    a.read(MemorySpace_Host);
    a.write(MemorySpace_Host);
    a.writeDiscard(MemorySpace_Host);
    a.readWrite(MemorySpace_Host);
    a.prefetch(MemorySpace_Host);

    a.clear(MemorySpace_Host, 0);
    a.copy(MemorySpace_Host, a, MemorySpace_Host);

    // Data access helpers.

    a.readHost();
    a.readLWDA();
    a.writeHost();
    a.writeLWDA();
    a.writeDiscardHost();
    a.writeDiscardLWDA();
    a.readWriteHost();
    a.readWriteLWDA();
    a.prefetchHost();
    a.prefetchLWDA();

    a.clearHost(0);
    a.clearLWDA(0);
    a.clear(0);
    a.copy(a);

    // Operators.

    b = a;
    BufferRef<const int2> dc; dc = d;
    (const BufferRef<const int2>)(const BufferRef<int2>)d;

    a[1];
    *a;

    // Colwenience funcs (copy).

    memcpyHtoH(a, (BufferRef<const unsigned char>)a);
    memcpyHtoD(a, (BufferRef<const unsigned char>)a);
    memcpyDtoH(a, (BufferRef<const unsigned char>)a);
    memcpyDtoD(a, (BufferRef<const unsigned char>)a);

    memcpyHtoH(a, a);
    memcpyHtoD(a, a);
    memcpyDtoH(a, a);
    memcpyDtoD(a, a);

    // Colwenience funcs (aggregation).

    aggregate(BufferRef<int>());
    aggregate(BufferRef<int>(), BufferRef<short>());
    aggregate(BufferRef<int>(), BufferRef<short>(), BufferRef<char>());
    aggregate(BufferRef<int>(), BufferRef<short>(), BufferRef<char>(), BufferRef<float>());

    overlay(BufferRef<int>());
    overlay(BufferRef<int>(), BufferRef<short>());
    overlay(BufferRef<int>(), BufferRef<short>(), BufferRef<char>());
    overlay(BufferRef<int>(), BufferRef<short>(), BufferRef<char>(), BufferRef<float>());
}

//------------------------------------------------------------------------
