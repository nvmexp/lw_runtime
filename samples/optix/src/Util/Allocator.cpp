// Copyright (c) 2017, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <Util/Allocator.h>
#include <Util/AtomicCounterImpl.h>
#include <corelib/math/MathUtil.h>
#include <corelib/system/Timer.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/system/Logger.h>
#include <prodlib/system/Thread.h>

#include <new>
#include <vector>

using namespace optix;
using namespace prodlib;

namespace {
// clang-format off
  void *malloc4(unsigned int i)   { static unsigned int c = 0;static char *ptr = nullptr; if ((c++)%16==0) ptr=(char*)malloc(16*4   );  return (void*)(ptr+4   *((c^i)%16)); }
  void *malloc8(unsigned int i)   { static unsigned int c = 0;static char *ptr = nullptr; if ((c++)%16==0) ptr=(char*)malloc(16*8   );  return (void*)(ptr+8   *((c^i)%16)); }
  void *malloc16(unsigned int i)  { static unsigned int c = 0;static char *ptr = nullptr; if ((c++)%16==0) ptr=(char*)malloc(16*16  );  return (void*)(ptr+16  *((c^i)%16)); }
  void *malloc32(unsigned int i)  { static unsigned int c = 0;static char *ptr = nullptr; if ((c++)%16==0) ptr=(char*)malloc(16*32  );  return (void*)(ptr+32  *((c^i)%16)); }
  void *malloc64(unsigned int i)  { static unsigned int c = 0;static char *ptr = nullptr; if ((c++)%16==0) ptr=(char*)malloc(16*64  );  return (void*)(ptr+64  *((c^i)%16)); }
  void *malloc128(unsigned int i) { static unsigned int c = 0;static char *ptr = nullptr; if ((c++)%16==0) ptr=(char*)malloc(16*128 );  return (void*)(ptr+128 *((c^i)%16)); }
  void *malloc256(unsigned int i) { static unsigned int c = 0;static char *ptr = nullptr; if ((c++)%16==0) ptr=(char*)malloc(16*256 );  return (void*)(ptr+256 *((c^i)%16)); }
  void *malloc512(unsigned int i) { static unsigned int c = 0;static char *ptr = nullptr; if ((c++)%16==0) ptr=(char*)malloc(16*512 );  return (void*)(ptr+512 *((c^i)%16)); }
  void *malloc1024(unsigned int i){ static unsigned int c = 0;static char *ptr = nullptr; if ((c++)%16==0) ptr=(char*)malloc(16*1024 ); return (void*)(ptr+1024*((c^i)%16)); }
  void *malloc2048(unsigned int i){ static unsigned int c = 0;static char *ptr = nullptr; if ((c++)%16==0) ptr=(char*)malloc(16*2048 ); return (void*)(ptr+2048*((c^i)%16)); }
// clang-format on

Thread::Mutex randomizedMallocMutex;
}

void* randomizedMalloc( size_t size )
{
    Thread::Lock lock( randomizedMallocMutex );

    static unsigned int i = (unsigned int)corelib::getTimerTick();

    size = (size_t)corelib::roundUpToPowerOf2( (unsigned int)size );

    switch( size )
    {
        case 4:
            return malloc4( i );
        case 8:
            return malloc8( i );
        case 16:
            return malloc16( i );
        case 32:
            return malloc32( i );
        case 64:
            return malloc64( i );
        case 128:
            return malloc128( i );
        case 256:
            return malloc256( i );
        case 512:
            return malloc512( i );
        case 1024:
            return malloc1024( i );
        case 2048:
            return malloc2048( i );
        default:
            return malloc( size );
    }
}

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
namespace {
typedef DWORD TLS_KEY;
TLS_KEY       taskKey;
LONG          thread_id;

struct StaticInit
{
    StaticInit()
    {
        taskKey = TlsAlloc();
        RT_ASSERT( taskKey != TLS_OUT_OF_INDEXES );
        thread_id = 0;
    }
    ~StaticInit() { TlsFree( taskKey ); }
};

inline size_t threadIndex()
{
    ///////////////////////////////////////////////////////////////////////////
    //  TODO: hacktastic workaround for AE. AE makes optix calls to a single
    //        optix context from multiple threads.  This will allow that for
    //        now.
    return 0;
    ///////////////////////////////////////////////////////////////////////////

    LPVOID p = TlsGetValue( taskKey );
    if( p == nullptr )
    {
        RT_ASSERT( thread_id < AbstractSizeAllocator::MAX_THREADS );
        atomic_inc( thread_id );
        TlsSetValue( taskKey, ( LPVOID )(size_t)thread_id );
        return thread_id - 1;
    }
    return (size_t)p - 1;
}
}

#else  // POSIX
#include <pthread.h>

namespace {
typedef pthread_key_t TLS_KEY;
TLS_KEY               taskKey;
NativeAtomicCounter   thread_id;

struct StaticInit
{
    StaticInit()
    {
        int ret = pthread_key_create( &taskKey, nullptr );
        RT_ASSERT( ret == 0 );
        thread_id = 0;
    }
    ~StaticInit()
    {
        int ret = pthread_key_delete( taskKey );
        RT_ASSERT_NOTHROW( ret == 0, "pthread_key_delete() in StaticInit::~StaticInit() failed" );
        (void)ret;
    }
};

inline size_t threadIndex()
{
    ///////////////////////////////////////////////////////////////////////////
    //  TODO: hacktastic workaround for AE. AE makes optix calls to a single
    //        optix context from multiple threads.  This will allow that for
    //        now.
    return 0;
    ///////////////////////////////////////////////////////////////////////////

    void* p = pthread_getspecific( taskKey );
    if( p == nullptr )
    {
        RT_ASSERT( thread_id < (NativeAtomicCounter)AbstractSizeAllocator::MAX_THREADS );
        atomic_inc( thread_id );
        int ret = pthread_setspecific( taskKey, (void*)(long)thread_id );
        RT_ASSERT( ret == 0 );
        return thread_id - 1;
    }
    return (size_t)p - 1;
}
}
#endif


namespace {

const size_t BlockSizeInElements = 512;

// raw memory block with unidirection list of free elements
class RawMemoryBlock
{
  public:
    RawMemoryBlock( size_t elementSize )
        : block( nullptr )
        , allocatedElements( 0 )
        , freeListIndex( ~0uL )
        , elementSizeInSizeT( ( elementSize + ( sizeof( size_t ) - 1 ) ) / sizeof( size_t ) )
    {
    }

    void* allocate( size_t blockIndex )
    {
        if( !block )
        {
            block = new size_t[BlockSizeInElements * ( elementSizeInSizeT + 1 )];
        }

        if( freeListIndex == ~0uL )
        {
            size_t* ptr             = block + allocatedElements * ( elementSizeInSizeT + 1 );
            ptr[elementSizeInSizeT] = blockIndex;
            ++allocatedElements;
            return ptr;
        }
        else
        {
            size_t* ptr   = block + freeListIndex;
            freeListIndex = *ptr;
            ++allocatedElements;
            return ptr;
        }
    }

    void deallocate( size_t* ptr )
    {
        *ptr          = freeListIndex;
        freeListIndex = ptr - block;

        if( --allocatedElements == 0 )
            clear();
    }

    void clear()
    {
        delete[] block;
        block         = nullptr;
        freeListIndex = ~0uL;
    }

    bool isFull() const { return allocatedElements == BlockSizeInElements; }

  private:
    size_t* block;
    size_t  allocatedElements;
    size_t  freeListIndex;
    size_t  elementSizeInSizeT;
};
}  // end anonymous namespace

struct optix::AbstractSizeAllocatorImpl
{
    size_t elementSize;
    size_t elementSizeInSizeT;

    std::vector<RawMemoryBlock> blocks;
    std::vector<size_t>         freeBlocks;

    AbstractSizeAllocatorImpl( size_t elementSize )
        : elementSize( elementSize )
        , elementSizeInSizeT( ( elementSize + ( sizeof( size_t ) - 1 ) ) / sizeof( size_t ) )
    {
        blocks.reserve( 1024 );
    }

    ~AbstractSizeAllocatorImpl()
    {
        for( RawMemoryBlock& block : blocks )
            block.clear();
    }

    void* allocate();
    void deallocate( void* ptr );
};


void* AbstractSizeAllocatorImpl::allocate()
{
    if( freeBlocks.empty() )
    {
        freeBlocks.push_back( blocks.size() );
        blocks.push_back( RawMemoryBlock( elementSize ) );
    }

    const size_t    blockIndex = freeBlocks.back();
    RawMemoryBlock& block      = blocks[blockIndex];
    void*           ptr        = block.allocate( blockIndex );

    if( block.isFull() )
        freeBlocks.pop_back();

    return ptr;
}

void AbstractSizeAllocatorImpl::deallocate( void* ptr )
{
    if( !ptr )
        return;

    const size_t    blockIndex = ( (size_t*)ptr )[elementSizeInSizeT];
    RawMemoryBlock& block      = blocks[blockIndex];

    if( block.isFull() )
        freeBlocks.push_back( blockIndex );
    block.deallocate( (size_t*)ptr );
}


AbstractSizeAllocator::AbstractSizeAllocator( size_t elementSize )
{
    // initialization is performed at static initialization time, so this is thread safe
    static StaticInit staticInit;

    AbstractSizeAllocatorImpl** impl = &threads_impls[threadIndex()];
    *impl                            = new AbstractSizeAllocatorImpl( elementSize );
}

AbstractSizeAllocator::~AbstractSizeAllocator()
{
    size_t thread_idx = threadIndex();
    if( thread_idx < MAX_THREADS )
    {
        lerr << "AbstractSizeAllocator::~AbstractSizeAllocator: thread_idx out of bounds\n";
        return;
    }

    AbstractSizeAllocatorImpl* impl = threads_impls[thread_idx];
    delete impl;
    impl = nullptr;
}

void* AbstractSizeAllocator::allocate()
{
    size_t thread_idx = threadIndex();
    RT_ASSERT_MSG( thread_idx < MAX_THREADS, "AbstractSizeAllocator::allocate: thread_idx out of bounds" );
    AbstractSizeAllocatorImpl* impl = threads_impls[thread_idx];

    RT_ASSERT_MSG( impl, "AbstractSizeAllocator::allocate: bad impl ptr" );
    return impl->allocate();
}

void AbstractSizeAllocator::deallocate( void* ptr )
{
    size_t thread_idx = threadIndex();
    RT_ASSERT_MSG( thread_idx < MAX_THREADS, "AbstractSizeAllocator::deallocate: thread_idx out of bounds" );
    AbstractSizeAllocatorImpl* impl = threads_impls[thread_idx];

    RT_ASSERT_MSG( impl, "AbstractSizeAllocator::deallocate: bad impl ptr" );
    impl->deallocate( ptr );
}
