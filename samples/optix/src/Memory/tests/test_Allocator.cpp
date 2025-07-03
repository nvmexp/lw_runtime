#include <srcTests.h>

#include <Context/Context.h>
#include <Context/UpdateManager.h>
#include <Device/DeviceManager.h>
#include <Memory/Allocator.h>
#include <Memory/BackedAllocator.h>
#include <Memory/MemoryManager.h>

#include <prodlib/math/Bits.h>
#include <prodlib/system/Knobs.h>

using namespace optix;
using namespace prodlib;
using namespace testing;


TEST( Allocator, CanAllocate )
{
    Allocator         allocator( 1000 );
    Allocator::Handle block0 = allocator.alloc( 200 );
    Allocator::Handle block1 = allocator.alloc( 200 );
    EXPECT_THAT( *block0, Eq( 0u ) );
    EXPECT_THAT( *block1, Eq( 200u ) );

    EXPECT_THAT( allocator.freeSpace(), Eq( 600u ) );
}

TEST( Allocator, ThrowsOnExcessiveAllocation )
{
    Allocator allocator( 1024 );

    EXPECT_ANY_THROW( allocator.alloc( 2048 ) );
}

TEST( Allocator, FailsOnExcessiveAllocation )
{
    bool      succeeded;
    Allocator allocator( 1024 );

    allocator.alloc( 2048, &succeeded );

    EXPECT_FALSE( succeeded );
}

TEST( Allocator, AlignsProperly )
{
    size_t            alignment = 16;
    Allocator         allocator( 1000, alignment );
    Allocator::Handle block1 = allocator.alloc( 3 );
    Allocator::Handle block2 = allocator.alloc( 24 );

    EXPECT_TRUE( isAligned( *block2, alignment ) );
}

TEST( Allocator, CoalescesWithPrevBlock )
{
    Allocator         allocator( 300 );
    Allocator::Handle block1 = allocator.alloc( 100 );
    Allocator::Handle block2 = allocator.alloc( 100 );
    Allocator::Handle block3 = allocator.alloc( 100 );

    block1.reset();  // force a free
    block2.reset();

    EXPECT_NO_THROW( allocator.alloc( 200 ) );
}

TEST( Allocator, CoalescesWithNextBlock )
{
    Allocator         allocator( 300 );
    Allocator::Handle block1 = allocator.alloc( 100 );
    Allocator::Handle block2 = allocator.alloc( 100 );
    Allocator::Handle block3 = allocator.alloc( 100 );

    block2.reset();  // force a free
    block1.reset();

    EXPECT_NO_THROW( allocator.alloc( 200 ) );
}

TEST( Allocator, CoalescesWithPrevAndNextBlock )
{
    Allocator         allocator( 300 );
    Allocator::Handle block1 = allocator.alloc( 100 );
    Allocator::Handle block2 = allocator.alloc( 100 );
    Allocator::Handle block3 = allocator.alloc( 100 );

    block1.reset();  // force a free
    block3.reset();
    block2.reset();

    EXPECT_NO_THROW( allocator.alloc( 300 ) );
}

TEST( Allocator, CannotCoalesceFragmentedBlocks )
{
    Allocator         allocator( 300 );
    Allocator::Handle block1 = allocator.alloc( 100 );
    Allocator::Handle block2 = allocator.alloc( 100 );
    Allocator::Handle block3 = allocator.alloc( 100 );

    block1.reset();  // force a free
    block3.reset();

    EXPECT_ANY_THROW( allocator.alloc( 200 ) );
}

TEST( Allocator, CanExpand )
{
    Allocator         allocator( 100 );
    Allocator::Handle block1 = allocator.alloc( 80 );
    EXPECT_ANY_THROW( allocator.alloc( 200 ) );  // only 20 bytes left

    allocator.expand( 300 );

    EXPECT_NO_THROW( allocator.alloc( 200 ) );
}

// This is a regression test for the fracture traces.
TEST( Allocator, LargestUsedAddressIsCorrectAfterCoalescingInitialBlock )
{
    Allocator         allocator( 300 );
    Allocator::Handle block1 = allocator.alloc( 100 );
    Allocator::Handle block2 = allocator.alloc( 100 );
    Allocator::Handle block3 = allocator.alloc( 100 );

    block1.reset();  // force a free
    block3.reset();

    EXPECT_EQ( (size_t)200, allocator.getUsedAddressRangeEnd() );
}

TEST( Allocator, CallsFreeBlockCallback )
{
    size_t    passedOffset = -1;
    size_t    passedSize   = -1;
    Allocator allocator( 100 );
    allocator.setFreeBlockCallback( [&]( size_t offset, size_t size ) {
        passedOffset = offset;
        passedSize   = size;
    } );
    Allocator::Handle block1 = allocator.alloc( 100 );

    block1.reset();

    EXPECT_THAT( passedOffset, Ne( static_cast<size_t>( -1 ) ) );
    EXPECT_THAT( passedSize, Eq( 100u ) );
}

TEST( Allocator, LastFreedMemoryShouldBeReusedAfterCoalescingWithPrev )
{
    Allocator         allocator( 400 );
    Allocator::Handle block1 = allocator.alloc( 100 );
    Allocator::Handle block2 = allocator.alloc( 100 );
    Allocator::Handle block3 = allocator.alloc( 100 );
    Allocator::Handle block4 = allocator.alloc( 100 );

    block1.reset();
    block4.reset();

    // offset   0  FREEFREE
    // offset 100  XXXXXXXX <- block2
    // offset 200  XXXXXXXX
    // offset 300  FREEFREE <- m_freeHead->freeNext
    block2.reset();
    // now, block1 and block2 are merged and should actually be used by next allocation
    // offset   0  FREEFREE <- m_freeHead->freeNext
    // offset 200  XXXXXXXX
    // offset 300  FREEFREE
    Allocator::Handle block5 = allocator.alloc( 100 );
    EXPECT_THAT( *block5, Eq( 0u ) );
}

TEST( Allocator, LastFreedMemoryShouldBeReusedAfterCoalescingWithNext )
{
    Allocator         allocator( 400 );
    Allocator::Handle block1 = allocator.alloc( 100 );
    Allocator::Handle block2 = allocator.alloc( 100 );
    Allocator::Handle block3 = allocator.alloc( 100 );
    Allocator::Handle block4 = allocator.alloc( 100 );

    block2.reset();
    block4.reset();

    // offset   0  XXXXXXXX <- block1
    // offset 100  FREEFREE
    // offset 200  XXXXXXXX
    // offset 300  FREEFREE <- m_freeHead->freeNext
    block1.reset();
    // now, block1 and block2 are merged and should actually be used by next allocation
    // offset   0  FREEFREE <- m_freeHead->freeNext
    // offset 200  XXXXXXXX
    // offset 300  FREEFREE
    Allocator::Handle block5 = allocator.alloc( 100 );
    EXPECT_THAT( *block5, Eq( 0u ) );
}


template <typename AllocatorT = BackedAllocator>
class ManagedHostMemoryFixture : public Test
{
  public:
    ManagedHostMemoryFixture()
        : m_allowCPUKnob( "deviceManager.allowCPUFallback", true )
        , m_forceCPUKnob( "deviceManager.forceCPUFallback", true )
    {
    }

    virtual void SetUp()
    {
        RTcontext ctx_api;
        ASSERT_EQ( RT_SUCCESS, rtContextCreate( &ctx_api ) );
        m_context = reinterpret_cast<Context*>( ctx_api );
    }

    virtual void TearDown()
    {
        allocator.reset();
        RTcontext ctx_api = RTcontext( m_context );
        ASSERT_EQ( RT_SUCCESS, rtContextDestroy( ctx_api ) );
        m_context = nullptr;
    }

    MBufferHandle allocHost( size_t size )
    {
        BufferDimensions BD;
        BD.setFormat( RT_FORMAT_BYTE, 1 );
        BD.setSize( size );
        return m_context->getMemoryManager()->allocateMBuffer( BD, MBufferPolicy::internal_readwrite );
    }

    void createAllocator( size_t size )
    {
        allocator.reset( new AllocatorT( size, 1, MBufferPolicy::internal_readwrite, m_context->getMemoryManager() ) );
    }

    MemoryManager* getMemoryManager() { return m_context->getMemoryManager(); }

    Context*                    m_context = nullptr;
    std::unique_ptr<AllocatorT> allocator;

  protected:
    ScopedKnobSetter m_allowCPUKnob;
    ScopedKnobSetter m_forceCPUKnob;
};

class TheMemoryManager : public ManagedHostMemoryFixture<>
{
};

TEST_F_DEV( TheMemoryManager, CanMapManagedMemory )
{
    MBufferHandle mem = allocHost( 100 );

    EXPECT_NO_THROW( getMemoryManager()->mapToHost( mem, MAP_WRITE_DISCARD ) );
    EXPECT_NO_THROW( getMemoryManager()->unmapFromHost( mem ) );
    EXPECT_NO_THROW( mem.reset() );
}


class ABackedAllocator : public ManagedHostMemoryFixture<BackedAllocator>
{
};

TEST_F_DEV( ABackedAllocator, CanGrowMemorySize )
{
    const size_t initialSize = 100;
    createAllocator( initialSize );
    EXPECT_THAT( allocator->memorySize(), Eq( initialSize ) );
    bool                    backingReallocated = false;
    BackedAllocator::Handle block1             = allocator->alloc( initialSize / 2, &backingReallocated );

    char* block1_value = getMemoryManager()->mapToHost( allocator->memory(), MAP_WRITE_DISCARD ) + *block1;
    *block1_value      = 111;
    getMemoryManager()->unmapFromHost( allocator->memory() );

    BackedAllocator::Handle block2 = allocator->alloc( 16 * initialSize, &backingReallocated );

    EXPECT_THAT( allocator->memorySize(), Gt( 16 * initialSize ) );

    block1_value = getMemoryManager()->mapToHost( allocator->memory(), MAP_READ ) + *block1;
    EXPECT_EQ( 111, *block1_value );
    getMemoryManager()->unmapFromHost( allocator->memory() );
}

TEST_F_DEV( ABackedAllocator, CanClearFreedBlocks )
{
    ScopedKnobSetter knob( "mem.clearBuffersOnAllocation", true );

    size_t totalSize = 16;
    size_t blockSize = 8;
    createAllocator( totalSize );
    BackedAllocator::Handle block    = allocator->alloc( blockSize );
    size_t                  offset   = *block;
    char*                   blockPtr = getMemoryManager()->mapToHost( allocator->memory(), MAP_WRITE_DISCARD ) + offset;
    *blockPtr                        = 123;
    getMemoryManager()->unmapFromHost( allocator->memory() );

    block.reset();

    blockPtr = getMemoryManager()->mapToHost( allocator->memory(), MAP_READ ) + offset;
    EXPECT_THAT( *blockPtr, Eq( 0 ) );
    getMemoryManager()->unmapFromHost( allocator->memory() );
}
