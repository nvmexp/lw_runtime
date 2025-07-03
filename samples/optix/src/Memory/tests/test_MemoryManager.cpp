#include <srcTests.h>

#include <LWCA/Memory.h>  // remove when no longer needed.
#include <Context/Context.h>
#include <Context/UpdateManager.h>
#include <Device/CPUDevice.h>
#include <Device/LWDADevice.h>
#include <Device/Device.h>
#include <Device/DeviceManager.h>
#include <Memory/MBuffer.h>
#include <Memory/MBufferPolicy.h>
#include <Memory/MapMode.h>
#include <Memory/MemoryManager.h>

#include <corelib/system/LwdaDriver.h>
#include <prodlib/exceptions/MemoryAllocationFailed.h>

using namespace optix;
using namespace testing;


#define REQUIRE_LWDA_DEVICE()                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        if( !requireAtLeastDevices( 2 ) )                                                                              \
            return;                                                                                                    \
    } while( 0 )

#define REQUIRE_MULTIPLE_LWDA_DEVICES()                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        if( !requireAtLeastDevices( 3 ) )                                                                              \
            return;                                                                                                    \
    } while( 0 )


std::ostream& operator<<( std::ostream& out, MBufferPolicy policy )
{
#define CASE( x )                                                                                                      \
    case MBufferPolicy::x:                                                                                             \
        out << #x;                                                                                                     \
        break;
    switch( policy )
    {
        CASE( readonly );
        CASE( readonly_discard_hostmem );
        CASE( readwrite );
        CASE( writeonly );

        CASE( readonly_raw );
        CASE( readonly_discard_hostmem_raw );
        CASE( readwrite_raw );
        CASE( writeonly_raw );

        CASE( readonly_lwdaInterop );
        CASE( readwrite_lwdaInterop );
        CASE( writeonly_lwdaInterop );

        CASE( readonly_lwdaInterop_copyOnDirty );
        CASE( readwrite_lwdaInterop_copyOnDirty );
        CASE( writeonly_lwdaInterop_copyOnDirty );

        CASE( readonly_demandload );
        CASE( texture_readonly_demandload );
        CASE( tileArray_readOnly_demandLoad );
        CASE( readonly_sparse_backing );

        CASE( gpuLocal );

        CASE( readonly_gfxInterop );
        CASE( readwrite_gfxInterop );
        CASE( writeonly_gfxInterop );

        CASE( texture_linear );
        CASE( texture_linear_discard_hostmem );
        CASE( texture_array );
        CASE( texture_array_discard_hostmem );
        CASE( texture_gfxInterop );

        CASE( internal_readonly );
        CASE( internal_readwrite );
        CASE( internal_writeonly );
        CASE( internal_hostonly );
        CASE( internal_readonly_deviceonly );

        CASE( internal_readonly_manualSync );
        CASE( internal_readwrite_manualSync );

        CASE( internal_texheapBacking );
        CASE( internal_preferTexheap );

        CASE( unused );
    }
#undef CASE

    return out;
}

class TestMemoryManager : public ::testing::Test
{
  public:
    Context*         m_context       = nullptr;
    DeviceManager*   m_deviceManager = nullptr;
    UpdateManager*   m_updateManager = nullptr;
    MemoryManager*   m_memoryManager = nullptr;
    BufferDimensions m_defaultBufferDims;
    BufferDimensions m_zeroSizeBufferDims;
    unsigned int     m_cpuIndex = 0;

    TestMemoryManager()
        : m_defaultBufferDims( RT_FORMAT_BYTE,   // RTformat     format
                               1,                // size_t       elementSize
                               1,                // unsigned int dimensionality
                               128,              // size_t       width
                               1,                // size_t       height
                               1 )               // size_t       depth
        , m_zeroSizeBufferDims( RT_FORMAT_BYTE,  // RTformat     format
                                1,               // size_t       elementSize
                                1,               // unsigned int dimensionality
                                0,               // size_t       width
                                0,               // size_t       height
                                0 )              // size_t       depth
    {
    }

    void SetUp()
    {
        RTcontext ctx_api;
        ASSERT_EQ( RT_SUCCESS, rtContextCreate( &ctx_api ) );
        m_context       = reinterpret_cast<Context*>( ctx_api );
        m_deviceManager = m_context->getDeviceManager();
        m_updateManager = m_context->getUpdateManager();
        m_memoryManager = m_context->getMemoryManager();

        m_deviceManager->enableActiveDevices();

        ASSERT_TRUE( m_zeroSizeBufferDims.zeroSized() );
        m_cpuIndex = m_deviceManager->cpuDevice()->allDeviceListIndex();
    }

    void TearDown()
    {
        RTcontext ctx_api = RTcontext( m_context );
        ASSERT_EQ( RT_SUCCESS, rtContextDestroy( ctx_api ) );
        m_context = nullptr;
    }

    bool requireAtLeastDevices( unsigned int N )
    {
        const unsigned int numDevices = m_deviceManager->allDevices().size();

        if( numDevices < N )
        {
            std::cout << "\n\n\t[WARNING] Test could not complete since not enough LWCA devices were found! (expected: " << N
                      << ", detected: " << numDevices << ")\n\n";
            return false;
        }

        return true;
    }
};

// Memory Manager API usage.

TEST_F( TestMemoryManager, CanSyncMemoryWithoutMappedBuffer )
{
    EXPECT_NO_THROW( m_memoryManager->syncAllMemoryBeforeLaunch() );
}

TEST_F( TestMemoryManager, CanDoSyncFinalizeReleaseWithoutMappedBuffer )
{
    EXPECT_NO_THROW( m_memoryManager->syncAllMemoryBeforeLaunch() );
    EXPECT_NO_THROW( m_memoryManager->finalizeTransfersBeforeLaunch() );
    EXPECT_NO_THROW( m_memoryManager->releaseMemoryAfterLaunch() );
}

TEST_F( TestMemoryManager, ReleaseWithoutSyncThrows )
{
    EXPECT_ANY_THROW( m_memoryManager->releaseMemoryAfterLaunch() );
}

TEST_F( TestMemoryManager, SyncWithoutUnmapFromHostThrows )
{
    MBufferPolicy policy = MBufferPolicy::readonly;
    MBufferHandle buf1   = m_memoryManager->allocateMBuffer( m_defaultBufferDims, policy );
    m_memoryManager->mapToHost( buf1, MAP_WRITE_DISCARD );
    EXPECT_ANY_THROW( m_memoryManager->syncAllMemoryBeforeLaunch() );
    m_memoryManager->unmapFromHost( buf1 );
}

// Misc.

// Host to host copies.

TEST_F( TestMemoryManager, DISABLED_CanCopyBetweenHostAllocations )
{
    ASSERT_GE( m_deviceManager->allDevices().size(), 1u );

    MBufferPolicy policy1  = MBufferPolicy::readonly;
    MBufferPolicy policy2  = MBufferPolicy::writeonly;
    MBufferHandle readBuf  = m_memoryManager->allocateMBuffer( m_defaultBufferDims, policy1 );
    MBufferHandle writeBuf = m_memoryManager->allocateMBuffer( m_defaultBufferDims, policy2 );

    // Fill the read buffer.
    char* readBufData = m_memoryManager->mapToHost( readBuf, MAP_WRITE_DISCARD );
    char  str[]       = "helloWorld!";
    memcpy( readBufData, reinterpret_cast<void*>( str ), sizeof( str ) * sizeof( char ) );
    m_memoryManager->unmapFromHost( readBuf );

    ASSERT_NO_THROW( m_memoryManager->syncAllMemoryBeforeLaunch() );
    ASSERT_NO_THROW( m_memoryManager->finalizeTransfersBeforeLaunch() );

#if 0
  m_memoryManager->copy( writeBuf, readBuf );   // doesn't exist. TODO: kill this test.
#endif

    char* writeBufData = m_memoryManager->mapToHost( writeBuf, MAP_READ );
    EXPECT_THAT( memcmp( (void*)str, reinterpret_cast<void*>( writeBufData ), sizeof( str ) * sizeof( char ) ), Eq( 0 ) );
}

// Zero Copy tests
// Basic allocate/deallocate/reallocate tests are done by writeonly parameterized tests.

TEST_F( TestMemoryManager, DISABLED_WriteOnlyDoesZeroCopy )
{
    // This is a very low level test. Any test of Buffer Write from Device should also test this more effectively.
    MBufferPolicy policy = MBufferPolicy::writeonly;
    //  PolicyDetails details = optix::PolicyDetails::policies[policy];

    MBufferHandle buf = m_memoryManager->allocateMBuffer( m_defaultBufferDims, policy );
    ASSERT_THAT( buf.get(), NotNull() );

    // Fake-Write to the buffer from the host. This is a hack.
    char reference[128];
    for( size_t i    = 0; i < 128; ++i )
        reference[i] = 127 - i;

    // Get first GPU device
    LWDADevice* gpu = m_deviceManager->primaryLWDADevice();
    ASSERT_THAT( gpu, NotNull() );
    gpu->makeLwrrent();

    // Map to host to trigger allocations.
    char* host_ptr = nullptr;
    ASSERT_NO_THROW( host_ptr = m_memoryManager->mapToHost( buf, MAP_READ ) );
    ASSERT_NO_THROW( m_memoryManager->unmapFromHost( buf ) );

    const MAccess gpuAccess = buf->getAccess( gpu );
    ASSERT_THAT( gpuAccess.getPitchedLinear( 0 ).ptr, NotNull() );

    ASSERT_NO_THROW( lwca::memcpyHtoD( (LWdeviceptr)gpuAccess.getPitchedLinear( 0 ).ptr, reference, 128 ) );

    // Read and compare

    // Compare
    ASSERT_NO_THROW( host_ptr = m_memoryManager->mapToHost( buf, MAP_READ ) );

    for( size_t i = 0; i < 128; ++i )
        EXPECT_EQ( reference[i], host_ptr[i] );

    // Unmap
    ASSERT_NO_THROW( m_memoryManager->unmapFromHost( buf ) );
}

// Test buffer policy changes.

class TestMemoryManagerChangePolicy : public TestMemoryManager, public ::testing::WithParamInterface<MBufferPolicy>
{
  public:
    static std::vector<MBufferPolicy> basicPolicies()
    {
        std::vector<MBufferPolicy> retval;
        retval.push_back( MBufferPolicy::readonly );
        retval.push_back( MBufferPolicy::readwrite );
        retval.push_back( MBufferPolicy::writeonly );
        return retval;
    }
};

// TODO: this does not look right. changing the policy of a mapped buffer can change from malloc to zeroCopy,
// so this test can fail in some cases. need to fix.
TEST_P( TestMemoryManagerChangePolicy, DISABLED_CanChangePolicyOfMappedBuffer )
{
    MBufferHandle buffer = m_memoryManager->allocateMBuffer( m_zeroSizeBufferDims, MBufferPolicy::unused );
    ASSERT_THAT( buffer.get(), NotNull() );
    m_memoryManager->mapToHost( buffer, MAP_READ );

    EXPECT_NO_THROW( m_memoryManager->changePolicy( buffer, GetParam() ) );

    m_memoryManager->unmapFromHost( buffer );
}

INSTANTIATE_TEST_SUITE_P( Policies, TestMemoryManagerChangePolicy, ValuesIn( TestMemoryManagerChangePolicy::basicPolicies() ) );


// Parameterize some tests based on policy details.
// Test against the memory manager taking into account the policy details, and ensure it matches.
// e.g. If a Host map tries to write to a write only buffer, the test passes if it throws an exception; as it should.

class TestMemoryManagerMatchesPolicy : public TestMemoryManager, public ::testing::WithParamInterface<MBufferPolicy>
{
};

TEST_P( TestMemoryManagerMatchesPolicy, CanAllocateMBuffer )
{
    MBufferPolicy policy = GetParam();
    MBufferHandle buf    = m_memoryManager->allocateMBuffer( m_defaultBufferDims, policy );
    EXPECT_THAT( buf.get(), NotNull() );
}

TEST_P( TestMemoryManagerMatchesPolicy, CanAllocateZeroSizeMBuffer )
{
    MBufferPolicy policy = GetParam();
    MBufferHandle buf    = m_memoryManager->allocateMBuffer( m_zeroSizeBufferDims, policy );
    EXPECT_THAT( buf.get(), NotNull() );
}

TEST_P( TestMemoryManagerMatchesPolicy, DefaultCPUAccessIsNone )
{
    MBufferPolicy policy  = GetParam();
    MBufferHandle buf     = m_memoryManager->allocateMBuffer( m_defaultBufferDims, policy );
    const MAccess hostMem = buf->getAccess( m_cpuIndex );
    EXPECT_THAT( hostMem.getKind(), Eq( MAccess::NONE ) );
}

TEST_P( TestMemoryManagerMatchesPolicy, BufferDimensionsOfAllocatedBufferMatch )
{
    MBufferPolicy           policy        = GetParam();
    MBufferHandle           buf           = m_memoryManager->allocateMBuffer( m_defaultBufferDims, policy );
    const BufferDimensions& allocatedDims = buf->getDimensions();
    EXPECT_THAT( m_defaultBufferDims, Eq( allocatedDims ) );
}

TEST_P( TestMemoryManagerMatchesPolicy, HostCanWriteToBuffer )
{
    MBufferPolicy policy = GetParam();
    MBufferHandle buf    = m_memoryManager->allocateMBuffer( m_defaultBufferDims, policy );

    if( m_memoryManager->policyAllowsMapToHostForWrite( buf ) )
        EXPECT_NO_THROW( m_memoryManager->mapToHost( buf, MAP_WRITE_DISCARD ) );
    else
        EXPECT_THROW( m_memoryManager->mapToHost( buf, MAP_WRITE_DISCARD ), prodlib::MemoryAllocationFailed );

    m_memoryManager->unmapFromHost( buf );
}

TEST_P( TestMemoryManagerMatchesPolicy, HostCanReadFromBuffer )
{
    MBufferPolicy policy = GetParam();
    MBufferHandle buf    = m_memoryManager->allocateMBuffer( m_defaultBufferDims, policy );

    if( m_memoryManager->policyAllowsMapToHostForRead( buf ) )
        EXPECT_NO_THROW( m_memoryManager->mapToHost( buf, MAP_READ ) );
    else
        EXPECT_THROW( m_memoryManager->mapToHost( buf, MAP_READ ), prodlib::MemoryAllocationFailed );

    m_memoryManager->unmapFromHost( buf );
}

// Allocate / deallocate

TEST_P( TestMemoryManagerMatchesPolicy, CanAllocateAndDeallocate )
{
    MBufferPolicy policy = GetParam();
    MBufferHandle buf    = m_memoryManager->allocateMBuffer( m_defaultBufferDims, policy );
    ASSERT_THAT( buf.get(), NotNull() );

    if( m_memoryManager->policyAllowsMapToHostForRead( buf ) )
    {
        m_memoryManager->mapToHost( buf, MAP_READ );
        m_memoryManager->unmapFromHost( buf );
    }
    else if( m_memoryManager->policyAllowsMapToHostForWrite( buf ) )
    {
        m_memoryManager->mapToHost( buf, MAP_WRITE_DISCARD );
        m_memoryManager->unmapFromHost( buf );
    }

    EXPECT_NO_THROW( buf.reset() );
}

// test has issues with multiGPU (probably Malloc/ZeroCopy issues). Disabled for the time being.
TEST_P( TestMemoryManagerMatchesPolicy, DISABLED_CanAllocateAndReallocate )
{
    MBufferPolicy old_policy = GetParam();

    MBufferHandle buf = m_memoryManager->allocateMBuffer( m_defaultBufferDims, old_policy );
    ASSERT_THAT( buf.get(), NotNull() );

    // Interestingly, some tests fail if we add MBufferPolicy_unused to this list. This is probably an issue interpreting the settings of the unused policy.
    MBufferPolicy new_policies[] = {MBufferPolicy::readonly, MBufferPolicy::writeonly, MBufferPolicy::readwrite,
                                    MBufferPolicy::gpuLocal};
    for( unsigned int i = 0; i < ( sizeof( new_policies ) / sizeof( MBufferPolicy ) ); ++i )
    {
        MBufferPolicy new_policy = new_policies[i];

        m_memoryManager->changePolicy( buf, new_policy );

        if( m_memoryManager->policyAllowsMapToHostForRead( buf ) )
        {
            EXPECT_NO_THROW( m_memoryManager->mapToHost( buf, MAP_READ ) ) << "Reallocated to Policy: "
                                                                           << optix::toString( new_policy );
            EXPECT_NO_THROW( m_memoryManager->unmapFromHost( buf ) );
        }

        if( !m_memoryManager->policyAllowsMapToHostForRead( buf ) )
        {
            EXPECT_THROW( m_memoryManager->mapToHost( buf, MAP_READ ), prodlib::MemoryAllocationFailed )
                << "Reallocated to Policy: " << optix::toString( new_policy );
            EXPECT_NO_THROW( m_memoryManager->unmapFromHost( buf ) );
        }

        if( m_memoryManager->policyAllowsMapToHostForWrite( buf ) )
        {
            EXPECT_NO_THROW( m_memoryManager->mapToHost( buf, MAP_WRITE_DISCARD ) ) << "Reallocated to Policy: "
                                                                                    << optix::toString( new_policy );
            EXPECT_NO_THROW( m_memoryManager->unmapFromHost( buf ) );
        }

        if( !m_memoryManager->policyAllowsMapToHostForWrite( buf ) )
        {
            EXPECT_THROW( m_memoryManager->mapToHost( buf, MAP_WRITE_DISCARD ), prodlib::MemoryAllocationFailed )
                << "Reallocated to Policy: " << optix::toString( new_policy );
            EXPECT_NO_THROW( m_memoryManager->unmapFromHost( buf ) );
        }
    }
    EXPECT_NO_THROW( buf.reset() );
}

// Test if written and read values are correct.

TEST_P( TestMemoryManagerMatchesPolicy, CanWriteAndReadOnHost )
{
    MBufferPolicy policy = GetParam();
    MBufferHandle buf    = m_memoryManager->allocateMBuffer( m_defaultBufferDims, policy );

    if( m_memoryManager->policyAllowsMapToHostForWrite( buf ) && m_memoryManager->policyAllowsMapToHostForRead( buf ) )
    {
        char* mem1                        = m_memoryManager->mapToHost( buf, MAP_WRITE_DISCARD );
        reinterpret_cast<int*>( mem1 )[0] = 13;
        m_memoryManager->unmapFromHost( buf );

        char*     mem2 = m_memoryManager->mapToHost( buf, MAP_READ );
        const int val  = reinterpret_cast<int*>( mem2 )[0];
        EXPECT_THAT( val, Eq( 13 ) );
        m_memoryManager->unmapFromHost( buf );
    }
    else
        ::testing::PrintToString( "Buffer Policy does not allow this test. Skipping." );
}

TEST_P( TestMemoryManagerMatchesPolicy, CPUAccessAfterMappingMatchesLinear )
{
    MBufferPolicy policy = GetParam();

    MBufferHandle buf = m_memoryManager->allocateMBuffer( m_defaultBufferDims, policy );
    ASSERT_THAT( buf.get(), NotNull() );

    if( m_memoryManager->policyAllowsMapToHostForWrite( buf ) )
    {
        m_memoryManager->mapToHost( buf, MAP_WRITE_DISCARD );
        const MAccess& hostMem = buf->getAccess( m_cpuIndex );
        EXPECT_EQ( hostMem.getKind(), MAccess::LINEAR );
        m_memoryManager->unmapFromHost( buf );
    }
    else
        ::testing::PrintToString( "Buffer Policy does not allow this test. Skipping." );
}

// Instantiate cases against defined policies

INSTANTIATE_TEST_SUITE_P( ReadOnly, TestMemoryManagerMatchesPolicy, ::testing::Values( MBufferPolicy::readonly ) );
INSTANTIATE_TEST_SUITE_P( ReadWrite, TestMemoryManagerMatchesPolicy, ::testing::Values( MBufferPolicy::readwrite ) );
INSTANTIATE_TEST_SUITE_P( WriteOnly, TestMemoryManagerMatchesPolicy, ::testing::Values( MBufferPolicy::writeonly ) );
INSTANTIATE_TEST_SUITE_P( GpuLocal, TestMemoryManagerMatchesPolicy, ::testing::Values( MBufferPolicy::gpuLocal ) );
INSTANTIATE_TEST_SUITE_P( DISABLED_TextureBackingArray, TestMemoryManagerMatchesPolicy, ::testing::Values( MBufferPolicy::texture_array ) );
INSTANTIATE_TEST_SUITE_P( DISABLED_TextureBackingLinear, TestMemoryManagerMatchesPolicy, ::testing::Values( MBufferPolicy::texture_linear ) );

INSTANTIATE_TEST_SUITE_P( DISABLED_Unused, TestMemoryManagerMatchesPolicy, ::testing::Values( MBufferPolicy::unused ) );

// Parameterize tests based on policy details.
// Test that the allocation of a buffer with the given policy succeed.

class TestMemoryManagerAllocation : public TestMemoryManager, public ::testing::WithParamInterface<MBufferPolicy>
{
};

TEST_P( TestMemoryManagerAllocation, CanDoCPUAllocation )
{
    MBufferPolicy policy = GetParam();
    MBufferHandle buffer = m_memoryManager->allocateMBuffer( m_defaultBufferDims, policy );
    ASSERT_THAT( buffer.get(), NotNull() );

    EXPECT_NO_THROW( m_memoryManager->syncAllMemoryBeforeLaunch() );
}

TEST_P( TestMemoryManagerAllocation, CanDoZeroSizeCPUAllocation )
{
    MBufferPolicy policy = GetParam();
    MBufferHandle buffer = m_memoryManager->allocateMBuffer( m_zeroSizeBufferDims, policy );
    ASSERT_THAT( buffer.get(), NotNull() );

    EXPECT_NO_THROW( m_memoryManager->syncAllMemoryBeforeLaunch() );
}

TEST_P( TestMemoryManagerAllocation, CanDoGPUAllocation )
{
    REQUIRE_LWDA_DEVICE();

    Device* device = m_deviceManager->primaryLWDADevice();
    ASSERT_THAT( device, NotNull() );

    MBufferPolicy policy = GetParam();
    MBufferHandle buffer = m_memoryManager->allocateMBuffer( m_defaultBufferDims, policy, device );
    ASSERT_THAT( buffer.get(), NotNull() );
    ASSERT_THAT( buffer->getPolicy(), Eq( policy ) );

    EXPECT_NO_THROW( m_memoryManager->syncAllMemoryBeforeLaunch() );
}

TEST_P( TestMemoryManagerAllocation, CanDoZeroSizeGPUAllocation )
{
    REQUIRE_LWDA_DEVICE();

    Device* device = m_deviceManager->primaryLWDADevice();
    ASSERT_THAT( device, NotNull() );

    MBufferPolicy policy = GetParam();
    MBufferHandle buffer = m_memoryManager->allocateMBuffer( m_zeroSizeBufferDims, policy, device );
    ASSERT_THAT( buffer.get(), NotNull() );
    ASSERT_THAT( buffer->getPolicy(), Eq( policy ) );

    EXPECT_NO_THROW( m_memoryManager->syncAllMemoryBeforeLaunch() );
}

INSTANTIATE_TEST_SUITE_P( ReadOnly, TestMemoryManagerAllocation, ::testing::Values( MBufferPolicy::readonly ) );
INSTANTIATE_TEST_SUITE_P( ReadWrite, TestMemoryManagerAllocation, ::testing::Values( MBufferPolicy::readwrite ) );
INSTANTIATE_TEST_SUITE_P( WriteOnly, TestMemoryManagerAllocation, ::testing::Values( MBufferPolicy::writeonly ) );
INSTANTIATE_TEST_SUITE_P( GpuLocal, TestMemoryManagerAllocation, ::testing::Values( MBufferPolicy::gpuLocal ) );
INSTANTIATE_TEST_SUITE_P( TextureBackingArray, TestMemoryManagerAllocation, ::testing::Values( MBufferPolicy::texture_array ) );
INSTANTIATE_TEST_SUITE_P( TextureBackingLinear, TestMemoryManagerAllocation, ::testing::Values( MBufferPolicy::texture_linear ) );
INSTANTIATE_TEST_SUITE_P( Unused, TestMemoryManagerAllocation, ::testing::Values( MBufferPolicy::unused ) );


#if 0
//
// Tests that use the old memory manager API.
// These are still here for reference until corresponding tests are implemented.
//
TEST_F( TheMemoryManager, MapHostToGlobalDevice0 ) // Assumes there's at least one GPU available on the system
{
  REQUIRE_LWDA_DEVICE();

  shared_ptr<MappedSet> mappedSet = memMgr->createEmptyMappedSet();
  optix::Device* device0 = static_cast<optix::Device*>( devMgr->primaryLWDADevice() );
  ASSERT_NE( device0, (Device*)0 );

  MemorySpaceSet destSpace( MemorySpaceSet::HOST_MALLOCED, device0->allDeviceListIndex() );
  MemorySpaceSet allowedSpaces = destSpace;
  allowedSpaces.insert( MemorySpaceSet::GLOBAL, device0->allDeviceListIndex() );

  MBufferHandle alloc = memMgr->allocate( defaultBufferDims, OPTIX_API_BUFFER_GROUP, allowedSpaces );

  // Populate host allocation
  memMgr->map( alloc, MAP_WRITE, destSpace, mappedSet );
  memMgr->finalizeAllocations( mappedSet );
  char *hostMemory = alloc->getPointer<MemorySpaceSet::HOST_MALLOCED>( device0->allDeviceListIndex() );
  char bytes[] = {'H', 'E', 'L', 'L', 'O'};
  memcpy( hostMemory, bytes, sizeof(bytes)*sizeof(char) );

  // Map to device0 global
  destSpace.insert( MemorySpaceSet::GLOBAL, device0->allDeviceListIndex() );
  mappedSet = memMgr->createEmptyMappedSet();
  memMgr->map( alloc, MAP_READ, destSpace, mappedSet );
  memMgr->finalizeAllocations( mappedSet );
  memMgr->finalizeTransfers( mappedSet );

  // Verify the memory directly from the device
  char *dev0Memory = mapToHost<MemorySpaceSet::GLOBAL>( alloc, device0->allDeviceListIndex() );
  memset( bytes, 0, sizeof(bytes)*sizeof(char) );
  lwca::Driver lwdaLateBinder;
  lwdaLateBinder.lwMemcpyDtoH( bytes, (LWdeviceptr)dev0Memory, sizeof(bytes)*sizeof(char) );

  ASSERT_THAT( memcmp(bytes, (void*)"HELLO", sizeof(bytes)*sizeof(char) ), 0 );
}


TEST_F( TheMemoryManager, PropagateCopiesMultipleGPUs ) // Assumes there are at least two GPUs available on the system
{
  REQUIRE_MULTIPLE_LWDA_DEVICES();

  Device* dev1 = devMgr->primaryLWDADevice();
  ASSERT_NE( dev1, (Device*)0 );
  Device* dev2 = 0;
  DeviceArray devices = devMgr->allDevices();
  for( DeviceArray::iterator it = devices.begin(), itEnd = devices.end(); it != itEnd; ++it ) {
    if( (dev2 = dynamic_cast<LWDADevice*>(*it)) && dev2 != dev1 )
      break;
  }
  ASSERT_NE( dev2, (Device*)0 );

  shared_ptr<MappedSet> mappedSet = memMgr->createEmptyMappedSet();

  MemorySpaceSet destSpace( MemorySpaceSet::HOST_MALLOCED, dev1->allDeviceListIndex() );
  MemorySpaceSet allowedSpaces = destSpace;
  allowedSpaces.insert( MemorySpaceSet::HOST_MALLOCED, dev2->allDeviceListIndex() );
  allowedSpaces.insert( MemorySpaceSet::GLOBAL, dev1->allDeviceListIndex() );
  allowedSpaces.insert( MemorySpaceSet::GLOBAL, dev2->allDeviceListIndex() );

  MBufferHandle alloc = memMgr->allocate( defaultBufferDims, OPTIX_API_BUFFER_GROUP, allowedSpaces );

  char bytes[] = "BufferEvacTest##0[]";
  populateHostMemory( alloc, destSpace, mappedSet, dev1, bytes, sizeof(bytes)*sizeof(char) );

  mapToDeviceGlobal( alloc, destSpace, mappedSet, dev1);

  removeHostSpace( alloc, destSpace, mappedSet );

  mapToDeviceGlobal( alloc, destSpace, mappedSet, dev2);

  compareGlobalWith(alloc, dev2, bytes, sizeof(bytes)*sizeof(char));
}
#endif
