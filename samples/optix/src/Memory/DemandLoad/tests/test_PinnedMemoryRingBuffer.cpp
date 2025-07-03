//
// Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
//
// LWPU Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software, related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from LWPU Corporation is strictly
// prohibited.
//
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
// AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
// INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
// SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
// LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
// BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
// INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGES
//

#include <Memory/DemandLoad/PinnedMemoryRingBuffer.h>

#include <LWCA/Event.h>
#include <LWCA/Memory.h>
#include <LWCA/Stream.h>
#include <Context/Context.h>
#include <Device/LWDADevice.h>
#include <Device/DeviceManager.h>
#include <ThreadPool/Job.h>
#include <ThreadPool/ThreadPool.h>

#include <srcTests.h>

#include <prodlib/exceptions/LwdaError.h>

#include <lwda_runtime.h>

#include <chrono>
#include <thread>

using namespace optix;
using namespace testing;

namespace {

class ManagedStream : public lwca::Stream
{
  public:
    ManagedStream()
        : Stream( create().get() )
    {
    }
    ~ManagedStream() { destroy(); }
};

class ManagedEvent : public lwca::Event
{
  public:
    ManagedEvent()
        : Event( create().get() )
    {
    }
    ~ManagedEvent() { destroy(); }
};

const size_t       RING_BUFFER_SIZE  = 256U;
const unsigned int NUM_RING_EVENTS   = 4U;
const unsigned int NUM_RING_REQUESTS = 256U;

void fillSequence( unsigned char* dest, size_t size )
{
    unsigned char counter = 0;
    for( int i = 0; i < size; ++i )
    {
        dest[i] = counter;
        counter = ( counter + 1U ) % 256U;
    }
}

void fillSequence( void* dest, size_t size )
{
    fillSequence( static_cast<unsigned char*>( dest ), size );
}

void fillSequenceDescending( unsigned char* dest, size_t size )
{
    unsigned char counter = 255;
    for( size_t i = 0; i < size; ++i )
    {
        dest[i] = counter;
        counter = ( counter - 1U ) % 256U;
    }
}

void fillSequenceDescending( void* dest, size_t size )
{
    fillSequenceDescending( static_cast<unsigned char*>( dest ), size );
}

std::vector<unsigned char> generateSequence( size_t size )
{
    std::vector<unsigned char> sequence( size );
    fillSequence( sequence.data(), sequence.size() );
    return sequence;
}

class TestPinnedMemoryRingBuffer : public Test
{
  public:
    static void SetUpTestCase()
    {
        RTcontext apiContext = nullptr;
        rtContextCreate( &apiContext );
        s_context = reinterpret_cast<Context*>( apiContext );
        s_context->getDeviceManager()->enableActiveDevices();
        for( Device* device : s_context->getDeviceManager()->activeDevices() )
        {
            if( LWDADevice* lwdaDevice = deviceCast<LWDADevice>( device ) )
            {
                s_allDeviceListIndex = lwdaDevice->allDeviceListIndex();
                lwdaDevice->makeLwrrent();
                break;
            }
        }
    }

    static void TearDownTestCase() { rtContextDestroy( reinterpret_cast<RTcontext>( s_context ) ); }

    void SetUp() override
    {
        m_buffer.init( s_context->getDeviceManager(), RING_BUFFER_SIZE, NUM_RING_EVENTS, NUM_RING_REQUESTS );
        m_buffer.setActiveDevices( s_context->getDeviceManager()->allDevices(), DeviceSet{static_cast<int>( s_allDeviceListIndex )} );
        m_devBuffer = lwca::memAlloc( RING_BUFFER_SIZE );
    }
    void TearDown() override
    {
        lwca::memFree( m_devBuffer );
        m_buffer.destroy();
    }

    static void assertDeviceMemoryContainsSequence( LWdeviceptr devDest, size_t size )
    {
        std::vector<unsigned char> hostReadBack( size );
        lwca::memcpyDtoH( hostReadBack.data(), devDest, size );
        // validate read memory contains same pattern
        unsigned char counter = 0;

        // Buffer should have thread 1's result in it.
        for( size_t j = 0; j < size; ++j )
        {
            ASSERT_EQ( hostReadBack[j], counter ) << " at index " << j << " and size " << size;
            counter = ( counter + 1U ) % 256U;
        }
    }

    static Context*        s_context;
    static unsigned int    s_allDeviceListIndex;
    PinnedMemoryRingBuffer m_buffer;
    LWdeviceptr            m_devBuffer{};
};

Context*     TestPinnedMemoryRingBuffer::s_context            = nullptr;
unsigned int TestPinnedMemoryRingBuffer::s_allDeviceListIndex = ~0U;

}  // namespace

TEST_F( TestPinnedMemoryRingBuffer, construct_init_destroy )
{
    // Everything is handled by the fixture
}

TEST_F( TestPinnedMemoryRingBuffer, single_thread_acquire_release_entire_buffer )
{
    const size_t               size       = RING_BUFFER_SIZE;
    std::vector<unsigned char> hostSource = generateSequence( size );
    ManagedEvent               evt;
    ManagedStream              str;

    StagingPageAllocation allocation = m_buffer.acquireResource( size );
    memcpy( allocation.address, hostSource.data(), size );
    memcpyHtoDAsync( m_devBuffer, allocation.address, size, str );
    m_buffer.recordEvent( str, s_allDeviceListIndex, allocation );
    m_buffer.releaseResource( allocation );
    evt.record( str );
    evt.synchronize();

    ASSERT_NE( nullptr, allocation.address );
    assertDeviceMemoryContainsSequence( m_devBuffer, size );
}

TEST_F( TestPinnedMemoryRingBuffer, releasing_without_recording_events_is_ok )
{
    const size_t          size       = RING_BUFFER_SIZE;
    StagingPageAllocation allocation = m_buffer.acquireResource( size );

    m_buffer.releaseResource( allocation );
}

namespace {

class TestDisjointRangeJob : public FragmentedJob
{
  public:
    TestDisjointRangeJob( unsigned int allDeviceListIndex, PinnedMemoryRingBuffer& buffer, ManagedStream& stream, LWdeviceptr devDest, size_t count )
        : FragmentedJob( count )
        , m_allDeviceListIndex( allDeviceListIndex )
        , m_buffer( buffer )
        , m_stream( stream )
        , m_devDest( devDest )
    {
    }
    ~TestDisjointRangeJob() override = default;

    void exelwteFragment( size_t index, size_t count ) noexcept override
    {
        // Acquire host memory from ring buffer, fill with pattern, async copy to device, and release ring buffer memory.
        const size_t          size       = RING_BUFFER_SIZE / count;
        const LWdeviceptr     devDest    = m_devDest + index * size;
        StagingPageAllocation allocation = m_buffer.acquireResource( size );
        ASSERT_NE( nullptr, allocation.address );
        fillSequence( allocation.address, size );
        memcpyHtoDAsync( devDest, allocation.address, size, m_stream );
        m_buffer.recordEvent( m_stream, m_allDeviceListIndex, allocation );
        m_buffer.releaseResource( allocation );
    }

  private:
    const unsigned int      m_allDeviceListIndex;
    PinnedMemoryRingBuffer& m_buffer;
    ManagedStream&          m_stream;
    LWdeviceptr             m_devDest;
};

}  // namespace

TEST_F( TestPinnedMemoryRingBuffer, two_threads_acquire_release_conlwrrently )
{
    const size_t  NUM_THREADS = 2;
    ThreadPool    pool( static_cast<float>( NUM_THREADS ), 1.0f, NUM_THREADS );
    ManagedStream str;

    std::shared_ptr<TestDisjointRangeJob> job =
        std::make_shared<TestDisjointRangeJob>( s_allDeviceListIndex, m_buffer, str, m_devBuffer, NUM_THREADS );
    pool.submitJobAndWait( job );

    // validate read memory contains same pattern
    const size_t chunkSize = RING_BUFFER_SIZE / NUM_THREADS;
    for( size_t i = 0; i < NUM_THREADS; ++i )
    {
        assertDeviceMemoryContainsSequence( m_devBuffer + chunkSize * i, chunkSize );
    }
}

namespace {

class TestOverlappedRangeJob : public FragmentedJob
{
  public:
    TestOverlappedRangeJob( unsigned int allDeviceListIndex, PinnedMemoryRingBuffer& buffer, ManagedStream& stream, LWdeviceptr devDest, size_t count )
        : FragmentedJob( count )
        , m_allDeviceListIndex( allDeviceListIndex )
        , m_buffer( buffer )
        , m_stream( stream )
        , m_devDest( devDest )
    {
    }
    ~TestOverlappedRangeJob() override = default;

    void exelwteFragment( size_t index, size_t count ) noexcept override
    {
        // Acquire host memory from ring buffer, fill with pattern, async copy to device, and release ring buffer memory.
        const size_t          size    = RING_BUFFER_SIZE;
        const LWdeviceptr     devDest = m_devDest;
        StagingPageAllocation allocation;
        {
            // force thread 1 to wait for thread 0 before acquiring.
            std::unique_lock<std::mutex> lock( m_mutex );
            if( index == 1 )
            {
                m_pause.wait( lock, [this] { return m_proceed; } );
            }
            allocation = m_buffer.acquireResource( size );
            // tell thread 1 to proceed
            if( index == 0 )
            {
                m_proceed = true;
            }
        }

        if( index == 0 )
        {
            // signal thread 1
            m_pause.notify_all();
            std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) );
        }

        ASSERT_NE( nullptr, allocation.address );
        if( index == 0 )
        {
            // thread 0 counts down, thread 1 counts up.
            fillSequenceDescending( allocation.address, size );
        }
        else
        {
            fillSequence( allocation.address, size );
        }
        memcpyHtoDAsync( devDest, allocation.address, size, m_stream );
        m_buffer.recordEvent( m_stream, m_allDeviceListIndex, allocation );
        m_buffer.releaseResource( allocation );
    }

  private:
    const unsigned int      m_allDeviceListIndex;
    PinnedMemoryRingBuffer& m_buffer;
    ManagedStream&          m_stream;
    LWdeviceptr             m_devDest;
    std::mutex              m_mutex;
    std::condition_variable m_pause;
    bool                    m_proceed = false;
};

}  // namespace

TEST_F( TestPinnedMemoryRingBuffer, new_request_overlaps_pending_requests )
{
    const size_t  NUM_THREADS = 2;
    ThreadPool    pool( static_cast<float>( NUM_THREADS ), 1.0f, NUM_THREADS );
    ManagedStream str;

    std::shared_ptr<TestOverlappedRangeJob> job =
        std::make_shared<TestOverlappedRangeJob>( s_allDeviceListIndex, m_buffer, str, m_devBuffer, NUM_THREADS );
    pool.submitJobAndWait( job );

    assertDeviceMemoryContainsSequence( m_devBuffer, RING_BUFFER_SIZE );
}
