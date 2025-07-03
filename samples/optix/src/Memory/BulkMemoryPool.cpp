// Copyright (c) 2018, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <Memory/BulkMemoryPool.h>

#include <LWCA/Memory.h>
#include <Context/Context.h>
#include <Device/LWDADevice.h>
#include <Device/DeviceManager.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/system/Knobs.h>
#include <prodlib/system/Logger.h>

using namespace optix;
using namespace prodlib;

namespace {
Knob<size_t> k_bulkAllocationPoolSize( RT_DSTRING( "mem.bulkAllocationPool" ),
                                       16u * 1024u * 1024u,
                                       RT_DSTRING( "Size of memory pools for for bulk allocator" ) );
Knob<size_t> k_maxPinnedMemory( RT_DSTRING( "mem.maxPinnedMemory" ),
                                2u * 1024u * 1024u * 1024u,
                                RT_DSTRING( "Only attempt to allocate this much pinned memory" ) );
Knob<int> k_bulkAllocationMaxEmptyPools( RT_DSTRING( "mem.bulkAllocationMaxFreePools" ),
                                         2,
                                         RT_DSTRING( "Maximum number of memory pools to leave empty after free" ) );
};

void BulkMemoryPool::initialize( Context* context, int allDeviceIndex, size_t alignment )
{
    std::lock_guard<std::mutex> guard( m_poolMutex );

    m_context        = context;
    m_allDeviceIndex = allDeviceIndex;
    m_alignment      = alignment;
}

void BulkMemoryPool::clear()
{
    for( BulkMemoryPoolEntry& entry : m_entries )
    {
        RT_ASSERT_MSG( entry.allocator->empty(), "Non-empty BulkMemoryPoolEntry detected" );

        if( entry.deviceIsHost )
        {
            lwca::memFreeHost( (void*)entry.basePointer );
        }
        else
        {
            getLWDADevice( m_allDeviceIndex )->makeLwrrent();
            lwca::memFree( entry.basePointer );
        }
    }

    m_entries.clear();
}

LWdeviceptr BulkMemoryPool::allocate( size_t nbytes, bool deviceIsHost )
{
    std::lock_guard<std::mutex> guard( m_poolMutex );

    if( !deviceIsHost )
        getLWDADevice( m_allDeviceIndex )->makeLwrrent();

    // Linear search for a free pool. We only expect to have a handful
    // of them, but start at the back to ensure efficiency.
    for( int idx = m_entries.size() - 1; idx >= 0; --idx )
    {
        size_t offset = m_entries[idx].allocator->alloc( nbytes, m_alignment );
        if( offset != LightweightAllocator::BAD_ADDR )
        {
            return m_entries[idx].basePointer + offset;
        }
    }

    size_t      poolSize    = k_bulkAllocationPoolSize.get();
    LWdeviceptr basePointer = 0ull;
    LWresult    lwresult    = LWDA_SUCCESS;

    if( deviceIsHost )
    {
        // Allocate on host
        if( poolSize * m_entries.size() < k_maxPinnedMemory.get() )
        {
            // Allocate pinned memory if we have not used too much yet.
            basePointer = (LWdeviceptr)lwca::memAllocHost( poolSize, &lwresult );
        }
        if( lwresult != LWDA_SUCCESS || poolSize * m_entries.size() >= k_maxPinnedMemory.get() )
        {
            basePointer = (LWdeviceptr)malloc( nbytes );
            return basePointer;
        }
    }
    else
    {
        // Allocate on device
        basePointer = lwca::memAlloc( poolSize, &lwresult );
        if( lwresult != LWDA_SUCCESS )
        {
            return 0ull;
        }
    }

    m_entries.push_back( BulkMemoryPoolEntry{
        basePointer, std::unique_ptr<LightweightAllocator>( new LightweightAllocator( poolSize ) ), deviceIsHost} );
    size_t offset = m_entries.back().allocator->alloc( nbytes, m_alignment );
    RT_ASSERT_MSG( offset != LightweightAllocator::BAD_ADDR, "Bad allocation from virgin pool" );
    return m_entries.back().basePointer + offset;
}

void BulkMemoryPool::free( LWdeviceptr ptr, size_t nbytes, bool deviceIsHost )
{
    std::lock_guard<std::mutex> guard( m_poolMutex );

    size_t poolSize = k_bulkAllocationPoolSize.get();

    // Find pool
    unsigned int idx = 0;
    for( ; idx < m_entries.size(); ++idx )
    {
        if( ptr >= m_entries[idx].basePointer && ptr < m_entries[idx].basePointer + poolSize )
            break;
    }

    // Some host allocations are not pinned, so use free to release them.
    if( deviceIsHost && idx >= m_entries.size() )
    {
        ::free( (void*)ptr );
        return;
    }

    // Any allocations that reach this point should be in a pool.
    RT_ASSERT_MSG( idx < m_entries.size(), "Could not find device allocation in pools" );

    // Free the block, and free a pool if there are too many empty ones.
    m_entries[idx].allocator->free( ptr - m_entries[idx].basePointer, nbytes, m_alignment );
    if( m_entries[idx].allocator->empty() )
    {
        int freeCount = 0;
        int maxFree   = k_bulkAllocationMaxEmptyPools.get();
        for( unsigned int i = 0; i < m_entries.size(); i++ )
        {
            if( m_entries[i].allocator->empty() && ++freeCount > maxFree )
            {
                if( deviceIsHost )
                {
                    lwca::memFreeHost( (void*)m_entries[i].basePointer );
                }
                else
                {
                    getLWDADevice( m_allDeviceIndex )->makeLwrrent();
                    lwca::memFree( m_entries[i].basePointer );
                }
                m_entries.erase( m_entries.begin() + i );
                break;
            }
        }
    }
}

// Utility functions
Device* BulkMemoryPool::getDevice( unsigned int allDeviceIndex ) const
{
    RT_ASSERT_MSG( m_context != nullptr, "Using a BulkMemoryPool without being initialized" );
    Device* device = m_context->getDeviceManager()->allDevices()[allDeviceIndex];
    return device;
}

LWDADevice* BulkMemoryPool::getLWDADevice( unsigned int allDeviceIndex ) const
{
    Device*     device     = getDevice( allDeviceIndex );
    LWDADevice* lwdaDevice = deviceCast<LWDADevice>( device );
    RT_ASSERT_MSG( lwdaDevice != nullptr, "Invalid LWCA device" );
    return lwdaDevice;
}
