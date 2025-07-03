/*
* Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
*
* LWPU Corporation and its licensors retain all intellectual property and proprietary
* rights in and to this software, related documentation and any modifications thereto.
* Any use, reproduction, disclosure or distribution of this software and related
* documentation without an express license agreement from LWPU Corporation is strictly
* prohibited.
*
* TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
* INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
* PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
* SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
* LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
* BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
* INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
* SUCH DAMAGES
*/

#include <Memory/DeviceSpecificTableBase.h>

#include <LWCA/Memory.h>
#include <Context/Context.h>
#include <Device/CPUDevice.h>
#include <Device/LWDADevice.h>
#include <Device/DeviceManager.h>
#include <Memory/MBuffer.h>
#include <Memory/MemoryManager.h>
#include <Util/ContainerAlgorithm.h>

#include <cstring>
#include <iterator>

using namespace optix;

static BufferDimensions tableDimensions( const size_t elementSize, const unsigned height )
{
    return BufferDimensions( RT_FORMAT_USER, elementSize, 1U, height, 1U, 1U );
}

DeviceSpecificTableBase::DeviceSpecificTableBase( Context* context, size_t diSize, size_t diOffset, size_t ddSize, size_t ddOffset, size_t interleavedSize )
    : m_context( context )
    , m_diSize( diSize )
    , m_diOffset( diOffset )
    , m_ddSize( ddSize )
    , m_ddOffset( ddOffset )
    , m_interleavedSize( interleavedSize )
{
    const size_t numAllDevices = m_context->getDeviceManager()->allDevices().size();
    m_deviceDependentTables.resize( numAllDevices );
    m_deviceDependentPtrs.resize( numAllDevices, nullptr );
    m_deviceDependentTablesStatus.resize( numAllDevices );
    m_deviceDependentTablesDirtyIndices.resize( numAllDevices );
    m_deviceInterleavedTables.resize( numAllDevices );

    m_deviceIndependentTableDirtyIndices.fill( std::make_pair( -1, -1 ) );
    for( int i = 0; i < numAllDevices; ++i )
    {
        m_deviceDependentTablesStatus[i] = TableStatus::NotDirty;
        m_deviceDependentTablesDirtyIndices[i].fill( std::make_pair( -1, -1 ) );
    }
}

DeviceSpecificTableBase::~DeviceSpecificTableBase()
{
    if( m_deviceIndependentPtr != nullptr )
    {
        unmapDeviceIndependentPtr();
    }
    for( size_t i = 0; i < m_deviceDependentPtrs.size(); ++i )
    {
        if( m_deviceDependentPtrs[i] != nullptr )
        {
            unmapDeviceDependentPtr( i );
        }
    }
}

bool DeviceSpecificTableBase::resize( size_t size )
{
    if( size == m_size )
    {
        return false;
    }
    m_size = size;
    if( !m_deviceIndependentTable )
    {
        allocateMBuffers();
        return true;
    }

    const bool growing = size > m_capacity;
    if( growing )
    {
        m_capacity = std::max( m_capacity * 2, m_size );
        resizeMBuffers();
    }
    return growing;
}

void DeviceSpecificTableBase::sync()
{
    if( !dirty() )
    {
        return;
    }

    const char* diPtr = m_deviceIndependentTableStatus != TableStatus::NotDirty ? mapDeviceIndependentPtrReadOnly() : nullptr;
    for( Device* device : m_context->getDeviceManager()->activeDevices() )
    {
        const unsigned int allDevicesIndex = device->allDeviceListIndex();
        if( LWDADevice* lwdaDevice = deviceCast<LWDADevice>( device ) )
        {
            syncLwdaDevice( lwdaDevice, allDevicesIndex, diPtr );
        }
        else if( deviceCast<CPUDevice>( device ) )
        {
            syncCpuDevice( allDevicesIndex, diPtr );
        }
        else
        {
            throw prodlib::AssertionFailure( RT_EXCEPTION_INFO,
                                             "Illegal destination device.  Only LWCA and CPU devices supported." );
        }
        m_deviceDependentTablesStatus[allDevicesIndex] = TableStatus::NotDirty;
        m_deviceDependentTablesDirtyIndices[allDevicesIndex].fill( std::make_pair( -1, -1 ) );
    }
    if( diPtr )
    {
        unmapDeviceIndependentPtr();
    }
    m_deviceIndependentTableStatus = TableStatus::NotDirty;
    m_deviceIndependentTableDirtyIndices.fill( std::make_pair( -1, -1 ) );
}

void DeviceSpecificTableBase::syncLwdaDevice( LWDADevice* lwdaDevice, const unsigned int allDevicesIndex, const char* diSrcPtr )
{
    lwdaDevice->makeLwrrent();

    char* dst = getInterleavedDevicePtr( allDevicesIndex );

    if( m_deviceIndependentTableStatus != TableStatus::NotDirty )
    {
        syncTableLwdaDevice( m_deviceIndependentTableStatus, m_deviceIndependentTableDirtyIndices, diSrcPtr,
                             &dst[m_diOffset], m_diSize );
    }
    if( m_deviceDependentTablesStatus[allDevicesIndex] != TableStatus::NotDirty )
    {
        const char* src = mapDeviceDependentPtrReadOnly( allDevicesIndex );
        syncTableLwdaDevice( m_deviceDependentTablesStatus[allDevicesIndex],
                             m_deviceDependentTablesDirtyIndices[allDevicesIndex], src, &dst[m_ddOffset], m_ddSize );
        unmapDeviceDependentPtr( allDevicesIndex );
    }
}

void DeviceSpecificTableBase::syncCpuDevice( const unsigned int allDevicesIndex, const char* diSrcPtr )
{
    char* const dst = getInterleavedDevicePtr( allDevicesIndex );
    if( m_deviceIndependentTableStatus != TableStatus::NotDirty )
    {
        char* const diDest = &dst[m_diOffset];
        syncTableCpuDevice( m_deviceIndependentTableStatus, m_deviceIndependentTableDirtyIndices, diSrcPtr, diDest, m_diSize );
    }
    if( m_deviceDependentTablesStatus[allDevicesIndex] != TableStatus::NotDirty )
    {
        char* const ddDest   = &dst[m_ddOffset];
        const char* ddSrcPtr = mapDeviceDependentPtrReadOnly( allDevicesIndex );
        syncTableCpuDevice( m_deviceDependentTablesStatus[allDevicesIndex],
                            m_deviceDependentTablesDirtyIndices[allDevicesIndex], ddSrcPtr, ddDest, m_ddSize );
        unmapDeviceDependentPtr( allDevicesIndex );
    }
}

void DeviceSpecificTableBase::markRecordsDirty( TableStatus& tableStatus,
                                                std::array<std::pair<int, int>, 4>& dirtyRecordIndices,
                                                int dirtyRecordIndexBegin,
                                                int dirtyRecordIndexEnd )
{
    if( dirtyRecordIndexBegin < 0 )  // we don't know what records are being written, just assume the entire table is dirty
        tableStatus = TableStatus::NeedsFullCopy;

    if( tableStatus == TableStatus::NeedsFullCopy )
        return;

    tableStatus = TableStatus::NeedsPartialCopy;
    for( int i = 0; i < dirtyRecordIndices.size(); ++i )
    {
        if( dirtyRecordIndices[i].first == -1 )  // found an empty slot to store a range of dirty entires to be copied
        {
            dirtyRecordIndices[i] = std::make_pair( dirtyRecordIndexBegin, dirtyRecordIndexEnd );
            return;
        }

        if( dirtyRecordIndexEnd + 1 >= dirtyRecordIndices[i].first
            && dirtyRecordIndexBegin - 1 <= dirtyRecordIndices[i].second )  // extend the range of an existing pair to include the new dirty record indices
        {
            bool coalesce = false;
            if( dirtyRecordIndexBegin < dirtyRecordIndices[i].first )
            {
                dirtyRecordIndices[i].first = dirtyRecordIndexBegin;
                coalesce                    = true;
            }

            if( dirtyRecordIndexEnd > dirtyRecordIndices[i].second )
            {
                dirtyRecordIndices[i].second = dirtyRecordIndexEnd;
                coalesce                     = true;
            }

            if( coalesce )
            {
                for( int j = i + 1; j < dirtyRecordIndices.size(); ++j )  // coalesce with other pairs if possible
                {
                    if( dirtyRecordIndices[j].first == -1 )
                        break;

                    if( dirtyRecordIndices[j].second + 1 >= dirtyRecordIndices[i].first
                        && dirtyRecordIndices[j].first - 1 <= dirtyRecordIndices[i].second )
                    {
                        dirtyRecordIndices[i].first = std::min( dirtyRecordIndices[i].first, dirtyRecordIndices[j].first );
                        dirtyRecordIndices[i].second = std::max( dirtyRecordIndices[i].second, dirtyRecordIndices[j].second );

                        for( int k = dirtyRecordIndices.size() - 1; k >= j; --k )  // swap another pair in the place of the coalesced pair (or just set it to [-1, -1] if there are no other pairs)
                        {
                            if( dirtyRecordIndices[k].first != -1 )
                            {
                                dirtyRecordIndices[j] = dirtyRecordIndices[k];
                                dirtyRecordIndices[k] = std::make_pair( -1, -1 );
                                j--;  // decrement j so that we keep looking for pairs to coalesce starting at whatever got swapped in (otherwise it would get skipped)
                                break;
                            }
                        }
                    }
                }
            }
            return;
        }
    }
    tableStatus = TableStatus::NeedsFullCopy;  // didn't find a slot (too many dirty entires to copy individually)
}

void DeviceSpecificTableBase::syncTableLwdaDevice( TableStatus tableStatus,
                                                   std::array<std::pair<int, int>, 4>& dirtyRecordIndices,
                                                   const char* src,
                                                   char* const dst,
                                                   size_t      srcPitch )
{
    LWDA_MEMCPY2D args{0};
    args.srcMemoryType = LW_MEMORYTYPE_HOST;
    args.srcPitch      = srcPitch;
    args.dstMemoryType = LW_MEMORYTYPE_DEVICE;
    args.dstPitch      = m_interleavedSize;
    args.WidthInBytes  = srcPitch;

    if( tableStatus == TableStatus::NeedsFullCopy )
    {
        args.srcHost   = src;
        args.dstDevice = reinterpret_cast<LWdeviceptr>( dst );
        args.Height    = m_size;
        lwca::memcpy2D( &args );
    }
    else
    {
        for( int i = 0; i < dirtyRecordIndices.size(); ++i )
        {
            if( dirtyRecordIndices[i].first == -1 )  // done copying
                break;

            // copy a range of dirty records
            args.srcHost   = src + dirtyRecordIndices[i].first * srcPitch;
            args.dstDevice = reinterpret_cast<LWdeviceptr>( dst + dirtyRecordIndices[i].first * m_interleavedSize );
            args.Height    = dirtyRecordIndices[i].second - dirtyRecordIndices[i].first + 1;

            lwca::memcpy2D( &args );
        }
    }
}

void DeviceSpecificTableBase::syncTableCpuDevice( TableStatus tableStatus,
                                                  std::array<std::pair<int, int>, 4>& dirtyRecordIndices,
                                                  const char* src,
                                                  char* const dst,
                                                  size_t      recordSize )
{
    if( tableStatus == TableStatus::NeedsFullCopy )
    {
        std::memcpy( dst, src, m_size * recordSize );
    }
    else
    {
        for( size_t i = 0; i < dirtyRecordIndices.size(); ++i )
        {
            if( dirtyRecordIndices[i].first == -1 )  // done copying
                break;

            // copy a range of dirty records
            size_t copySize = ( dirtyRecordIndices[i].second - dirtyRecordIndices[i].first + 1 ) * recordSize;
            std::memcpy( &dst[dirtyRecordIndices[i].first * recordSize], &src[dirtyRecordIndices[i].first * recordSize], copySize );
        }
    }
}

size_t DeviceSpecificTableBase::getTableSizeInBytes() const
{
    return m_interleavedSize * m_size;
}

void DeviceSpecificTableBase::activeDeviceRemoved( const unsigned int allDevicesIndex )
{
    if( m_deviceIndependentTable )
    {
        if( m_deviceIndependentPtr != nullptr )
        {
            unmapDeviceIndependentPtr();
        }
        m_deviceIndependentTable.reset();
    }
    if( m_deviceDependentTables[allDevicesIndex] )
    {
        if( m_deviceDependentPtrs[allDevicesIndex] != nullptr )
        {
            unmapDeviceDependentPtr( allDevicesIndex );
        }
        m_deviceDependentTables[allDevicesIndex].reset();
        m_deviceInterleavedTables[allDevicesIndex].reset();
    }
}

void DeviceSpecificTableBase::setActiveDevices()
{
    MemoryManager* const mm{m_context->getMemoryManager()};
    if( !m_deviceIndependentTable )
    {
        m_deviceIndependentTable = mm->allocateMBuffer( tableDimensions( m_diSize, m_capacity ), MBufferPolicy::internal_hostonly );
    }
    for( Device* device : m_context->getDeviceManager()->activeDevices() )
    {
        const unsigned int deviceIndex{device->allDeviceListIndex()};
        if( !m_deviceDependentTables[deviceIndex] )
        {
            m_deviceDependentTables[deviceIndex] =
                mm->allocateMBuffer( tableDimensions( m_ddSize, m_capacity ), MBufferPolicy::internal_hostonly );
            const MBufferPolicy interleavedPolicy = deviceCast<CPUDevice>( device ) != nullptr ?
                                                        MBufferPolicy::internal_hostonly :
                                                        MBufferPolicy::internal_readonly_deviceonly;
            m_deviceInterleavedTables[deviceIndex] =
                mm->allocateMBuffer( tableDimensions( m_interleavedSize, m_capacity ), interleavedPolicy );
        }
    }
}

bool DeviceSpecificTableBase::allocatedToDevice( const unsigned int allDeviceListIndex ) const
{
    return static_cast<bool>( m_deviceDependentTables[allDeviceListIndex] );
}

char* DeviceSpecificTableBase::mapDeviceIndependentPtr( int dirtyRecordIndexBegin, int dirtyRecordIndexEnd )
{
    markRecordsDirty( m_deviceIndependentTableStatus, m_deviceIndependentTableDirtyIndices, dirtyRecordIndexBegin,
                      dirtyRecordIndexEnd < 0 ? dirtyRecordIndexBegin : dirtyRecordIndexEnd );

    if( m_deviceIndependentPtr == nullptr )
    {
        m_deviceIndependentPtr = m_context->getMemoryManager()->mapToHost( m_deviceIndependentTable, MAP_READ_WRITE );
    }
    return m_deviceIndependentPtr + std::max( 0, dirtyRecordIndexBegin ) * m_diSize;
}

// Note: We map the pointer READ_WRITE so that if someone calls mapDeviceIndependentPtr they will
// get a pointer that they can write through and we don't have to unmap/remap just to change the
// mapping mode.  This mapping mode is advisory to memoryManager() to tell it when/how to sync the
// memory, but we are taking over the sync responsibility ourselves, so this mode is purely advisory
// from our point of view.
const char* DeviceSpecificTableBase::mapDeviceIndependentPtrReadOnly()
{
    if( m_deviceIndependentPtr == nullptr )
    {
        m_deviceIndependentPtr = m_context->getMemoryManager()->mapToHost( m_deviceIndependentTable, MAP_READ_WRITE );
    }
    return m_deviceIndependentPtr;
}

void DeviceSpecificTableBase::unmapDeviceIndependentPtr()
{
    RT_ASSERT( m_deviceIndependentPtr != nullptr );
    m_context->getMemoryManager()->unmapFromHost( m_deviceIndependentTable );
    m_deviceIndependentPtr = nullptr;
}

char* DeviceSpecificTableBase::mapDeviceDependentPtr( const unsigned int allDevicesIndex, int dirtyRecordIndexBegin, int dirtyRecordIndexEnd )
{
    markRecordsDirty( m_deviceDependentTablesStatus[allDevicesIndex], m_deviceDependentTablesDirtyIndices[allDevicesIndex],
                      dirtyRecordIndexBegin, dirtyRecordIndexEnd < 0 ? dirtyRecordIndexBegin : dirtyRecordIndexEnd );

    if( m_deviceDependentPtrs[allDevicesIndex] == nullptr )
    {
        m_deviceDependentPtrs[allDevicesIndex] =
            m_context->getMemoryManager()->mapToHost( m_deviceDependentTables[allDevicesIndex], MAP_READ_WRITE );
    }
    return m_deviceDependentPtrs[allDevicesIndex] + std::max( 0, dirtyRecordIndexBegin ) * m_ddSize;
}

// Note: We map the pointer READ_WRITE so that if someone calls mapDeviceIndependentPtr they will
// get a pointer that they can write through and we don't have to unmap/remap just to change the
// mapping mode.  This mapping mode is advisory to memoryManager() to tell it when/how to sync the
// memory, but we are taking over the sync responsibility ourselves, so this mode is purely advisory
// from our point of view.
const char* DeviceSpecificTableBase::mapDeviceDependentPtrReadOnly( unsigned int allDevicesIndex )
{
    if( m_deviceDependentPtrs[allDevicesIndex] == nullptr )
    {
        m_deviceDependentPtrs[allDevicesIndex] =
            m_context->getMemoryManager()->mapToHost( m_deviceDependentTables[allDevicesIndex], MAP_READ_WRITE );
    }
    return m_deviceDependentPtrs[allDevicesIndex];
}

void DeviceSpecificTableBase::unmapDeviceDependentPtr( const unsigned int allDevicesIndex )
{
    RT_ASSERT( m_deviceDependentPtrs[allDevicesIndex] != nullptr );
    m_context->getMemoryManager()->unmapFromHost( m_deviceDependentTables[allDevicesIndex] );
    m_deviceDependentPtrs[allDevicesIndex] = nullptr;
}

char* DeviceSpecificTableBase::getInterleavedDevicePtr( const unsigned int allDevicesIndex )
{
    auto& ptr = m_deviceInterleavedTables[allDevicesIndex];
    return ptr ? ptr->getAccess( m_context->getDeviceManager()->allDevices()[allDevicesIndex] ).getLinearPtr() : nullptr;
}

const std::array<std::pair<int, int>, 4>& DeviceSpecificTableBase::getDeviceIndependentTableDirtyIndicesForTest()
{
    return m_deviceIndependentTableDirtyIndices;
}

void DeviceSpecificTableBase::unmapFromHost( unsigned int allDevicesIndex )
{
    if( m_deviceIndependentPtr != nullptr )
    {
        unmapDeviceIndependentPtr();
    }
    if( m_deviceDependentPtrs[allDevicesIndex] != nullptr )
    {
        unmapDeviceDependentPtr( allDevicesIndex );
    }
}

bool DeviceSpecificTableBase::dirty() const
{
    return m_deviceIndependentTableStatus != TableStatus::NotDirty
           || algorithm::find( m_deviceDependentTablesStatus, TableStatus::NotDirty ) != std::end( m_deviceDependentTablesStatus );
}

void DeviceSpecificTableBase::allocateMBuffers()
{
    RT_ASSERT( m_deviceIndependentPtr == nullptr );
    MemoryManager* mm{m_context->getMemoryManager()};
    m_capacity               = m_size;
    m_deviceIndependentTable = mm->allocateMBuffer( tableDimensions( m_diSize, m_capacity ), MBufferPolicy::internal_hostonly );
    m_deviceIndependentPtr = nullptr;
    for( Device* device : m_context->getDeviceManager()->activeDevices() )
    {
        const unsigned int deviceIndex = device->allDeviceListIndex();
        RT_ASSERT( !m_deviceDependentTables[deviceIndex] );
        RT_ASSERT( m_deviceDependentPtrs[deviceIndex] == nullptr );
        m_deviceDependentTables[deviceIndex] =
            mm->allocateMBuffer( tableDimensions( m_ddSize, m_capacity ), MBufferPolicy::internal_hostonly );
        m_deviceDependentPtrs[deviceIndex] = nullptr;

        const MBufferPolicy interleavedPolicy = deviceCast<CPUDevice>( device ) != nullptr ?
                                                    MBufferPolicy::internal_hostonly :
                                                    MBufferPolicy::internal_readonly_deviceonly;
        m_deviceInterleavedTables[deviceIndex] =
            mm->allocateMBuffer( tableDimensions( m_interleavedSize, m_capacity ), interleavedPolicy );
    }
}

void DeviceSpecificTableBase::resizeMBuffers()
{
    if( m_deviceIndependentPtr != nullptr )
    {
        unmapDeviceIndependentPtr();
    }
    resizeMBuffer( m_deviceIndependentTable );

    // Mark all records from the pre-resized table dirty because they will NOT get copied when changing the size of the
    // MBuffer (preserveContents does nothing since the internal_hostonly and internal_readonly_deviceonly policies of
    // the tables do not require a copy on launch, and thus the validSet for their MBuffers is always empty)
    if( m_deviceIndependentTable->getDimensions().getTotalSizeInBytes() > 0 )
        markRecordsDirty( m_deviceIndependentTableStatus, m_deviceIndependentTableDirtyIndices, 0, m_size - 1 );
    for( Device* device : m_context->getDeviceManager()->activeDevices() )
    {
        const unsigned int deviceIndex = device->allDeviceListIndex();
        if( m_deviceDependentPtrs[deviceIndex] != nullptr )
        {
            unmapDeviceDependentPtr( deviceIndex );
        }
        resizeMBuffer( m_deviceDependentTables[deviceIndex] );
        resizeMBuffer( m_deviceInterleavedTables[deviceIndex] );

        if( m_deviceDependentTables[deviceIndex]->getDimensions().getTotalSizeInBytes() > 0 )
            markRecordsDirty( m_deviceDependentTablesStatus[deviceIndex],
                              m_deviceDependentTablesDirtyIndices[deviceIndex], 0, m_size - 1 );
    }
}

void DeviceSpecificTableBase::resizeMBuffer( const MBufferHandle& buffer )
{
    // Can we use the already allocated space?
    BufferDimensions size = buffer->getDimensions();
    size.setSize( m_capacity );
    MemoryManager* mm{m_context->getMemoryManager()};
    if( mm->isMappedToHost( buffer ) )
    {
        mm->unmapFromHost( buffer );
    }
    mm->changeSize( buffer, size, /*preserveContents=*/true );
}
