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

#include <Memory/MTextureSampler.h>

#include <Device/Device.h>
#include <Memory/DemandLoad/PagingService.h>
#include <Memory/MBufferListener.h>

#include <prodlib/exceptions/Assert.h>

#include <algorithm>

namespace optix {

MTextureSampler::MTextureSampler( const MBufferHandle& backing, const TextureDescriptor& texDesc )
    : m_backing( backing )
    , m_texDesc( texDesc )
{
    // Make space for 3 devices (2 GPU + 1 CPU) as a common default.
    m_access.reserve( 3 );
}

MTextureSampler::~MTextureSampler()
{
    /*
    All data that is wrapped by this class is freed by MemoryManager::freeTextureSampler
    when the shared pointer deleter is ilwoked.

    We don't have to delete anything here!
    */
}

void MTextureSampler::growAccess( int count )
{
    RT_ASSERT( count >= 0 && count <= OPTIX_MAX_DEVICES );

    if( count > (int)m_access.size() )
        m_access.resize( count );
}

MAccess MTextureSampler::getAccess( int allDeviceIndex ) const
{
    const int count = allDeviceIndex + 1;
    RT_ASSERT( count >= 0 && count <= OPTIX_MAX_DEVICES );
    if( count > m_access.size() )
        return {};

    return m_access[allDeviceIndex];
}

MAccess MTextureSampler::getAccess( const Device* device ) const
{
    return getAccess( device->allDeviceListIndex() );
}

bool MTextureSampler::isDemandLoad( const Device* device ) const
{
    const MAccess::Kind kind = m_backing->getAccess( device ).getKind();
    return kind == MAccess::DEMAND_LOAD_ARRAY;
}

unsigned int MTextureSampler::getDemandLoadMinMipLevel( int allDeviceListIndex ) const
{
    return m_backing->getAccess( allDeviceListIndex ).getDemandLoadArray().minMipLevel;
}

unsigned int MTextureSampler::allocateVirtualPages( PagingService* pagingMgr, unsigned int& numPages )
{
    if( !m_startPage )
    {
        m_numPages  = pagingMgr->computeSoftwareNumDemandTexturePages( m_backing->getDimensions() );
        m_startPage = pagingMgr->reservePageTableEntries( m_numPages );
    }
    numPages = m_numPages;
    return static_cast<unsigned int>( *m_startPage );
}

void MTextureSampler::releaseVirtualPages( PagingService* pagingMgr )
{
    if( m_startPage )
        pagingMgr->releasePageTableEntries( *m_startPage, m_numPages );
    m_numPages  = 0U;
    m_startPage = nullptr;
}

void MTextureSampler::addListener( MTextureSamplerListener* listener )
{
    m_listeners.push_back( listener );
}

void MTextureSampler::removeListener( MTextureSamplerListener* listener )
{
    m_listeners.erase( std::remove_if( m_listeners.begin(), m_listeners.end(),
                                       [listener]( MTextureSamplerListener* item ) { return item == listener; } ),
                       m_listeners.end() );
}

void MTextureSampler::setAccess( const Device* device, const MAccess& newMA )
{
    const int allDeviceIndex = static_cast<int>( device->allDeviceListIndex() );
    growAccess( allDeviceIndex + 1 );
    MAccess& oldMA = m_access[allDeviceIndex];
    for( MTextureSamplerListener* listener : m_listeners )
        listener->eventMTextureSamplerMAccessDidChange( device, oldMA, newMA );
    oldMA = newMA;
}

}  // namespace optix
