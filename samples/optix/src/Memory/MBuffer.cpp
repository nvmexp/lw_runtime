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

#include <Memory/MBuffer.h>

#include <Device/Device.h>
#include <Device/DeviceManager.h>
#include <Memory/MBufferListener.h>
#include <Memory/MResources.h>
#include <Memory/MemoryManager.h>

#include <corelib/misc/String.h>
#include <prodlib/exceptions/Assert.h>

#include <algorithm>
#include <sstream>
#include <string>

using namespace optix;


const BufferDimensions& MBuffer::getDimensions() const
{
    return m_dims;
}

BufferDimensions MBuffer::getNonZeroDimensions() const
{
    BufferDimensions dims = getDimensions();
    if( dims.zeroSized() )
        dims.setSize( 1 );
    return dims;
}

MBufferPolicy MBuffer::getPolicy() const
{
    return m_policy;
}

MBuffer::MBuffer( MemoryManager* manager, const BufferDimensions& size, MBufferPolicy policy, const DeviceSet& allowedSet )
    : m_policy( policy )
    , m_dims( size )
    , m_memoryManager( manager )
    , m_allowedSet( allowedSet )
{
    m_resources.reset( new MResources( this ) );

    // Make space for 3 devices (2 GPU + 1 CPU) as a common default.
    m_access.reserve( 3 );
}

MBuffer::~MBuffer()
{
    /*
    All data that is wrapped by this class is freed by
    MemoryManager::freeMBuffer when the shared pointer deleter is
    ilwoked.

    We don't have to delete anything here!
  */

    RT_ASSERT_NOTHROW( m_backingClientsList.empty(), "Backing store being destroyed while there are client buffers" );
}

void MBuffer::attachBackingStore( const MBufferHandle& backing )
{
    RT_ASSERT_MSG( !m_backing, "Backing store already set" );
    m_backing = backing;
    m_backing->m_backingClientsList.addItem( this );
}

void MBuffer::detachBackingStore()
{
    RT_ASSERT_MSG( m_backing, "Trying to detach backing store, but none attached" );
    m_backing->m_backingClientsList.removeItem( this );
    m_backing.reset();
}

void MBuffer::growAccess( int count )
{
    RT_ASSERT( count >= 0 && count <= OPTIX_MAX_DEVICES );
    RT_ASSERT_MSG( !m_notifyingListeners, "Growing access while notifying listeners of an access change" );

    if( count > (int)m_access.size() )
        m_access.resize( count );
}

// Public, return a value
MAccess MBuffer::getAccess( int allDeviceIndex ) const
{
    const int count = allDeviceIndex + 1;
    RT_ASSERT( count >= 0 && count <= OPTIX_MAX_DEVICES );
    if( count > m_access.size() )
        return {};

    return m_access[allDeviceIndex];
}

MAccess MBuffer::getAccess( const Device* device ) const
{
    return getAccess( device->allDeviceListIndex() );
}

const GfxInteropResource& MBuffer::getGfxInteropResource() const
{
    return m_resources->m_gfxInteropResource;
}

void MBuffer::addListener( MBufferListener* listener )
{
    if( !listener )
        return;
    m_listeners.push_back( listener );
}

void MBuffer::removeListener( MBufferListener* listener )
{
    m_listeners.erase( std::remove( m_listeners.begin(), m_listeners.end(), listener ), m_listeners.end() );
}

bool optix::MBuffer::isZeroCopy( int allDeviceIndex ) const
{
    return m_resources->m_resourceKind[allDeviceIndex] == MResources::ZeroCopy;
}

bool optix::MBuffer::isAllocatedP2P( int allDeviceIndex ) const
{
    return ( m_resources->m_p2pAllocatedSet & DeviceSet( allDeviceIndex ) ) == DeviceSet( allDeviceIndex );
}

namespace {

class FlagToggler
{
  public:
    FlagToggler( bool& flag )
        : m_flag( flag )
    {
        m_flag = true;
    }
    ~FlagToggler() { m_flag = false; }
  private:
    bool& m_flag;
};

}  // namespace

void MBuffer::setAccess( int allDeviceIndex, const MAccess& newMA )
{
    RT_ASSERT( !m_notifyingListeners );
    Device* device = m_memoryManager->getDevice( allDeviceIndex );
    growAccess( allDeviceIndex + 1 );
    MAccess&    oldMA = m_access[allDeviceIndex];
    FlagToggler notifying( m_notifyingListeners );
    for( MBufferListener* listener : m_listeners )
    {
        listener->eventMBufferMAccessDidChange( this, device, oldMA, newMA );
    }
    oldMA = newMA;
}

void MBuffer::setAccess( const Device* device, const MAccess& newMA )
{
    setAccess( device->allDeviceListIndex(), newMA );
}

// Assumes that the device allocation exists, will throw otherwise.
// The memory managaer must be manually synced before trying to access a device pointer.
char* MBuffer::getDevicePtr( Device* device )
{
    const DeviceSet theDevice( device );
    RT_ASSERT_MSG( m_pinnedSet == theDevice, "Attemp to get device pointer on non-pinned buffer" );

    MAccess access = getAccess( device );

    if( m_backing )
        RT_ASSERT( m_backing->getAccess( device ).getKind() == MAccess::LINEAR );

    size_t offset = 0;
    char*  ptr    = nullptr;

    switch( access.getKind() )
    {
        case MAccess::LINEAR:
            offset = 0;
            ptr    = access.getLinearPtr();
            break;
        case MAccess::TEX_OBJECT:
            offset = access.getTexObject().indexOffset;
            ptr    = m_backing->getAccess( device ).getLinearPtr();
            break;
        case MAccess::DEMAND_TEX_OBJECT:
            offset = access.getDemandTexObject().indexOffset;
            ptr    = m_backing->getAccess( device ).getLinearPtr();
            break;
        case MAccess::TEX_REFERENCE:
            offset = access.getTexReference().indexOffset;
            ptr    = m_backing->getAccess( device ).getLinearPtr();
            break;
        default:
            RT_ASSERT_FAIL_MSG( "Invalid Access kind for device pointer" );
            break;
    }

    offset *= m_dims.elementSize();
    return ptr + offset;
}

// Update the minMipLevel field in DemandLoadArrayAccess from info in MResources.
// Called from MemoryManager::reallocDemandLoadLwdaArray.
void MBuffer::updateDemandLoadArrayAccess( int allDeviceListIndex )
{
    DemandLoadArrayAccess oldAccess = getAccess( allDeviceListIndex ).getDemandLoadArray();
    MAccess               newAccess = MAccess::makeDemandLoadArray( oldAccess.virtualPageBegin, oldAccess.numPages,
                                                      m_resources->m_demandTextureMinMipLevel[allDeviceListIndex] );
    setAccess( allDeviceListIndex, newAccess );
}

LWDA_ARRAY_SPARSE_PROPERTIES MBuffer::getSparseTextureProperties( int allDeviceListIndex ) const
{
    return m_resources->m_lwdaArrays[allDeviceListIndex].getSparseProperties();
}

static bool isLwdaSparseArray( const lwca::MipmappedArray& mipmapped )
{
    const lwca::Array             level = mipmapped.getLevel( 0 );
    const LWDA_ARRAY3D_DESCRIPTOR desc  = level.getDescriptor3D();
    return desc.Flags & LWDA_ARRAY3D_SPARSE;
}

LWDA_ARRAY_SPARSE_PROPERTIES MBuffer::getSparseTextureProperties() const
{
    for( const lwca::MipmappedArray& lwrrArray : m_resources->m_lwdaArrays )
        if( lwrrArray.get() != nullptr && isLwdaSparseArray( lwrrArray ) )
            return lwrrArray.getSparseProperties();

    return m_memoryManager->getSparseTexturePropertiesFromMBufferProperties( this );
}

unsigned int MBuffer::getDemandLoadStartPage() const
{
    return *m_resources->m_demandLoadAllocation;
}

void MBuffer::switchLwdaSparseArrayToLwdaArray( DeviceSet devices )
{
    RT_ASSERT( m_policy == MBufferPolicy::texture_readonly_demandload );
    m_memoryManager->switchLwdaSparseArrayToLwdaArray( this, devices );
}

std::string MBuffer::stateString() const
{
    std::stringstream out;

    out << "M" << m_serialNumber << ":"
        << " policy=" << toString( m_policy ) << " size=" << corelib::formatSize( m_dims.getTotalSizeInBytes() )
        << " dims=" << m_dims.toString() << " exactSz=" << m_dims.getTotalSizeInBytes() << " p2pReq=" << m_p2pRequested
        << " allowedSet=" << m_allowedSet.toString() << " allocatedSet=" << m_allocatedSet.toString()
        << " validSet=" << m_validSet.toString() << " pinnedSet=" << m_pinnedSet.toString()
        << " frozenSet=" << m_frozenSet.toString() << " mapped=" << m_mappedToHost
        << " hasBacking=" << ( m_backing != nullptr );

    out << " accesses=";

    // Note: do not call getAccess here, which would resize m_access.
    for( const MAccess& acces : m_access )
        out << (int)acces.getKind();
    for( int i = (int)m_access.size(); i < OPTIX_MAX_DEVICES; ++i )
        out << (int)MAccess::NONE;

    out << " resKind=";
    for( MResources::ResourceKind kind : m_resources->m_resourceKind )
        out << MResources::toString( kind ) << ",";

    return out.str();
}
