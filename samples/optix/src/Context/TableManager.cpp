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

#include <Context/TableManager.h>

#include <Context/Context.h>
#include <Context/ObjectManager.h>
#include <Context/UpdateManager.h>
#include <Device/CPUDevice.h>
#include <Device/LWDADevice.h>
#include <Device/DeviceManager.h>
#include <ExelwtionStrategy/CORTTypes.h>
#include <FrontEnd/Canonical/CanonicalProgram.h>
#include <Memory/BackedAllocator.h>
#include <Memory/MemoryManager.h>
#include <Objects/Buffer.h>
#include <Objects/GlobalScope.h>
#include <Objects/LexicalScope.h>
#include <Objects/Program.h>
#include <Objects/TextureSampler.h>
#include <Util/LayoutPrinter.h>
#include <prodlib/misc/TimeViz.h>

#include <prodlib/system/Knobs.h>

#include <cstdint>
#include <cstring>  // memcpy
#include <limits>
#include <sstream>

using namespace prodlib;

static const size_t INITIAL_ALLOCATOR_SIZE = 32 * 1024;
static const size_t ALLOCATOR_ALIGNMENT    = 16;

namespace {
// clang-format off
  Knob<std::string> k_saveObjectRecords( RT_DSTRING("launch.saveObjectRecords"), "", RT_DSTRING( "Save object records in file at launch" ) );
// clang-format on
}  // namespace

namespace optix {

static void fillBufferHeaderForDevice( cort::Buffer::DeviceDependent& bh, const MBufferHandle& mbuf, const MAccess& memAccess );

TableManager::TableManager( Context* context )
    : m_context( context )
    , m_bufferHeaders( context )
    , m_textureHeaders( context )
    , m_programHeaders( context )
    , m_traversableHeaders( context, 0, 0, sizeof( cort::TraversableHeader ), 0, sizeof( cort::TraversableHeader ) )
{
    MBufferPolicy policy = MBufferPolicy::internal_readonly_manualSync;
    m_objectRecordAllocator.reset(
        new BackedAllocator( INITIAL_ALLOCATOR_SIZE, ALLOCATOR_ALIGNMENT, policy, m_context->getMemoryManager() ) );
    // don't use ALLOCATOR_ALIGNMENT here - lwrrently we are allocating blocks holding short, hence we don't really care
    m_dynamicVariableTableAllocator.reset(
        new BackedAllocator( INITIAL_ALLOCATOR_SIZE, sizeof( unsigned short ), policy, m_context->getMemoryManager() ) );

    m_context->getUpdateManager()->registerUpdateListener( this );
}

TableManager::~TableManager()
{
    m_context->getUpdateManager()->unregisterUpdateListener( this );
    unmapFromHost();
}

void TableManager::unmapFromHost()
{
    unmapObjectRecordAllocator( /*syncOnUnmap=*/false );
    unmapDynamicVariableTableAllocator( /*syncOnUnmap=*/false );

    for( const Device* device : m_context->getDeviceManager()->activeDevices() )
    {
        const unsigned int allDeviceIndex = device->allDeviceListIndex();

        m_bufferHeaders.unmapFromHost( allDeviceIndex );
        m_textureHeaders.unmapFromHost( allDeviceIndex );
        m_programHeaders.unmapFromHost( allDeviceIndex );
        m_traversableHeaders.unmapFromHost( allDeviceIndex );
    }
}

void TableManager::preSetActiveDevices( const DeviceArray& removedDevices )
{
    unmapFromHost();

    // Release resources from devices to be removed.
    for( const auto& device : removedDevices )
    {
        const unsigned int allDeviceIndex = device->allDeviceListIndex();

        m_bufferHeaders.activeDeviceRemoved( allDeviceIndex );
        m_textureHeaders.activeDeviceRemoved( allDeviceIndex );
        m_programHeaders.activeDeviceRemoved( allDeviceIndex );
        m_traversableHeaders.activeDeviceRemoved( allDeviceIndex );
    }
}

void TableManager::postSetActiveDevices()
{
    m_bufferHeaders.setActiveDevices();
    m_textureHeaders.setActiveDevices();
    m_programHeaders.setActiveDevices();
    m_traversableHeaders.setActiveDevices();

    resizeBufferHeaders();
    resizeTextureHeaders();
    resizeProgramHeaders();
    resizeTraversableHeaders();

    mapObjectRecordToHost();  // needed to force sync later

    // Fill in all the data
    fillBufferHeaders();
    fillTextureHeaders();
    fillProgramHeaders();
    fillTraversableHeaders();
}

void TableManager::allocateTables()
{
    TIMEVIZ_FUNC;
    // We have to call resizeTextureHeaders() at least once. If no textures were
    // created, notifyCreateTextureSampler() has never been called, so we need to do it here.
    if( m_context->getObjectManager()->getTextureSamplers().empty() )
        resizeTextureHeaders();
    if( m_context->getObjectManager()->getTraversables().empty() )
        resizeTraversableHeaders();
}

void TableManager::syncTablesForLaunch()
{
    TIMEVIZ_FUNC;
    RT_ASSERT( !m_launching );

    ObjectManager* om = m_context->getObjectManager();
    DeviceManager* dm = m_context->getDeviceManager();

    RT_ASSERT( om->getBuffers().linearArraySize() <= m_bufferHeaders.size() );
    RT_ASSERT( om->getTextureSamplers().linearArraySize() <= m_textureHeaders.size() );
    RT_ASSERT( om->getPrograms().linearArraySize() <= m_programHeaders.size() );
    RT_ASSERT( om->getTraversables().linearArraySize() <= m_traversableHeaders.size() );

    // Unmap the host-side copies, then copy to device.
    syncTablesFromHost();

#if defined( DEBUG ) || defined( DEVELOP )
    // Print object records for debugging if necessary.  Do this after syncing, because we
    // don't want the side effect of the LayoutPrinter mapping the tables to cause extra
    // syncing that can mask bugs.  We can clear the host pointers again here, and only
    // inlwr this cost when we are printing the object records.
    if( !k_saveObjectRecords.get().empty() )
    {
        const std::string& filename = k_saveObjectRecords.get();
        std::ostringstream out;
        LayoutPrinter      dumper( out, dm->activeDevices(), om, this, m_context->useRtxDataModel() );  // TODO: This happens every launch and imposes high
        // overhead. It would be great if we could tell whether
        // the tables changed without printing them first (hash
        // the tables themselves?)
        dumper.run();
        std::string contents( out.str() );
        if( contents != m_printedLayoutCache )
        {
            std::ofstream out_file( filename.c_str(),
                                    m_context->getKernelLaunchCount() == 0 ? std::ios_base::out : std::ios_base::app );
            out_file << "======================================================\n";
            out_file << "Launch:" << m_context->getKernelLaunchCount() << "\n";
            out_file << contents;
            m_printedLayoutCache.swap( contents );
        }

        unmapObjectRecordAllocator( /*syncOnUnmap=*/false );
        unmapDynamicVariableTableAllocator( /*syncOnUnmap=*/false );
    }
#endif

    m_launching = true;
}


void TableManager::launchCompleted()
{
    RT_ASSERT( m_launching );
    m_launching = false;
}

std::shared_ptr<size_t> TableManager::allocateObjectRecord( size_t nbytes )
{
    RT_ASSERT( !m_launching );
    size_t                  oldSize         = getObjectRecordSize();
    bool                    backingUnmapped = false;
    BackedAllocator::Handle hdl             = m_objectRecordAllocator->alloc( nbytes, &backingUnmapped );
    size_t                  newSize         = getObjectRecordSize();
    if( backingUnmapped )
        m_objectRecordHostPtr = nullptr;
    if( oldSize != newSize )
        m_context->getUpdateManager()->eventTableManagerObjectRecordResized( oldSize, newSize );
    return hdl;
}

char* TableManager::getObjectRecordHostPointer()
{
    RT_ASSERT( !m_launching );
    mapObjectRecordToHost();
    return m_objectRecordHostPtr;
}

std::shared_ptr<size_t> optix::TableManager::allocateDynamicVariableTable( size_t bytes )
{
    RT_ASSERT( !m_launching );
    bool                    backingUnmapped = false;
    BackedAllocator::Handle hdl             = m_dynamicVariableTableAllocator->alloc( bytes, &backingUnmapped );
    if( backingUnmapped )
        m_dynamicVariableTableHostPtr = nullptr;
    // so far there is no need (aka no consumer) to notify the update manager about any change in size
    return hdl;
}

char* TableManager::getDynamicVariableTableHostPointer()
{
    RT_ASSERT( !m_launching );
    mapDynamicVariableTableToHost();
    return m_dynamicVariableTableHostPtr;
}

size_t TableManager::getObjectRecordSize()
{
    return m_objectRecordAllocator->getUsedAddressRangeEnd();
}

size_t TableManager::getDynamicVariableTableSize()
{
    return m_dynamicVariableTableAllocator->getUsedAddressRangeEnd();
}

char* TableManager::getObjectRecordDevicePointer( const Device* device )
{
    RT_ASSERT( m_launching );
    return m_objectRecordAllocator->memory()->getAccess( device ).getLinearPtr();
}

cort::Buffer* TableManager::getBufferHeaderDevicePointer( const Device* device )
{
    RT_ASSERT( m_launching );
    unsigned int d = device->allDeviceListIndex();
    return m_bufferHeaders.getInterleavedTableDevicePtr( d );
}

cort::TextureSamplerHost* TableManager::getTextureHeaderDevicePointer( const Device* device )
{
    RT_ASSERT( m_launching );
    unsigned int d = device->allDeviceListIndex();
    return m_textureHeaders.getInterleavedTableDevicePtr( d );
}

cort::ProgramHeader* TableManager::getProgramHeaderDevicePointer( const Device* device )
{
    RT_ASSERT( m_launching );
    unsigned int d = device->allDeviceListIndex();
    return m_programHeaders.getInterleavedTableDevicePtr( d );
}

cort::TraversableHeader* TableManager::getTraversableHeaderDevicePointer( const Device* device )
{
    RT_ASSERT( m_launching );
    unsigned int d = device->allDeviceListIndex();
    return reinterpret_cast<cort::TraversableHeader*>( m_traversableHeaders.getInterleavedDevicePtr( d ) );
}

RtcTraversableHandle TableManager::getTraversableHandleForTest( int id, unsigned int allDeviceIndex )
{
    return getTraversableHeaderDdHostPointerReadOnly( id, allDeviceIndex )->traversable;
}

char* TableManager::getDynamicVariableTableDevicePointer( const Device* device )
{
    RT_ASSERT( m_launching );
    return m_dynamicVariableTableAllocator->memory()->getAccess( device ).getLinearPtr();
}

void TableManager::writeBufferHeader( int id, size_t width, size_t height, size_t depth, unsigned int pageWidth, unsigned int pageHeight, unsigned int pageDepth )
{
    RT_ASSERT( !m_launching );
    cort::Buffer::DeviceIndependent* buf = getBufferHeaderDiHostPointer( id );
    buf->size.x                          = width;
    buf->size.y                          = height;
    buf->size.z                          = depth;
    buf->pageSize.x                      = pageWidth;
    buf->pageSize.y                      = pageHeight;
    buf->pageSize.z                      = pageDepth;
}

void TableManager::clearBufferHeader( int id )
{
    RT_ASSERT( !m_launching );
    writeBufferHeader( id, 0, 0, 0, 0, 0, 0 );
    for( auto device : m_context->getDeviceManager()->activeDevices() )
    {
        cort::Buffer::DeviceDependent* buf = getBufferHeaderDdHostPointer( id, device->allDeviceListIndex() );
        buf->data                          = nullptr;
        buf->texUnit                       = cort::Buffer::UseDataAsPointer;
    }
}

void TableManager::writeTextureHeader( int          id,
                                       unsigned int width,
                                       unsigned int height,
                                       unsigned int depth,
                                       unsigned int mipLevels,
                                       unsigned int format,
                                       unsigned int wrapMode0,
                                       unsigned int wrapMode1,
                                       unsigned int wrapMode2,
                                       unsigned int normCoord,
                                       unsigned int filterMode,
                                       unsigned int normRet,
                                       bool         isDemandLoad )
{
    RT_ASSERT( !m_launching );
    cort::TextureSamplerHost::DeviceIndependent* th = getTextureHeaderDiHostPointer( id );

    th->width        = width;
    th->height       = height;
    th->depth        = depth;
    th->mipLevels    = mipLevels;
    th->format       = format;
    th->wrapMode0    = wrapMode0;
    th->wrapMode1    = wrapMode1;
    th->wrapMode2    = wrapMode2;
    th->normCoord    = normCoord;
    th->filterMode   = filterMode;
    th->normRet      = normRet;
    th->isDemandLoad = static_cast<unsigned int>( isDemandLoad );
}

void optix::TableManager::writeDemandTextureHeader( int          id,
                                                    unsigned int mipTailFirstLevel,
                                                    float        ilwAnisotropy,
                                                    unsigned int tileWidth,
                                                    unsigned int tileHeight,
                                                    unsigned int tileGutterWidth,
                                                    unsigned int isInitialized,
                                                    unsigned int isSquarePowerOfTwo,
                                                    unsigned int mipmapFilterMode )
{
    RT_ASSERT( !m_launching );
    cort::TextureSamplerHost::DeviceIndependent* di = getTextureHeaderDiHostPointer( id );
    di->mipTailFirstLevel                           = mipTailFirstLevel;
    di->ilwAnisotropy                               = ilwAnisotropy;
    di->tileWidth                                   = tileWidth;
    di->tileHeight                                  = tileHeight;
    di->tileGutterWidth                             = tileGutterWidth;
    di->isInitialized                               = isInitialized;
    di->isSquarePowerOfTwo                          = isSquarePowerOfTwo;
    di->mipmapFilterMode                            = mipmapFilterMode;
}


void optix::TableManager::writeDemandTextureDeviceHeader( int id, unsigned int firstVirtualPage, unsigned int numPages, unsigned int minMipLevel )
{
    RT_ASSERT( !m_launching );

    for( Device* device : m_context->getDeviceManager()->activeDevices() )
    {
        cort::TextureSamplerHost::DeviceDependent* dd = getTextureHeaderDdHostPointer( id, device->allDeviceListIndex() );
        setDemandTextureDeviceHeader( dd, firstVirtualPage, numPages, minMipLevel );
    }
}


void TableManager::clearTextureHeader( int id )
{
    RT_ASSERT( !m_launching );
    for( auto device : m_context->getDeviceManager()->activeDevices() )
    {
        cort::TextureSamplerHost::DeviceDependent* th = getTextureHeaderDdHostPointer( id, device->allDeviceListIndex() );
        // Initialize texture 0 to use texref 0, so if it is erroneously used it doesn't
        // crash the driver.  See bug http://lwbugs/1722422
        th->texref = id != 0 ? cort::TextureSampler::IlwalidSampler : 0;
        th->swptr  = nullptr;
    }
}

void TableManager::writeProgramHeader( int id, unsigned int offset )
{
    RT_ASSERT( !m_launching );
    cort::ProgramHeader::DeviceIndependent* ph = getProgramHeaderDiHostPointer( id );
    ph->programOffset                          = offset;
}

void TableManager::clearProgramHeader( int id )
{
    RT_ASSERT( !m_launching );
    writeProgramHeader( id, 0U );
    for( auto device : m_context->getDeviceManager()->activeDevices() )
    {
        cort::ProgramHeader::DeviceDependent* ph = getProgramHeaderDdHostPointer( id, device->allDeviceListIndex() );
        ph->canonicalProgramID                   = 0;
    }
}

void TableManager::writeTraversableHeader( int id, RtcTraversableHandle travHandle, unsigned int allDeviceIndex )
{
    RT_ASSERT( !m_launching );
    cort::TraversableHeader* travHeaderPtr = getTraversableHeaderDdHostPointer( id, allDeviceIndex );
    travHeaderPtr->traversable             = travHandle;
}

void TableManager::clearTraversableHeader( int id )
{
    for( auto device : m_context->getDeviceManager()->activeDevices() )
        writeTraversableHeader( id, 0, device->allDeviceListIndex() );
}

void TableManager::notifyCanonicalProgramAddedToProgram( const Program* program )
{
    RT_ASSERT( !m_launching );
    writeProgramHeaderCanonicalProgram( program );
}

void TableManager::writeProgramHeaderCanonicalProgram( const Program* program )
{
    RT_ASSERT( !m_launching );
    int id = program->getId();
    for( auto device : m_context->getDeviceManager()->activeDevices() )
    {
        cort::ProgramHeader::DeviceDependent* ph = getProgramHeaderDdHostPointer( id, device->allDeviceListIndex() );
        const CanonicalProgram*               cp = program->getCanonicalProgram( device );
        ph->canonicalProgramID                   = cp->getID();
    }
}

void TableManager::notifyCreateBuffer( int bid, Buffer* buffer )
{
    resizeBufferHeaders();
}

void TableManager::notifyCreateTextureSampler( int tid, TextureSampler* sampler )
{
    resizeTextureHeaders();
}

void TableManager::notifyCreateProgram( int pid, Program* program )
{
    resizeProgramHeaders();
}

void TableManager::notifyCreateTraversableHandle( int tid, GraphNode* node )
{
    resizeTraversableHeaders();
}

void TableManager::eventBufferMAccessDidChange( const Buffer* buffer, const Device* device, const MAccess& oldMA, const MAccess& newMA )
{
    // If the device is inactive or we are in the middle of a device change we need to ignore the event.
    // We will pick up the changes later when we reinitialize everything.
    if( !device->isActive() || !m_bufferHeaders.allocatedToDevice( device->allDeviceListIndex() ) )
    {
        return;
    }
    cort::Buffer::DeviceDependent* bh = getBufferHeaderDdHostPointer( buffer->getId(), device->allDeviceListIndex() );
    fillBufferHeaderForDevice( *bh, buffer->getMBuffer(), newMA );
}

void TableManager::eventTextureSamplerMAccessDidChange( const TextureSampler* sampler,
                                                        const Device*         device,
                                                        const MAccess&        oldMA,
                                                        const MAccess&        newMA )
{
    // If the device is inactive or we are in the middle of a device change we need to ignore the event.
    // We will pick up the changes later when we reinitialize everything.
    if( !device->isActive() || !m_textureHeaders.allocatedToDevice( device->allDeviceListIndex() ) )
    {
        return;
    }
    cort::TextureSamplerHost::DeviceDependent* th =
        getTextureHeaderDdHostPointer( sampler->getId(), device->allDeviceListIndex() );
    fillTextureHeaderForDevice( *th, sampler, device, newMA );
}

void TableManager::resizeBufferHeaders()
{
    RT_ASSERT( !m_launching );

    const size_t     eltSize    = sizeof( cort::Buffer );
    const size_t     numBuffers = m_context->getObjectManager()->getBuffers().linearArraySize();
    const size_t     newCount   = std::max( numBuffers, static_cast<size_t>( 1 ) );  // need to allocate at least 1, since it is possible to have no Buffers if you resize devices early
    BufferDimensions oldSize( RT_FORMAT_USER, eltSize, 1, m_bufferHeaders.size(), 1, 1 );
    BufferDimensions newSize( RT_FORMAT_USER, eltSize, 1, newCount, 1, 1 );
    if( newCount != m_bufferHeaders.size() )
    {
        m_context->getUpdateManager()->eventTableManagerBufferHeaderTableResized( oldSize.getTotalSizeInBytes(),
                                                                                  newSize.getTotalSizeInBytes() );
    }

    if( m_bufferHeaders.resize( newCount ) )
    {
        // We need to initialize the new sections of the memory
        cort::Buffer::DeviceIndependent* di = getBufferHeaderDiHostPointer( 0 );
        for( size_t i = oldSize.width(); i < m_bufferHeaders.capacity(); ++i )
        {
            di[i] = cort::Buffer::DeviceIndependent{};
        }
        for( auto device : m_context->getDeviceManager()->activeDevices() )
        {
            cort::Buffer::DeviceDependent* headers = getBufferHeaderDdHostPointer( 0, device->allDeviceListIndex() );
            // Start at the oldSize, and work to the allocated size.
            for( size_t i = oldSize.width(); i < m_bufferHeaders.capacity(); ++i )
            {
                headers[i].data    = nullptr;
                headers[i].texUnit = cort::Buffer::UseDataAsPointer;
            }
        }
    }
}

void TableManager::resizeTextureHeaders()
{
    RT_ASSERT( !m_launching );

    const size_t     eltSize     = sizeof( cort::TextureSamplerHost );
    const size_t     numSamplers = m_context->getObjectManager()->getTextureSamplers().linearArraySize();
    const size_t     newCount    = std::max( numSamplers, static_cast<size_t>( 1 ) );  // need to allocate at least 1, since it is possible to have no TextureSamplers
    BufferDimensions oldSize( RT_FORMAT_USER, eltSize, 1, m_textureHeaders.size(), 1, 1 );
    BufferDimensions newSize( RT_FORMAT_USER, eltSize, 1, newCount, 1, 1 );
    if( newCount != m_textureHeaders.size() )
        m_context->getUpdateManager()->eventTableManagerTextureHeaderTableResized( oldSize.getTotalSizeInBytes(),
                                                                                   newSize.getTotalSizeInBytes() );

    if( m_textureHeaders.resize( newCount ) )
    {
        // We need to initialize the new sections of the memory
        cort::TextureSamplerHost::DeviceIndependent* di = getTextureHeaderDiHostPointer( 0 );
        for( size_t i = oldSize.width(); i < m_textureHeaders.capacity(); ++i )
        {
            di[i] = cort::TextureSamplerHost::DeviceIndependent{};
        }
        for( auto device : m_context->getDeviceManager()->activeDevices() )
        {
            cort::TextureSamplerHost::DeviceDependent* headers = getTextureHeaderDdHostPointer( 0, device->allDeviceListIndex() );
            // Start at the oldSize, and work to the allocated size.
            for( size_t i = oldSize.width(); i < m_textureHeaders.capacity(); i++ )
            {
                // Initialize texture 0 to use texref 0, so if it is erroneously used it doesn't
                // crash the driver.  See bug http://lwbugs/1722422
                headers[i].texref = i != 0 ? cort::TextureSampler::IlwalidSampler : 0;
                headers[i].swptr  = nullptr;
            }
        }
    }
}

void TableManager::resizeProgramHeaders()
{
    RT_ASSERT( !m_launching );

    const size_t     eltSize     = sizeof( cort::ProgramHeader );
    const size_t     numPrograms = m_context->getObjectManager()->getPrograms().linearArraySize();
    const size_t     newCount    = std::max( numPrograms, static_cast<size_t>( 1 ) );  // need to allocate at least 1, since it is possible to have no Programs if you resize devices early
    BufferDimensions oldSize( RT_FORMAT_USER, eltSize, 1, m_programHeaders.size(), 1, 1 );
    BufferDimensions newSize( RT_FORMAT_USER, eltSize, 1, newCount, 1, 1 );
    if( newCount != m_programHeaders.size() )
        m_context->getUpdateManager()->eventTableManagerProgramHeaderTableResized( oldSize.getTotalSizeInBytes(),
                                                                                   newSize.getTotalSizeInBytes() );
    if( m_programHeaders.resize( newCount ) )
    {
        // We need to initialize the new sections of the memory
        cort::ProgramHeader::DeviceIndependent* di = getProgramHeaderDiHostPointer( 0 );
        for( size_t i = oldSize.width(); i < m_programHeaders.capacity(); i++ )
        {
            di[i] = cort::ProgramHeader::DeviceIndependent{};
        }
        for( auto device : m_context->getDeviceManager()->activeDevices() )
        {
            cort::ProgramHeader::DeviceDependent* headers = getProgramHeaderDdHostPointer( 0, device->allDeviceListIndex() );
            // Start at the oldSize, and work to the allocated size.
            for( size_t i = oldSize.width(); i < m_programHeaders.capacity(); i++ )
            {
                headers[i] = cort::ProgramHeader::DeviceDependent{};
            }
        }
    }
}

void TableManager::resizeTraversableHeaders()
{
    RT_ASSERT( !m_launching );

    const size_t     eltSize         = sizeof( cort::TraversableHeader );
    const size_t     numTraversables = m_context->getObjectManager()->getTraversables().linearArraySize();
    const size_t     newCount        = std::max( numTraversables, static_cast<size_t>( 1 ) );  // need to allocate at least 1, since it is possible to have no Traversables if you resize devices early
    BufferDimensions oldSize( RT_FORMAT_USER, eltSize, 1, m_traversableHeaders.size(), 1, 1 );
    BufferDimensions newSize( RT_FORMAT_USER, eltSize, 1, newCount, 1, 1 );
    if( newCount != m_traversableHeaders.size() )
        m_context->getUpdateManager()->eventTableManagerTraversableHeaderTableResized( oldSize.getTotalSizeInBytes(),
                                                                                       newSize.getTotalSizeInBytes() );
    if( m_traversableHeaders.resize( newCount ) )
    {
        // We need to initialize the new sections of the memory
        for( auto device : m_context->getDeviceManager()->activeDevices() )
        {
            cort::TraversableHeader* travHandle = getTraversableHeaderDdHostPointer( 0, device->allDeviceListIndex() );
            // Start at the oldSize, and work to the allocated size.
            for( size_t i = oldSize.width(); i < m_traversableHeaders.capacity(); i++ )
            {
                travHandle[i] = {0};
            }
        }
    }
}

static void fillBufferHeaderForDevice( cort::Buffer::DeviceDependent& bh, const MBufferHandle& mbuf, const MAccess& memAccess )
{
    bh.data    = nullptr;
    bh.texUnit = cort::Buffer::UseDataAsPointer;
    if( !mbuf )
        return;

    switch( memAccess.getKind() )
    {
        case MAccess::LINEAR:
            bh.data = memAccess.getLinear().ptr;
            break;
        case MAccess::NONE:
            // Buffers may have an invalid accessor, but only if they
            // are zero-sized or unused.
            break;
        case MAccess::TEX_REFERENCE:
        {
            // Lwrrently, the only type of that can be a tex_reference
            // kind is the texture heap. The null pointer will cause it
            // to use texheap.
            RT_ASSERT_MSG( mbuf->getPolicy() == MBufferPolicy::internal_preferTexheap,
                           "Unexpected MAccess::TEX_REFERENCE pointer" );
            unsigned int texUnit = memAccess.getTexReference().texUnit;
            unsigned int offset  = memAccess.getTexReference().indexOffset;
            bh.texUnit           = texUnit;
            bh.data              = reinterpret_cast<char*>( static_cast<unsigned long long>( offset ) );
        }
        break;
        // Texture samplers contain the information for LWCA sparse arrays, so
        // they don't need to be handled here.
        case MAccess::LWDA_SPARSE:
            break;
        case MAccess::DEMAND_LOAD:
            bh.data = reinterpret_cast<char*>( memAccess.getDemandLoad().virtualPageBegin );
            break;
        case MAccess::DEMAND_LOAD_ARRAY:
            bh.data = reinterpret_cast<char*>( memAccess.getDemandLoadArray().virtualPageBegin );
            break;
        case MAccess::DEMAND_LOAD_TILE_ARRAY:
            // Intentionally do nothing; we get the begin virtual page for a buffer from DEMAND_LOAD_ARRAY
            // and not the tile arrays that are used for storage across multiple demand load textures.
            break;
        default:
            RT_ASSERT_FAIL_MSG( "Invalid buffer pointer kind" );
    }
}

void TableManager::fillBufferHeaders()
{
    const auto& buffers = m_context->getObjectManager()->getBuffers();
    int         last    = 0;
    for( auto id_val = buffers.mapBegin(), end = buffers.mapEnd(); id_val != end; ++id_val )
    {
        const int id     = id_val->first;
        Buffer*   buffer = id_val->second;

        // clear the header between the gaps
        for( ; last < id; ++last )
            clearBufferHeader( last );
        last = id + 1;

        buffer->writeHeader();

        // Fill in the device specific portion
        for( auto device : m_context->getDeviceManager()->activeDevices() )
        {
            const MBufferHandle&           mbuf = buffer->getMBuffer();
            cort::Buffer::DeviceDependent* bh   = getBufferHeaderDdHostPointer( id, device->allDeviceListIndex() );
            fillBufferHeaderForDevice( *bh, mbuf, mbuf ? mbuf->getAccess( device ) : MAccess::makeNone() );
        }
    }
}

void TableManager::fillTextureHeaderForDevice( cort::TextureSamplerHost::DeviceDependent& th,
                                               const TextureSampler*                      sampler,
                                               const Device*                              device,
                                               const MAccess&                             memAccess )
{
    const MTextureSamplerHandle& mtex = sampler->getMTextureSampler();

    th.texref = cort::TextureSampler::IlwalidSampler;
    if( !mtex )
    {
        th.swptr = nullptr;
        return;
    }

    switch( memAccess.getKind() )
    {
        case MAccess::LINEAR:
        case MAccess::MULTI_PITCHED_LINEAR:
        {
            // texref should be cort::TextureSampler::UseSoftwarePointer(-1), but at least two
            // tests from smoke fail: solid_angle_simple and glass_test0_adaptive.  This has
            // been -2 (IlwalidSampler) for a long time, and I suspect something is broken with
            // the software texturing.
            th.texref = cort::TextureSampler::IlwalidSampler;
            th.swptr  = memAccess.getPitchedLinear( 0 ).ptr;  // TODO SW MIP texture
        }
        break;
        case MAccess::DEMAND_TEX_OBJECT:
        {
            const DemandTexObjectAccess dto = memAccess.getDemandTexObject();

            th.texref = dto.texObject.get();
            th.swptr = reinterpret_cast<char*>( ( static_cast<unsigned long long>( dto.numPages ) << 32 ) | dto.startPage );
            th.minMipLevel = dto.minMipLevel;
        }
        break;
        case MAccess::TEX_OBJECT:
        {
            th.texref = memAccess.getTexObject().texObject.get();

            // Don't disturb the "swptr" field, which holds the first virtual
            // page number for a demand texture.
            if( mtex->isDemandLoad( device ) )
            {
                th.minMipLevel = mtex->getDemandLoadMinMipLevel( device->allDeviceListIndex() );
            }
        }
        break;
        case MAccess::TEX_REFERENCE:
        {
            th.texref = memAccess.getTexReference().texUnit;
            th.swptr  = nullptr;
        }
        break;
        case MAccess::NONE:
        {
            // TextureSamplers may have an empty accessor during the TextureSampler::reallocateTextureSampler() and tear down process.
        }
        break;
        default:
        {
            RT_ASSERT_FAIL_MSG( "Illegal texture access kind" );
        }
    }
}

void TableManager::fillTextureHeaders()
{
    const auto& textures = m_context->getObjectManager()->getTextureSamplers();
    int         last     = 0;
    for( auto id_val = textures.mapBegin(), end = textures.mapEnd(); id_val != end; ++id_val )
    {
        const int             id      = id_val->first;
        const TextureSampler* texture = id_val->second;

        // clear the header between the gaps
        for( ; last < id; ++last )
            clearTextureHeader( last );
        last = id + 1;

        texture->writeHeader();

        // Fill in the device specific portion
        for( auto device : m_context->getDeviceManager()->activeDevices() )
        {
            const MTextureSamplerHandle&               mtex = texture->getMTextureSampler();
            cort::TextureSamplerHost::DeviceDependent* th = getTextureHeaderDdHostPointer( id, device->allDeviceListIndex() );
            fillTextureHeaderForDevice( *th, texture, device, mtex ? mtex->getAccess( device ) : MAccess::makeNone() );
        }
    }
}

void TableManager::fillProgramHeaders()
{
    RT_ASSERT( !m_launching );
    // Reallocate the table
    const auto& programs = m_context->getObjectManager()->getPrograms();
    int         last     = 0;
    for( auto id_val = programs.mapBegin(), end = programs.mapEnd(); id_val != end; ++id_val )
    {
        const int      id      = id_val->first;
        const Program* program = id_val->second;

        for( ; last < id; ++last )
            clearProgramHeader( last );
        last = id + 1;

        program->writeHeader();

        // Fill in the device specific portion
        writeProgramHeaderCanonicalProgram( program );
    }
}

void TableManager::fillTraversableHeaders()
{
    RT_ASSERT( !m_launching );
    // Reallocate the table
    const auto& traversables = m_context->getObjectManager()->getTraversables();
    int         last         = 0;
    for( auto id_val = traversables.mapBegin(), end = traversables.mapEnd(); id_val != end; ++id_val )
    {
        const int        id   = id_val->first;
        const GraphNode* node = id_val->second;

        for( ; last < id; ++last )
            clearTraversableHeader( last );
        last = id + 1;

        for( auto device : m_context->getDeviceManager()->activeDevices() )
        {
            const unsigned int allDeviceIndex = device->allDeviceListIndex();
            writeTraversableHeader( id, node->getTraversableHandle( allDeviceIndex ), allDeviceIndex );
        }
    }
}

void TableManager::mapObjectRecordToHost()
{
    if( m_objectRecordHostPtr )
        return;
    MemoryManager* mm     = m_context->getMemoryManager();
    m_objectRecordHostPtr = mm->mapToHost( m_objectRecordAllocator->memory(), MAP_WRITE_DISCARD );
}

void TableManager::unmapObjectRecordAllocator( bool syncOnUnmap )
{
    MemoryManager* mm = m_context->getMemoryManager();

    if( m_objectRecordHostPtr )
    {
        mm->unmapFromHost( m_objectRecordAllocator->memory() );
        m_objectRecordHostPtr = nullptr;
        if( syncOnUnmap )
            mm->manualSynchronize( m_objectRecordAllocator->memory() );
    }
}

void TableManager::mapDynamicVariableTableToHost()
{
    if( m_dynamicVariableTableHostPtr )
        return;
    MemoryManager* mm             = m_context->getMemoryManager();
    m_dynamicVariableTableHostPtr = mm->mapToHost( m_dynamicVariableTableAllocator->memory(), MAP_WRITE_DISCARD );
}

void TableManager::unmapDynamicVariableTableAllocator( bool syncOnUnmap )
{
    MemoryManager* mm = m_context->getMemoryManager();

    if( m_dynamicVariableTableHostPtr )
    {
        mm->unmapFromHost( m_dynamicVariableTableAllocator->memory() );
        m_dynamicVariableTableHostPtr = nullptr;
        if( syncOnUnmap )
            mm->manualSynchronize( m_dynamicVariableTableAllocator->memory() );
    }
}

void TableManager::syncTablesFromHost()
{
    unmapObjectRecordAllocator( /*syncOnUnmap=*/true );
    unmapDynamicVariableTableAllocator( /*syncOnUnmap=*/true );

    // Unmap from all active devices
    m_bufferHeaders.sync();
    m_textureHeaders.sync();
    m_programHeaders.sync();
    m_traversableHeaders.sync();
}

cort::Buffer::DeviceIndependent* TableManager::getBufferHeaderDiHostPointer( int id )
{
    RT_ASSERT( !m_launching );
    RT_ASSERT( static_cast<size_t>( id ) < m_bufferHeaders.size() );
    return m_bufferHeaders.mapDeviceIndependentTable( id );
}

cort::Buffer::DeviceDependent* TableManager::getBufferHeaderDdHostPointer( int id, unsigned int allDeviceIndex )
{
    RT_ASSERT( !m_launching );
    RT_ASSERT( static_cast<size_t>( id ) < m_bufferHeaders.size() );
    return m_bufferHeaders.mapDeviceDependentTable( allDeviceIndex, id );
}

const cort::Buffer::DeviceIndependent* TableManager::getBufferHeaderDiHostPointerReadOnly( int id )
{
    RT_ASSERT( !m_launching );
    RT_ASSERT( static_cast<size_t>( id ) < m_bufferHeaders.size() );
    return &m_bufferHeaders.mapDeviceIndependentTableReadOnly()[id];
}

const cort::Buffer::DeviceDependent* TableManager::getBufferHeaderDdHostPointerReadOnly( int id, unsigned int allDeviceIndex )
{
    RT_ASSERT( !m_launching );
    RT_ASSERT( static_cast<size_t>( id ) < m_bufferHeaders.size() );
    return &m_bufferHeaders.mapDeviceDependentTableReadOnly( allDeviceIndex )[id];
}

cort::TextureSamplerHost::DeviceIndependent* TableManager::getTextureHeaderDiHostPointer( int id )
{
    RT_ASSERT( !m_launching );
    RT_ASSERT( static_cast<size_t>( id ) < m_textureHeaders.size() );
    return m_textureHeaders.mapDeviceIndependentTable( id );
}

cort::TextureSamplerHost::DeviceDependent* TableManager::getTextureHeaderDdHostPointer( int id, unsigned int allDeviceIndex )
{
    RT_ASSERT( !m_launching );
    RT_ASSERT( static_cast<size_t>( id ) < m_textureHeaders.size() );
    return m_textureHeaders.mapDeviceDependentTable( allDeviceIndex, id );
}

const cort::TextureSamplerHost::DeviceIndependent* TableManager::getTextureHeaderDiHostPointerReadOnly( int id )
{
    RT_ASSERT( !m_launching );
    RT_ASSERT( static_cast<size_t>( id ) < m_textureHeaders.size() );
    return &m_textureHeaders.mapDeviceIndependentTableReadOnly()[id];
}

const cort::TextureSamplerHost::DeviceDependent* TableManager::getTextureHeaderDdHostPointerReadOnly( int id, unsigned int allDeviceIndex )
{
    RT_ASSERT( !m_launching );
    RT_ASSERT( static_cast<size_t>( id ) < m_textureHeaders.size() );
    return &m_textureHeaders.mapDeviceDependentTableReadOnly( allDeviceIndex )[id];
}

cort::ProgramHeader::DeviceIndependent* TableManager::getProgramHeaderDiHostPointer( int id )
{
    RT_ASSERT( !m_launching );
    RT_ASSERT( static_cast<size_t>( id ) < m_programHeaders.size() );
    return m_programHeaders.mapDeviceIndependentTable( id );
}

cort::ProgramHeader::DeviceDependent* TableManager::getProgramHeaderDdHostPointer( int id, unsigned int allDeviceIndex )
{
    RT_ASSERT( !m_launching );
    RT_ASSERT( static_cast<size_t>( id ) < m_programHeaders.size() );
    return m_programHeaders.mapDeviceDependentTable( allDeviceIndex, id );
}

cort::TraversableHeader* TableManager::getTraversableHeaderDdHostPointer( int id, unsigned int allDeviceIndex )
{
    RT_ASSERT( !m_launching );
    RT_ASSERT( static_cast<size_t>( id ) < m_traversableHeaders.size() );
    return reinterpret_cast<cort::TraversableHeader*>( m_traversableHeaders.mapDeviceDependentPtr( allDeviceIndex, id ) );
}

const cort::ProgramHeader::DeviceIndependent* TableManager::getProgramHeaderDiHostPointerReadOnly( int id )
{
    RT_ASSERT( !m_launching );
    RT_ASSERT( static_cast<size_t>( id ) < m_programHeaders.size() );
    return &m_programHeaders.mapDeviceIndependentTableReadOnly()[id];
}

const cort::ProgramHeader::DeviceDependent* TableManager::getProgramHeaderDdHostPointerReadOnly( int id, unsigned int allDeviceIndex )
{
    RT_ASSERT( !m_launching );
    RT_ASSERT( static_cast<size_t>( id ) < m_programHeaders.size() );
    return &m_programHeaders.mapDeviceDependentTableReadOnly( allDeviceIndex )[id];
}

const cort::TraversableHeader* TableManager::getTraversableHeaderDdHostPointerReadOnly( int id, unsigned int allDeviceIndex )
{
    RT_ASSERT( !m_launching );
    RT_ASSERT( static_cast<size_t>( id ) < m_traversableHeaders.size() );
    return reinterpret_cast<const cort::TraversableHeader*>( m_traversableHeaders.mapDeviceDependentPtrReadOnly( allDeviceIndex ) ) + id;
}

size_t TableManager::getBufferHeaderTableSizeInBytes()
{
    return m_bufferHeaders.getTableSizeInBytes();
}

size_t TableManager::getTextureHeaderTableSizeInBytes()
{
    return m_textureHeaders.getTableSizeInBytes();
}

size_t TableManager::getTraversableTableSizeInBytes()
{
    return m_traversableHeaders.getTableSizeInBytes();
}

size_t TableManager::getProgramHeaderTableSizeInBytes()
{
    return m_programHeaders.getTableSizeInBytes();
}

size_t TableManager::getNumberOfBuffers( const Device* device )
{
    return m_bufferHeaders.size();
}

size_t TableManager::getNumberOfTextures( const Device* device )
{
    return m_textureHeaders.size();
}

size_t TableManager::getNumberOfPrograms( const Device* device )
{
    return m_programHeaders.size();
}

size_t TableManager::getNumberOfTraversables( const Device* device )
{
    return m_traversableHeaders.size();
}

}  // namespace optix
