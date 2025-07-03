//
//  Copyright (c) 2018 LWPU Corporation.  All rights reserved.
//
//  LWPU Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from LWPU Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
//
#include <Memory/DemandLoad/PagingManager.h>

#include <Context/Context.h>
#include <Device/LWDADevice.h>
#include <Device/Device.h>
#include <Device/DeviceManager.h>
#include <Memory/DemandLoad/PageRequestsSerial.h>
#include <Memory/DemandLoad/PageRequestsSimple.h>
#include <Memory/DemandLoad/PageRequestsThreaded.h>
#include <Memory/DemandLoad/PagingServiceLogging.h>
#include <Memory/DemandLoad/TileIndexing.h>
#include <Memory/DemandLoad/optixPaging/optixPaging.h>
#include <Memory/MAccess.h>
#include <Memory/MemoryManager.h>
#include <Objects/Buffer.h>
#include <Util/ElapsedTimeCapture.h>
#include <Util/MakeUnique.h>
#include <Util/Metrics.h>
#include <c-api/ApiCapture.h>

#include <corelib/misc/String.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/system/Knobs.h>
#include <prodlib/system/Logger.h>

#include <algorithm>
#include <chrono>
#include <cmath>

using namespace prodlib;

namespace {
// clang-format off
Knob<unsigned int> k_demandTextureTileWidth(            RT_DSTRING( "rtx.demandTextureTileWidth" ),              32,    RT_DSTRING( "Tile width for demand-loaded textures (must be a power of two)" ) );
Knob<unsigned int> k_demandTextureTileHeight(           RT_DSTRING( "rtx.demandTextureTileHeight" ),             32,    RT_DSTRING( "Tile height for demand-loaded textures (must be a power of two)" ) );
Knob<bool>         k_demandLoadForceSynchronous(        RT_DSTRING( "rtx.demandLoadForceSynchronous" ),          false, RT_DSTRING( "Force demand load paging to use synchronous copies" ) );
Knob<bool>         k_demandLoadForceSingleThreaded(     RT_DSTRING( "rtx.demandLoadForceSingleThreaded" ),       false, RT_DSTRING( "Force demand load paging to ilwoke callbacks from the single main thread" ) );
Knob<bool>         k_demandLoadForceMultiThreaded(      RT_DSTRING( "rtx.demandLoadForceMultiThreaded" ),        false, RT_DSTRING( "Force demand load paging to ilwoke callbacks from the thread pool") );
Knob<bool>         k_demandLoadForceSimplePageRequests( RT_DSTRING( "rtx.demandLoadForceSimplePageRequests" ),   false, RT_DSTRING( "Force demand load paging to ilwoke callbacks using a simple strategy" ) );
Knob<bool>         k_demandLoadForceAsynchronous(       RT_DSTRING( "rtx.demandLoadForceAsynchronous" ),         false, RT_DSTRING( "Force demand load paging to use asynchronous copies" ) );
PublicKnob<bool>   k_enableHardwareSparseTextures(      RT_PUBLIC_DSTRING( "rtx.enableHardwareSparseTextures" ), false, RT_PUBLIC_DSTRING( "Enable hardware sparse textures" ) );
Knob<bool>         k_forceLwdaHybridMode(               RT_DSTRING( "rtx.demandLoadForceLwdaHybrid" ),           true, RT_DSTRING( "Force the paging manager to use the hybrid LWCA path" ) );
// clang-format on
}

namespace {

class LwdaContextSaver
{
  public:
    LwdaContextSaver()
        : m_saved( optix::lwca::Context::getLwrrent() )
    {
    }
    ~LwdaContextSaver()
    {
        if( m_saved.isValid() )
            m_saved.setLwrrent();
    }
    optix::lwca::Context m_saved;
};

}  // namespace

namespace optix {

using namespace demandLoad;

static const unsigned int NUM_STAGING_PAGES = 8192;  // 800 MB

static const char* boolAsYesNo( bool value )
{
    return value ? "yes" : "no";
}

PagingManager::PagingManager( Context* context, bool ilwokeCallbacksPerTileNotPerMipLevel, unsigned int numVirtualPages )
    : m_context( context )
    , m_deviceState( context->getDeviceManager()->allDevices().size() )
    , m_heap( numVirtualPages )
    , m_ilwokeCallbacksPerTileNotPerMipLevel( ilwokeCallbacksPerTileNotPerMipLevel )
    , m_numVirtualPages( numVirtualPages )
    , m_activeDevicesSupportLwdaSparse( m_context->getDeviceManager()->activeDevicesSupportLwdaSparseTextures() )
    , m_lwdaSparseTexturesEnabled( k_enableHardwareSparseTextures.get() )
{
    LOG_NORMAL( "PagingManager::PagingManager() whole mip level callbacks: "
                << boolAsYesNo( m_ilwokeCallbacksPerTileNotPerMipLevel ) << ", numPages " << m_numVirtualPages
                << ", LWCA sparse supported: " << boolAsYesNo( m_activeDevicesSupportLwdaSparse )
                << ", LWCA sparse enabled: " << boolAsYesNo( m_lwdaSparseTexturesEnabled ) << '\n' );
    resetStagingPageAllocator( m_context );
    m_context->getUpdateManager()->registerUpdateListener( this );
    RequestHandler::setStampMemoryBlocks( m_context->getDemandLoadStampMemoryBlocks() );
}

PagingMode PagingManager::getLwrrentPagingMode() const
{
    if( !m_ilwokeCallbacksPerTileNotPerMipLevel )
        return PagingMode::WHOLE_MIPLEVEL;

    if( m_lwdaSparseTexturesEnabled && m_context->getDeviceManager()->activeDevicesSupportLwdaSparseTextures() )
    {
        if( m_context->getDeviceManager()->activeDevicesSupportTextureFootprint() && !k_forceLwdaHybridMode.get() )
            return PagingMode::LWDA_SPARSE_HARDWARE;

        return PagingMode::LWDA_SPARSE_HYBRID;
    }

    return PagingMode::SOFTWARE_SPARSE;
}

void PagingManager::enable()
{
    LOG_NORMAL( "PagingManager::enable\n" );

    // Activate the paging manager when a demand buffer/array is first created.
    m_isActive = true;
    if( !m_pageRequests )
    {
        // Lazily initialize the PageRequests processor because we might need to get the
        // thread pool from the context and that isn't available at the time PagingManager
        // is constructed due to order of initialization of items within the Context.
        createPageRequests();
    }
}

void PagingManager::tearDown()
{
    LOG_NORMAL( "PagingManager::tearDown\n" );

    for( DevicePaging& state : m_deviceState )
    {
        state.deactivate();
    }

    m_stagingPages->tearDown();
}

void PagingManager::launchPrepare( const DeviceSet& devices )
{
    if( !m_isActive )
        return;

    LOG_NORMAL( "PagingManager::launchPrepare\n" );

    m_stagingPages->initializeDeferred();
    m_stagingPages->clear();

    LwdaContextSaver savedContext;

    // Accumulate device set from potentially overlapping async launches.
    const DeviceSet newDevices = m_launchDevices | devices;
    if( m_launchDevices != newDevices )
    {
        m_launchDevices = newDevices;
        activateDevices();
    }

    // Always push mappings and synchronize tiles
    for( unsigned int allDeviceListIndex : newDevices )
    {
        m_deviceState[allDeviceListIndex].pushMappings();
        m_deviceState[allDeviceListIndex].synchronizeTiles();
    }
}

void PagingManager::launchComplete()
{
    if( !m_isActive )
        return;

    LOG_NORMAL( "PagingManager::launchComplete\n" );

    LwdaContextSaver savedContext;
    pullRequests();
    processRequests();
    reportMetrics();
}

// Get page requests from all devices, building a vector of RequestHandlers, each of which
// specifies a pageId and a DeviceSet.
void PagingManager::pullRequests()
{
    LOG_MEDIUM_VERBOSE( "PagingManager::pullRequests\n" );

    m_pageRequests->clear();

    // Get requests from each active device (aclwmulated during potentially overlapping async launches).
    for( unsigned int allDeviceListIndex : m_launchDevices )
    {
        // Pull requests from the specified device. The raw request vector is
        // reused to amortize allocation overhead.
        m_rawRequests.clear();
        m_deviceState[allDeviceListIndex].pullRequests( m_rawRequests );

        m_pageRequests->addRequests( m_rawRequests.data(), static_cast<unsigned int>( m_rawRequests.size() ), allDeviceListIndex );
    }
}

void PagingManager::processRequests()
{
    LOG_MEDIUM_VERBOSE( "PagingManager::processRequests\n" );

    processPageRequests();
    copyPageMappingsToDevices();
}

void PagingManager::reportMetrics() const
{
    MetricsScope metricsScope;
    m_pageRequests->reportMetrics();
    Metrics::logInt( "demand_texture_copy_msec", m_copyMsec );
}

void PagingManager::processPageRequests()
{
    LOG_MEDIUM_VERBOSE( "PagingManager::processPageRequests\n" );

    m_pageRequests->processRequests( getLwrrentPagingMode(), m_stagingPages.get(), getApiCapture(), m_deviceState );
}

void PagingManager::copyPageMappingsToDevices()
{
    LOG_MEDIUM_VERBOSE( "PagingManager::copyPageMappingsToDevices\n" );

    ElapsedTimeCapture elapsed( m_copyMsec );
    for( unsigned int allDeviceListIndex : m_launchDevices )
    {
        DevicePaging& devicePaging = m_deviceState[allDeviceListIndex];

        devicePaging.copyPageMappingsToDevice();
    }
}

// Called from DeviceManager::setActiveDevices with the set of removed devices.
void PagingManager::preSetActiveDevices( const DeviceSet& removedDevices )
{
    if( !m_isActive || removedDevices.empty() )
        return;

    LOG_MEDIUM_VERBOSE( "PagingManager::preSetActiveDevices( " << removedDevices.toString() << " )\n" );

    LwdaContextSaver savedContext;
    for( unsigned int allDeviceListIndex : removedDevices )
    {
        if( !m_deviceState[allDeviceListIndex].isInitialized() )
            continue;

        Device* device = m_context->getDeviceManager()->allDevices()[allDeviceListIndex];
        if( LWDADevice* lwdaDevice = deviceCast<LWDADevice>( device ) )
        {
            if( lwdaDevice->lwdaContext().isValid() )
            {
                m_deviceState[allDeviceListIndex].deactivate();
            }
        }
    }
    m_launchDevices -= removedDevices;

    m_stagingPages->removeActiveDevices( removedDevices );
}

// Called with the set of active devices
void PagingManager::activateDevices()
{
    LOG_MEDIUM_VERBOSE( "PagingManager::activateDevices( " << m_launchDevices.toString() << " )\n" );

    LwdaContextSaver savedContext;
    for( unsigned int allDeviceListIndex : m_launchDevices )
    {
        Device* device = m_context->getDeviceManager()->allDevices()[allDeviceListIndex];
        if( m_deviceState[allDeviceListIndex].isInitialized() )
            continue;

        if( LWDADevice* lwdaDevice = deviceCast<LWDADevice>( device ) )
        {
            m_deviceState[allDeviceListIndex].activate( lwdaDevice, m_context->getMemoryManager(), allDeviceListIndex, m_numVirtualPages );
        }
    }

    m_stagingPages->setActiveDevices( m_launchDevices );
}

bool PagingManager::getMultiThreadedCallbacksEnabled() const
{
    return m_multiThreadedCallbacksEnabled;
}

bool PagingManager::getAsynchronousCopies() const
{
    // The synchronous knob overrides the context attribute, the asynchronous knob overrides everything.
    return ( m_context->getDemandLoadUseAsyncCopies() && !k_demandLoadForceSimplePageRequests.get() )
           || k_demandLoadForceAsynchronous.get();
}

void PagingManager::resetStagingPageAllocator( Context* context )
{
    m_stagingPages = createStagingPageAllocator( m_context->getDeviceManager(), NUM_STAGING_PAGES, PAGE_SIZE_IN_BYTES,
                                                 getAsynchronousCopies() );
}

void PagingManager::createPageRequests()
{
    // The knob overrides the context attribute.
    const bool forceSynchronous = m_context->getDemandLoadForceSynchronous() || k_demandLoadForceSynchronous.get();
    // Forcing multithreaded wins over forcing single threaded, although why would you try to force both?
    const bool multiThreaded = ( m_multiThreadedCallbacksEnabled && !k_demandLoadForceSingleThreaded.get() )
                               || k_demandLoadForceMultiThreaded.get();

    // Need to have the correct staging page allocator strategy for the selected PageRequests strategy.
    resetStagingPageAllocator( m_context );

    // Forcing simple handling takes precedence over everything.
    if( !getAsynchronousCopies() )
    {
        LOG_NORMAL( "PagingManager::createPageRequests: "
                    << ( k_demandLoadForceSimplePageRequests.get() ? "(forced) " : "" ) << "simple "
                    << ( multiThreaded ? ( k_demandLoadForceMultiThreaded.get() ? "(forced) threaded " : "threaded " ) : "" )
                    << "always synchronous\n" );
        m_pageRequests.reset( new PageRequestsSimple( m_context->getThreadPool(), &m_heap,
                                                      m_ilwokeCallbacksPerTileNotPerMipLevel, multiThreaded ) );
    }
    else if( multiThreaded )
    {
        LOG_NORMAL( "PagingManager::createPageRequests: "
                    << ( k_demandLoadForceMultiThreaded.get() ? "(forced) threaded " : "threaded " )
                    << ( forceSynchronous ?
                             ( k_demandLoadForceSynchronous.get() ? "(forced) synchronous\n" : "synchronous\n" ) :
                             "asynchronous\n" ) );
        m_pageRequests.reset( new PageRequestsThreaded( m_context->getDeviceManager(), m_context->getThreadPool(), &m_heap,
                                                        m_ilwokeCallbacksPerTileNotPerMipLevel, forceSynchronous ) );
    }
    else
    {
        LOG_NORMAL( "PagingManager::createPageRequests: "
                    << ( k_demandLoadForceSingleThreaded.get() ? "(forced) serial " : "serial " )
                    << ( forceSynchronous ?
                             ( k_demandLoadForceSynchronous.get() ? "(forced) synchronous\n" : "synchronous\n" ) :
                             "asynchronous\n" ) );
        m_pageRequests.reset( new PageRequestsSerial( m_context->getDeviceManager(), &m_heap,
                                                      m_ilwokeCallbacksPerTileNotPerMipLevel, forceSynchronous ) );
    }
}

void PagingManager::setMultiThreadedCallbacksEnabled( bool enabled )
{
    LOG_NORMAL( "PagingManager::setMultiThreadedCallbacksEnabled: " << ( enabled ? "yes\n" : "no\n" ) );
    if( m_multiThreadedCallbacksEnabled != enabled )
    {
        m_multiThreadedCallbacksEnabled = enabled;
        createPageRequests();
    }
}

void PagingManager::setLwdaSparseTexturesEnabled( bool enabled )
{
    const PagingMode oldMode = getLwrrentPagingMode();

    if( !k_enableHardwareSparseTextures.isSet() )
        m_lwdaSparseTexturesEnabled = enabled;

    const PagingMode newMode = getLwrrentPagingMode();
    if( newMode != oldMode )
    {
        LOG_NORMAL( "PagingManager::setLwdaSparseTexturesEnabled( " << enabled << " ):  pagingMode = " << newMode << '\n' );
        m_context->getUpdateManager()->eventPagingModeDidChange( newMode );
    }
}

inline bool isPowerOfTwo( unsigned int value )
{
    return value > 0 && ( value & ( value - 1 ) ) == 0;
}

unsigned int PagingManager::getTileWidth() const
{
    const unsigned int width = k_demandTextureTileWidth.get();
    RT_ASSERT_MSG( isPowerOfTwo( width ), "Texture tile width must be a power of two." );
    return width;
}

unsigned int PagingManager::getTileHeight() const
{
    const unsigned int height = k_demandTextureTileHeight.get();
    RT_ASSERT_MSG( isPowerOfTwo( height ), "Texture tile height must be a power of two." );
    return height;
}

void PagingManager::forceSynchronousRequestsChanged()
{
    createPageRequests();
}

void PagingManager::useAsynchronousCopiesChanged()
{
    createPageRequests();
}

void PagingManager::stampMemoryBlocksChanged()
{
    const bool stampMemoryBlocks = m_context->getDemandLoadStampMemoryBlocks();
    RequestHandler::setStampMemoryBlocks( stampMemoryBlocks );
    LOG_MEDIUM_VERBOSE( "PagingManager::stampMemoryBlocksChanged: " << boolAsYesNo( stampMemoryBlocks ) << '\n' );
}

std::shared_ptr<size_t> PagingManager::reservePageTableEntries( size_t numPages )
{
    const size_t startPage = m_heap.allocate( numPages );
    return std::make_shared<size_t>( startPage );
}

void PagingManager::releasePageTableEntries( size_t startPage, size_t numPages )
{
    // TODO: remove pages from m_heap and record them for the next call to optixPagingPushMappings
}

static std::ostream& operator<<( std::ostream& stream, MAccess::Kind kind )
{
    switch( kind )
    {
        case MAccess::DEMAND_LOAD:
            return stream << "DEMAND_LOAD";
        case MAccess::LWDA_SPARSE:
            return stream << "LWDA_SPARSE";
        case MAccess::LINEAR:
            return stream << "LINEAR";
        case MAccess::MULTI_PITCHED_LINEAR:
            return stream << "MULTI_PITCHED_LINEAR";
        case MAccess::TEX_OBJECT:
            return stream << "TEX_OBJECT";
        case MAccess::TEX_REFERENCE:
            return stream << "TEX_REFERENCE";
        case MAccess::DEMAND_LOAD_ARRAY:
            return stream << "DEMAND_LOAD_ARRAY";
        case MAccess::DEMAND_LOAD_TILE_ARRAY:
            return stream << "DEMAND_LOAD_TILE_ARRAY";
        case MAccess::DEMAND_TEX_OBJECT:
            return stream << "DEMAND_TEX_OBJECT";
        case MAccess::NONE:
            return stream << "NONE";
        default:
            return stream << static_cast<int>( kind );
    }
}

inline std::string toStringFromPtr( const void* p )
{
    std::ostringstream str;
    str << std::hex << "0x" << reinterpret_cast<unsigned long long>( p );
    return str.str();
}

void PagingManager::bufferMAccessDidChange( Buffer* buffer, const Device* device, const MAccess& oldMA, const MAccess& newMA )
{
    LOG_NORMAL( "PagingManager::bufferMAccessDidChange( " << toStringFromPtr( buffer ) << ", " << device->allDeviceListIndex() << ", "
                                                          << oldMA.getKind() << " -> " << newMA.getKind() << " )\n" );

    if( oldMA.getKind() == MAccess::NONE )
    {
        if( newMA.getKind() == MAccess::DEMAND_LOAD )
            m_heap.associateBuffer( buffer, newMA.getDemandLoad().virtualPageBegin );
        else if( newMA.getKind() == MAccess::LWDA_SPARSE )
        {
            m_heap.associateLwdaSparse( buffer, newMA.getDemandLoad().virtualPageBegin );
        }
    }
    else if( newMA.getKind() == MAccess::NONE )
    {
        // TODO: mark these pages as evictable next time we do optixPagingPushMappings?
        if( oldMA.getKind() == MAccess::DEMAND_LOAD || oldMA.getKind() == MAccess::LWDA_SPARSE )
            m_heap.freeBuffer( buffer, oldMA.getDemandLoad().virtualPageBegin );
    }
}

void PagingManager::eventTextureSamplerMAccessDidChange( const TextureSampler* sampler,
                                                         const Device*         device,
                                                         const MAccess&        oldMA,
                                                         const MAccess&        newMA )
{
    LOG_NORMAL( "PagingManager::eventTextureSamplerMAccessDidChange( "
                << toStringFromPtr( sampler ) << ", " << device->allDeviceListIndex() << ", " << oldMA.getKind()
                << " -> " << newMA.getKind() << " )\n" );

    if( sampler == nullptr || sampler->isInteropTexture() )
        return;
    const Buffer* buffer = sampler->getBuffer();
    if( buffer == nullptr )
        return;
    if( !buffer->isDemandLoad() )
        return;

    switch( oldMA.getKind() )
    {
        case MAccess::NONE:
        case MAccess::TEX_OBJECT:
            // NONE -> DEMAND_TEX_OBJECT: Remember the sampler associated with the mip tail.
            // TEX_OBJECT -> DEMAND_TEX_OBJECT happens when we recognize a small non-mipmapped
            // texture would fit into a LWCA HW sparse texture miptail and we switch from LWCA
            // HW sparse textures to truncated mip pyramid demand loaded texture.
            if( newMA.getKind() == MAccess::DEMAND_TEX_OBJECT )
                m_heap.associateSampler( sampler, newMA.getDemandTexObject().startPage );
            break;

        case MAccess::DEMAND_TEX_OBJECT:
            // DEMAND_TEX_OBJECT -> NONE: Forget about the associated sampler that has the mip tail.
            if( newMA.getKind() == MAccess::NONE )
                m_heap.freeSampler( sampler, oldMA.getDemandTexObject().startPage );
            break;

        default:
            break;
    }
}

inline unsigned int ratioCeiling( unsigned int x, unsigned int y )
{
    return static_cast<unsigned int>( std::ceil( static_cast<float>( x ) / static_cast<float>( y ) ) );
}

inline unsigned int logSqrtRounded( unsigned int x )
{
    return static_cast<unsigned int>( std::roundf( std::log2f( std::sqrt( static_cast<float>( x ) ) ) ) );
}

inline unsigned int ratioLogSqrtRounded( unsigned int x, unsigned int y )
{
    return static_cast<unsigned int>( std::roundf( std::log2f( std::sqrt( static_cast<float>( x ) / static_cast<float>( y ) ) ) ) );
}

inline unsigned int logLwbeRootRounded( unsigned int x )
{
    return static_cast<unsigned int>( std::roundf( std::log2f( std::cbrt( static_cast<float>( x ) ) ) ) );
}

// Get page dimensions for a demand-loaded buffer with the specified dimensionality and element size.
uint3 PagingManager::getPageDimensions( const BufferDimensions& dims ) const
{
    // User format buffers might have an element size of zero when the format is
    // set before the element size is set.
    if( dims.elementSize() == 0 )
        return make_uint3( 0, 0, 0 );

    const unsigned int elementsPerPage = PAGE_SIZE_IN_BYTES / dims.elementSize();
    switch( dims.dimensionality() )
    {
        case 1:
        {
            return make_uint3( elementsPerPage, 1, 1 );
        }
        case 2:
        {
            // Round the width to a power of two.  For element sizes of 1 through 10 this yields:
            // 256x256, 256x128, 128x171, 128x128, 128x103, 128x86, 128x74, 128x64, 64x114,  64x103
            const unsigned int width  = 1U << logSqrtRounded( elementsPerPage );
            const unsigned int height = ratioCeiling( elementsPerPage, width );
            return make_uint3( width, height, 1 );
        }
        case 3:
        {
            // Round width and height to a power of two.  For element sizes of 1 through 8 this yields:
            // 32x64x32, 32x32x32, 32x32x22, 32x32x16, 32x16x26, 16x32x22, 16x32x19, 16x32x16
            const unsigned int width  = 1U << logLwbeRootRounded( elementsPerPage );
            const unsigned int height = 1U << ratioLogSqrtRounded( elementsPerPage, width );
            const unsigned int depth  = ratioCeiling( elementsPerPage, width * height );
            return make_uint3( width, height, depth );
        }
        default:
        {
            RT_ASSERT_FAIL_MSG( "Invalid dimensionality" );
        }
    }
}

size_t PagingManager::computeNumDemandBufferPages( const BufferDimensions& dims ) const
{
    const uint3        pageDims = getPageDimensions( dims );
    const unsigned int tilesX   = ratioCeiling( dims.width(), pageDims.x );
    const unsigned int tilesY   = ratioCeiling( dims.height(), pageDims.y );
    const unsigned int tilesZ   = ratioCeiling( dims.depth(), pageDims.z );
    const size_t       numPages = tilesX * tilesY * tilesZ;

    LOG_MEDIUM_VERBOSE( "PagingManager::computeNumDemandBufferPages( { " << dims.toString() << " } ) = " << numPages << '\n' );

    return numPages;
}

unsigned int PagingManager::getSoftwareMipTailFirstLevel( const BufferDimensions& dims ) const
{
    return TileIndexing::getSoftwareMipTailFirstLevel( dims.width(), dims.height(), getTileWidth(), getTileHeight() );
}

size_t PagingManager::computeSoftwareNumDemandTexturePages( const BufferDimensions& dims ) const
{
    unsigned int mipTailFirstLevel = getSoftwareMipTailFirstLevel( dims );
    return computeNumPages( dims, getTileWidth(), getTileHeight(), mipTailFirstLevel );
}

size_t PagingManager::computeNumPages( const BufferDimensions& dims, unsigned int tileWidth, unsigned int tileHeight, unsigned int mipTailFirstLevel ) const
{
    size_t numPages;

    if( getLwrrentPagingMode() == PagingMode::WHOLE_MIPLEVEL )
        numPages = dims.mipLevelCount();
    else
        numPages = static_cast<size_t>(
            TileIndexing( dims.width(), dims.height(), tileWidth, tileHeight ).callwlateNumPages( mipTailFirstLevel ) );

    LOG_MEDIUM_VERBOSE( "PagingManager::computeNumPages( { " << dims.toString() << " } ) = " << numPages << '\n' );

    return numPages;
}

unsigned int* PagingManager::getUsageBits( const Device* device ) const
{
    return m_isActive ? m_deviceState[device->allDeviceListIndex()].getUsageBits() : nullptr;
}
unsigned int* PagingManager::getResidenceBits( const Device* device ) const
{
    return m_isActive ? m_deviceState[device->allDeviceListIndex()].getResidenceBits() : nullptr;
}
unsigned long long* PagingManager::getPageTable( const Device* device ) const
{
    return m_isActive ? m_deviceState[device->allDeviceListIndex()].getPageTable() : nullptr;
}

unsigned long long* PagingManager::getTileArrays( const Device* device ) const
{
    return m_isActive ? m_deviceState[device->allDeviceListIndex()].getTileArrays() : nullptr;
}

void PagingManager::eventActiveDevicesSupportLwdaSparseTexturesDidChange( bool supported )
{
    PagingMode oldMode = getLwrrentPagingMode();

    m_activeDevicesSupportLwdaSparse = supported;

    PagingMode newMode = getLwrrentPagingMode();
    if( newMode != oldMode )
    {
        LOG_NORMAL( "PagingManager::eventActiveDevicesSupportLwdaSparseTexturesDidChange( "
                    << supported << " ): pagingMode = " << newMode << '\n' );
        m_context->getUpdateManager()->eventPagingModeDidChange( newMode );
    }
}

}  // namespace optix
