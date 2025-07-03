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

#include <Memory/ResourceManager.h>

#include <LWCA/Memory.h>
#include <LWCA/Stream.h>
#include <Context/Context.h>
#include <Device/CPUDevice.h>
#include <Device/LWDADevice.h>
#include <Device/DeviceManager.h>
#include <Memory/Allocator.h>
#include <Memory/BulkMemoryPool.h>
#include <Memory/DemandLoad/PagingService.h>
#include <Memory/MResources.h>
#include <Memory/MemoryManager.h>
#include <Memory/PolicyDetails.h>
#include <corelib/system/LwdaDriver.h>
#include <prodlib/misc/TimeViz.h>

#include <Exceptions/ResourceAlreadyRegistered.h>
#include <Exceptions/ResourceNotRegistered.h>
#include <corelib/math/MathUtil.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/exceptions/MemoryAllocationFailed.h>
#include <prodlib/misc/BufferFormats.h>
#include <prodlib/misc/GLFunctions.h>
#include <prodlib/misc/GLInternalFormats.h>
#include <prodlib/misc/RTFormatUtil.h>
#include <prodlib/system/Knobs.h>
#include <prodlib/system/Logger.h>

#include <algorithm>
#include <cstring>
#include <functional>

using namespace corelib;
using namespace prodlib;

namespace {
// clang-format off
  Knob<size_t>       k_maxTexHeapSize( RT_DSTRING("mem.maxTexHeapSize"), 2147483648u, RT_DSTRING( "Maximum size of texture heap in bytes. 0 = disable texheap."));
  Knob<int>          k_mmll( RT_DSTRING("mem.logLevel"), 30, RT_DSTRING( "Log level used for MemoryManager logging"));
  Knob<bool>         k_clearBuffersOnAlloc( RT_DSTRING("mem.clearBuffersOnAllocation"), false, RT_DSTRING( "Clears a buffer to mem.clearValue during allocation."));
  Knob<int>          k_clearValue( RT_DSTRING("mem.clearValue"), 0, RT_DSTRING( "Value (byte) to clear with in mem.clearBuffersOnAllocation."));
  Knob<size_t>       k_bulkAllocationThresholdRtx( RT_DSTRING( "mem.bulkAllocationThresholdRtx" ), 1024u*1024u, RT_DSTRING( "Use bulk allocator for allocations below this number of bytes in Rtx mode." ) );
  Knob<size_t>       k_bulkAllocationThresholdMegakernel( RT_DSTRING( "mem.bulkAllocationThresholdMegakernel" ), 128u, RT_DSTRING( "Use bulk allocator for allocations below this number of bytes in Megakernel mode." ) );
  Knob<size_t>       k_hostBulkAllocationThreshold( RT_DSTRING( "mem.hostBulkAllocationThreshold" ), 0, RT_DSTRING( "Use bulk allocator for host allocations below this number of bytes (4MB is good)" ) );
// clang-format on
}

// Float4 elements
static const size_t TEXHEAP_ELEMENT_SIZE = 16;

// Align to this boundary (in bytes). Must be a multiple of element size
static const size_t TEXHEAP_ALIGNMENT = 512;

// If texheap allocation fails and is less than N% full then defragment the
// texheap and retry the allocation
static const float TEXHEAP_DEFRAG_THRESHOLD = 0.95f;

// If texheap is less than half full, shrink it.
static const float TEXHEAP_SHRINK_THRESHOLD = 0.5f;

// Bulk allocations alignments (for small and large sizes)
static const size_t BULK_ALLOCATION_SMALL_ALIGNMENT = 64;
static const size_t BULK_ALLOCATION_TRANSITION      = 256;
static const size_t BULK_ALLOCATION_LARGE_ALIGNMENT = 256;

namespace optix {

/****************************************************************
 *
 * Initialization / shutdown
 *
 ****************************************************************/

ResourceManager::ResourceManager( MemoryManager* mm, Context* context )
    : m_context( context )
    , m_memoryManager( mm )
    , m_deviceManager( context->getDeviceManager() )
{
}

ResourceManager::~ResourceManager() = default;

void ResourceManager::initialize()
{
    // Disable the texture heap when using the rtx data model.
    size_t texHeapSizeKnob = k_maxTexHeapSize.get();
    if( m_context->useRtxDataModel() )
        texHeapSizeKnob = 0;

    size_t texHeapSize = idivCeil( texHeapSizeKnob, TEXHEAP_ELEMENT_SIZE );

    m_texHeapAllocator.reset( new Allocator( texHeapSize, TEXHEAP_ALIGNMENT / TEXHEAP_ELEMENT_SIZE ) );
    m_texHeapEnabled = texHeapSizeKnob > 0;

    BufferDimensions texheapDims( RT_FORMAT_FLOAT4, getElementSize( RT_FORMAT_FLOAT4 ), 1, 0, 1, 1 );
    m_texHeapBacking = m_memoryManager->allocateMBuffer( texheapDims, MBufferPolicy::internal_texheapBacking );
    TIMEVIZ_COUNT( "Texheap Bytes", texheapDims.getTotalSizeInBytes() );

    TextureDescriptor texDesc;
    for( RTwrapmode& wrapMode : texDesc.wrapMode )
        wrapMode          = RT_WRAP_REPEAT;
    texDesc.minFilterMode = RT_FILTER_NEAREST;
    texDesc.magFilterMode = RT_FILTER_NEAREST;
    texDesc.mipFilterMode = RT_FILTER_NONE;
    texDesc.maxAnisotropy = 1.f;
    texDesc.indexMode     = RT_TEXTURE_INDEX_ARRAY_INDEX;
    texDesc.readMode      = RT_TEXTURE_READ_ELEMENT_TYPE;
    m_texHeapSampler      = m_memoryManager->attachMTextureSampler( m_texHeapBacking, texDesc );


    // Subscribe to events from the texture sampler
    m_texHeapSampler->addListener( this );
}

void optix::ResourceManager::freeBulkMemoryPools( int allDeviceIndex )
{
    lazyInitializeBulkMemoryPools( allDeviceIndex );  // needed for proper destruction.
    m_bulkMemoryPools_small[allDeviceIndex].clear();
    m_bulkMemoryPools_large[allDeviceIndex].clear();
}

void ResourceManager::shutdown()
{
    m_texHeapSampler.reset();
    m_texHeapBacking.reset();
    m_texHeapAllocator.reset();

    // Free any remaining pools. They must already be empty at this point.
    for( int allDeviceIndex = 0; allDeviceIndex < OPTIX_MAX_DEVICES; ++allDeviceIndex )
    {
        freeBulkMemoryPools( allDeviceIndex );
    }
}

void optix::ResourceManager::removedDevices( DeviceSet& removedDevices )
{
    for( DeviceSet::position deviceIndex : removedDevices )
    {
        freeBulkMemoryPools( deviceIndex );
    }
}

/****************************************************************
 *
 * Release resources (generic version)
 *
 ****************************************************************/

void ResourceManager::releaseResourcesOnDevices( MResources* res, const DeviceSet& onDevices, const PolicyDetails& policy )
{
    // Free individually the per-device type resources.
    for( int deviceIndex : onDevices )
    {
        switch( res->m_resourceKind[deviceIndex] )
        {
            case MResources::HostMalloc:
                releaseHostMalloc( res, deviceIndex );
                break;
            case MResources::LwdaMalloc:
                releaseLwdaMalloc( res, deviceIndex );
                break;
            case MResources::LwdaArray:
                releaseLwdaArray( res, deviceIndex );
                break;
            case MResources::LwdaSparseArray:
                releaseLwdaSparseArray( res, deviceIndex );
                break;
            case MResources::DemandLoadArray:
                releaseDemandLoadArray( res, deviceIndex );
                break;
            case MResources::DemandLoadTileArray:
                releaseDemandLoadTileArray( res, deviceIndex );
                break;
            case MResources::LwdaSparseBacking:
                releaseLwdaSparseBacking( res, deviceIndex );
                break;
            case MResources::TexHeap:               // fall through
            case MResources::ZeroCopy:              // fall through
            case MResources::LwdaMallocP2P:         // fall through
            case MResources::LwdaArrayP2P:          // fall through
            case MResources::LwdaMallocSingleCopy:  // fall through
            case MResources::DemandLoad:            // fall through
            case MResources::None:
                // Do nothing - handled below
                break;
                // Default case intentionally omitted
        }
    }

    // The cases below either have "resource-global" data that needs to be freed,
    // or needs to be handled all devices at once.
    releaseTexHeapOnDevices( res, onDevices );
    releaseZeroCopyOnDevices( res, onDevices );
    releaseLwdaMallocP2POnDevices( res, onDevices );
    releaseLwdaArrayP2POnDevices( res, onDevices );
    releaseLwdaSingleCopyOnDevices( res, onDevices, policy );
    releaseDemandLoadOnDevices( res, onDevices );
}

/****************************************************************
 *
 * Move resources and high-level memory manager functions. Note: Copy
 * is broken out into ResourceCopying.cpp
 *
 ****************************************************************/

void ResourceManager::moveHostResources( MResources* dst, MResources* src )
{
    // This is a special function that Memory Manager uses to move
    // resources from one MResources object to another. It is used to
    // transition data without requiring additional temporaries.
    // Rules:
    // Ownership and all associated member variables transfers from "other" to "this"
    // No other resources should be disrupted

    const unsigned int index = m_deviceManager->cpuDevice()->allDeviceListIndex();
    RT_ASSERT_MSG( dst->m_resourceKind[index] == MResources::None,
                   "Move host resources requires an empty desintation" );
    if( src->m_resourceKind[index] == MResources::HostMalloc )
    {
        //  No need to update MAccess, since the parent buffer did not
        // change.
    }
    else if( src->m_resourceKind[index] == MResources::ZeroCopy )
    {
        RT_ASSERT( dst->m_zeroCopyAllocatedSet.empty() );
        std::swap( dst->m_zeroCopyAllocatedSet, src->m_zeroCopyAllocatedSet );
        RT_ASSERT( dst->m_zeroCopyHostPtr == nullptr );
        std::swap( dst->m_zeroCopyHostPtr, src->m_zeroCopyHostPtr );
        RT_ASSERT( dst->m_zeroCopyRegistrarDevice == nullptr );
        std::swap( dst->m_zeroCopyRegistrarDevice, src->m_zeroCopyRegistrarDevice );
    }
    else
    {
        RT_ASSERT_FAIL_MSG( "Invalid host resources" );
    }
    std::swap( dst->m_resourceKind[index], src->m_resourceKind[index] );
}


void ResourceManager::moveDeviceResources( MResources* dst, MResources* src, LWDADevice* lwdaDevice )
{
    const unsigned int index = lwdaDevice->allDeviceListIndex();
    RT_ASSERT_MSG( dst->m_resourceKind[index] == MResources::None,
                   "Move device resources requires an empty destination" );
    RT_ASSERT_MSG( src->m_resourceKind[index] == MResources::LwdaMalloc,
                   "Move device resources requires an LwdaMalloc source" );

    std::swap( dst->m_resourceKind[index], src->m_resourceKind[index] );
}


void ResourceManager::expandValidSet( MResources* res, DeviceSet& validSet )
{
    // If one device is made valid on zero copy, then other devices are
    // also valid.
    const DeviceSet& zeroCopyDevices = res->m_zeroCopyAllocatedSet;
    if( validSet.overlaps( zeroCopyDevices ) )
        validSet |= zeroCopyDevices;
}


void ResourceManager::updateRegistrations( MResources* res, LWDADevice* newRegistrar )
{
    // When the list of active devices change, it is possible that the primary LWCA device will change too.
    // This can lead to a resource that is registered with a non-active device, which will fail when deregistering.
    // here we migrate the registration to a new registrar.

    // no registrar or has not changed, nothing to do.
    if( !res->m_zeroCopyRegistrarDevice || res->m_zeroCopyRegistrarDevice == newRegistrar )
        return;

    // host pointer is null, nothing to modify
    if( !res->m_zeroCopyHostPtr )
        return;

    llog( k_mmll.get() ) << " - update zero-copy registration from: " << res->m_zeroCopyRegistrarDevice->allDeviceListIndex()
                         << " to: " << newRegistrar->allDeviceListIndex() << '\n';

    // Unregister from the old registrar.
    if( res->m_zeroCopyRegistrarDevice )
    {
        res->m_zeroCopyRegistrarDevice->makeLwrrent();
        lwca::memHostUnregister( res->m_zeroCopyHostPtr );
    }

    BufferDimensions dims   = res->m_buf->getNonZeroDimensions();
    size_t           nbytes = dims.getTotalSizeInBytes();

    // register in the new registrar.
    newRegistrar->makeLwrrent();
    const unsigned int flags = LW_MEMHOSTREGISTER_PORTABLE | LW_MEMHOSTREGISTER_DEVICEMAP;
    lwca::memHostRegister( res->m_zeroCopyHostPtr, nbytes, flags );
    res->m_zeroCopyRegistrarDevice = newRegistrar;
}

static LWarray_format getLwdaArrayFormat( RTformat format )
{
    switch( format )
    {
        case RT_FORMAT_HALF:
        case RT_FORMAT_HALF2:
        case RT_FORMAT_HALF3:
        case RT_FORMAT_HALF4:
            return LW_AD_FORMAT_HALF;

        case RT_FORMAT_FLOAT:
        case RT_FORMAT_FLOAT2:
        case RT_FORMAT_FLOAT3:
        case RT_FORMAT_FLOAT4:
            return LW_AD_FORMAT_FLOAT;

        case RT_FORMAT_BYTE:
        case RT_FORMAT_BYTE2:
        case RT_FORMAT_BYTE3:
        case RT_FORMAT_BYTE4:
            return LW_AD_FORMAT_SIGNED_INT8;

        case RT_FORMAT_UNSIGNED_BYTE:
        case RT_FORMAT_UNSIGNED_BYTE2:
        case RT_FORMAT_UNSIGNED_BYTE3:
        case RT_FORMAT_UNSIGNED_BYTE4:
            return LW_AD_FORMAT_UNSIGNED_INT8;

        case RT_FORMAT_SHORT:
        case RT_FORMAT_SHORT2:
        case RT_FORMAT_SHORT3:
        case RT_FORMAT_SHORT4:
            return LW_AD_FORMAT_SIGNED_INT16;

        case RT_FORMAT_UNSIGNED_SHORT:
        case RT_FORMAT_UNSIGNED_SHORT2:
        case RT_FORMAT_UNSIGNED_SHORT3:
        case RT_FORMAT_UNSIGNED_SHORT4:
            return LW_AD_FORMAT_UNSIGNED_INT16;

        case RT_FORMAT_INT:
        case RT_FORMAT_INT2:
        case RT_FORMAT_INT3:
        case RT_FORMAT_INT4:
            return LW_AD_FORMAT_SIGNED_INT32;

        case RT_FORMAT_UNSIGNED_INT:
        case RT_FORMAT_UNSIGNED_INT2:
        case RT_FORMAT_UNSIGNED_INT3:
        case RT_FORMAT_UNSIGNED_INT4:
        // BC compressed formats require UNSIGNED_INT32 regardless of signedness
        case RT_FORMAT_UNSIGNED_BC1:
        case RT_FORMAT_UNSIGNED_BC2:
        case RT_FORMAT_UNSIGNED_BC3:
        case RT_FORMAT_UNSIGNED_BC4:
        case RT_FORMAT_BC4:
        case RT_FORMAT_UNSIGNED_BC5:
        case RT_FORMAT_BC5:
        case RT_FORMAT_UNSIGNED_BC6H:
        case RT_FORMAT_BC6H:
        case RT_FORMAT_UNSIGNED_BC7:
            return LW_AD_FORMAT_UNSIGNED_INT32;

        case RT_FORMAT_USER:
        case RT_FORMAT_BUFFER_ID:
        case RT_FORMAT_PROGRAM_ID:
        case RT_FORMAT_UNKNOWN:
            throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Unsupported texture format for LWCA array: ", format );

        default:
            throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Unknown buffer format: ", format );
    }
}

LWresourceViewFormat getViewFormat( const RTformat format )
{
    switch( format )
    {
        case RT_FORMAT_FLOAT:
            return LW_RES_VIEW_FORMAT_FLOAT_1X32;
        case RT_FORMAT_FLOAT2:
            return LW_RES_VIEW_FORMAT_FLOAT_2X32;
        case RT_FORMAT_FLOAT4:
            return LW_RES_VIEW_FORMAT_FLOAT_4X32;
        case RT_FORMAT_BYTE:
            return LW_RES_VIEW_FORMAT_SINT_1X8;
        case RT_FORMAT_BYTE2:
            return LW_RES_VIEW_FORMAT_SINT_2X8;
        case RT_FORMAT_BYTE4:
            return LW_RES_VIEW_FORMAT_SINT_4X8;
        case RT_FORMAT_UNSIGNED_BYTE:
            return LW_RES_VIEW_FORMAT_UINT_1X8;
        case RT_FORMAT_UNSIGNED_BYTE2:
            return LW_RES_VIEW_FORMAT_UINT_2X8;
        case RT_FORMAT_UNSIGNED_BYTE4:
            return LW_RES_VIEW_FORMAT_UINT_4X8;
        case RT_FORMAT_SHORT:
            return LW_RES_VIEW_FORMAT_SINT_1X16;
        case RT_FORMAT_SHORT2:
            return LW_RES_VIEW_FORMAT_SINT_2X16;
        case RT_FORMAT_SHORT4:
            return LW_RES_VIEW_FORMAT_SINT_4X16;
        case RT_FORMAT_UNSIGNED_SHORT:
            return LW_RES_VIEW_FORMAT_UINT_1X16;
        case RT_FORMAT_UNSIGNED_SHORT2:
            return LW_RES_VIEW_FORMAT_UINT_2X16;
        case RT_FORMAT_UNSIGNED_SHORT4:
            return LW_RES_VIEW_FORMAT_UINT_4X16;
        case RT_FORMAT_INT:
            return LW_RES_VIEW_FORMAT_SINT_1X32;
        case RT_FORMAT_INT2:
            return LW_RES_VIEW_FORMAT_SINT_2X32;
        case RT_FORMAT_INT4:
            return LW_RES_VIEW_FORMAT_SINT_4X32;
        case RT_FORMAT_UNSIGNED_INT:
            return LW_RES_VIEW_FORMAT_UINT_1X32;
        case RT_FORMAT_UNSIGNED_INT2:
            return LW_RES_VIEW_FORMAT_UINT_2X32;
        case RT_FORMAT_UNSIGNED_INT4:
            return LW_RES_VIEW_FORMAT_UINT_4X32;
        case RT_FORMAT_HALF:
            return LW_RES_VIEW_FORMAT_FLOAT_1X16;
        case RT_FORMAT_HALF2:
            return LW_RES_VIEW_FORMAT_FLOAT_2X16;
        case RT_FORMAT_HALF4:
            return LW_RES_VIEW_FORMAT_FLOAT_4X16;
        case RT_FORMAT_UNSIGNED_BC1:
            return LW_RES_VIEW_FORMAT_UNSIGNED_BC1;
        case RT_FORMAT_UNSIGNED_BC2:
            return LW_RES_VIEW_FORMAT_UNSIGNED_BC2;
        case RT_FORMAT_UNSIGNED_BC3:
            return LW_RES_VIEW_FORMAT_UNSIGNED_BC3;
        case RT_FORMAT_UNSIGNED_BC4:
            return LW_RES_VIEW_FORMAT_UNSIGNED_BC4;
        case RT_FORMAT_BC4:
            return LW_RES_VIEW_FORMAT_SIGNED_BC4;
        case RT_FORMAT_UNSIGNED_BC5:
            return LW_RES_VIEW_FORMAT_UNSIGNED_BC5;
        case RT_FORMAT_BC5:
            return LW_RES_VIEW_FORMAT_SIGNED_BC5;
        case RT_FORMAT_UNSIGNED_BC6H:
            return LW_RES_VIEW_FORMAT_UNSIGNED_BC6H;
        case RT_FORMAT_BC6H:
            return LW_RES_VIEW_FORMAT_SIGNED_BC6H;
        case RT_FORMAT_UNSIGNED_BC7:
            return LW_RES_VIEW_FORMAT_UNSIGNED_BC7;

        case RT_FORMAT_FLOAT3:
        case RT_FORMAT_BYTE3:
        case RT_FORMAT_UNSIGNED_BYTE3:
        case RT_FORMAT_SHORT3:
        case RT_FORMAT_UNSIGNED_SHORT3:
        case RT_FORMAT_INT3:
        case RT_FORMAT_UNSIGNED_INT3:
        case RT_FORMAT_USER:
        case RT_FORMAT_BUFFER_ID:
        case RT_FORMAT_PROGRAM_ID:
        case RT_FORMAT_HALF3:
        case RT_FORMAT_LONG_LONG:
        case RT_FORMAT_LONG_LONG2:
        case RT_FORMAT_LONG_LONG3:
        case RT_FORMAT_LONG_LONG4:
        case RT_FORMAT_UNSIGNED_LONG_LONG:
        case RT_FORMAT_UNSIGNED_LONG_LONG2:
        case RT_FORMAT_UNSIGNED_LONG_LONG3:
        case RT_FORMAT_UNSIGNED_LONG_LONG4:
        case RT_FORMAT_UNKNOWN:
            throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Unsupported buffer format: ", format );
    }

    return LW_RES_VIEW_FORMAT_NONE;
}

static unsigned int getLwdaBufferFlags( bool lwbe, bool layered )
{
    unsigned int lwflags = 0;
    if( lwbe )
        lwflags |= LWDA_ARRAY3D_LWBEMAP;
    if( layered )
        lwflags |= LWDA_ARRAY3D_LAYERED;
    return lwflags;
}

lwca::TexObject ResourceManager::createTexObject( const MResources* res, LWDADevice* lwdaDevice, const TextureDescriptor& texDesc )
{
    // Build the LWCA resource descriptor
    lwdaDevice->makeLwrrent();
    LWDA_RESOURCE_DESC       descRes{};
    LWDA_RESOURCE_VIEW_DESC  descView{};
    LWDA_RESOURCE_VIEW_DESC* pDescView = nullptr;

    const unsigned int             deviceIndex = lwdaDevice->allDeviceListIndex();
    const BufferDimensions&        dims        = res->m_buf->getDimensions();
    const bool                     compressed  = prodlib::isCompressed( dims.format() );
    const MResources::ResourceKind kind        = res->m_resourceKind[deviceIndex];
    const bool isArrayResourceKind             = kind == MResources::LwdaArray || kind == MResources::LwdaArrayP2P
                                     || kind == MResources::DemandLoadArray || kind == MResources::DemandLoadTileArray
                                     || kind == MResources::LwdaSparseArray;

    if( isArrayResourceKind )
    {
        // block compressed textures are only supported for arrays
        if( compressed )
        {
            const bool layered1D      = ( dims.isLayered() && dims.height() == 1 && dims.dimensionality() == 3 );
            pDescView                 = &descView;
            descView.format           = getViewFormat( dims.format() );
            descView.width            = dims.width() * 4;
            descView.height           = ( dims.dimensionality() > 1 && !layered1D ) ? dims.height() * 4 : 0;
            descView.depth            = dims.dimensionality() > 2 ? dims.depth() : 0;
            descView.firstMipmapLevel = 0;
            descView.lastMipmapLevel  = dims.mipLevelCount() - 1;
            descView.firstLayer       = 0;
            descView.lastLayer        = dims.isLayered() ? dims.depth() - 1 : 0;
        }

        descRes.resType                    = LW_RESOURCE_TYPE_MIPMAPPED_ARRAY;
        descRes.res.mipmap.hMipmappedArray = res->m_lwdaArrays[deviceIndex].get();
    }
    else
    {
        RT_ASSERT_MSG( !compressed, "Compressed textures can only be created from arrays." );
        const MAccess& memAccess = res->m_buf->getAccess( static_cast<int>( deviceIndex ) );
        descRes.resType          = LW_RESOURCE_TYPE_LINEAR;

        if( dims.dimensionality() == 1 )
        {
            descRes.res.linear.devPtr = reinterpret_cast<LWdeviceptr>( memAccess.getLinearPtr() );
            RT_ASSERT_MSG( isAligned( descRes.res.linear.devPtr, LW_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT ),
                           "Pointer to linear memory isn't aligned sufficiently" );
            descRes.res.linear.format      = getLwdaArrayFormat( dims.format() );
            descRes.res.linear.numChannels = getNumElements( dims.format() );
            descRes.res.linear.sizeInBytes = dims.getTotalSizeInBytes();
            llog( k_mmll.get() ) << "descRes.res.linear.sizeInBytes = " << descRes.res.linear.sizeInBytes << "\n";
        }
        else if( dims.dimensionality() == 2 )
        {
            RT_ASSERT_MSG( dims.mipLevelCount() == 1, "Linear memory can't have MIP levels" );
            const PitchedLinearAccess& ptAccess = memAccess.getPitchedLinear( 0 );
            descRes.res.pitch2D.devPtr          = reinterpret_cast<LWdeviceptr>( ptAccess.ptr );
            RT_ASSERT_MSG( isAligned( descRes.res.pitch2D.devPtr, LW_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT ),
                           "Pointer to linear memory isn't aligned sufficiently" );
            descRes.res.pitch2D.format       = getLwdaArrayFormat( dims.format() );
            descRes.res.pitch2D.numChannels  = getNumElements( dims.format() );
            descRes.res.pitch2D.width        = dims.width();
            descRes.res.pitch2D.height       = dims.height();
            descRes.res.pitch2D.pitchInBytes = ptAccess.pitch;
            llog( k_mmll.get() ) << "descRes.res.pitch2D.sizeInBytes = " << dims.getTotalSizeInBytes() << "\n";
        }
        else
        {
            RT_ASSERT_FAIL_MSG( "Invalid linear texture dimension" );
        }
    }
    descRes.flags = 0;

    // Build the LWCA texture descriptor
    LWDA_TEXTURE_DESC descTex;
    texDesc.getLwdaTextureDescriptor( &descTex );

    // Return a new tex object that must be destroyed by the caller
    return lwca::TexObject::create( descRes, descTex, pDescView );
}

void ResourceManager::bindTexReference( const MResources* res, LWDADevice* lwdaDevice, const TextureDescriptor& texDesc, lwca::TexRef texRef )
{
    lwdaDevice->makeLwrrent();
    unsigned int deviceIndex = lwdaDevice->allDeviceListIndex();

    // Fill in texref sampling modes
    texDesc.fillLwdaTexRef( texRef );

    const BufferDimensions& dims = res->m_buf->getDimensions();
    RT_ASSERT_MSG( !prodlib::isCompressed( dims.format() ), "Compressed textures cannot be bound." );

    // Attach storage
    if( res->m_resourceKind[deviceIndex] == MResources::LwdaArray )
    {
        texRef.setMipmappedArray( res->m_lwdaArrays[deviceIndex], LW_TRSA_OVERRIDE_FORMAT );
    }
    else
    {
        const MAccess& memAccess = res->m_buf->getAccess( static_cast<int>( deviceIndex ) );
        RT_ASSERT_MSG( dims.mipLevelCount() == 1, "Linear memory can't have MIP levels" );
        texRef.setFormat( getLwdaArrayFormat( dims.format() ), getNumElements( dims.format() ) );
        if( dims.dimensionality() == 1 )
        {
            texRef.setAddress( reinterpret_cast<LWdeviceptr>( memAccess.getLinearPtr() ),
                               res->m_buf->getDimensions().getTotalSizeInBytes() );
        }
        else if( dims.dimensionality() == 2 )
        {
            const PitchedLinearAccess& ptAccess = memAccess.getPitchedLinear( 0 );
            LWDA_ARRAY_DESCRIPTOR      desc;
            desc.Width       = dims.width();
            desc.Height      = dims.height();
            desc.Format      = getLwdaArrayFormat( dims.format() );
            desc.NumChannels = getNumElements( dims.format() );
            texRef.setAddress2D( desc, reinterpret_cast<LWdeviceptr>( ptAccess.ptr ), ptAccess.pitch );
        }
        else
        {
            RT_ASSERT_FAIL_MSG( "Invalid linear texture dimension" );
        }
    }
}


/****************************************************************
 *
 * LWCA arrays
 *
 ****************************************************************/

void ResourceManager::acquireLwdaArrayOnDevices( MResources* res, DeviceSet& onDevices )
{
    const BufferDimensions dims = res->m_buf->getNonZeroDimensions();
    llog( k_mmll.get() ) << " - allocate lwca array: " << dims.toString() << "(" << dims.getTotalSizeInBytes()
                         << " bytes)"
                         << " on devices: " << onDevices.toString() << '\n';

    for( int devIdx : onDevices )
    {
        if( acquireLwdaArray( res, devIdx, false ) )
            onDevices -= DeviceSet( devIdx );
    }
}

lwca::MipmappedArray createSparseLwdaArrayFromDimensions( const BufferDimensions& dims, LWDADevice* device, LWresult* lwres = nullptr )
{
    // Allocate memory
    device->makeLwrrent();
    const LWarray_format format      = getLwdaArrayFormat( dims.format() );
    const unsigned int   numChannels = getNumElements( dims.format() );
    const unsigned int   flags       = LWDA_ARRAY3D_SPARSE;

    const size_t lwdaWidth  = dims.width();
    const size_t lwdaHeight = dims.dimensionality() > 1 ? dims.height() : 0;
    const size_t lwdaDepth  = 0;  // Sparse textures are always 2D, for now

    LWDA_ARRAY3D_DESCRIPTOR ad;
    ad.Width       = lwdaWidth;
    ad.Height      = lwdaHeight;
    ad.Depth       = lwdaDepth;
    ad.Format      = format;
    ad.NumChannels = numChannels;
    ad.Flags       = flags;

    return lwca::MipmappedArray::create( ad, dims.mipLevelCount(), lwres );
}

bool ResourceManager::createSparseLwdaArray( MResources* res, const BufferDimensions& dims, int onDevice )
{
    LWresult             lwres = LWDA_SUCCESS;
    lwca::MipmappedArray array = createSparseLwdaArrayFromDimensions( dims, getLWDADevice( onDevice ), &lwres );
    if( lwres != LWDA_SUCCESS )
        return false;

    // The array is not stored in MAccess, so store it in a private array
    RT_ASSERT( res->m_lwdaArrays[onDevice].get() == nullptr );
    res->m_lwdaArrays[onDevice]      = array;
    res->m_numMipmapLevels[onDevice] = dims.mipLevelCount();

    return true;
}

void ResourceManager::acquireLwdaSparseArrayOnDevices( MResources* res, DeviceSet& onDevices )
{
    const BufferDimensions dims          = res->m_buf->getNonZeroDimensions();
    PagingService*         pagingManager = m_context->getPagingManager();
    llog( k_mmll.get() ) << " - acquire sparse lwca array: " << dims.toString() << "(" << dims.getTotalSizeInBytes() << " bytes)"
                         << " on devices: " << onDevices.toString() << '\n';

    for( int devIdx : onDevices )
    {
        if( createSparseLwdaArray( res, dims, devIdx ) )
        {
            if( !res->m_demandLoadAllocation )
            {
                LWDA_ARRAY_SPARSE_PROPERTIES sparseTexProps = res->m_lwdaArrays[devIdx].getSparseProperties();
                const size_t                 numPages =
                    pagingManager->computeNumPages( dims, sparseTexProps.tileExtent.width,
                                                    sparseTexProps.tileExtent.height, sparseTexProps.miptailFirstLevel );
                res->m_demandLoadAllocation = pagingManager->reservePageTableEntries( numPages );
            }

            MAccess memAccess = MAccess::makeLwdaSparse( *res->m_demandLoadAllocation );

            res->setResource( devIdx, MResources::LwdaSparseArray, memAccess );
            res->m_demandLoadAllocatedSet |= DeviceSet( devIdx );

            onDevices -= DeviceSet( devIdx );
        }
    }
}

void ResourceManager::unbindLwdaSparseArray( MResources* res, int onDevice )
{
    LWDADevice* lwdaDevice = getLWDADevice( onDevice );
    lwdaDevice->makeLwrrent();
    lwca::MipmappedArray sparseArray = res->m_lwdaArrays[onDevice];
    if( sparseArray.isSparse() )
    {
        LWDA_ARRAY_SPARSE_PROPERTIES sparseTexProps = sparseArray.getSparseProperties();
        const int                    deviceOrdinal  = lwdaDevice->lwdaOrdinal();

        // Unbind non-miptail mip levels.
        for( int i = 0; i < sparseTexProps.miptailFirstLevel && i < res->m_numMipmapLevels[onDevice] - 1; ++i )
        {
            LWresult bindResult = LWDA_SUCCESS;
            sparseArray.unmapSparseLevel( i, deviceOrdinal, &bindResult );
            if( bindResult != LWDA_SUCCESS )
            {
                lerr << "Failed to unbind sparse mip level " << i << "\n";
            }
        }

        if( sparseTexProps.miptailFirstLevel < res->m_numMipmapLevels[onDevice] - 1 )
        {
            LWresult bindResult = LWDA_SUCCESS;
            sparseArray.unmapSparseMipTail( deviceOrdinal, &bindResult );
            if( bindResult != LWDA_SUCCESS )
            {
                lerr << "Failed to unbind sparse texture mip tail\n";
            }
        }
    }

    res->m_numMipmapLevels[onDevice] = 0;
}

void ResourceManager::releaseLwdaSparseArray( MResources* res, int onDevice )
{
    unbindLwdaSparseArray( res, onDevice );
    destroyLwdaArray( res, onDevice );

    // Reset the resource kind and access kind
    res->setResource( onDevice, MResources::None, MAccess::makeNone() );
}

bool ResourceManager::createLwdaArray( MResources* res, const BufferDimensions& dims, int onDevice )
{
    // Allocate memory
    getLWDADevice( onDevice )->makeLwrrent();
    const LWarray_format format      = getLwdaArrayFormat( dims.format() );
    const unsigned int   numChannels = getNumElements( dims.format() );
    const unsigned int   flags       = getLwdaBufferFlags( dims.isLwbe(), dims.isLayered() );

    const bool   layered1D  = ( dims.isLayered() && dims.height() == 1 && dims.dimensionality() == 3 );
    const size_t lwdaWidth  = dims.width();
    const size_t lwdaHeight = ( dims.dimensionality() > 1 && !layered1D ) ? dims.height() : 0;
    const size_t lwdaDepth  = dims.dimensionality() > 2 ? dims.depth() : 0;

    LWDA_ARRAY3D_DESCRIPTOR ad;
    ad.Width       = lwdaWidth;
    ad.Height      = lwdaHeight;
    ad.Depth       = lwdaDepth;
    ad.Format      = format;
    ad.NumChannels = numChannels;
    ad.Flags       = flags;

    LWresult             lwres = LWDA_SUCCESS;
    lwca::MipmappedArray array = lwca::MipmappedArray::create( ad, dims.mipLevelCount(), &lwres );
    if( lwres != LWDA_SUCCESS )
        return false;

    // The array is not stored in MAccess, so store it in a private array
    RT_ASSERT( res->m_lwdaArrays[onDevice].get() == nullptr );
    res->m_lwdaArrays[onDevice] = array;

    return true;
}

bool ResourceManager::acquireLwdaArray( MResources* res, int onDevice, bool p2p )
{
    if( !createLwdaArray( res, res->m_buf->getNonZeroDimensions(), onDevice ) )
        return false;

    res->setResource( onDevice, p2p ? MResources::LwdaArrayP2P : MResources::LwdaArray, MAccess::makeNone() );
    return true;
}

void ResourceManager::destroyLwdaArray( MResources* res, int onDevice )
{
    const BufferDimensions dims = res->m_buf->getNonZeroDimensions();
    llog( k_mmll.get() ) << " - destroy lwca array: " << dims.toString() << "(" << dims.getTotalSizeInBytes()
                         << " bytes) on device: " << onDevice << '\n';

    // Free the resource
    getLWDADevice( onDevice )->makeLwrrent();
    lwca::MipmappedArray array = res->m_lwdaArrays[onDevice];
    array.destroy();
    res->m_lwdaArrays[onDevice] = lwca::MipmappedArray();
}

void ResourceManager::releaseLwdaArray( MResources* res, int onDevice )
{
    destroyLwdaArray( res, onDevice );

    // Reset the resource kind and access kind
    res->setResource( onDevice, MResources::None, MAccess::makeNone() );
}


/****************************************************************
 *
 * Lwca array peer-to-peer
 *
 ****************************************************************/

void ResourceManager::acquireLwdaArrayP2POnDevices( MResources* res, DeviceSet& onDevices )
{
    BufferDimensions dims   = res->m_buf->getNonZeroDimensions();
    const size_t     nbytes = dims.getTotalSizeInBytes();
    llog( k_mmll.get() ) << " - allocate p2p lwca array: " << dims.toString() << "(" << nbytes << " bytes)"
                         << " on devices: " << onDevices.toString() << '\n';

    // For each island, allocate on a single device. Note that the allocation
    // must either fully succeed or fully fail on the entire island.
    for( DeviceSet island : m_deviceManager->getLwlinkIslands() )
    {
        if( ( island & onDevices ).empty() )
            continue;

        int deviceContainingAllocIdx = -1;

        // Check if this island already contains an allocation for the given
        // resource. If so, use that instead of allocating a new array.
        for( int devIdx : island )
        {
            if( res->m_resourceKind[devIdx] != MResources::ResourceKind::None )
            {
                deviceContainingAllocIdx = devIdx;

                // Switch the resource kind to reflect the allocation's new use
                // Don't switch the access. We didn't do anything that will
                // affect it, and will be accessing the array in the same
                // manner.
                res->m_resourceKind[devIdx] = MResources::LwdaArrayP2P;

                break;
            }
        }

        // If we didn't find an existing allocation, acquire one
        if( deviceContainingAllocIdx == -1 )
        {
            RT_ASSERT_MSG( ( island & onDevices ) == island,
                           "ArrayP2P allocations must lwrrently happen for entire islands at once" );
            RT_ASSERT_MSG( ( res->m_p2pAllocatedSet & island ) == DeviceSet(), "Double p2p allocation" );

            // Start with a hash to "randomize" the preferred device, then try others
            // in case of failure.
            const int h = std::hash<size_t>{}( res->m_buf->m_serialNumber );

            for( int i = 0; i < (int)island.count(); ++i )
            {
                const int devIdx = island[( i + h ) % static_cast<int>( island.count() )];

                if( acquireLwdaArray( res, devIdx, true ) )
                {
                    deviceContainingAllocIdx = devIdx;
                    break;
                }
            }
        }

        // If we failed to find or acquire an allocation, there's nothing to do.
        if( deviceContainingAllocIdx == -1 )
            continue;

        // Set all devices on the island to use the allocation
        for( int peerIdx : island )
        {
            if( peerIdx == deviceContainingAllocIdx )
                continue;

            res->m_lwdaArrays[peerIdx] = res->m_lwdaArrays[deviceContainingAllocIdx];
            res->setResource( peerIdx, MResources::LwdaArrayP2P, MAccess::makeNone() );
        }

        // Remember which device owns the allocation
        res->m_p2pAllocatedSet |= DeviceSet( deviceContainingAllocIdx );
        onDevices -= island;
    }
}

void ResourceManager::releaseLwdaArrayP2POnDevices( MResources* res, const DeviceSet& onDevices )
{
    // Find the devices with a P2P allocation
    DeviceSet toClear;
    for( int i = 0; i < OPTIX_MAX_DEVICES; ++i )
    {
        if( res->m_resourceKind[i] == MResources::LwdaArrayP2P )
            toClear |= DeviceSet( i );
    }

    toClear &= onDevices;
    if( toClear.empty() )
        return;

    BufferDimensions dims = res->m_buf->getNonZeroDimensions();
    llog( k_mmll.get() ) << " - release p2p lwca array: " << dims.toString() << " on devices: " << toClear.toString() << '\n';

    for( DeviceSet island : m_deviceManager->getLwlinkIslands() )
    {
        if( ( island & toClear ).empty() )
            continue;

        RT_ASSERT_MSG( ( island & toClear ) == island,
                       "ArrayP2P releases must lwrrently happen for entire islands at once" );

        // Find the device that owns the allocation
        DeviceSet alloc = res->m_p2pAllocatedSet & island;
        RT_ASSERT( alloc.count() == 1 );
        const int devIdx = alloc[0];

        // Free the resource
        releaseLwdaArray( res, devIdx );

        // Reset all resources in the island
        for( int peerIdx : island )
            res->setResource( peerIdx, MResources::None, MAccess::makeNone() );

        // Clear owner bit
        res->m_p2pAllocatedSet -= alloc;
    }
}


/****************************************************************
 *
 * LWCA malloc
 *
 ****************************************************************/

void ResourceManager::acquireLwdaMallocOnDevices( MResources* res, DeviceSet& onDevices, const PolicyDetails& policy )
{
    BufferDimensions dims   = res->m_buf->getNonZeroDimensions();
    const size_t     nbytes = dims.getTotalSizeInBytes();
    llog( k_mmll.get() ) << " - allocate lwca malloc: " << dims.toString() << "(" << nbytes << " bytes)"
                         << " on devices: " << onDevices.toString() << '\n';

    for( int devIdx : onDevices )
    {
        if( acquireLwdaMalloc( res, devIdx, policy ) )
            onDevices -= DeviceSet( devIdx );
    }
}

bool ResourceManager::acquireLwdaMalloc( MResources* res, int onDevice, const PolicyDetails& policy )
{
    TIMEVIZ_FUNC;

    getLWDADevice( onDevice )->makeLwrrent();

    // Allocate memory
    BufferDimensions dims   = res->m_buf->getNonZeroDimensions();
    const size_t     nbytes = dims.getTotalSizeInBytes();
    LWdeviceptr      ptr    = 0;

    if( nbytes <= getAllocationThreshold() )
    {
        size_t alignment = BULK_ALLOCATION_SMALL_ALIGNMENT;
        if( nbytes > BULK_ALLOCATION_TRANSITION )
            alignment = BULK_ALLOCATION_LARGE_ALIGNMENT;

        size_t nb = align( nbytes, alignment );
        ptr       = bulkAllocate( onDevice, nb, alignment, false );
        if( !ptr )
            return false;
    }
    else
    {
        LWresult lwres = LWDA_SUCCESS;
        ptr            = lwca::memAlloc( nbytes, &lwres );
        if( lwres != LWDA_SUCCESS )
            return false;
    }

    // Clear if required
    if( policy.clearOnAlloc || k_clearBuffersOnAlloc.get() )
        lwca::memsetD8( ptr, k_clearValue.get(), nbytes );

    // Store the pointer in MAccess
    MAccess memAccess = MAccess::makeLinear( (char*)ptr );
    res->setResource( onDevice, MResources::LwdaMalloc, memAccess );

    return true;
}

void ResourceManager::acquireLwdaMallocExternalOnDevices( MResources* res, DeviceSet& onDevices, char* ptr )
{
    llog( k_mmll.get() ) << " - allocate lwca malloc external on devices: " << onDevices.toString() << '\n';

    MAccess memAccess = MAccess::makeLinear( ptr );

    for( int devIdx : onDevices )
    {
        res->setResource( devIdx, MResources::LwdaMalloc, memAccess );
        res->m_lwdaMallocExternalSet |= DeviceSet( devIdx );
        onDevices -= DeviceSet( devIdx );
    }
}

void ResourceManager::releaseLwdaMalloc( MResources* res, int onDevice )
{
    BufferDimensions dims   = res->m_buf->getNonZeroDimensions();
    const size_t     nbytes = dims.getTotalSizeInBytes();
    llog( k_mmll.get() ) << " - release lwca malloc: " << dims.toString() << "(" << nbytes << " bytes)"
                         << " on device: " << onDevice << '\n';

    // Free the LWCA memory if we own it
    if( !res->m_lwdaMallocExternalSet.overlaps( DeviceSet( onDevice ) ) )
    {
        LWdeviceptr ptr = (LWdeviceptr)res->m_buf->getAccess( onDevice ).getLinearPtr();
        // Small allocations are in memory pools, and larger ones are not
        if( nbytes <= getAllocationThreshold() )
        {
            size_t alignment = BULK_ALLOCATION_SMALL_ALIGNMENT;
            if( nbytes > BULK_ALLOCATION_TRANSITION )
                alignment = BULK_ALLOCATION_LARGE_ALIGNMENT;
            size_t nb     = align( nbytes, alignment );
            bulkFree( onDevice, ptr, nb, alignment, false );
        }
        else
        {
            getLWDADevice( onDevice )->makeLwrrent();
            lwca::memFree( ptr );
        }
    }

    // Reset the resource kind and access kind
    res->setResource( onDevice, MResources::None, MAccess::makeNone() );
    res->m_lwdaMallocExternalSet -= DeviceSet( onDevice );
}


/****************************************************************
 *
 * Lwca malloc peer-to-peer
 *
 ****************************************************************/

void ResourceManager::acquireLwdaMallocP2POnDevices( MResources* res, DeviceSet& onDevices, const PolicyDetails& policy )
{
    BufferDimensions dims   = res->m_buf->getNonZeroDimensions();
    const size_t     nbytes = dims.getTotalSizeInBytes();
    llog( k_mmll.get() ) << " - allocate p2p lwca malloc: " << dims.toString() << "(" << nbytes << " bytes)"
                         << " on devices: " << onDevices.toString() << '\n';

    // For each island, allocate on a single device. Note that the allocation
    // must either fully succeed or fully fail on the entire island.
    for( DeviceSet island : m_deviceManager->getLwlinkIslands() )
    {
        if( ( island & onDevices ).empty() )
            continue;

        int deviceContainingAllocIdx = -1;

        // Check if any of the devices on the island have an allocation. If so, use that.
        for( int devIdx : island )
        {
            if( res->m_resourceKind[devIdx] != MResources::ResourceKind::None )
            {
                deviceContainingAllocIdx = devIdx;

                // Update the device's resource kind to reflect its new usage
                res->m_resourceKind[devIdx] = MResources::ResourceKind::LwdaMallocP2P;
                break;
            }
        }

        // If we didn't find an existing allocation, acquire one.
        if( deviceContainingAllocIdx == -1 )
        {
            RT_ASSERT_MSG( ( island & onDevices ) == island,
                           "P2P allocations must lwrrently happen for entire islands at once" );
            RT_ASSERT_MSG( ( res->m_p2pAllocatedSet & island ) == DeviceSet(), "Double p2p allocation" );

            // Start with a hash to "randomize" the preferred device, then try others
            // in case of failure.
            const int h = std::hash<size_t>{}( res->m_buf->m_serialNumber );

            for( int i = 0; i < (int)island.count(); ++i )
            {
                const int devIdx = island[( i + h ) % static_cast<int>( island.count() )];

                // Allocate
                getLWDADevice( devIdx )->makeLwrrent();
                LWresult    lwres = LWDA_SUCCESS;
                LWdeviceptr ptr   = lwca::memAlloc( nbytes, &lwres );

                // Clear if required
                if( policy.clearOnAlloc || k_clearBuffersOnAlloc.get() )
                    lwca::memsetD8( ptr, k_clearValue.get(), nbytes );

                if( lwres == LWDA_SUCCESS )
                {
                    // Set the same pointer on all devices in the island
                    MAccess memAccess = MAccess::makeLinear( (char*)ptr );
                    res->setResource( devIdx, MResources::LwdaMallocP2P, memAccess );
                }
            }
        }

        // If we failed to locate or acquire an allocation, there's nothing to do.
        if( deviceContainingAllocIdx == -1 )
            continue;

        MAccess memAccess = res->m_buf->getAccess( deviceContainingAllocIdx );

        // Set all devices to use the new allocation.
        for( int peerIdx : island )
        {
            if( peerIdx == deviceContainingAllocIdx )
                continue;
            res->setResource( peerIdx, MResources::LwdaMallocP2P, memAccess );
        }

        // Remember which device owns the allocation
        res->m_p2pAllocatedSet |= DeviceSet( deviceContainingAllocIdx );
        onDevices -= island;
        break;
    }
}

void ResourceManager::releaseLwdaMallocP2POnDevices( MResources* res, const DeviceSet& onDevices )
{
    // Find the devices with a P2P allocation
    DeviceSet toClear;
    for( int i = 0; i < OPTIX_MAX_DEVICES; ++i )
    {
        if( res->m_resourceKind[i] == MResources::LwdaMallocP2P )
            toClear |= DeviceSet( i );
    }

    toClear &= onDevices;
    if( toClear.empty() )
        return;

    BufferDimensions dims = res->m_buf->getNonZeroDimensions();
    llog( k_mmll.get() ) << " - release p2p lwca malloc: " << dims.toString() << " on devices: " << toClear.toString() << '\n';

    for( DeviceSet island : m_deviceManager->getLwlinkIslands() )
    {
        if( ( island & toClear ).empty() )
            continue;

        RT_ASSERT_MSG( ( island & toClear ) == island,
                       "P2P releases must lwrrently happen for entire islands at once" );

        // Find the device that owns the allocation
        DeviceSet alloc = res->m_p2pAllocatedSet & island;
        RT_ASSERT( alloc.count() == 1 );
        const int devIdx = alloc[0];

        // Free the resource
        getLWDADevice( devIdx )->makeLwrrent();
        char* ptr = res->m_buf->getAccess( devIdx ).getLinearPtr();
        lwca::memFree( (LWdeviceptr)ptr );

        // Reset all resources in the island
        for( int peerIdx : island )
            res->setResource( peerIdx, MResources::None, MAccess::makeNone() );

        // Clear owner bit
        res->m_p2pAllocatedSet -= alloc;
    }
}

/****************************************************************
*
* Bulk Allocation
*
****************************************************************/

size_t ResourceManager::getAllocationThreshold()
{
    if( m_context->useRtxDataModel() )
        return k_bulkAllocationThresholdRtx.get();

    return k_bulkAllocationThresholdMegakernel.get();
}

LWdeviceptr ResourceManager::bulkAllocate( int allDeviceListIdx, size_t nbytes, size_t alignment, bool deviceIsHost )
{
    lazyInitializeBulkMemoryPools( allDeviceListIdx );
    switch( alignment )
    {
        case BULK_ALLOCATION_SMALL_ALIGNMENT:
            return m_bulkMemoryPools_small[allDeviceListIdx].allocate( nbytes, deviceIsHost );
        case BULK_ALLOCATION_LARGE_ALIGNMENT:
            return m_bulkMemoryPools_large[allDeviceListIdx].allocate( nbytes, deviceIsHost );
        default:
            RT_ASSERT_FAIL_MSG( "Wrong alignment requested" );
    }
}

void ResourceManager::bulkFree( int allDeviceListIdx, LWdeviceptr ptr, size_t nbytes, size_t alignment, bool deviceIsHost )
{
    switch( alignment )
    {
        case BULK_ALLOCATION_SMALL_ALIGNMENT:
            RT_ASSERT_MSG( m_bulkMemoryPools_small[allDeviceListIdx].isInitialized(),
                           "Trying to free memory from an uninitialized pool. Calling free before allocate?" );
            m_bulkMemoryPools_small[allDeviceListIdx].free( ptr, nbytes, deviceIsHost );
            break;
        case BULK_ALLOCATION_LARGE_ALIGNMENT:
            RT_ASSERT_MSG( m_bulkMemoryPools_large[allDeviceListIdx].isInitialized(),
                           "Trying to free memory from an uninitialized pool. Calling free before allocate?" );
            m_bulkMemoryPools_large[allDeviceListIdx].free( ptr, nbytes, deviceIsHost );
            break;
        default:
            RT_ASSERT_FAIL_MSG( "Wrong alignment requested" );
    }
    return;
}

/****************************************************************
 *
 * Demand load
 *
 ****************************************************************/

void ResourceManager::acquireDemandLoadOnDevices( MResources* res, DeviceSet& onDevices )
{
    const BufferDimensions dims          = res->m_buf->getNonZeroDimensions();
    PagingService*         pagingManager = m_context->getPagingManager();
    const size_t           numPages      = pagingManager->computeNumDemandBufferPages( dims );
    llog( k_mmll.get() ) << " - allocate demandload: " << dims.toString() << "(" << dims.getTotalSizeInBytes()
                         << " bytes, " << numPages << " pages) on devices: " << onDevices.toString() << '\n';

    if( !res->m_demandLoadAllocation )
    {
        res->m_demandLoadAllocation = pagingManager->reservePageTableEntries( numPages );
    }

    for( DeviceSet::const_iterator iter = onDevices.begin(); iter != onDevices.end(); ++iter )
    {
        // Set the resource kind and pointer
        MAccess memAccess = MAccess::makeDemandLoad( *res->m_demandLoadAllocation );
        res->setResource( *iter, MResources::DemandLoad, memAccess );

        // Update allocated sets
        res->m_demandLoadAllocatedSet |= DeviceSet( iter );
        onDevices -= DeviceSet( iter );
    }
}

void ResourceManager::releaseDemandLoadOnDevices( MResources* res, const DeviceSet& onDevices )
{
    DeviceSet toClear = onDevices & res->m_demandLoadAllocatedSet;
    if( toClear.empty() )
        return;

    const BufferDimensions dims = res->m_buf->getNonZeroDimensions();
    llog( k_mmll.get() ) << " - release demandload: " << dims.toString() << "(" << dims.getTotalSizeInBytes()
                         << " bytes) on devices: " << toClear.toString() << '\n';

    res->m_demandLoadAllocatedSet -= toClear;
    if( res->m_demandLoadAllocatedSet.empty() )
    {
        res->m_demandLoadAllocation.reset();
    }
    for( unsigned int onDevice : onDevices )
        res->setResource( onDevice, MResources::None, MAccess::makeNone() );
}

/****************************************************************
 *
 * Demand load array
 *
 ****************************************************************/

BufferDimensions ResourceManager::actualDimsForDemandLoadTexture( const BufferDimensions& nominalDims,
                                                                  unsigned int            minDemandLevel,
                                                                  unsigned int            maxDemandLevel )
{
    BufferDimensions dims = nominalDims;
    dims.setMipLevelCount( maxDemandLevel - minDemandLevel + 1 );
    const size_t newWidth  = std::max( static_cast<size_t>( 1UL ), dims.levelWidth( 0 ) >> minDemandLevel );
    const size_t newHeight = std::max( static_cast<size_t>( 1UL ), dims.levelHeight( 0 ) >> minDemandLevel );
    dims.setSize( newWidth, newHeight );
    return dims;
}

void ResourceManager::acquireDemandLoadArrayOnDevices( MResources* res, DeviceSet& onDevices )
{
    BufferDimensions nominalDims   = res->m_buf->getDimensions();
    PagingService*   pagingManager = m_context->getPagingManager();
    const size_t     numPages      = pagingManager->computeSoftwareNumDemandTexturePages( nominalDims );
    llog( k_mmll.get() ) << " - allocate demandload array: " << nominalDims.toString() << "("
                         << nominalDims.getTotalSizeInBytes() << " bytes, " << numPages
                         << " pages) on devices: " << onDevices.toString() << '\n';

    if( !res->m_demandLoadAllocation )
    {
        res->m_demandLoadAllocation = pagingManager->reservePageTableEntries( numPages );
    }

    // When a demand-loaded texture is first created, we allocate a 1x1 LWCA array to serve as a
    // placeholder resource.  This might not be necessary.
    BufferDimensions minimalDims( nominalDims );
    minimalDims.setSize( 1, 1 );
    minimalDims.setMipLevelCount( 1 );

    for( int allDeviceListIndex : onDevices )
    {
        if( !createLwdaArray( res, minimalDims, allDeviceListIndex ) )
            continue;

        // Set the resource kind and pointer.
        MAccess memAccess = MAccess::makeDemandLoadArray( *res->m_demandLoadAllocation, numPages,
                                                          res->m_demandTextureMinMipLevel[allDeviceListIndex] );
        res->setResource( allDeviceListIndex, MResources::DemandLoadArray, memAccess );

        // Update allocated sets
        res->m_demandLoadAllocatedSet |= DeviceSet( allDeviceListIndex );
        onDevices -= DeviceSet( allDeviceListIndex );
    }
}

void ResourceManager::acquireDemandLoadTileArrayOnDevices( MResources* res, DeviceSet& onDevices )
{
    // TODO: This scenario might never occur if the MTextureSampler/MBuffer that represents a TileArray is manually synched...
    // there should be no deferred allocations to satisfy.
    BufferDimensions bufferDims = res->m_buf->getDimensions();

    for( int allDeviceListIndex : onDevices )
    {
        if( !createLwdaArray( res, bufferDims, allDeviceListIndex ) )
            continue;

        // Set the resource kind and pointer.
        MAccess memAccess = MAccess::makeDemandLoadTileArray();
        res->setResource( allDeviceListIndex, MResources::DemandLoadTileArray, memAccess );

        // Update allocated sets
        res->m_demandLoadAllocatedSet |= DeviceSet( allDeviceListIndex );
        onDevices -= DeviceSet( allDeviceListIndex );
    }
}

bool ResourceManager::createLwdaSparseBacking( MResources* res, const BufferDimensions& dims, int onDevice )
{

    getLWDADevice( onDevice )->makeLwrrent();

    const size_t allocationSize = dims.getTotalSizeInBytes();

    // Create memory allocation handle.
    LWmemAllocationProp allocationProps{};
    allocationProps.type             = LW_MEM_ALLOCATION_TYPE_PINNED;
    allocationProps.location         = {LW_MEM_LOCATION_TYPE_DEVICE, onDevice};
    allocationProps.allocFlags.usage = LW_MEM_CREATE_USAGE_TILE_POOL;

    LWresult                     result = LWDA_SUCCESS;
    LWmemGenericAllocationHandle memHandle = lwca::memCreate( allocationSize, (LWmemAllocationProp*)&allocationProps, 0, &result );
    if( result != LWDA_SUCCESS )
        return false;

    MAccess texAccess = MAccess::makeLwdaSparseBacking( memHandle );
    res->setResource( onDevice, MResources::LwdaSparseBacking, texAccess );

    return true;
}

void ResourceManager::acquireDemandLoadTileArraySparseOnDevices( MResources* res, DeviceSet& onDevices )
{
    // TODO: This scenario might never occur if the MTextureSampler/MBuffer that represents a TileArray is manually synched...
    // there should be no deferred allocations to satisfy.
    BufferDimensions bufferDims = res->m_buf->getDimensions();

    for( int allDeviceListIndex : onDevices )
    {
        if( !createLwdaSparseBacking( res, bufferDims, allDeviceListIndex ) )
            continue;

        // Update allocated sets
        res->m_demandLoadAllocatedSet |= DeviceSet( allDeviceListIndex );
        onDevices -= DeviceSet( allDeviceListIndex );
    }
}

void ResourceManager::switchLwdaSparseArrayToLwdaArray( MResources* resources, DeviceSet devices )
{
    MBuffer* buffer = resources->m_buf;
    for( int allDeviceListIndex : devices )
    {
        const LwdaSparseAccess demandLoadAccess = buffer->getAccess( allDeviceListIndex ).getLwdaSparse();
        unbindLwdaSparseArray( resources, allDeviceListIndex );
        destroyLwdaArray( resources, allDeviceListIndex );
        createLwdaArray( resources, resources->m_buf->getNonZeroDimensions(), allDeviceListIndex );  // TODO: return value
        resources->m_demandTextureMinMipLevel[allDeviceListIndex] = 0;
        resources->m_demandTextureMaxMipLevel[allDeviceListIndex] = 0;
        buffer->setAccess( allDeviceListIndex, MAccess::makeDemandLoadArray( demandLoadAccess.virtualPageBegin, 1, 0 ) );
    }
}

LWDA_ARRAY_SPARSE_PROPERTIES ResourceManager::getSparseTexturePropertiesFromMBufferProperties( const MBuffer* buffer )
{
    LWDADevice* onDevice = nullptr;
    for( Device* device : m_deviceManager->activeDevices() )
    {
        if( LWDADevice* lwdaDevice = deviceCast<LWDADevice>( device ) )
        {
            onDevice = lwdaDevice;
            break;
        }
    }
    if( onDevice == nullptr )
        return LWDA_ARRAY_SPARSE_PROPERTIES{};

    lwca::MipmappedArray               tmp = createSparseLwdaArrayFromDimensions( buffer->getDimensions(), onDevice );
    const LWDA_ARRAY_SPARSE_PROPERTIES properties = tmp.getSparseProperties();
    tmp.destroy();
    return properties;
}

void ResourceManager::reallocDemandLoadLwdaArray( MResources* resources, unsigned int allDeviceListIndex, unsigned int minLevel, unsigned int maxLevel )
{
    // When no miplevels have been loaded, the min miplevel is initially UINT_MAX, and the max miplevel is zero.
    const unsigned int oldMinMipLevel = resources->m_demandTextureMinMipLevel[allDeviceListIndex];
    const unsigned int oldMaxMipLevel = resources->m_demandTextureMaxMipLevel[allDeviceListIndex];
    if( minLevel < oldMinMipLevel || maxLevel > oldMaxMipLevel )
    {
        // acquire new lwca array
        const unsigned int      newMinMipLevel = std::min( oldMinMipLevel, minLevel );
        const unsigned int      newMaxMipLevel = std::max( oldMaxMipLevel, maxLevel );
        const BufferDimensions& nominalDims    = resources->m_buf->getDimensions();
        const BufferDimensions actualDims = actualDimsForDemandLoadTexture( nominalDims, newMinMipLevel, newMaxMipLevel );

        lwca::MipmappedArray oldArray               = resources->m_lwdaArrays[allDeviceListIndex];
        resources->m_lwdaArrays[allDeviceListIndex] = lwca::MipmappedArray();

        if( !createLwdaArray( resources, actualDims, static_cast<int>( allDeviceListIndex ) ) )
            throw MemoryAllocationFailed( RT_EXCEPTION_INFO, "Demand load allocation failed for device " + std::to_string( allDeviceListIndex )
                                                                 + ' ' + actualDims.toString() );
        lwca::MipmappedArray newArray = resources->m_lwdaArrays[allDeviceListIndex];

        // copy from old lwca array to new lwca array
        LWDADevice* device = getLWDADevice( allDeviceListIndex );
        device->makeLwrrent();
        for( unsigned int nominalLevel = oldMinMipLevel; nominalLevel <= oldMaxMipLevel; ++nominalLevel )
        {
            unsigned int sourceLevel = nominalLevel - oldMinMipLevel;
            unsigned int destLevel   = nominalLevel - newMinMipLevel;

            // Get the source and destination arrays for this miplevel.
            lwca::Array source = oldArray.getLevel( sourceLevel );
            lwca::Array dest   = newArray.getLevel( destLevel );

            // Get the miplevel dimensions.
            size_t width       = nominalDims.levelWidth( nominalLevel );
            size_t height      = nominalDims.levelHeight( nominalLevel );
            size_t elementSize = nominalDims.elementSize();

#if defined( DEBUG ) || defined( DEVELOP )
            // Sanity check: lwca arrays should have matching dimensions.
            LWDA_ARRAY_DESCRIPTOR sourceDesc = source.getDescriptor();
            LWDA_ARRAY_DESCRIPTOR destDesc   = dest.getDescriptor();
            RT_ASSERT( width == sourceDesc.Width && height == sourceDesc.Height );
            RT_ASSERT( width == destDesc.Width && height == destDesc.Height );
#endif

            // Copy the miplevel.
            LWDA_MEMCPY2D copy{};
            copy.Height        = height;
            copy.WidthInBytes  = width * elementSize;
            copy.dstArray      = dest.get();
            copy.dstDevice     = device->lwdaDevice().get();
            copy.dstHost       = nullptr;
            copy.dstMemoryType = LW_MEMORYTYPE_ARRAY;
            copy.dstPitch      = 0;  // ignored for array
            copy.dstXInBytes   = 0;
            copy.dstY          = 0;
            copy.srcArray      = source.get();
            copy.srcDevice     = device->lwdaDevice().get();
            copy.srcHost       = nullptr;
            copy.srcMemoryType = LW_MEMORYTYPE_ARRAY;
            copy.srcPitch      = 0;  // ignored for array
            copy.srcXInBytes   = 0;
            copy.srcY          = 0;
            lwca::memcpy2D( &copy );
        }

        // release old lwca array
        oldArray.destroy();

        resources->m_demandTextureMinMipLevel[allDeviceListIndex] = newMinMipLevel;
        resources->m_demandTextureMaxMipLevel[allDeviceListIndex] = newMaxMipLevel;
    }
}

LWDA_MEMCPY2D ResourceManager::getSyncDemandLoadMipLevelCopyArgs( MResources*  res,
                                                                  void*        baseAddress,
                                                                  size_t       byteCount,
                                                                  unsigned int allDeviceListIndex,
                                                                  int          nominalMipLevel ) const
{
    const BufferDimensions dimensions = res->m_buf->getDimensions();

    RT_ASSERT( static_cast<unsigned int>( nominalMipLevel ) >= res->m_demandTextureMinMipLevel[allDeviceListIndex] );
    const int actualMipLevel = nominalMipLevel - static_cast<int>( res->m_demandTextureMinMipLevel[allDeviceListIndex] );

    LWDA_MEMCPY2D args{};
    args.srcMemoryType = LW_MEMORYTYPE_HOST;
    args.srcHost       = baseAddress;
    args.srcPitch      = dimensions.getLevelNaturalPitchInBytes( nominalMipLevel );
    args.dstMemoryType = LW_MEMORYTYPE_ARRAY;
    args.dstArray      = res->m_lwdaArrays[allDeviceListIndex].getLevel( actualMipLevel ).get();
    args.WidthInBytes  = dimensions.levelWidth( nominalMipLevel ) * dimensions.elementSize();
    args.Height        = dimensions.levelHeight( nominalMipLevel );
    RT_ASSERT( byteCount == args.WidthInBytes * args.Height );
    return args;
}

void ResourceManager::syncDemandLoadMipLevel( MResources* res, void* baseAddress, size_t byteCount, unsigned int allDeviceListIndex, int nominalMipLevel )
{
    getLWDADevice( allDeviceListIndex )->makeLwrrent();
    const LWDA_MEMCPY2D args = getSyncDemandLoadMipLevelCopyArgs( res, baseAddress, byteCount, allDeviceListIndex, nominalMipLevel );
    lwca::memcpy2D( &args );
}

void ResourceManager::syncDemandLoadMipLevelAsync( lwca::Stream& stream,
                                                   MResources*   res,
                                                   void*         baseAddress,
                                                   size_t        byteCount,
                                                   unsigned int  allDeviceListIndex,
                                                   int           nominalMipLevel )
{
    getLWDADevice( allDeviceListIndex )->makeLwrrent();
    const LWDA_MEMCPY2D args = getSyncDemandLoadMipLevelCopyArgs( res, baseAddress, byteCount, allDeviceListIndex, nominalMipLevel );
    lwca::memcpy2DAsync( &args, stream );
}

LWDA_MEMCPY3D ResourceManager::getFillTileCopyArgs( MResources*  resources,
                                                    unsigned int allDeviceListIndex,
                                                    unsigned int layer,
                                                    const void*  data ) const
{
    LWDA_MEMCPY3D args{};
    args.srcMemoryType = LW_MEMORYTYPE_HOST;
    args.srcHost       = data;
    args.srcPitch      = resources->m_buf->getDimensions().getNaturalPitchInBytes();
    args.srcHeight     = resources->m_buf->getDimensions().height();
    args.dstMemoryType = LW_MEMORYTYPE_ARRAY;
    args.dstArray      = resources->m_lwdaArrays[allDeviceListIndex].getLevel( 0 ).get();
    args.dstXInBytes   = 0;
    args.dstY          = 0;
    args.dstZ          = layer;
    args.WidthInBytes  = args.srcPitch;
    args.Height        = resources->m_buf->getDimensions().height();
    args.Depth         = 1;
    return args;
}

void ResourceManager::fillTile( MResources* resources, unsigned int allDeviceListIndex, unsigned int layer, const void* data )
{
    getLWDADevice( allDeviceListIndex )->makeLwrrent();
    const LWDA_MEMCPY3D args = getFillTileCopyArgs( resources, allDeviceListIndex, layer, data );
    lwca::memcpy3D( &args );
}

void ResourceManager::fillTileAsync( lwca::Stream& stream, MResources* resources, unsigned int allDeviceListIndex, unsigned int layer, const void* data )
{
    getLWDADevice( allDeviceListIndex )->makeLwrrent();
    const LWDA_MEMCPY3D args = getFillTileCopyArgs( resources, allDeviceListIndex, layer, data );
    memcpy3DAsync( &args, stream );
}

LWarrayMapInfo ResourceManager::getFillHardwareTileBindInfo( MResources*          arrayResources,
                                                             MResources*          backingStoreResources,
                                                             unsigned int         allDeviceListIndex,
                                                             const RTmemoryblock& memBlock,
                                                             int                  offset ) const
{
    // Map backing memory to array
    LWmipmappedArray sparseArray = arrayResources->m_lwdaArrays[allDeviceListIndex].get();
    RT_ASSERT( arrayResources->m_resourceKind[allDeviceListIndex] == MResources::LwdaSparseArray );

    LWarrayMapInfo mapInfo{};
    mapInfo.resourceType    = LW_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    mapInfo.resource.mipmap = sparseArray;

    mapInfo.subresourceType               = LW_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL;
    mapInfo.subresource.sparseLevel.level = memBlock.mipLevel;

    mapInfo.subresource.sparseLevel.offsetX = memBlock.x;
    mapInfo.subresource.sparseLevel.offsetY = memBlock.y;

    mapInfo.subresource.sparseLevel.extentWidth  = memBlock.width;
    mapInfo.subresource.sparseLevel.extentHeight = memBlock.height;
    mapInfo.subresource.sparseLevel.extentDepth  = 1;

    MAccess                      memAccess   = backingStoreResources->m_buf->getAccess( allDeviceListIndex );
    LWmemGenericAllocationHandle allocHandle = memAccess.getLwdaSparseBacking().handle;

    mapInfo.memOperationType    = LW_MEM_OPERATION_TYPE_MAP;
    mapInfo.memHandleType       = LW_MEM_HANDLE_TYPE_GENERIC;
    mapInfo.memHandle.memHandle = allocHandle;
    mapInfo.offset              = offset;
    mapInfo.deviceBitMask       = 1U << getLWDADevice( allDeviceListIndex )->lwdaOrdinal();
    return mapInfo;
}

LWDA_MEMCPY2D ResourceManager::getFillHardwareTileCopyArgs( MResources*          arrayResources,
                                                            unsigned int         allDeviceListIndex,
                                                            const RTmemoryblock& memBlock ) const
{
    // Copy to array
    LWarray mipLevelArray = arrayResources->m_lwdaArrays[allDeviceListIndex].getLevel( memBlock.mipLevel ).get();

    LWDA_MEMCPY2D copyArgs{};
    copyArgs.srcMemoryType = LW_MEMORYTYPE_HOST;
    copyArgs.srcHost       = memBlock.baseAddress;
    copyArgs.srcPitch      = memBlock.rowPitch;

    copyArgs.dstMemoryType = LW_MEMORYTYPE_ARRAY;
    copyArgs.dstArray      = mipLevelArray;

    copyArgs.WidthInBytes = memBlock.rowPitch;
    copyArgs.Height       = memBlock.height;

    copyArgs.dstXInBytes = memBlock.x * prodlib::getElementSize( memBlock.format );
    copyArgs.dstY        = memBlock.y;
    return copyArgs;
}

void ResourceManager::fillHardwareTileAsync( lwca::Stream&        stream,
                                             MResources*          arrayResources,
                                             MResources*          backingStoreResources,
                                             unsigned int         allDeviceListIndex,
                                             const RTmemoryblock& memBlock,
                                             int                  offset )
{
    getLWDADevice( allDeviceListIndex )->makeLwrrent();
    // Map backing memory to array
    LWarrayMapInfo mapInfo =
        getFillHardwareTileBindInfo( arrayResources, backingStoreResources, allDeviceListIndex, memBlock, offset );
    lwca::memMapArrayAsync( &mapInfo, 1, stream );

    const LWDA_MEMCPY2D copyArgs = getFillHardwareTileCopyArgs( arrayResources, allDeviceListIndex, memBlock );
    lwca::memcpy2DAsync( &copyArgs, stream );
}

LWarrayMapInfo ResourceManager::getBindHardwareMipTailBindInfo( MResources*  arrayResources,
                                                                MResources*  backingStoreResources,
                                                                unsigned int allDeviceListIndex,
                                                                int          mipTailSizeInBytes,
                                                                int          offset ) const
{
    LWmipmappedArray sparseArray = arrayResources->m_lwdaArrays[allDeviceListIndex].get();
    RT_ASSERT( arrayResources->m_resourceKind[allDeviceListIndex] == MResources::LwdaSparseArray );

    LWarrayMapInfo mapInfo{};
    mapInfo.resourceType    = LW_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    mapInfo.resource.mipmap = sparseArray;

    mapInfo.subresourceType            = LW_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL;
    mapInfo.subresource.miptail.offset = 0;
    mapInfo.subresource.miptail.size   = mipTailSizeInBytes;

    MAccess                      memAccess   = backingStoreResources->m_buf->getAccess( allDeviceListIndex );
    LWmemGenericAllocationHandle allocHandle = memAccess.getLwdaSparseBacking().handle;

    mapInfo.memOperationType    = LW_MEM_OPERATION_TYPE_MAP;
    mapInfo.memHandleType       = LW_MEM_HANDLE_TYPE_GENERIC;
    mapInfo.memHandle.memHandle = allocHandle;
    mapInfo.offset              = offset;
    mapInfo.deviceBitMask       = 1U << getLWDADevice( allDeviceListIndex )->lwdaOrdinal();
    return mapInfo;
}

void ResourceManager::bindHardwareMipTailAsync( lwca::Stream& stream,
                                                MResources*   arrayResources,
                                                MResources*   backingStoreResources,
                                                unsigned int  allDeviceListIndex,
                                                int           mipTailSizeInBytes,
                                                int           offset )
{
    getLWDADevice( allDeviceListIndex )->makeLwrrent();
    LWarrayMapInfo mapInfo = getBindHardwareMipTailBindInfo( arrayResources, backingStoreResources, allDeviceListIndex,
                                                             mipTailSizeInBytes, offset );
    lwca::memMapArrayAsync( &mapInfo, 1, stream );
}

LWDA_MEMCPY2D ResourceManager::getFillHardwareMipTailCopyArgs( MResources*          arrayResources,
                                                               unsigned int         allDeviceListIndex,
                                                               const RTmemoryblock& memBlock ) const
{
    LWarray mipLevelArray = arrayResources->m_lwdaArrays[allDeviceListIndex].getLevel( memBlock.mipLevel ).get();

    LWDA_MEMCPY2D copyArgs{};
    copyArgs.srcMemoryType = LW_MEMORYTYPE_HOST;
    copyArgs.srcHost       = memBlock.baseAddress;
    copyArgs.srcPitch      = memBlock.rowPitch;

    copyArgs.dstMemoryType = LW_MEMORYTYPE_ARRAY;
    copyArgs.dstArray      = mipLevelArray;

    copyArgs.WidthInBytes = memBlock.width * prodlib::getElementSize( memBlock.format );
    copyArgs.Height       = memBlock.height;
    return copyArgs;
}

void ResourceManager::fillHardwareMipTail( MResources* arrayResources, unsigned int allDeviceListIndex, const RTmemoryblock& memBlock )
{
    const LWDA_MEMCPY2D copyArgs = getFillHardwareMipTailCopyArgs( arrayResources, allDeviceListIndex, memBlock );
    lwca::memcpy2D( &copyArgs );
}

void ResourceManager::fillHardwareMipTailAsync( lwca::Stream&        stream,
                                                MResources*          arrayResources,
                                                unsigned int         allDeviceListIndex,
                                                const RTmemoryblock& memBlock )
{
    const LWDA_MEMCPY2D copyArgs = getFillHardwareMipTailCopyArgs( arrayResources, allDeviceListIndex, memBlock );
    memcpy2DAsync( &copyArgs, stream );
}

void ResourceManager::releaseDemandLoadArray( MResources* res, int onDevice )
{
    if( res->m_lwdaArrays[onDevice].get() != nullptr )
        destroyLwdaArray( res, onDevice );
}

void ResourceManager::releaseDemandLoadTileArray( MResources* res, int onDevice )
{
    if( res->m_lwdaArrays[onDevice].get() != nullptr )
        destroyLwdaArray( res, onDevice );
}

void ResourceManager::releaseLwdaSparseBacking( MResources* res, int onDevice )
{
    MAccess                      memAccess   = res->m_buf->getAccess( onDevice );
    LWmemGenericAllocationHandle allocHandle = memAccess.getLwdaSparseBacking().handle;

    // We need to ensure all MemMapArrayAsync calls are complete before releasing
    // backing storage they might be operating on.
    lwca::Context::synchronize();

    lwca::memRelease( allocHandle );

    res->setResource( onDevice, MResources::None, MAccess::makeNone() );
}

/****************************************************************
 *
 * Graphics interop
 *
 ****************************************************************/

void ResourceManager::setupGfxInteropResource( MResources* res, const GfxInteropResource& resource, Device* device )
{
    RT_ASSERT_MSG( device != nullptr, "Illegal interop device" );
    res->m_gfxInteropResource = resource;
    res->m_gfxInteropDevice   = device;
}

Device* ResourceManager::getGfxInteropDevice( MResources* res )
{
    return res->m_gfxInteropDevice;
}

void ResourceManager::registerGfxInteropResource( MResources* res )
{
    if( res->m_gfxInteropRegistered )
        throw ResourceAlreadyRegistered( RT_EXCEPTION_INFO );
    res->m_gfxInteropRegistered = true;

    // Postpone actual resource registration till LWCA resource mapping
    // in order to support 0-sized gfx interop buffers when they are unmapped to LWCA
    // for backward compatibility with 3.x
}

void ResourceManager::unregisterGfxInteropResource( MResources* res )
{
    if( !res->m_gfxInteropRegistered )
        throw ResourceNotRegistered( RT_EXCEPTION_INFO );
    res->m_gfxInteropRegistered = false;

    freeGfxInteropResource( res );
}

void ResourceManager::freeGfxInteropResource( MResources* res )
{
    // do nothing if the resource hasn't been actually registered
    if( !res->m_gfxInteropLWDARegisteredResource.get() )
        return;

    // If this is real interop with a LWCA device, unregister with LWCA.
    if( LWDADevice* lwdaDevice = deviceCast<LWDADevice>( res->m_gfxInteropDevice ) )
    {
        // The interop device must be set as current
        lwdaDevice->makeLwrrent();

        const GfxInteropResource& resource = res->m_gfxInteropResource;
        if( resource.isOGL() )
        {
            res->m_gfxInteropLWDARegisteredResource.unregister();
            res->m_gfxInteropLWDARegisteredResource = lwca::GraphicsResource();
        }
        else
        {
            RT_ASSERT_FAIL_MSG( "DX interop" );
        }
    }
}

lwca::GraphicsResource ResourceManager::registerGLResource( const GfxInteropResource& resource, const PolicyDetails& policy )
{
    unsigned int flags = LW_GRAPHICS_MAP_RESOURCE_FLAGS_NONE;
    if( policy.activeDeviceAccess == PolicyDetails::R )
        flags = LW_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY;
    else if( policy.activeDeviceAccess == PolicyDetails::W )
    {
// This ifndef is a temporary workaround for http://lwbugs/2063599
// which affects Optix in subtle ways. Tracked in JIRA OP-2243
// We can remove it when the driver bug gets fixed.
#ifndef __APPLE__
        flags = LW_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD;
#endif
    }
    if( resource.kind == GfxInteropResource::OGL_BUFFER_OBJECT )
        return lwca::GraphicsResource::GLRegisterBuffer( resource.gl.glId, flags );

    return lwca::GraphicsResource::GLRegisterImage( resource.gl.glId, getGLTextureTarget( resource.gl.target ), flags );
}

void ResourceManager::mapGfxInteropResourceBatch( GfxInteropResourceBatch& gfxResourceBatch )
{
    for( unsigned int deviceIndex = 0; deviceIndex < gfxResourceBatch.resourceData.size(); ++deviceIndex )
    {
        std::vector<void*>& rawGfxResources = gfxResourceBatch.resourceData[deviceIndex];
        if( rawGfxResources.empty() )
            continue;

        LWDADevice* lwdeDevice = getLWDADevice( deviceIndex );
        lwdeDevice->makeLwrrent();

        lwca::GraphicsResource::map( rawGfxResources.size(), (LWgraphicsResource*)rawGfxResources.data(),
                                     lwdeDevice->primaryStream() );
    }

    for( MBuffer* buf : gfxResourceBatch.bufferData )
    {
        MResources*        res         = buf->m_resources.get();
        const unsigned int deviceIndex = res->m_gfxInteropDevice->allDeviceListIndex();
        if( res->m_gfxInteropResource.isArray() )
        {
            lwca::MipmappedArray array = res->m_gfxInteropLWDARegisteredResource.getMappedMipmappedArray();
            RT_ASSERT( res->m_lwdaArrays[deviceIndex].get() == nullptr );
            res->m_lwdaArrays[deviceIndex] = array;

            // Set the resource kind and access kind
            res->setResource( deviceIndex, MResources::LwdaArray, MAccess::makeNone() );
        }
        else
        {
            // Get the LWCA pointer
            size_t      sizeAccordingToLwda;
            LWdeviceptr interopPtr = res->m_gfxInteropLWDARegisteredResource.getMappedPointer( &sizeAccordingToLwda );

            // Store directly in MAccess for the interop device
            MAccess memAccess = MAccess::makeLinear( (char*)interopPtr );
            res->setResource( deviceIndex, MResources::LwdaMalloc, memAccess );
        }
    }
}

void ResourceManager::unmapGfxInteropResourceBatch( GfxInteropResourceBatch& gfxResourceBatch )
{
    for( unsigned int deviceIndex = 0; deviceIndex < gfxResourceBatch.resourceData.size(); ++deviceIndex )
    {
        std::vector<void*>& rawGfxResources = gfxResourceBatch.resourceData[deviceIndex];
        if( rawGfxResources.empty() )
            continue;

        LWDADevice* lwdeDevice = getLWDADevice( deviceIndex );
        lwdeDevice->makeLwrrent();

        lwca::GraphicsResource::unmap( rawGfxResources.size(), (LWgraphicsResource*)rawGfxResources.data(),
                                       lwdeDevice->primaryStream() );
    }
}

void ResourceManager::mapGfxInteropResource( MResources* res, const PolicyDetails& policy, GfxInteropResourceBatch* gfxResourceBatch )
{
    if( !res->m_gfxInteropRegistered )
        throw ResourceNotRegistered( RT_EXCEPTION_INFO );

    BufferDimensions dims        = res->m_buf->getDimensions();
    size_t           sizeInBytes = dims.getTotalSizeInBytes();
    // Sanity check on size
    if( sizeInBytes == 0 )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Graphics resource with a size of 0 cannot be mapped" );

    if( LWDADevice* lwdaDevice = deviceCast<LWDADevice>( res->m_gfxInteropDevice ) )
    {
        // Native interop: map the resource
        lwdaDevice->makeLwrrent();

        // do a postponed resource registration
        if( !res->m_gfxInteropLWDARegisteredResource.get() )
        {
            const GfxInteropResource& resource = res->m_gfxInteropResource;
            if( resource.isOGL() )
            {
                res->m_gfxInteropLWDARegisteredResource = registerGLResource( resource, policy );
            }
            else
            {
                RT_ASSERT_FAIL_MSG( "DX interop" );
            }
        }

        // Batch immediate HW interop map calls.
        if( gfxResourceBatch && isGfxInteropResourceImmediate( res, policy ) )
        {
            const unsigned int deviceIndex = lwdaDevice->allDeviceListIndex();
            gfxResourceBatch->resourceData[deviceIndex].push_back( res->m_gfxInteropLWDARegisteredResource.get() );
            gfxResourceBatch->bufferData.push_back( res->m_buf );
            return;
        }
        res->m_gfxInteropLWDARegisteredResource.map( lwdaDevice->primaryStream() );

        if( res->m_gfxInteropResource.isArray() )
        {
            lwca::MipmappedArray array       = res->m_gfxInteropLWDARegisteredResource.getMappedMipmappedArray();
            unsigned int         deviceIndex = lwdaDevice->allDeviceListIndex();
            RT_ASSERT( res->m_lwdaArrays[deviceIndex].get() == nullptr );
            res->m_lwdaArrays[deviceIndex] = array;

            // Set the resource kind and access kind
            res->setResource( deviceIndex, MResources::LwdaArray, MAccess::makeNone() );
        }
        else
        {
            // Get the LWCA pointer
            size_t      sizeAccordingToLwda;
            LWdeviceptr interopPtr = res->m_gfxInteropLWDARegisteredResource.getMappedPointer( &sizeAccordingToLwda );

            // Sanity check on size
            if( sizeInBytes != sizeAccordingToLwda )
                throw IlwalidValue( RT_EXCEPTION_INFO, "Graphics resource must be the same size as OptiX buffer" );

            if( policy.interopMode == PolicyDetails::DIRECT )
            {
                // Store directly in MAccess for the interop device
                MAccess      memAccess   = MAccess::makeLinear( (char*)interopPtr );
                unsigned int deviceIndex = lwdaDevice->allDeviceListIndex();
                res->setResource( deviceIndex, MResources::LwdaMalloc, memAccess );
            }
            else
            {
                // Indirect interop. The MAccess should already be
                // allocated. Copy it using a device-to-device copy.
                RT_ASSERT( policy.allowsActiveDeviceWriteAccess() );

                // WAR for http://lwbugs/2414270. Copies from OpenGL back to LWCA are not synchronized correctly,
                // causing the OpenGL buffer to overwrite the OptiX buffer with the results of the previous frame.
                // No one seems to be using this feature, so we're disabling it for now.

                // MAccess memAccess = res->m_buf->getAccess( res->m_gfxInteropDevice );
                // char*   dstPtr    = memAccess.getLinearPtr();
                // lwca::memcpyDtoD( (LWdeviceptr)dstPtr, interopPtr, sizeInBytes );
            }
        }
    }
    else
    {
        // Foreign interop
        MAccess memAccess = res->m_buf->getAccess( res->m_gfxInteropDevice );
        char*   dstPtr    = memAccess.getLinearPtr();

        if( policy.allowsActiveDeviceReadAccess() )
        {
            // Copy from the resource readable on active device
            if( res->m_gfxInteropResource.isOGL() )
                res->m_gfxInteropResource.copyToOrFromGLResource( GfxInteropResource::FromResource, dstPtr, sizeInBytes,
                                                                  &res->m_gfxInteropFBO );
            else
                RT_ASSERT_FAIL_MSG( "non-OGL interop" );
        }
    }
}

void ResourceManager::unmapGfxInteropResource( MResources* res, const PolicyDetails& policy, GfxInteropResourceBatch* gfxResourceBatch )
{
    if( !res->m_gfxInteropRegistered )
        throw ResourceNotRegistered( RT_EXCEPTION_INFO );

    BufferDimensions dims        = res->m_buf->getDimensions();
    size_t           sizeInBytes = dims.getTotalSizeInBytes();

    if( LWDADevice* lwdaDevice = deviceCast<LWDADevice>( res->m_gfxInteropDevice ) )
    {
        lwdaDevice->makeLwrrent();

        if( policy.interopMode == PolicyDetails::DIRECT )
        {
            // Reset the MAccess for the interop device. SGP: I am concerned
            // that this will trigger unncessary plan ilwalidations if we
            // specialize on interop resources. Perhaps we could assume that the
            // pointer will stay the same and only update when changed in
            // mapGraphicsResources(). Re-evelaute if this shows up in a
            // profile.
            const unsigned int deviceIndex = lwdaDevice->allDeviceListIndex();
            // reset LWCA array
            if( res->m_gfxInteropResource.isArray() )
            {
                res->m_lwdaArrays[deviceIndex] = lwca::MipmappedArray();
            }
            res->setResource( deviceIndex, MResources::None, MAccess::makeNone() );
        }
        else
        {
            // Indirect interop.  Copy the data back from the interop resource.
            RT_ASSERT( policy.allowsActiveDeviceWriteAccess() );
            MAccess     memAccess  = res->m_buf->getAccess( res->m_gfxInteropDevice );
            char*       srcPtr     = memAccess.getLinearPtr();
            LWdeviceptr interopPtr = res->m_gfxInteropLWDARegisteredResource.getMappedPointer( nullptr );
            lwca::memcpyDtoD( (LWdeviceptr)interopPtr, (LWdeviceptr)srcPtr, sizeInBytes );
        }

        // Batch immediate HW interop unmap calls.
        if( gfxResourceBatch && isGfxInteropResourceImmediate( res, policy ) )
        {
            const unsigned int deviceIndex = lwdaDevice->allDeviceListIndex();
            gfxResourceBatch->resourceData[deviceIndex].push_back( res->m_gfxInteropLWDARegisteredResource.get() );
            return;
        }

        // Unmap the LWCA resource
        res->m_gfxInteropLWDARegisteredResource.unmap( lwdaDevice->primaryStream() );
    }
    else
    {
        // Foreign interop
        MAccess memAccess = res->m_buf->getAccess( res->m_gfxInteropDevice );
        char*   srcPtr    = memAccess.getLinearPtr();

        if( policy.allowsActiveDeviceWriteAccess() )
        {
            // Copy the resource back to the interop resource writable on active device
            if( res->m_gfxInteropResource.isOGL() )
                res->m_gfxInteropResource.copyToOrFromGLResource( GfxInteropResource::ToResource, srcPtr, sizeInBytes,
                                                                  &res->m_gfxInteropFBO );
            else
                RT_ASSERT_FAIL_MSG( "non-OGL interop" );
        }
    }
}

BufferDimensions ResourceManager::queryGfxInteropResourceSize( MResources* res )
{
    // Query the size of the interop resource.

    GfxInteropResource::Properties props;
    props.glRenderbufferFBO = &res->m_gfxInteropFBO;  // use cached FBO
    res->m_gfxInteropResource.queryProperties( &props );

    return BufferDimensions( props.format, props.elementSize, props.dimensionality, props.width, props.height, props.depth );
}

bool ResourceManager::doesGfxInteropResourceSizeNeedUpdate( MResources* res ) const
{
    // If it is going to be registered (when mapping) and it is array.
    return !res->m_gfxInteropLWDARegisteredResource.get() && res->m_gfxInteropResource.isArray();
}

bool ResourceManager::isGfxInteropResourceImmediate( MResources* res, const PolicyDetails& policy ) const
{
    // If there is no intermediate copy
    // Not foreign with copy through host and not indirect with zero-copy memory
    return policy.interopMode == PolicyDetails::DIRECT && deviceCast<LWDADevice>( res->m_gfxInteropDevice );
}

/****************************************************************
 *
 * Host malloc
 *
 ****************************************************************/

void ResourceManager::acquireHostMallocOnDevices( MResources* res, DeviceSet& onDevices, const PolicyDetails& policy )
{
    BufferDimensions dims   = res->m_buf->getNonZeroDimensions();
    const size_t     nbytes = dims.getTotalSizeInBytes();
    llog( k_mmll.get() ) << " - allocate host malloc: " << dims.toString() << "(" << nbytes << " bytes)"
                         << " on devices: " << onDevices.toString() << '\n';

    for( DeviceSet::const_iterator iter = onDevices.begin(); iter != onDevices.end(); ++iter )
    {
        if( acquireHostMalloc( res, *iter, policy ) )
            onDevices -= DeviceSet( iter );
    }
}

bool ResourceManager::acquireHostMalloc( MResources* res, int onDevice, const PolicyDetails& policy )
{
    // Allocate memory
    BufferDimensions   dims   = res->m_buf->getNonZeroDimensions();
    const unsigned int levels = dims.mipLevelCount();
    const size_t       nbytes = dims.getTotalSizeInBytes();
    char*              ptr    = nullptr;

    // Small allocations come from pools, larger ones are individual.
    if( nbytes <= k_hostBulkAllocationThreshold.get() )
    {
        size_t nb = align( nbytes, BULK_ALLOCATION_SMALL_ALIGNMENT );
        ptr       = (char*)( bulkAllocate( onDevice, nb, BULK_ALLOCATION_SMALL_ALIGNMENT, true ) );
    }
    else
    {
        ptr = static_cast<char*>( malloc( nbytes ) );
    }
    if( ptr == nullptr )
        return false;

    // Clear if required
    if( policy.clearOnAlloc || k_clearBuffersOnAlloc.get() )
        memset( ptr, k_clearValue.get(), nbytes );

    if( levels == 1 )
    {
        // Store the pointer in MAccess
        MAccess memAccess = MAccess::makeLinear( ptr );
        res->setResource( onDevice, MResources::HostMalloc, memAccess );
    }
    else
    {
        // Store the pointers in MAccess
        PitchedLinearAccess pitchedLinear[MAccess::OPTIX_MAX_MIP_LEVELS];
        size_t              offset = 0;
        for( unsigned int level = 0; level < levels; level++ )
        {
            pitchedLinear[level].ptr   = ptr + offset;
            pitchedLinear[level].pitch = dims.getLevelNaturalPitchInBytes( level );
            offset += dims.getLevelSizeInBytes( level );
        }
        MAccess memAccess = MAccess::makeMultiPitchedLinear( pitchedLinear, static_cast<int>( dims.mipLevelCount() ) );
        res->setResource( onDevice, MResources::HostMalloc, memAccess );
    }

    return true;
}

void ResourceManager::releaseHostMalloc( MResources* res, int onDevice )
{
    BufferDimensions dims   = res->m_buf->getNonZeroDimensions();
    const size_t     nbytes = dims.getTotalSizeInBytes();
    llog( k_mmll.get() ) << " - release host malloc: " << dims.toString() << "(" << nbytes << " bytes)"
                         << " on device: " << onDevice << '\n';

    // Free the resource
    char*          ptr    = nullptr;
    const MAccess& access = res->m_buf->getAccess( onDevice );
    if( access.getKind() == MAccess::LINEAR )
        ptr = access.getLinearPtr();
    else
        ptr = access.getPitchedLinear( 0 ).ptr;

    // Small allocations are in pools, and large ones are individiual.
    if( nbytes <= k_hostBulkAllocationThreshold.get() )
    {
        size_t nb = align( nbytes, BULK_ALLOCATION_SMALL_ALIGNMENT );
        bulkFree( onDevice, (LWdeviceptr)ptr, nb, BULK_ALLOCATION_SMALL_ALIGNMENT, true );
    }
    else
    {
        free( ptr );
    }

    // Reset the resource kind and access kind
    res->setResource( onDevice, MResources::None, MAccess::makeNone() );
}

/****************************************************************
 *
 * Texture heap
 *
 ****************************************************************/

void ResourceManager::acquireTexHeapOnDevices( MResources* res, DeviceSet& onDevices )
{
    BufferDimensions dims   = res->m_buf->getNonZeroDimensions();
    const size_t     nbytes = dims.getTotalSizeInBytes();
    llog( k_mmll.get() ) << " - allocate texheap: " << dims.toString() << "(" << nbytes << " bytes)"
                         << " on devices: " << onDevices.toString() << '\n';

    if( !res->m_texHeapAllocation )
    {
        // Acquire the resource from the allocator
        if( reserveTexHeap( res, dims ) == false )
            return;

        // Attach the texheap backing as a dependency
        res->m_buf->attachBackingStore( m_texHeapBacking );
    }

    // Update allocated sets
    res->m_texHeapAllocatedSet |= onDevices;

    // Update the allocation on each device
    for( DeviceSet::position deviceIndex : onDevices )
    {
        MAccess heapAccess = makeTexHeapAccess( m_texHeapSampler->getAccess( deviceIndex ), *res->m_texHeapAllocation );
        res->setResource( deviceIndex, MResources::TexHeap, heapAccess );
    }

    onDevices.clear();
}

void ResourceManager::setTexHeapEnabled( bool enable )
{
    if( enable )
    {
        llog( k_mmll.get() ) << "TexHeap is being (re-)enabled\n";
    }
    else
    {
        llog( k_mmll.get() ) << "TexHeap is being disabled\n";
        RT_ASSERT_MSG( m_texHeapAllocator->empty(), "Trying to disable TexHeap, but allocator not empty" );
    }
    m_texHeapEnabled = enable;
}

bool ResourceManager::isTexHeapEnabled() const
{
    return m_texHeapEnabled;
}

DeviceSet ResourceManager::getLwdaMallocExternalSet( MResources* res ) const
{
    return res->m_lwdaMallocExternalSet;
}

void ResourceManager::releaseTexHeapOnDevices( MResources* res, const DeviceSet& onDevices )
{
    DeviceSet toClear = onDevices & res->m_texHeapAllocatedSet;
    if( toClear.empty() )
        return;

    BufferDimensions dims   = res->m_buf->getNonZeroDimensions();
    const size_t     nbytes = dims.getTotalSizeInBytes();
    llog( k_mmll.get() ) << " - release texheap: " << dims.toString() << "(" << nbytes << " bytes)"
                         << " on devices: " << toClear.toString() << '\n';

    // Reset the resource kind and acceess on each device
    for( DeviceSet::position deviceIndex : toClear )
        res->setResource( deviceIndex, MResources::None, MAccess::makeNone() );

    // Update allocated sets and release main resource if no more allocations
    res->m_texHeapAllocatedSet -= toClear;
    if( res->m_texHeapAllocatedSet.empty() )
    {
        // Release the reservation
        res->m_texHeapAllocation.reset();
        size_t allocElements = idivCeil( dims.getTotalSizeInBytes(), TEXHEAP_ELEMENT_SIZE );
        m_texHeapTotalSize -= allocElements;

        // Detach from the texheap backing
        res->m_buf->detachBackingStore();

        // Unsubscribe from events
        m_texHeapSampler->removeListener( this );

        // If the texture help drops below full then re-enable it.
        // TODO: where to put re-enabling of the texheap. It doesn't belong here (too low level)
        /*
    if( !m_texHeapEnabled && texHeapPercentFull() < TEXHEAP_REENABLE_THRESHOLD ){
      setTexHeapEnabled( true );
      const bool success = m_memoryManager->refillTexHeapAllocations();
      RT_ASSERT_MSG( success, "Texheap refill failed unexpectedly" );
    }
    */
    }
}

bool ResourceManager::reserveTexHeap( MResources* res, const BufferDimensions& dims )
{
    // Texture heap is not responding to new allocations
    if( !m_texHeapEnabled )
        return false;

    // There is a single texheap with 16 byte elements. Size the
    // allocation appropriately.
    RT_ASSERT_MSG( dims.elementSize() % TEXHEAP_ELEMENT_SIZE == 0,
                   "Texheap buffer elements must be a multiple of 16 bytes" );
    const size_t allocElements = idivCeil( dims.getTotalSizeInBytes(), TEXHEAP_ELEMENT_SIZE );

    // Allocate space in the virtual index space (throws on OOM)
    bool success             = false;
    res->m_texHeapAllocation = m_texHeapAllocator->alloc( allocElements, &success );
    if( !success )
        return false;

    m_texHeapTotalSize += allocElements;

    return true;
}


/*
void ResourceManager::defragTexHeap()
{
// To defrag, we just release and refill all texheap allocations. Attempt this
// only if the texheap is full enough to be worth defragging (threshold check),
// but not completely full which would make it hopeless.

const float percentFull = texHeapPercentFull();
if( percentFull > TEXHEAP_DEFRAG_THRESHOLD && percentFull <= 1.0f ) {
m_memoryManager->releaseTexHeapAllocations();
RT_ASSERT_MSG( m_texHeapAllocator->empty(), "Tex heap not empty after flushing" );
const bool success = m_memoryManager->refillTexHeapAllocations();
if( success ) {
  llog(k_mmll.get()) << "TexHeap defrag successful\n";
  return;
}
}

// If the refill failed or the texheap wasn't eligible for defragging to begin
// with, flush it again and disable the texheap.
llog(k_mmll.get()) << "TexHeap defrag attempt failed\n";
m_memoryManager->releaseTexHeapAllocations();
setTexHeapEnabled( false );
}
*/

bool ResourceManager::isTexHeapEligibleForDefrag()
{
    if( !m_texHeapEnabled )
        return false;

    // Return whether a defrag is reasonable
    const float percentFull = static_cast<float>( m_texHeapTotalSize )
                              / static_cast<float>( m_texHeapAllocator->size() - m_texHeapAllocator->freeSpace() );
    return percentFull > TEXHEAP_DEFRAG_THRESHOLD && percentFull <= 1.0f;
}

MAccess ResourceManager::makeTexHeapAccess( const MAccess& texAccess, unsigned int newOffset )
{
    RT_ASSERT_MSG( texAccess.getKind() != MAccess::DEMAND_TEX_OBJECT, "Invalid MAccess kind" );
    if( texAccess.getKind() == MAccess::TEX_OBJECT )
        return MAccess::makeTexObject( texAccess.getTexObject().texObject, newOffset + texAccess.getTexObject().indexOffset );
    if( texAccess.getKind() == MAccess::TEX_REFERENCE )
        return MAccess::makeTexReference( texAccess.getTexReference().texUnit, newOffset + texAccess.getTexReference().indexOffset );
    return texAccess;
}

void ResourceManager::getTexHeapSizeRequest( MBufferHandle* texHeapBacking, size_t* requestedSize )
{
    *texHeapBacking = m_texHeapBacking;
    if( !m_texHeapBacking )
        return;

    const size_t lwrrentSize         = m_texHeapBacking->getDimensions().width();
    const size_t usedAddressRangeEnd = m_texHeapAllocator->getUsedAddressRangeEnd();
    size_t       newSize             = lwrrentSize;

    if( usedAddressRangeEnd > lwrrentSize )
    {
        // Grow the texheap if necessary. This will grow on every
        // allocation. Consider something more aggressive if resizes are
        // excessive for large scenes.
        newSize = usedAddressRangeEnd;
    }
    else
    {
        // Consider shrinking the texheap. Lwrrently a simple metric:
        // shrink if less than half full.
        const size_t inUse = m_texHeapAllocator->size() - m_texHeapAllocator->freeSpace();
        if( inUse < lwrrentSize * TEXHEAP_SHRINK_THRESHOLD )
            newSize = usedAddressRangeEnd;
    }

    *requestedSize = newSize;
}

char* ResourceManager::getTexHeapBackingPointer( Device* device, const MAccess& access )
{
    RT_ASSERT( m_texHeapBacking && access.getKind() == MAccess::TEX_REFERENCE );

    // Ilwalidate other device copies of the texheap
    m_texHeapBacking->m_validSet = DeviceSet( device );

    char* base = m_texHeapBacking->getAccess( device ).getLinearPtr();
    return base + TEXHEAP_ELEMENT_SIZE * access.getTexReference().indexOffset;
}

/****************************************************************
 *
 * Zero copy
 *
 ****************************************************************/

void ResourceManager::acquireZeroCopyOnDevices( MResources* res, DeviceSet& onDevices, const PolicyDetails& policy )
{
    BufferDimensions dims   = res->m_buf->getNonZeroDimensions();
    const size_t     nbytes = dims.getTotalSizeInBytes();
    llog( k_mmll.get() ) << " - allocate zero-copy: " << dims.toString() << "(" << nbytes << " bytes)"
                         << " on devices: " << onDevices.toString() << '\n';

    // The main zero copy allocation is not stored in the host device
    // unless it is allocated on the host.  Allocate it once for all
    // devices.
    if( !res->m_zeroCopyHostPtr )
    {
        // Allocate memory
        res->m_zeroCopyHostPtr = static_cast<char*>( malloc( nbytes ) );
        if( res->m_zeroCopyHostPtr == nullptr )
            return;

        // Clear if required
        if( policy.clearOnAlloc || k_clearBuffersOnAlloc.get() )
            memset( res->m_zeroCopyHostPtr, k_clearValue.get(), nbytes );

        // Register with LWCA using the context on the primary device
        LWDADevice* device = m_deviceManager->primaryLWDADevice();
        RT_ASSERT_MSG( device && device->isEnabled(),
                       "No primary LWCA Device enabled, but a zero copy register is needed." );
        device->makeLwrrent();
        const unsigned int flags = LW_MEMHOSTREGISTER_PORTABLE | LW_MEMHOSTREGISTER_DEVICEMAP;

        LWresult lwres = LWDA_SUCCESS;
        lwca::memHostRegister( res->m_zeroCopyHostPtr, nbytes, flags, &lwres );
        if( lwres != LWDA_SUCCESS )
            return;

        res->m_zeroCopyRegistrarDevice = device;
    }

    // Map to each requested device as appropriate
    for( DeviceSet::const_iterator iter = onDevices.begin(); iter != onDevices.end(); ++iter )
    {
        Device* device  = getDevice( *iter );
        char*   dev_ptr = nullptr;
        if( deviceCast<CPUDevice>( device ) )
        {
            // The resource is also the cpu pointer
            dev_ptr = res->m_zeroCopyHostPtr;
        }
        else if( LWDADevice* lwdaDevice = deviceCast<LWDADevice>( device ) )
        {
            // Get the GPU pointer from LWCA
            lwdaDevice->makeLwrrent();
            unsigned int flags = 0;
            LWresult     lwres = LWDA_SUCCESS;
            dev_ptr            = (char*)lwca::memHostGetDevicePointer( res->m_zeroCopyHostPtr, flags, &lwres );
            if( lwres != LWDA_SUCCESS )
                continue;
        }
        else
        {
            RT_ASSERT_FAIL_MSG( "Unknown device type" );
        }

        // Set the resource kind and pointer
        MAccess memAccess = MAccess::makeLinear( dev_ptr );
        res->setResource( *iter, MResources::ZeroCopy, memAccess );

        // Update allocated sets
        res->m_zeroCopyAllocatedSet |= DeviceSet( iter );
        onDevices -= DeviceSet( iter );
    }
}


void ResourceManager::releaseZeroCopyOnDevices( MResources* res, const DeviceSet& onDevices )
{
    DeviceSet toClear = onDevices & res->m_zeroCopyAllocatedSet;
    if( toClear.empty() )
        return;

    BufferDimensions dims   = res->m_buf->getNonZeroDimensions();
    size_t           nbytes = dims.getTotalSizeInBytes();

    llog( k_mmll.get() ) << " - release zero-copy: " << dims.toString() << "(" << nbytes << " bytes)"
                         << " on devices: " << toClear.toString() << '\n';

    for( DeviceSet::position deviceIndex : toClear )
        // Reset the resource kind and acceess
        res->setResource( deviceIndex, MResources::None, MAccess::makeNone() );

    res->m_zeroCopyAllocatedSet -= toClear;
    if( res->m_zeroCopyAllocatedSet.empty() )
    {
        // it is possible that no zero-copy mapping exists yet, because it is a zero-copy buffer but no allocation on a
        // device
        // has taken place yet.
        if( res->m_zeroCopyRegistrarDevice )
        {
            LWDADevice* device = res->m_zeroCopyRegistrarDevice;
            // LWDADevice* device = m_deviceManager->primaryLWDADevice();
            RT_ASSERT_MSG( device && device->isEnabled(),
                           "Trying to unregister zero-copy buffer, but regitrar device is not enabled. Something went "
                           "wrong "
                           "with updateRegistrations()." );
            // Unregister the host pointer
            device->makeLwrrent();
            lwca::memHostUnregister( res->m_zeroCopyHostPtr );
            res->m_zeroCopyRegistrarDevice = nullptr;
        }

        // Free the resource
        free( res->m_zeroCopyHostPtr );
        res->m_zeroCopyHostPtr = nullptr;
    }
}

/****************************************************************
*
* Single Copy
*
****************************************************************/

LWdeviceptr ResourceManager::allocateLwdaSingleCopy( MResources*          res,
                                                     const DeviceSet&     possibleDevices,
                                                     const PolicyDetails& policy,
                                                     const size_t         nbytes )
{
    llog( k_mmll.get() ) << " - allocating new copy of size " << nbytes << " bytes\n";

    LWdeviceptr ptr = 0;

    Device*    interopDevice     = res->m_gfxInteropDevice;
    const bool haveInteropDevice = interopDevice != nullptr && interopDevice->isActive();
    const bool canUseInteropDevice =
        haveInteropDevice && possibleDevices[static_cast<int>( interopDevice->allDeviceListIndex() )] != -1;

    int lwrDevice;
    if( canUseInteropDevice && policy.interopMode != PolicyDetails::NOINTEROP )
        lwrDevice = static_cast<int>( interopDevice->allDeviceListIndex() );
    else
    {
        // Start with a hash to "randomize" the preferred device
        lwrDevice = static_cast<int>( std::hash<size_t>{}( res->m_buf->m_serialNumber ) % possibleDevices.count() );
    }

    // Acquire a single allocation across all devices.
    for( int i = 0; i < static_cast<int>( possibleDevices.count() ); ++i )
    {
        int devIdx = possibleDevices[lwrDevice];
        lwrDevice  = ( lwrDevice + 1 ) % static_cast<int>( possibleDevices.count() );

        LWresult lwres = LWDA_SUCCESS;

        getLWDADevice( devIdx )->makeLwrrent();
        ptr = lwca::memAlloc( nbytes, &lwres );

        // Remember which device owns the allocation.
        res->m_singleCopyAllocatedSet |= DeviceSet( devIdx );

        if( lwres == LWDA_SUCCESS )
        {
            // Clear if required
            if( policy.clearOnAlloc || k_clearBuffersOnAlloc.get() )
                lwca::memsetD8( ptr, k_clearValue.get(), nbytes );

            return ptr;
        }
    }

    return ptr;
}

void ResourceManager::acquireLwdaSingleCopyOnDevices( MResources* res, DeviceSet& onDevices, const PolicyDetails& policy )
{
    // Continue only if the GPUs are fully connected.
    if( !shouldAllocateWritableBufferOnDevice() )
        return;

    BufferDimensions dims   = res->m_buf->getNonZeroDimensions();
    const size_t     nbytes = dims.getTotalSizeInBytes();
    llog( k_mmll.get() ) << " - acquire single copy lwca malloc: " << dims.toString() << "(" << nbytes << " bytes)"
                         << " on devices: " << onDevices.toString() << '\n';

    if( !res->m_singleCopyPtr )
        res->m_singleCopyPtr = (char*)allocateLwdaSingleCopy( res, onDevices, policy, nbytes );

    if( res->m_singleCopyPtr )
    {
        // Set the same pointer on all devices
        MAccess memAccess = MAccess::makeLinear( res->m_singleCopyPtr );
        for( int peerIdx : onDevices )
        {
            res->setResource( peerIdx, MResources::LwdaMallocSingleCopy, memAccess );
            onDevices -= DeviceSet( peerIdx );
        }
    }
}

void ResourceManager::releaseLwdaSingleCopyOnDevices( MResources* res, const DeviceSet& onDevices, const PolicyDetails& policy )
{
    // Find the devices with a single copy allocation
    DeviceSet lwrrentDevices;
    for( int i = 0; i < OPTIX_MAX_DEVICES; ++i )
    {
        if( res->m_resourceKind[i] == MResources::LwdaMallocSingleCopy )
            lwrrentDevices |= DeviceSet( i );
    }

    DeviceSet toClear = lwrrentDevices & onDevices;
    if( toClear.empty() )
        return;

    BufferDimensions dims = res->m_buf->getNonZeroDimensions();
    llog( k_mmll.get() ) << " - release single copy lwca malloc: " << dims.toString()
                         << " on devices: " << toClear.toString() << '\n';

    // If the device with the allocation is being released, we need to deallocate.
    DeviceSet alloc = res->m_singleCopyAllocatedSet & onDevices;
    RT_ASSERT( alloc.count() <= 1U );
    if( !alloc.empty() )
    {
        // Information about the allocation.
        const size_t nbytes    = dims.getTotalSizeInBytes();
        const int    oldDevIdx = alloc[0];
        char*        oldPtr    = res->m_buf->getAccess( oldDevIdx ).getLinearPtr();

        llog( k_mmll.get() ) << " - freeing single copy allocation on device " << oldDevIdx << " \n";

        // If there will be active devices after this release, allocate on a new device,
        if( toClear != lwrrentDevices )
        {
            llog( k_mmll.get() ) << " - active devices will remain after release. Moving to new device...\n";

            LWdeviceptr newPtr = allocateLwdaSingleCopy( res, lwrrentDevices - onDevices, policy, nbytes );
            RT_ASSERT( newPtr != 0 );

            // Copy the data from the old buffer to the new buffer.
            Device*     srcDevice = getLWDADevice( oldDevIdx );
            LWDADevice* srcLwda   = deviceCast<LWDADevice>( srcDevice );

            DeviceSet   newAlloc  = res->m_singleCopyAllocatedSet - alloc;
            Device*     dstDevice = getLWDADevice( newAlloc[0] );
            LWDADevice* dstLwda   = deviceCast<LWDADevice>( dstDevice );

            llog( k_mmll.get() ) << " - new device: " << newAlloc[0] << " old device: " << oldDevIdx << "\n";

            lwca::memcpyPeer( newPtr, dstLwda->lwdaContext(), (LWdeviceptr)res->m_singleCopyPtr, srcLwda->lwdaContext(), nbytes );

            // Update the pointers on peer devices.
            MAccess memAccess = MAccess::makeLinear( (char*)newPtr );
            for( int peerIdx : lwrrentDevices )
            {
                res->setResource( peerIdx, MResources::None, MAccess::makeNone() );
                res->setResource( peerIdx, MResources::LwdaMallocSingleCopy, memAccess );
            }

            res->m_singleCopyPtr = (char*)newPtr;
        }
        else
        {
            res->m_singleCopyPtr = nullptr;
        }

        // Free the old allocation.
        getLWDADevice( oldDevIdx )->makeLwrrent();
        lwca::memFree( (LWdeviceptr)oldPtr );
        res->m_singleCopyAllocatedSet -= alloc;
    }

    // Update the devices on which we're releasing the resources.
    for( int peerIdx : onDevices )
        res->setResource( peerIdx, MResources::None, MAccess::makeNone() );
}

void ResourceManager::acquireHostSingleCopyOnDevices( MResources* res, DeviceSet& onDevices, const PolicyDetails& policy )
{
    if( shouldAllocateWritableBufferOnDevice() )
        acquireHostMallocOnDevices( res, onDevices, policy );
}

/****************************************************************
 *
 * Utility functions and access change event
 *
 ****************************************************************/

Device* ResourceManager::getDevice( unsigned int allDeviceIndex ) const
{
    Device* device = m_deviceManager->allDevices()[allDeviceIndex];
    return device;
}

LWDADevice* ResourceManager::getLWDADevice( unsigned int allDeviceIndex ) const
{
    Device*     device     = getDevice( allDeviceIndex );
    LWDADevice* lwdaDevice = deviceCast<LWDADevice>( device );
    RT_ASSERT_MSG( lwdaDevice != nullptr, "Invalid LWCA device" );
    return lwdaDevice;
}

void ResourceManager::lazyInitializeBulkMemoryPools( unsigned int allDeviceIndex )
{
    if( !m_bulkMemoryPools_small[allDeviceIndex].isInitialized() )
        m_bulkMemoryPools_small[allDeviceIndex].initialize( m_context, static_cast<int>( allDeviceIndex ),
                                                            BULK_ALLOCATION_SMALL_ALIGNMENT );
    if( !m_bulkMemoryPools_large[allDeviceIndex].isInitialized() )
        m_bulkMemoryPools_large[allDeviceIndex].initialize( m_context, static_cast<int>( allDeviceIndex ),
                                                            BULK_ALLOCATION_LARGE_ALIGNMENT );
}

void ResourceManager::eventMTextureSamplerMAccessDidChange( const Device* device, const MAccess&, const MAccess& texAccess )
{
    // Received notification that the texture sampler address
    // changed. Update pointers for all texheap buffers.  This should
    // happen rarely, so no need to maintain a separate list of buffers
    // allocated in texture.
    const unsigned int           allDeviceListIndex = device->allDeviceListIndex();
    const std::vector<MBuffer*>& buffers            = m_memoryManager->getMasterList();
    for( MBuffer* buf : buffers )
    {
        MResources* res = buf->m_resources.get();
        if( res->m_resourceKind[allDeviceListIndex] == MResources::TexHeap )
        {
            MAccess heapAccess = makeTexHeapAccess( texAccess, *res->m_texHeapAllocation );
            res->m_buf->setAccess( static_cast<int>( allDeviceListIndex ), heapAccess );
        }
    }
}

bool ResourceManager::shouldAllocateWritableBufferOnDevice() const
{
    return m_deviceManager->isPeerToPeerFullyConnected();
}


/****************************************************************
 *
 * Debug Utility functions
 *
 ****************************************************************/
void* ResourceManager::reallocateMemoryForDebug( void* ptr, size_t size )
{
    lwca::memFree( (LWdeviceptr)ptr );
    if( size == 0 )
        return nullptr;
    return (void*)lwca::memAlloc( size );
}

}  // namespace optix
