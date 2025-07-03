//
//  Copyright (c) 2020 LWPU Corporation.  All rights reserved.
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

#include <srcTests.h>

#include <LWCA/Array.h>
#include <LWCA/Context.h>
#include <LWCA/Device.h>
#include <LWCA/Memory.h>
#include <LWCA/Stream.h>

#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/LwdaError.h>

#include <corelib/system/LwdaDriver.h>

#include <lwda_etbl/lwda_feature_toggle.h>
#include <lwda_etbl/memmap.h>
#include <lwda_etbl/texture.h>

#define CHECK_LWDA( call ) prodlib::checkLwdaError( call, #call, RT_FILE_NAME, RT_LINE, nullptr )

using namespace optix;
using namespace testing;

static int g_sparseOrdinal = -1;

static bool supportsSparseTextures()
{
    corelib::LwdaDriver& driver = corelib::lwdaDriver();
    CHECK_LWDA( driver.LwInit( 0 ) );
    int numDevices = 0;
    CHECK_LWDA( driver.LwDeviceGetCount( &numDevices ) );
    for( int i = 0; i < numDevices; ++i )
    {
        LWdevice device = 0;
        CHECK_LWDA( driver.LwDeviceGet( &device, i ) );
        int supportsSparseTextures = 0;
        CHECK_LWDA( driver.LwDeviceGetAttribute( &supportsSparseTextures, LW_DEVICE_ATTRIBUTE_SPARSE_LWDA_ARRAY_SUPPORTED, device ) );
        if( supportsSparseTextures )
        {
            g_sparseOrdinal = i;
            return true;
        }
    }

    return false;
}

static bool supportsSparseMipmappedArrays()
{
    corelib::LwdaDriver& driver = corelib::lwdaDriver();
    CHECK_LWDA( driver.LwInit( 0 ) );
    int numDevices = 0;
    CHECK_LWDA( driver.LwDeviceGetCount( &numDevices ) );
    for( int i = 0; i < numDevices; ++i )
    {
        LWdevice device = 0;
        CHECK_LWDA( driver.LwDeviceGet( &device, i ) );

        // LW_ARRAY3D_SPARSE flag for LwMipmappedArrayCreate is only supported on Pascal and later
        int major, minor;
        CHECK_LWDA( driver.LwDeviceComputeCapability( &major, &minor, device ) );
        if( major >= 6 )
        {
            if( g_sparseOrdinal < 0 )
                g_sparseOrdinal = i;
            return true;
        }
    }

    return false;
}

static std::vector<std::string> probeSparseTextureSupport()
{
    std::vector<std::string> filters;

    if( !supportsSparseTextures() )
    {
        filters.emplace_back( "TestLWDASparseTexture*" );
        std::cerr << "WARNING: filtering tests based on LWCA sparse texture support\n\n";
    }
    if( !supportsSparseMipmappedArrays() )
    {
        filters.emplace_back( "*TestSparseMipMappedArray*" );
        std::cerr << "WARNING: filtering tests based on LWCA sparse mipmapped array support\n\n";
    }
    if( g_sparseOrdinal < 0 )
        g_sparseOrdinal = 0;

    return filters;
}

static SrcTestFilter s_noSparseTextures( probeSparseTextureSupport );

namespace {

class TestLWDASparseTexture : public Test
{
  public:
    ~TestLWDASparseTexture() override = default;

    void SetUp() override;

    void TearDown() override;

  protected:
    lwca::Device        m_device;
    lwca::Context       m_context;
    optix::lwca::Stream m_stream;
};

void TestLWDASparseTexture::SetUp()
{
    m_device  = lwca::Device::get( g_sparseOrdinal );
    m_context = lwca::Context::create( 0, m_device );
    m_context.setLwrrent();
}

void TestLWDASparseTexture::TearDown()
{
    m_context.destroy();
}

class TestSparseMipMappedArray : public TestLWDASparseTexture, public WithParamInterface<LWarray_format>
{
  protected:
    void TearDown() override;

    size_t getPixelSize() const;
    size_t getWholePageSizeInBytes( unsigned int sizeInBytes ) const;
    lwca::MipmappedArray         createMipMappedArray();
    LWmemGenericAllocationHandle createBackingStore( unsigned int mipTailSizeInBytes );
    LWmemGenericAllocationHandle createMipTailBackingStore();
    LWmemGenericAllocationHandle createMipLevelBackingStore();
    LWarrayMapInfo               createMipTailBindInfo() const;
    LWarrayMapInfo createMipLevelBindInfo( unsigned int mipLevel ) const;
    LWarrayMapInfo createMipTailUnbindInfo() const;
    LWarrayMapInfo createMipLevelUnbindInfo( unsigned int mipLevel ) const;

    lwca::MipmappedArray         m_mipMappedArray;
    LWmemGenericAllocationHandle m_backingStore = 0;
};

void TestSparseMipMappedArray::TearDown()
{
    if( m_backingStore )
    {
        lwca::memRelease( m_backingStore );
    }
    if( m_mipMappedArray.get() )
    {
        m_mipMappedArray.destroy();
    }
    TestLWDASparseTexture::TearDown();
}

size_t TestSparseMipMappedArray::getPixelSize() const
{
    switch( GetParam() )
    {
        case LW_AD_FORMAT_UNSIGNED_INT8:
            return sizeof( std::uint8_t );
        case LW_AD_FORMAT_UNSIGNED_INT16:
            return sizeof( std::uint16_t );
        case LW_AD_FORMAT_UNSIGNED_INT32:
            return sizeof( std::uint32_t );
        case LW_AD_FORMAT_SIGNED_INT8:
            return sizeof( std::int8_t );
        case LW_AD_FORMAT_SIGNED_INT16:
            return sizeof( std::int16_t );
        case LW_AD_FORMAT_SIGNED_INT32:
            return sizeof( std::int32_t );
        case LW_AD_FORMAT_HALF:
            return sizeof( float ) / 2;
        case LW_AD_FORMAT_FLOAT:
            return sizeof( float );
        default:
            throw prodlib::AssertionFailure( RT_EXCEPTION_INFO,
                                             "Unknown format " + std::to_string( static_cast<int>( GetParam() ) ) );
    }
}

size_t roundUp( size_t value, size_t increment )
{
    return ( ( value + increment - 1 ) / increment ) * increment;
}

size_t TestSparseMipMappedArray::getWholePageSizeInBytes( unsigned int sizeInBytes ) const
{
    LWmemAllocationProp        prop;
    size_t                     pageSize = 0;
    corelib::lwdaDriver().LwMemGetAllocationGranularity( &pageSize, reinterpret_cast<LWmemAllocationProp*>( &prop ),
                                                         LW_MEM_ALLOC_GRANULARITY_RECOMMENDED );
    return roundUp( static_cast<size_t>( sizeInBytes ), pageSize );
}

lwca::MipmappedArray TestSparseMipMappedArray::createMipMappedArray()
{
    LWDA_ARRAY3D_DESCRIPTOR desc;
    desc.Width       = 1024;
    desc.Height      = 1024;
    desc.Depth       = 0;
    desc.Format      = GetParam();
    desc.NumChannels = 4;
    desc.Flags       = LWDA_ARRAY3D_SPARSE;
    return lwca::MipmappedArray::create( desc, 10 );
}

LWmemGenericAllocationHandle TestSparseMipMappedArray::createBackingStore( unsigned int mipTailSizeInBytes )
{
    LWmemAllocationProp allocationProps{};
    allocationProps.type             = LW_MEM_ALLOCATION_TYPE_PINNED;
    allocationProps.location         = {LW_MEM_LOCATION_TYPE_DEVICE, g_sparseOrdinal};
    allocationProps.allocFlags.usage = LW_MEM_CREATE_USAGE_TILE_POOL;

    return lwca::memCreate( getWholePageSizeInBytes( mipTailSizeInBytes ),
                            reinterpret_cast<LWmemAllocationProp*>( &allocationProps ), 0 );
}

LWmemGenericAllocationHandle TestSparseMipMappedArray::createMipTailBackingStore()
{
    return createBackingStore( m_mipMappedArray.getSparseProperties().miptailSize );
}

LWmemGenericAllocationHandle TestSparseMipMappedArray::createMipLevelBackingStore()
{
    LWDA_ARRAY_DESCRIPTOR levelDesc  = m_mipMappedArray.getLevel( 0 ).getDescriptor();
    unsigned int mipLevelSizeInBytes = levelDesc.Width * levelDesc.Height * levelDesc.NumChannels * getPixelSize();
    return createBackingStore( mipLevelSizeInBytes );
}

LWarrayMapInfo TestSparseMipMappedArray::createMipTailBindInfo() const
{
    LWarrayMapInfo mapInfo{};
    mapInfo.resourceType               = LW_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    mapInfo.resource.mipmap            = m_mipMappedArray.get();
    mapInfo.subresourceType            = LW_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL;
    mapInfo.subresource.miptail.offset = 0;
    mapInfo.subresource.miptail.size   = m_mipMappedArray.getSparseProperties().miptailSize;
    mapInfo.memOperationType           = LW_MEM_OPERATION_TYPE_MAP;
    mapInfo.memHandleType              = LW_MEM_HANDLE_TYPE_GENERIC;
    mapInfo.memHandle.memHandle        = m_backingStore;
    mapInfo.offset                     = 0;
    mapInfo.deviceBitMask              = 1 << g_sparseOrdinal;
    return mapInfo;
}

LWarrayMapInfo TestSparseMipMappedArray::createMipTailUnbindInfo() const
{
    LWarrayMapInfo mapInfo{};
    mapInfo.resourceType               = LW_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    mapInfo.resource.mipmap            = m_mipMappedArray.get();
    mapInfo.subresourceType            = LW_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL;
    mapInfo.subresource.miptail.offset = 0;
    mapInfo.subresource.miptail.size   = m_mipMappedArray.getSparseProperties().miptailSize;
    mapInfo.memOperationType           = LW_MEM_OPERATION_TYPE_UNMAP;
    mapInfo.memHandleType              = LW_MEM_HANDLE_TYPE_GENERIC;
    mapInfo.memHandle.memHandle        = 0;
    mapInfo.offset                     = 0;
    mapInfo.deviceBitMask              = 1 << g_sparseOrdinal;
    return mapInfo;
}

LWarrayMapInfo TestSparseMipMappedArray::createMipLevelBindInfo( unsigned int mipLevel ) const
{
    LWDA_ARRAY_DESCRIPTOR levelDesc = m_mipMappedArray.getLevel( mipLevel ).getDescriptor();
    LWarrayMapInfo        mapInfo{};
    mapInfo.resourceType                         = LW_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    mapInfo.resource.mipmap                      = m_mipMappedArray.get();
    mapInfo.subresourceType                      = LW_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL;
    mapInfo.subresource.sparseLevel.level        = mipLevel;
    mapInfo.subresource.sparseLevel.offsetX      = 0;
    mapInfo.subresource.sparseLevel.offsetY      = 0;
    mapInfo.subresource.sparseLevel.extentWidth  = levelDesc.Width;
    mapInfo.subresource.sparseLevel.extentHeight = levelDesc.Height;
    mapInfo.subresource.sparseLevel.extentDepth  = 1;
    mapInfo.memOperationType                     = LW_MEM_OPERATION_TYPE_MAP;
    mapInfo.memHandleType                        = LW_MEM_HANDLE_TYPE_GENERIC;
    mapInfo.memHandle.memHandle                  = m_backingStore;
    mapInfo.offset                               = 0;
    mapInfo.deviceBitMask                        = 1 << g_sparseOrdinal;
    
    return mapInfo;
}

LWarrayMapInfo TestSparseMipMappedArray::createMipLevelUnbindInfo( unsigned int mipLevel ) const
{
    LWDA_ARRAY_DESCRIPTOR levelDesc = m_mipMappedArray.getLevel( mipLevel ).getDescriptor();
    LWarrayMapInfo        mapInfo{};
    mapInfo.resourceType                         = LW_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    mapInfo.resource.mipmap                      = m_mipMappedArray.get();
    mapInfo.subresourceType                      = LW_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL;
    mapInfo.subresource.sparseLevel.level        = mipLevel;
    mapInfo.subresource.sparseLevel.offsetX      = 0;
    mapInfo.subresource.sparseLevel.offsetY      = 0;
    mapInfo.subresource.sparseLevel.extentWidth  = levelDesc.Width;
    mapInfo.subresource.sparseLevel.extentHeight = levelDesc.Height;
    mapInfo.subresource.sparseLevel.extentDepth  = 1;
    mapInfo.memOperationType                     = LW_MEM_OPERATION_TYPE_UNMAP;
    mapInfo.memHandleType                        = LW_MEM_HANDLE_TYPE_GENERIC;
    mapInfo.memHandle.memHandle                  = 0;
    mapInfo.offset                               = 0;
    mapInfo.deviceBitMask                        = 1 << g_sparseOrdinal;
    return mapInfo;
}

}  // namespace

TEST_P( TestSparseMipMappedArray, construct )
{
    lwca::MipmappedArray array = createMipMappedArray();

    array.destroy();
}

TEST_P( TestSparseMipMappedArray, allocate_backing_store )
{
    const LWmemGenericAllocationHandle memHandle = createBackingStore( 64 * 1024 /* 64K */ );

    lwca::memRelease( memHandle );
}

// bug 2925967 bind/unbind lwca sparse arrays crashes in a worker thread
TEST_P( TestSparseMipMappedArray, bind_unbind_mip_tail )
{
    m_mipMappedArray = createMipMappedArray();
    m_backingStore   = createMipTailBackingStore();

    LWarrayMapInfo mapInfo = createMipTailBindInfo();
    lwca::memMapArrayAsync(&mapInfo, 1, m_stream);
    m_context.synchronize();
    mapInfo = createMipTailUnbindInfo();
    lwca::memMapArrayAsync( &mapInfo, 1, m_stream );
    m_context.synchronize();
}

// bug 2925967 bind/unbind lwca sparse arrays crashes in a worker thread
TEST_P( TestSparseMipMappedArray, bind_unbind_mip_level )
{
    m_mipMappedArray = createMipMappedArray();
    m_backingStore   = createMipLevelBackingStore();

    LWarrayMapInfo mapInfo = createMipLevelBindInfo( 0 );
    lwca::memMapArrayAsync( &mapInfo, 1, m_stream );
    m_context.synchronize();
    mapInfo = createMipLevelUnbindInfo( 0 );
    lwca::memMapArrayAsync( &mapInfo, 1, m_stream );
    m_context.synchronize();
}

static const LWarray_format s_arrayFormats[] = {
    // clang-format off
    LW_AD_FORMAT_UNSIGNED_INT8,
    LW_AD_FORMAT_UNSIGNED_INT16,
    LW_AD_FORMAT_UNSIGNED_INT32,
    LW_AD_FORMAT_SIGNED_INT8,
    LW_AD_FORMAT_SIGNED_INT16,
    LW_AD_FORMAT_SIGNED_INT32,
    LW_AD_FORMAT_HALF,
    LW_AD_FORMAT_FLOAT
    // clang-format on
};

// googletest will use this automatically
std::string PrintToString( LWarray_format format )
{
    switch( format )
    {
        case LW_AD_FORMAT_UNSIGNED_INT8:
            return "uint8";
        case LW_AD_FORMAT_UNSIGNED_INT16:
            return "uint16";
        case LW_AD_FORMAT_UNSIGNED_INT32:
            return "uint32";
        case LW_AD_FORMAT_SIGNED_INT8:
            return "int8";
        case LW_AD_FORMAT_SIGNED_INT16:
            return "int16";
        case LW_AD_FORMAT_SIGNED_INT32:
            return "int32";
        case LW_AD_FORMAT_HALF:
            return "half";
        case LW_AD_FORMAT_FLOAT:
            return "float";
        default:
            return "unknown: " + std::to_string( static_cast<int>( format ) );
    }
}

INSTANTIATE_TEST_SUITE_P( ArrayFormats, TestSparseMipMappedArray, ValuesIn( s_arrayFormats ) );
