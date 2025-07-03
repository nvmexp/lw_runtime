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

#include <LWCA/Array.h>

#include <LWCA/ErrorCheck.h>
#include <LWCA/Memory.h>
#include <LWCA/Stream.h>

#include <prodlib/exceptions/Assert.h>

#include <corelib/system/LwdaDriver.h>

using namespace optix;
using namespace optix::lwca;
using namespace corelib;

Array::Array()
    : m_array( nullptr )
{
}

Array::Array( LWarray array )
    : m_array( array )
{
}

LWarray Array::get() const
{
    return m_array;
}

Array Array::create( const LWDA_ARRAY_DESCRIPTOR& pAllocateArray, LWresult* returnResult )
{
    LWarray result = nullptr;
    CHECK( lwdaDriver().LwArrayCreate( &result, &pAllocateArray ) );
    return Array( result );
}

Array Array::create( const LWDA_ARRAY3D_DESCRIPTOR& pAllocateArray, LWresult* returnResult )
{
    LWarray result = nullptr;
    CHECK( lwdaDriver().LwArray3DCreate( &result, &pAllocateArray ) );
    return Array( result );
}

Array Array::create( LWgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel, LWresult* returnResult /*= 0 */ )
{
    LWarray result = nullptr;
    CHECK( lwdaDriver().LwGraphicsSubResourceGetMappedArray( &result, resource, arrayIndex, mipLevel ) );
    return Array( result );
}

void Array::destroy( LWresult* returnResult )
{
    RT_ASSERT( m_array != nullptr );
    CHECK( lwdaDriver().LwArrayDestroy( m_array ) );
}

// Get a 1D or 2D LWCA array descriptor.
LWDA_ARRAY_DESCRIPTOR
Array::getDescriptor( LWresult* returnResult ) const
{
    RT_ASSERT( m_array != nullptr );
    LWDA_ARRAY_DESCRIPTOR result;
    CHECK( lwdaDriver().LwArrayGetDescriptor( &result, m_array ) );
    return result;
}

// Get a 3D LWCA array descriptor.
LWDA_ARRAY3D_DESCRIPTOR
Array::getDescriptor3D( LWresult* returnResult ) const
{
    RT_ASSERT( m_array != nullptr );
    LWDA_ARRAY3D_DESCRIPTOR result;
    CHECK( lwdaDriver().LwArray3DGetDescriptor( &result, m_array ) );
    return result;
}

MipmappedArray::MipmappedArray()
    : m_mmarray( nullptr )
{
}

MipmappedArray::MipmappedArray( LWmipmappedArray mmarray )
    : m_mmarray( mmarray )
{
}

LWmipmappedArray MipmappedArray::get() const
{
    return m_mmarray;
}

// Creates a LWCA mipmapped array.
MipmappedArray MipmappedArray::create( const LWDA_ARRAY3D_DESCRIPTOR& pMipmappedArrayDesc, unsigned int numMipmapLevels, LWresult* returnResult )
{
    LWmipmappedArray result = nullptr;
    CHECK( lwdaDriver().LwMipmappedArrayCreate( &result, &pMipmappedArrayDesc, numMipmapLevels ) );
    return MipmappedArray( result );
}

MipmappedArray MipmappedArray::create( LWgraphicsResource resource, LWresult* returnResult /*= 0 */ )
{
    LWmipmappedArray result = nullptr;
    CHECK( lwdaDriver().LwGraphicsResourceGetMappedMipmappedArray( &result, resource ) );
    return MipmappedArray( result );
}

// Destroys a LWCA mipmapped array.
void MipmappedArray::destroy( LWresult* returnResult )
{
    RT_ASSERT( m_mmarray != nullptr );
    CHECK( lwdaDriver().LwMipmappedArrayDestroy( m_mmarray ) );
}

// Gets a mipmap level of a LWCA mipmapped array.
Array MipmappedArray::getLevel( unsigned int level, LWresult* returnResult ) const
{
    RT_ASSERT( m_mmarray != nullptr );
    LWarray result = nullptr;
    CHECK( lwdaDriver().LwMipmappedArrayGetLevel( &result, m_mmarray, level ) );
    return Array( result );
}

bool MipmappedArray::isSparse() const
{
    if( m_mmarray == nullptr )
        return false;

    LWDA_ARRAY3D_DESCRIPTOR desc{};
    LWresult* returnResult = nullptr;
    CHECK( lwdaDriver().LwArray3DGetDescriptor( &desc, getLevel( 0 ).get() ) );
    return ( desc.Flags & LWDA_ARRAY3D_SPARSE ) != 0;
}

void MipmappedArray::unmapSparseLevel( int mipLevel, int deviceOrdinal, LWresult* returnResult )
{
    LWDA_ARRAY_DESCRIPTOR levelDesc = getLevel( mipLevel ).getDescriptor();

    // Unmap all memory associated with this mip level.
    LWarrayMapInfo mapInfo{};

    mapInfo.resourceType    = LW_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    mapInfo.resource.mipmap = m_mmarray;

    mapInfo.subresourceType               = LW_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL;
    mapInfo.subresource.sparseLevel.level = mipLevel;

    mapInfo.subresource.sparseLevel.offsetX = 0;
    mapInfo.subresource.sparseLevel.offsetY = 0;

    mapInfo.subresource.sparseLevel.extentWidth  = levelDesc.Width;
    mapInfo.subresource.sparseLevel.extentHeight = levelDesc.Height;
    mapInfo.subresource.sparseLevel.extentDepth  = 1;

    mapInfo.memOperationType = LW_MEM_OPERATION_TYPE_UNMAP;
    mapInfo.memHandleType = LW_MEM_HANDLE_TYPE_GENERIC;
    mapInfo.memHandle.memHandle = 0;

    mapInfo.deviceBitMask = 1U << deviceOrdinal;

    // A null stream indicates that the default stream should be used.
    memMapArrayAsync( &mapInfo, 1, Stream(), returnResult );
}

void MipmappedArray::unmapSparseMipTail( int deviceOrdinal, LWresult* returnResult )
{
    LWarrayMapInfo mapInfo{};

    mapInfo.resourceType    = LW_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    mapInfo.resource.mipmap = m_mmarray;

    mapInfo.subresourceType            = LW_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL;
    mapInfo.subresource.miptail.offset = 0;
    mapInfo.subresource.miptail.size   = getSparseProperties().miptailSize;

    mapInfo.memOperationType = LW_MEM_OPERATION_TYPE_UNMAP;
    mapInfo.memHandleType = LW_MEM_HANDLE_TYPE_GENERIC;
    mapInfo.memHandle.memHandle = 0;

    mapInfo.deviceBitMask = 1U << deviceOrdinal;

    // A null stream indicates that the default stream should be used.
    memMapArrayAsync( &mapInfo, 1, Stream(), returnResult );
}

LWDA_ARRAY_SPARSE_PROPERTIES MipmappedArray::getSparseProperties( LWresult* returnResult ) const
{
    RT_ASSERT( m_mmarray != nullptr );
#if defined( DEBUG ) || defined( DEVELOP )
    LWDA_ARRAY3D_DESCRIPTOR desc{};
    CHECK( lwdaDriver().LwArray3DGetDescriptor( &desc, getLevel( 0, returnResult ).get() ) );
    RT_ASSERT( ( desc.Flags & LWDA_ARRAY3D_SPARSE ) != 0 );
#endif
    LWDA_ARRAY_SPARSE_PROPERTIES props;
    CHECK( lwdaDriver().LwMipmappedArrayGetSparseProperties( &props, m_mmarray ) );
    return props;
}
