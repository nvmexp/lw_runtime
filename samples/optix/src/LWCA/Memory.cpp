// Copyright (c) 2017, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER(INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <LWCA/Memory.h>

#include <LWCA/Array.h>
#include <LWCA/Context.h>
#include <LWCA/ErrorCheck.h>
#include <LWCA/Stream.h>
#include <corelib/system/LwdaDriver.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/system/Knobs.h>

using namespace corelib;

namespace
{
// clang-format off
Knob<size_t> k_maximumDeviceMemory( RT_DSTRING( "lwca.maximumDeviceMemory" ), 0, RT_DSTRING( "Set a limit on the visible device memory. Default is 0, which means use what the driver reports." ) );
// clang-format on
}

namespace optix {
namespace lwca {

LWdeviceptr memAlloc( size_t byteCount, LWresult* returnResult )
{
    LWdeviceptr result = 0;
    CHECK( lwdaDriver().LwMemAlloc( &result, byteCount ) );
    return result;
}

LWdeviceptr memAllocPitch( size_t* pPitch, size_t WidthInBytes, size_t height, unsigned int ElementSizeBytes, LWresult* returnResult )
{
    LWdeviceptr result = 0;
    CHECK( lwdaDriver().LwMemAllocPitch( &result, pPitch, WidthInBytes, height, ElementSizeBytes ) );
    return result;
}

void memFree( LWdeviceptr dptr, LWresult* returnResult )
{
    CHECK( lwdaDriver().LwMemFree( dptr ) );
}

void memGetAddressRange( LWdeviceptr* pbase, size_t* psize, LWdeviceptr dptr, LWresult* returnResult )
{
    CHECK( lwdaDriver().LwMemGetAddressRange( pbase, psize, dptr ) );
}

void memGetInfo( size_t* free, size_t* total, LWresult* returnResult )
{
    const size_t memLimit = k_maximumDeviceMemory.get();
    CHECK( lwdaDriver().LwMemGetInfo( free, total ) );
    *total = ( memLimit > 0 && memLimit < *total ) ? memLimit : *total;
}

void* memAllocHost( size_t byteCount, LWresult* returnResult )
{
    void* result = nullptr;
    CHECK( lwdaDriver().LwMemAllocHost( &result, byteCount ) );
    return result;
}

void* memHostAlloc( size_t byteCount, unsigned int flags, LWresult* returnResult )
{
    void* result = nullptr;
    CHECK( lwdaDriver().LwMemHostAlloc( &result, byteCount, flags ) );
    return result;
}

void memFreeHost( void* p, LWresult* returnResult )
{
    CHECK( lwdaDriver().LwMemFreeHost( p ) );
}

LWdeviceptr memHostGetDevicePointer( void* p, unsigned int flags, LWresult* returnResult )
{
    LWdeviceptr result = 0;
    CHECK( lwdaDriver().LwMemHostGetDevicePointer( &result, p, flags ) );
    return result;
}

unsigned int memHostGetFlags( void* p, LWresult* returnResult )
{
    unsigned int result = 0;
    CHECK( lwdaDriver().LwMemHostGetFlags( &result, p ) );
    return result;
}

void memHostRegister( void* p, size_t byteCount, unsigned int flags, LWresult* returnResult )
{
    CHECK( lwdaDriver().LwMemHostRegister( p, byteCount, flags ) );
}

void memHostUnregister( void* p, LWresult* returnResult )
{
    CHECK( lwdaDriver().LwMemHostUnregister( p ) );
}

void memcpy( LWdeviceptr dst, LWdeviceptr src, size_t byteCount, LWresult* returnResult )
{
    CHECK( lwdaDriver().LwMemcpy( dst, src, byteCount ) );
}

void memcpyAsync( LWdeviceptr dst, LWdeviceptr src, size_t byteCount, const Stream& stream, LWresult* returnResult )
{
    RT_ASSERT_FAIL_MSG( "optix::lwca::memcpyAsync not implemented" );
#if 0
  RT_ASSERT( stream.get() != 0);
  CHECK(lwdaDriver().lwMemcpyAsync( dst, src, byteCount, stream.get() ) );
#endif
}

void memcpyDtoD( LWdeviceptr dstDevice, LWdeviceptr srcDevice, size_t byteCount, LWresult* returnResult )
{
    CHECK( lwdaDriver().LwMemcpyDtoD( dstDevice, srcDevice, byteCount ) );
}

void memcpyDtoH( void* dstHost, LWdeviceptr srcDevice, size_t byteCount, LWresult* returnResult )
{
    CHECK( lwdaDriver().LwMemcpyDtoH( dstHost, srcDevice, byteCount ) );
}

void memcpyHtoD( LWdeviceptr dstDevice, const void* srcHost, size_t byteCount, LWresult* returnResult )
{
    CHECK( lwdaDriver().LwMemcpyHtoD( dstDevice, srcHost, byteCount ) );
}

void memcpyDtoDAsync( LWdeviceptr dstDevice, LWdeviceptr srcDevice, size_t byteCount, const Stream& stream, LWresult* returnResult )
{
    RT_ASSERT( stream.get() != nullptr );
    CHECK( lwdaDriver().LwMemcpyDtoDAsync( dstDevice, srcDevice, byteCount, stream.get() ) );
}

void memcpyDtoHAsync( void* dstHost, LWdeviceptr srcDevice, size_t byteCount, const Stream& stream, LWresult* returnResult )
{
    RT_ASSERT( stream.get() != nullptr );
    CHECK( lwdaDriver().LwMemcpyDtoHAsync( dstHost, srcDevice, byteCount, stream.get() ) );
}

void memcpyHtoDAsync( LWdeviceptr dstDevice, const void* srcHost, size_t byteCount, const Stream& stream, LWresult* returnResult )
{
    RT_ASSERT( stream.get() != nullptr );
    CHECK( lwdaDriver().LwMemcpyHtoDAsync( dstDevice, srcHost, byteCount, stream.get() ) );
}

void memcpy2D( const LWDA_MEMCPY2D* pCopy, LWresult* returnResult )
{
    CHECK( lwdaDriver().LwMemcpy2D( pCopy ) );
}

void memcpy2DUnaligned( const LWDA_MEMCPY2D* pCopy, LWresult* returnResult )
{
    CHECK( lwdaDriver().LwMemcpy2DUnaligned( pCopy ) );
}

void memcpy2DAsync( const LWDA_MEMCPY2D* pCopy, const Stream& stream, LWresult* returnResult )
{
    RT_ASSERT( stream.get() != nullptr );
    CHECK( lwdaDriver().LwMemcpy2DAsync( pCopy, stream.get() ) );
}

void memcpy3D( const LWDA_MEMCPY3D* pCopy, LWresult* returnResult )
{
    CHECK( lwdaDriver().LwMemcpy3D( pCopy ) );
}

void memcpy3DAsync( const LWDA_MEMCPY3D* pCopy, const Stream& stream, LWresult* returnResult )
{
    RT_ASSERT( stream.get() != nullptr );
    CHECK( lwdaDriver().LwMemcpy3DAsync( pCopy, stream.get() ) );
}

void memcpyAtoA( const Array& dstArray, size_t dstOffset, const Array& srcArray, size_t srcOffset, size_t byteCount, LWresult* returnResult )
{
    RT_ASSERT( dstArray.get() != nullptr );
    RT_ASSERT( srcArray.get() != nullptr );
    CHECK( lwdaDriver().LwMemcpyAtoA( dstArray.get(), dstOffset, srcArray.get(), srcOffset, byteCount ) );
}

void memcpyAtoD( LWdeviceptr dstDevice, const Array& srcArray, size_t srcOffset, size_t byteCount, LWresult* returnResult )
{
    RT_ASSERT( srcArray.get() != nullptr );
    CHECK( lwdaDriver().LwMemcpyAtoD( dstDevice, srcArray.get(), srcOffset, byteCount ) );
}

void memcpyAtoH( void* dstHost, const Array& srcArray, size_t srcOffset, size_t byteCount, LWresult* returnResult )
{
    RT_ASSERT( srcArray.get() != nullptr );
    CHECK( lwdaDriver().LwMemcpyAtoH( dstHost, srcArray.get(), srcOffset, byteCount ) );
}

void memcpyDtoA( const Array& dstArray, size_t dstOffset, LWdeviceptr srcDevice, size_t byteCount, LWresult* returnResult )
{
    RT_ASSERT( dstArray.get() != nullptr );
    CHECK( lwdaDriver().LwMemcpyDtoA( dstArray.get(), dstOffset, srcDevice, byteCount ) );
}

void memcpyHtoA( const Array& dstArray, size_t dstOffset, const void* srcHost, size_t byteCount, LWresult* returnResult )
{
    RT_ASSERT( dstArray.get() != nullptr );
    CHECK( lwdaDriver().LwMemcpyHtoA( dstArray.get(), dstOffset, srcHost, byteCount ) );
}

void memcpyAtoHAsync( void* dstHost, const Array& srcArray, size_t srcOffset, size_t byteCount, const Stream& stream, LWresult* returnResult )
{
    RT_ASSERT( srcArray.get() != nullptr );
    RT_ASSERT( stream.get() != nullptr );
    CHECK( lwdaDriver().LwMemcpyAtoHAsync( dstHost, srcArray.get(), srcOffset, byteCount, stream.get() ) );
}

void memcpyHtoAAsync( const Array& dstArray, size_t dstOffset, const void* srcHost, size_t byteCount, const Stream& stream, LWresult* returnResult )
{
    RT_ASSERT( dstArray.get() != nullptr );
    RT_ASSERT( stream.get() != nullptr );
    CHECK( lwdaDriver().LwMemcpyHtoAAsync( dstArray.get(), dstOffset, srcHost, byteCount, stream.get() ) );
}

void memcpyPeer( LWdeviceptr dstDevice, const Context& dstContext, LWdeviceptr srcDevice, const Context& srcContext, size_t byteCount, LWresult* returnResult )
{
    RT_ASSERT( dstContext.get() );
    RT_ASSERT( srcContext.get() );
    CHECK( lwdaDriver().LwMemcpyPeer( dstDevice, dstContext.get(), srcDevice, srcContext.get(), byteCount ) );
}

void memcpy3DPeer( const LWDA_MEMCPY3D_PEER* pCopy, LWresult* returnResult )
{
    CHECK( lwdaDriver().LwMemcpy3DPeer( pCopy ) );
}

void memcpyPeerAsync( LWdeviceptr    dstDevice,
                      const Context& dstContext,
                      LWdeviceptr    srcDevice,
                      const Context& srcContext,
                      size_t         byteCount,
                      const Stream&  stream,
                      LWresult*      returnResult )
{
    RT_ASSERT( dstContext.get() );
    RT_ASSERT( srcContext.get() );
    RT_ASSERT( stream.get() != nullptr );
    CHECK( lwdaDriver().LwMemcpyPeerAsync( dstDevice, dstContext.get(), srcDevice, srcContext.get(), byteCount, stream.get() ) );
}

void memcpy3DPeerAsync( const LWDA_MEMCPY3D_PEER* pCopy, const Stream& stream, LWresult* returnResult )
{
    RT_ASSERT( stream.get() != nullptr );
    CHECK( lwdaDriver().LwMemcpy3DPeerAsync( pCopy, stream.get() ) );
}

void memsetD8( LWdeviceptr dstDevice, unsigned char uc, size_t N, LWresult* returnResult )
{
    CHECK( lwdaDriver().LwMemsetD8( dstDevice, uc, N ) );
}

void memsetD16( LWdeviceptr dstDevice, unsigned short us, size_t N, LWresult* returnResult )
{
    CHECK( lwdaDriver().LwMemsetD16( dstDevice, us, N ) );
}

void memsetD32( LWdeviceptr dstDevice, unsigned int ui, size_t N, LWresult* returnResult )
{
    CHECK( lwdaDriver().LwMemsetD32( dstDevice, ui, N ) );
}

void memsetD8Async( LWdeviceptr dstDevice, unsigned char uc, size_t N, const Stream& stream, LWresult* returnResult )
{
#if 1
    RT_ASSERT( stream.get() != nullptr );
    CHECK( lwdaDriver().LwMemsetD8Async( dstDevice, uc, N, stream.get() ) );
#else
    memsetD8( dstDevice, uc, N, returnResult );
#endif
}

void memsetD16Async( LWdeviceptr dstDevice, unsigned short us, size_t N, const Stream& stream, LWresult* returnResult )
{
    RT_ASSERT_FAIL_MSG( "optix::lwca::memsetD16Async not implemented" );
#if 0
  RT_ASSERT( stream.get() != 0);
  CHECK(lwdaDriver().lwMemsetD16Async( dstDevice, us, N, stream.get()) );
#endif
}

void memsetD32Async( LWdeviceptr dstDevice, unsigned int ui, size_t N, const Stream& stream, LWresult* returnResult )
{
    RT_ASSERT( stream.get() != nullptr );
    CHECK( lwdaDriver().LwMemsetD32Async( dstDevice, ui, N, stream.get() ) );
}

void memsetD2D8( LWdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t width, size_t height, LWresult* returnResult )
{
    CHECK( lwdaDriver().LwMemsetD2D8( dstDevice, dstPitch, uc, width, height ) );
}

void memsetD2D16( LWdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t width, size_t height, LWresult* returnResult )
{
    CHECK( lwdaDriver().LwMemsetD2D16( dstDevice, dstPitch, us, width, height ) );
}

void memsetD2D32( LWdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t width, size_t height, LWresult* returnResult )
{
    CHECK( lwdaDriver().LwMemsetD2D32( dstDevice, dstPitch, ui, width, height ) );
}

void memsetD2D8Async( LWdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t width, size_t height, const Stream& stream, LWresult* returnResult )
{
    RT_ASSERT_FAIL_MSG( "optix::lwca::memsetD2D8Async not implemented" );
#if 0
  RT_ASSERT( stream.get() != 0);
  CHECK(lwdaDriver().lwMemsetD2D8Async( dstDevice, dstPitch, uc, width, height, stream.get()) );
#endif
}

void memsetD2D16Async( LWdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t width, size_t height, const Stream& stream, LWresult* returnResult )
{
    RT_ASSERT_FAIL_MSG( "optix::lwca::memsetD2D16Async not implemented" );
#if 0
  RT_ASSERT( stream.get() != 0);
  CHECK(lwdaDriver().lwMemsetD2D16Async( dstDevice, dstPitch, us, width, height, stream.get() );
#endif
}

void memsetD2D32Async( LWdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t width, size_t height, const Stream& stream, LWresult* returnResult )
{
    RT_ASSERT_FAIL_MSG( "optix::lwca::memsetD2D32Async not implemented" );
#if 0
  RT_ASSERT( stream.get() != 0);
  CHECK(lwdaDriver().lwMemsetD2D32Async( dstDevice, dstPitch, ui,  width, height, stream.get()) );
#endif
}

LWmemGenericAllocationHandle memCreate( size_t allocationSize, LWmemAllocationProp* allocationProps, unsigned long long flags, LWresult* returnResult )
{
    LWmemGenericAllocationHandle result = 0;
    CHECK( lwdaDriver().LwMemCreate( &result, allocationSize, allocationProps, flags ) );
    return result;
}

void memRelease( LWmemGenericAllocationHandle handle, LWresult* returnResult )
{
    CHECK( lwdaDriver().LwMemRelease( handle ) );
}

LWdeviceptr memAddressReserve( size_t size, size_t alignment, LWdeviceptr address, unsigned long long flags, LWresult* returnResult )
{
    LWdeviceptr result = 0;
    CHECK( lwdaDriver().LwMemAddressReserve( &result, size, alignment, address, flags ) );
    return result;
}

void memSetAccess( LWdeviceptr allocationPtr, size_t size, LWmemAccessDesc* accessDescriptor, size_t count, LWresult* returnResult )
{
    CHECK( lwdaDriver().LwMemSetAccess( allocationPtr, size, accessDescriptor, count ) );
}

void memMap( LWdeviceptr ptr, size_t size, size_t offset, LWmemGenericAllocationHandle handle, unsigned long long flags, LWresult* returnResult )
{
    CHECK( lwdaDriver().LwMemMap( ptr, size, offset, handle, flags ) );
}

void memMapArrayAsync( LWarrayMapInfo* mapInfoList, const unsigned int count, const Stream& stream, LWresult* returnResult )
{
    CHECK( lwdaDriver().LwMemMapArrayAsync( mapInfoList, count, stream.get() ) );
}

LWDA_ARRAY_SPARSE_PROPERTIES mipmappedArrayGetSparseProperties( LWmipmappedArray mipmap, LWresult* returnResult )
{
    LWDA_ARRAY_SPARSE_PROPERTIES result;
    CHECK( lwdaDriver().LwMipmappedArrayGetSparseProperties( &result, mipmap ) );
    return result;
}

}  // namespace lwca
}  // namespace optix
