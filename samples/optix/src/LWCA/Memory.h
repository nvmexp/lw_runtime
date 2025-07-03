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

#pragma once

#include <corelib/system/LwdaDriver.h>

#include <lwca.h>

namespace optix {
namespace lwca {

class Array;
class Context;
class Stream;

/*
     * Device memory
     */

// Allocates device memory.
LWdeviceptr memAlloc( size_t byteCount, LWresult* returnResult = nullptr );

// Allocates pitched device memory. Returns pitch in pPitch.
LWdeviceptr memAllocPitch( size_t* pPitch, size_t widthInBytes, size_t height, unsigned int elementSizeBytes, LWresult* returnResult = nullptr );

// Frees device memory.
void memFree( LWdeviceptr dptr, LWresult* returnResult = nullptr );

// Get information on memory allocations.
void memGetAddressRange( LWdeviceptr* pbase, size_t* psize, LWdeviceptr dptr, LWresult* returnResult = nullptr );

// Gets free and total memory.
void memGetInfo( size_t* free, size_t* total, LWresult* returnResult = nullptr );

/*
     * Host memory
     */

// Allocates page-locked host memory.
void* memAllocHost( size_t byteCount, LWresult* returnResult = nullptr );

// Allocates page-locked host memory.
void* memHostAlloc( size_t byteCount, unsigned int flags, LWresult* returnResult = nullptr );

// Frees page-locked host memory.
void memFreeHost( void* p, LWresult* returnResult = nullptr );

// Passes back device pointer of mapped pinned memory.
LWdeviceptr memHostGetDevicePointer( void* p, unsigned int flags, LWresult* returnResult = nullptr );

// Passes back flags that were used for a pinned allocation.
unsigned int memHostGetFlags( void* p, LWresult* returnResult = nullptr );

// Registers an existing host memory range for use by LWCA.
void memHostRegister( void* p, size_t byteCount, unsigned int flags, LWresult* returnResult = nullptr );

// Unregisters a memory range that was registered with lwMemHostRegister.
void memHostUnregister( void* p, LWresult* returnResult = nullptr );

/*
     * Memory copy
     */

// UVA copies
void memcpy( LWdeviceptr dst, LWdeviceptr src, size_t byteCount, LWresult* returnResult = nullptr );
void memcpyAsync( LWdeviceptr dst, LWdeviceptr src, size_t byteCount, const Stream& stream, LWresult* returnResult = nullptr );

// Directional copies
void memcpyDtoD( LWdeviceptr dstDevice, LWdeviceptr srcDevice, size_t byteCount, LWresult* returnResult = nullptr );
void memcpyDtoH( void* dstHost, LWdeviceptr srcDevice, size_t byteCount, LWresult* returnResult = nullptr );
void memcpyHtoD( LWdeviceptr dstDevice, const void* srcHost, size_t byteCount, LWresult* returnResult = nullptr );
void memcpyDtoDAsync( LWdeviceptr dstDevice, LWdeviceptr srcDevice, size_t byteCount, const Stream& stream, LWresult* returnResult = nullptr );
void memcpyDtoHAsync( void* dstHost, LWdeviceptr srcDevice, size_t byteCount, const Stream& stream, LWresult* returnResult = nullptr );
void memcpyHtoDAsync( LWdeviceptr dstDevice, const void* srcHost, size_t byteCount, const Stream& stream, LWresult* returnResult = nullptr );

// Copies memory for 2D arrays.
void memcpy2D( const LWDA_MEMCPY2D* pCopy, LWresult* returnResult = nullptr );
void memcpy2DUnaligned( const LWDA_MEMCPY2D* pCopy, LWresult* returnResult = nullptr );
void memcpy2DAsync( const LWDA_MEMCPY2D* pCopy, const Stream& stream, LWresult* returnResult = nullptr );

// Copies memory for 3D arrays.
void memcpy3D( const LWDA_MEMCPY3D* pCopy, LWresult* returnResult = nullptr );
void memcpy3DAsync( const LWDA_MEMCPY3D* pCopy, const Stream& stream, LWresult* returnResult = nullptr );

// Copies memory for Array objects
void memcpyAtoA( const Array& dstArray, size_t dstOffset, const Array& srcArray, size_t srcOffset, size_t byteCount, LWresult* returnResult = nullptr );
void memcpyAtoD( LWdeviceptr dstDevice, const Array& srcArray, size_t srcOffset, size_t byteCount, LWresult* returnResult = nullptr );
void memcpyAtoH( void* dstHost, const Array& srcArray, size_t srcOffset, size_t byteCount, LWresult* returnResult = nullptr );
void memcpyDtoA( const Array& dstArray, size_t dstOffset, LWdeviceptr srcDevice, size_t byteCount, LWresult* returnResult = nullptr );
void memcpyHtoA( const Array& dstArray, size_t dstOffset, const void* srcHost, size_t byteCount, LWresult* returnResult = nullptr );
void memcpyAtoHAsync( void* dstHost, const Array& srcArray, size_t srcOffset, size_t byteCount, const Stream& stream, LWresult* returnResult = nullptr );
void memcpyHtoAAsync( const Array& dstArray, size_t dstOffset, const void* srcHost, size_t byteCount, const Stream& stream, LWresult* returnResult = nullptr );

// Copies device memory between two contexts.
void memcpyPeer( LWdeviceptr    dstDevice,
                 const Context& dstContext,
                 LWdeviceptr    srcDevice,
                 const Context& srcContext,
                 size_t         byteCount,
                 LWresult*      returnResult = nullptr );
void memcpy3DPeer( const LWDA_MEMCPY3D_PEER* pCopy, LWresult* returnResult = nullptr );
void memcpyPeerAsync( LWdeviceptr    dstDevice,
                      const Context& dstContext,
                      LWdeviceptr    srcDevice,
                      const Context& srcContext,
                      size_t         byteCount,
                      const Stream&  stream,
                      LWresult*      returnResult = nullptr );
void memcpy3DPeerAsync( const LWDA_MEMCPY3D_PEER* pCopy, const Stream& stream, LWresult* returnResult = nullptr );

/*
     * Memset
     */
// Initializes device memory.
void memsetD8( LWdeviceptr dstDevice, unsigned char uc, size_t N, LWresult* returnResult = nullptr );
void memsetD16( LWdeviceptr dstDevice, unsigned short us, size_t N, LWresult* returnResult = nullptr );
void memsetD32( LWdeviceptr dstDevice, unsigned int ui, size_t N, LWresult* returnResult = nullptr );
void memsetD8Async( LWdeviceptr dstDevice, unsigned char uc, size_t N, const Stream& stream, LWresult* returnResult = nullptr );
void memsetD16Async( LWdeviceptr dstDevice, unsigned short us, size_t N, const Stream& stream, LWresult* returnResult = nullptr );
void memsetD32Async( LWdeviceptr dstDevice, unsigned int ui, size_t N, const Stream& stream, LWresult* returnResult = nullptr );

void memsetD2D8( LWdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t width, size_t height, LWresult* returnResult = nullptr );
void memsetD2D16( LWdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t width, size_t height, LWresult* returnResult = nullptr );
void memsetD2D32( LWdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t width, size_t height, LWresult* returnResult = nullptr );
void memsetD2D8Async( LWdeviceptr   dstDevice,
                      size_t        dstPitch,
                      unsigned char uc,
                      size_t        width,
                      size_t        height,
                      const Stream& stream,
                      LWresult*     returnResult = nullptr );
void memsetD2D16Async( LWdeviceptr    dstDevice,
                       size_t         dstPitch,
                       unsigned short us,
                       size_t         width,
                       size_t         height,
                       const Stream&  stream,
                       LWresult*      returnResult = nullptr );
void memsetD2D32Async( LWdeviceptr   dstDevice,
                       size_t        dstPitch,
                       unsigned int  ui,
                       size_t        width,
                       size_t        height,
                       const Stream& stream,
                       LWresult*     returnResult = nullptr );

/*
 * Generic memory allocation
 */
LWmemGenericAllocationHandle memCreate( size_t               allocationSize,
                                        LWmemAllocationProp* allocationProps,
                                        unsigned long long   flags,
                                        LWresult*            returnResult = nullptr );
void memRelease( LWmemGenericAllocationHandle handle, LWresult* returnResult = nullptr );
void memSetAccess( LWdeviceptr allocationPtr, size_t size, LWmemAccessDesc* accessDescriptor, size_t count, LWresult* returnResult = nullptr );
LWdeviceptr memAddressReserve( size_t size, size_t alignment, LWdeviceptr address, unsigned long long flags, LWresult* returnResult = nullptr );
void memMap( LWdeviceptr ptr, size_t size, size_t offset, LWmemGenericAllocationHandle handle, unsigned long long flags, LWresult* returnResult = nullptr );

/*
 * Sparse textures
 */
void memMapArrayAsync( LWarrayMapInfo* mapInfoList, unsigned int count, const Stream& stream, LWresult* returnResult = nullptr );

LWDA_ARRAY_SPARSE_PROPERTIES mipmappedArrayGetSparseProperties( LWmipmappedArray mipmap, LWresult* returnResult = nullptr );

}  // namespace lwca
}  // namespace optix
