// Copyright (c) 2017, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, LW_DEVICE_ATTRIBUTE_EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <LWCA/Device.h>

#include <LWCA/ComputeCapability.h>
#include <LWCA/ErrorCheck.h>
#include <corelib/system/LwdaDriver.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/system/Knobs.h>

#include <vector>

using namespace optix::lwca;
using namespace corelib;

namespace {
// clang-format off
Knob<size_t> k_maximumDeviceMemory( RT_DSTRING( "lwca.maximumDeviceMemory" ), 0, RT_DSTRING( "Set a limit on the visible device memory. Default is 0, which means use what the driver reports." ) );
// clang-format on
}

static const LWdevice ILLEGAL_DEVICE = 0xffff;

Device::Device()
    : m_device( ILLEGAL_DEVICE )
{
}

Device::Device( LWdevice device )
    : m_device( device )
{
}

LWdevice Device::get()
{
    return m_device;
}

const LWdevice Device::get() const
{
    return m_device;
}

bool Device::isValid() const
{
    return m_device != ILLEGAL_DEVICE;
}

Device Device::get( int ordinal, LWresult* returnResult )
{
    LWdevice device = ILLEGAL_DEVICE;
    CHECK( lwdaDriver().LwDeviceGet( &device, ordinal ) );
    return Device( device );
}

int Device::getCount( LWresult* returnResult )
{
    int count = 0;
    CHECK( lwdaDriver().LwDeviceGetCount( &count ) );
    return count;
}

ComputeCapability Device::computeCapability( LWresult* returnResult ) const
{
    RT_ASSERT( m_device != ILLEGAL_DEVICE );
    int major = 0, minor = 0;
    CHECK( lwdaDriver().LwDeviceComputeCapability( &major, &minor, m_device ) );
    return ComputeCapability( major, minor );
}

int Device::getAttribute( LWdevice_attribute attrib, LWresult* returnResult ) const
{
    RT_ASSERT( m_device != ILLEGAL_DEVICE );
    int result = 0;
    CHECK( lwdaDriver().LwDeviceGetAttribute( &result, attrib, m_device ) );
    return result;
}

std::string Device::getName( LWresult* returnResult ) const
{
    RT_ASSERT( m_device != ILLEGAL_DEVICE );
    const int MAXLEN = 256;
    char      name[MAXLEN];
    name[0] = '\0';
    CHECK( lwdaDriver().LwDeviceGetName( name, MAXLEN, m_device ) );
    return std::string( name );
}

size_t Device::totalMem( LWresult* returnResult ) const
{
    RT_ASSERT( m_device != ILLEGAL_DEVICE );
    size_t totalMem = 0;
    CHECK( lwdaDriver().LwDeviceTotalMem( &totalMem, m_device ) );
    const size_t memLimit = k_maximumDeviceMemory.get();
    return ( memLimit > 0 && memLimit < totalMem ) ? memLimit : totalMem;
}

Device Device::getByPCIBusId( const std::string& id, LWresult* returnResult )
{
    LWdevice result = ILLEGAL_DEVICE;
    CHECK( lwdaDriver().LwDeviceGetByPCIBusId( &result, id.c_str() ) );
    return Device( result );
}

std::string Device::getPCIBusId( LWresult* returnResult ) const
{
    RT_ASSERT( m_device != ILLEGAL_DEVICE );
    const int MAX_ID_LENGTH = 256;
    char      result[MAX_ID_LENGTH];
    CHECK( lwdaDriver().LwDeviceGetPCIBusId( result, MAX_ID_LENGTH, m_device ) );
    return std::string( result );
}

void Device::getLuidAndNodeMask( char* luid, unsigned int* deviceNodeMask, LWresult* returnResult ) const
{
    RT_ASSERT( m_device != ILLEGAL_DEVICE );
    const int LUID_LENGTH = 8;
    CHECK( lwdaDriver().LwDeviceGetLuid( luid, deviceNodeMask, m_device ) );
}

bool Device::canAccessPeer( const Device& peerDev, LWresult* returnResult ) const
{
    RT_ASSERT( m_device != ILLEGAL_DEVICE );
    int canAccessPeer = 0;
    CHECK( lwdaDriver().LwDeviceCanAccessPeer( &canAccessPeer, m_device, peerDev.get() ) );
    return canAccessPeer != 0;
}


/**< Maximum number of threads per block */
int Device::MAX_THREADS_PER_BLOCK() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK );
}

/**< Maximum block dimension X */
int Device::MAX_BLOCK_DIM_X() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X );
}

/**< Maximum block dimension Y */
int Device::MAX_BLOCK_DIM_Y() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y );
}

/**< Maximum block dimension Z */
int Device::MAX_BLOCK_DIM_Z() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z );
}

/**< Maximum grid dimension X */
int Device::MAX_GRID_DIM_X() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X );
}

/**< Maximum grid dimension Y */
int Device::MAX_GRID_DIM_Y() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y );
}

/**< Maximum grid dimension Z */
int Device::MAX_GRID_DIM_Z() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z );
}

/**< Maximum shared memory available per block in bytes */
int Device::MAX_SHARED_MEMORY_PER_BLOCK() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK );
}

/**< Deprecated, use LW_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK */
int Device::SHARED_MEMORY_PER_BLOCK() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK );
}

/**< Memory available on device for __constant__ variables in a LWCA C kernel in bytes */
int Device::TOTAL_CONSTANT_MEMORY() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY );
}

/**< Warp size in threads */
int Device::WARP_SIZE() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_WARP_SIZE );
}

/**< Maximum pitch in bytes allowed by memory copies */
int Device::MAX_PITCH() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAX_PITCH );
}

/**< Maximum number of 32-bit registers available per block */
int Device::MAX_REGISTERS_PER_BLOCK() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK );
}

/**< Deprecated, use LW_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK */
int Device::REGISTERS_PER_BLOCK() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK );
}

/**< Peak clock frequency in kilohertz */
int Device::CLOCK_RATE() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_CLOCK_RATE );
}

/**< Alignment requirement for textures */
int Device::TEXTURE_ALIGNMENT() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT );
}

/**< Device can possibly copy memory and execute a kernel conlwrrently. Deprecated. Use instead LW_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT. */
int Device::GPU_OVERLAP() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_GPU_OVERLAP );
}

/**< Number of multiprocessors on device */
int Device::MULTIPROCESSOR_COUNT() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT );
}

/**< Specifies whether there is a run time limit on kernels */
int Device::KERNEL_EXEC_TIMEOUT() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT );
}

/**< Device is integrated with host memory */
int Device::INTEGRATED() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_INTEGRATED );
}

/**< Device can map host memory into LWCA address space */
int Device::CAN_MAP_HOST_MEMORY() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY );
}

/**< Compute mode (See ::LWcomputemode for details) */
int Device::COMPUTE_MODE() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_COMPUTE_MODE );
}

/**< Maximum 1D texture width */
int Device::MAXIMUM_TEXTURE1D_WIDTH() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH );
}

/**< Maximum 2D texture width */
int Device::MAXIMUM_TEXTURE2D_WIDTH() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH );
}

/**< Maximum 2D texture height */
int Device::MAXIMUM_TEXTURE2D_HEIGHT() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT );
}

/**< Maximum 3D texture width */
int Device::MAXIMUM_TEXTURE3D_WIDTH() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH );
}

/**< Maximum 3D texture height */
int Device::MAXIMUM_TEXTURE3D_HEIGHT() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT );
}

/**< Maximum 3D texture depth */
int Device::MAXIMUM_TEXTURE3D_DEPTH() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH );
}

/**< Maximum 2D layered texture width */
int Device::MAXIMUM_TEXTURE2D_LAYERED_WIDTH() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH );
}

/**< Maximum 2D layered texture height */
int Device::MAXIMUM_TEXTURE2D_LAYERED_HEIGHT() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT );
}

/**< Maximum layers in a 2D layered texture */
int Device::MAXIMUM_TEXTURE2D_LAYERED_LAYERS() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS );
}

/**< Deprecated, use LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH */
int Device::MAXIMUM_TEXTURE2D_ARRAY_WIDTH() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH );
}

/**< Deprecated, use LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT */
int Device::MAXIMUM_TEXTURE2D_ARRAY_HEIGHT() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT );
}

/**< Deprecated, use LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS */
int Device::MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES );
}

/**< Alignment requirement for surfaces */
int Device::SURFACE_ALIGNMENT() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT );
}

/**< Device can possibly execute multiple kernels conlwrrently */
int Device::CONLWRRENT_KERNELS() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_CONLWRRENT_KERNELS );
}

/**< Device has ECC support enabled */
int Device::ECC_ENABLED() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_ECC_ENABLED );
}

/**< PCI bus ID of the device */
int Device::PCI_BUS_ID() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_PCI_BUS_ID );
}

/**< PCI device ID of the device */
int Device::PCI_DEVICE_ID() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_PCI_DEVICE_ID );
}

/**< Device is using TCC driver model */
int Device::TCC_DRIVER() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_TCC_DRIVER );
}

/**< Peak memory clock frequency in kilohertz */
int Device::MEMORY_CLOCK_RATE() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE );
}

/**< Global memory bus width in bits */
int Device::GLOBAL_MEMORY_BUS_WIDTH() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH );
}

/**< Size of L2 cache in bytes */
int Device::L2_CACHE_SIZE() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_L2_CACHE_SIZE );
}

/**< Maximum resident threads per multiprocessor */
int Device::MAX_THREADS_PER_MULTIPROCESSOR() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR );
}

/**< Number of asynchronous engines */
int Device::ASYNC_ENGINE_COUNT() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT );
}

/**< Device shares a unified address space with the host */
int Device::UNIFIED_ADDRESSING() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING );
}

/**< Maximum 1D layered texture width */
int Device::MAXIMUM_TEXTURE1D_LAYERED_WIDTH() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH );
}

/**< Maximum layers in a 1D layered texture */
int Device::MAXIMUM_TEXTURE1D_LAYERED_LAYERS() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS );
}

/**< Deprecated, do not use. */
int Device::CAN_TEX2D_GATHER() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER );
}

/**< Maximum 2D texture width if LWDA_ARRAY3D_TEXTURE_GATHER is set */
int Device::MAXIMUM_TEXTURE2D_GATHER_WIDTH() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH );
}

/**< Maximum 2D texture height if LWDA_ARRAY3D_TEXTURE_GATHER is set */
int Device::MAXIMUM_TEXTURE2D_GATHER_HEIGHT() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT );
}

/**< Alternate maximum 3D texture width */
int Device::MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE );
}

/**< Alternate maximum 3D texture height */
int Device::MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE );
}

/**< Alternate maximum 3D texture depth */
int Device::MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE );
}

/**< PCI domain ID of the device */
int Device::PCI_DOMAIN_ID() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID );
}

/**< Pitch alignment requirement for textures */
int Device::TEXTURE_PITCH_ALIGNMENT() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT );
}

/**< Maximum lwbemap texture width/height */
int Device::MAXIMUM_TEXTURELWBEMAP_WIDTH() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURELWBEMAP_WIDTH );
}

/**< Maximum lwbemap layered texture width/height */
int Device::MAXIMUM_TEXTURELWBEMAP_LAYERED_WIDTH() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURELWBEMAP_LAYERED_WIDTH );
}

/**< Maximum layers in a lwbemap layered texture */
int Device::MAXIMUM_TEXTURELWBEMAP_LAYERED_LAYERS() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURELWBEMAP_LAYERED_LAYERS );
}

/**< Maximum 1D surface width */
int Device::MAXIMUM_SURFACE1D_WIDTH() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH );
}

/**< Maximum 2D surface width */
int Device::MAXIMUM_SURFACE2D_WIDTH() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH );
}

/**< Maximum 2D surface height */
int Device::MAXIMUM_SURFACE2D_HEIGHT() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT );
}

/**< Maximum 3D surface width */
int Device::MAXIMUM_SURFACE3D_WIDTH() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH );
}

/**< Maximum 3D surface height */
int Device::MAXIMUM_SURFACE3D_HEIGHT() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT );
}

/**< Maximum 3D surface depth */
int Device::MAXIMUM_SURFACE3D_DEPTH() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH );
}

/**< Maximum 1D layered surface width */
int Device::MAXIMUM_SURFACE1D_LAYERED_WIDTH() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH );
}

/**< Maximum layers in a 1D layered surface */
int Device::MAXIMUM_SURFACE1D_LAYERED_LAYERS() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS );
}

/**< Maximum 2D layered surface width */
int Device::MAXIMUM_SURFACE2D_LAYERED_WIDTH() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH );
}

/**< Maximum 2D layered surface height */
int Device::MAXIMUM_SURFACE2D_LAYERED_HEIGHT() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT );
}

/**< Maximum layers in a 2D layered surface */
int Device::MAXIMUM_SURFACE2D_LAYERED_LAYERS() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS );
}

/**< Maximum lwbemap surface width */
int Device::MAXIMUM_SURFACELWBEMAP_WIDTH() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACELWBEMAP_WIDTH );
}

/**< Maximum lwbemap layered surface width */
int Device::MAXIMUM_SURFACELWBEMAP_LAYERED_WIDTH() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACELWBEMAP_LAYERED_WIDTH );
}

/**< Maximum layers in a lwbemap layered surface */
int Device::MAXIMUM_SURFACELWBEMAP_LAYERED_LAYERS() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_SURFACELWBEMAP_LAYERED_LAYERS );
}

/**< Maximum 1D linear texture width */
int Device::MAXIMUM_TEXTURE1D_LINEAR_WIDTH() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH );
}

/**< Maximum 2D linear texture width */
int Device::MAXIMUM_TEXTURE2D_LINEAR_WIDTH() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH );
}

/**< Maximum 2D linear texture height */
int Device::MAXIMUM_TEXTURE2D_LINEAR_HEIGHT() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT );
}

/**< Maximum 2D linear texture pitch in bytes */
int Device::MAXIMUM_TEXTURE2D_LINEAR_PITCH() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH );
}

/**< Maximum mipmapped 2D texture width */
int Device::MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH );
}

/**< Maximum mipmapped 2D texture height */
int Device::MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT );
}

/**< Major compute capability version number */
int Device::COMPUTE_CAPABILITY_MAJOR() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR );
}

/**< Minor compute capability version number */
int Device::COMPUTE_CAPABILITY_MINOR() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR );
}

/**< Maximum mipmapped 1D texture width */
int Device::MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH );
}

/**< Device supports stream priorities */
int Device::STREAM_PRIORITIES_SUPPORTED() const
{
    return getAttribute( LW_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED );
}


void Device::GLGetDevices( unsigned int* pDeviceCount, Device* pDevices, unsigned int lwdaDeviceCount, LWGLDeviceList deviceList, LWresult* returnResult /*= 0*/ )
{
    std::vector<LWdevice> result( lwdaDeviceCount );

    // Can't use CHECK, because if there is a failure the value of *pDeviceCount is undefined.
    LWresult err = lwdaDriver().LwGLGetDevices( pDeviceCount, &result[0], lwdaDeviceCount, deviceList );

    // Always initialize the return result if it was requested.
    if( returnResult )
        *returnResult = err;

    // If successful, write back the devices.
    if( err == LWDA_SUCCESS )
    {
        for( unsigned i = 0; i < *pDeviceCount; ++i )
            pDevices[i] = Device( result[i] );
        return;
    }

    // If there was no returnResult passed, throw an exception.
    // Otherwise, error handling is done by the caller.
    if( !returnResult )
        throw prodlib::LwdaError( RT_EXCEPTION_INFO, "lwGLGetDevices", err );
}

#ifdef _WIN32
Device Device::lwWGLGetDevice( HGPULW hGpu, LWresult* returnResult /*= 0*/ )
{
    LWdevice result;
    CHECK( lwdaDriver().LwWGLGetDevice( &result, hGpu ) );
    return Device( result );
}
#endif  // _WIN32

bool Device::operator==( const Device& other ) const
{
    return this->m_device == other.m_device;
}
