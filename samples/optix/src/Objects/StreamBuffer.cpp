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

#include <Objects/StreamBuffer.h>

#include <Objects/StreamBufferKernels_ptx_bin.h>  // generated from .lw

#include <LWCA/Function.h>
#include <LWCA/Memory.h>
#include <LWCA/Module.h>
#include <Context/Context.h>
#include <Context/ObjectManager.h>
#include <Device/LWDADevice.h>
#include <Device/DeviceManager.h>
#include <Memory/MemoryManager.h>
#include <Objects/Buffer.h>

#include <Exceptions/AlreadyMapped.h>
#include <corelib/math/MathUtil.h>
#include <prodlib/exceptions/LwdaError.h>
#include <prodlib/exceptions/IlwalidValue.h>

#include <algorithm>
#include <cstring>
#include <iostream>

using namespace optix;
using namespace prodlib;
using namespace corelib;

StreamBuffer::StreamBuffer( Context* context )
    : ManagedObject( context, RT_OBJECT_STREAM_BUFFER )
    , m_update_ready( false )
{
    m_id = context->getObjectManager()->registerObject( this );

    // Allocate empty device buffers to allow map() anytime after creation
    alloc();
}

StreamBuffer::~StreamBuffer()
{
    RT_ASSERT_MSG( m_linkedPointers.empty(), "References remain to stream buffer remain after destroy" );
    if( m_lwda_module.get() )
    {
        m_lwda_module.unload();
    }
}

void StreamBuffer::detachFromParents()
{
    // Lwrrently there are no linked pointers to stream buffers, but
    // this function is included for a consistent design pattern for
    // ManagtedObjects. Revisit this function if LinkedPtr is used.
}

void StreamBuffer::preSetActiveDevices( const DeviceSet& removedDevices )
{

    // If the primary device is going to be removed, then remove the MBuffers and kernel module
    // that we use for aclwmulation.

    LWDADevice* primaryDevice = m_context->getDeviceManager()->primaryLWDADevice();
    if( removedDevices.isSet( primaryDevice ) )
    {

        if( m_lwda_module.get() )
        {
            m_lwda_module.unload();
        }

        RT_ASSERT( m_aclwm_storage_host == nullptr );
        m_stream_storage_device.reset();
        m_aclwm_storage_device.reset();
    }
}

void StreamBuffer::postSetActiveDevices( const DeviceSet& removedDevices )
{
    alloc();
}

int StreamBuffer::getId() const
{
    RT_ASSERT( m_id != nullptr );
    return *m_id;
}

void StreamBuffer::setFormat( RTformat fmt )
{
    if( fmt != RT_FORMAT_UNSIGNED_BYTE4 )
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Invalid format for stream buffer" );
    m_format   = fmt;
    m_elemsize = 4;
    alloc();
}

RTformat StreamBuffer::getFormat()
{
    return m_format;
}

void StreamBuffer::setElementSize( RTsize sz )
{
    if( m_format != RT_FORMAT_USER )
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Cannot set element size of non user-typed buffer" );
    m_elemsize = sz;
    alloc();
}

RTsize StreamBuffer::getElementSize()
{
    return m_elemsize;
}

void StreamBuffer::setSize1D( RTsize w )
{
    throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Only 2D stream buffers are lwrrently supported" );

    m_width  = w;
    m_height = 1;
    m_depth  = 1;
    m_ndims  = 1;
    alloc();
}

void StreamBuffer::setSize2D( RTsize w, RTsize h )
{
    m_width  = w;
    m_height = h;
    m_depth  = 1;
    m_ndims  = 2;
    alloc();
}

void StreamBuffer::setSize3D( RTsize w, RTsize h, RTsize d )
{
    throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Only 2D stream buffers are lwrrently supported" );

    m_width  = w;
    m_height = h;
    m_depth  = d;
    m_ndims  = 3;
    alloc();
}

void StreamBuffer::setMipLevelCount( unsigned int levels )
{
    m_levels = levels;
    alloc();
}

RTsize StreamBuffer::getWidth()
{
    return m_width;
}

RTsize StreamBuffer::getHeight()
{
    return m_height;
}

RTsize StreamBuffer::getDepth()
{
    return m_depth;
}

RTsize StreamBuffer::getLevelWidth( unsigned int level )
{
    if( level != 0 )
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Only 2D stream buffers are lwrrently supported" );

    return m_width;
}

RTsize StreamBuffer::getLevelHeight( unsigned int level )
{
    if( level != 0 )
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Only 2D stream buffers are lwrrently supported" );
    return m_height;
}

RTsize StreamBuffer::getLevelDepth( unsigned int level )
{
    if( level != 0 )
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Only 2D stream buffers are lwrrently supported" );

    return m_depth;
}

unsigned int StreamBuffer::getMipLevelCount()
{
    return m_levels;
}


void StreamBuffer::setSize( int dimensionality, const RTsize* sz, unsigned int levels )
{
    if( dimensionality != 2 )
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Only 2D stream buffers are lwrrently supported" );

    if( dimensionality > 0 )
        m_width = sz[0];
    if( dimensionality > 1 )
        m_height = sz[1];
    if( dimensionality > 2 )
        m_depth = sz[2];
    m_ndims     = dimensionality;
    m_levels    = levels;
    alloc();
}

void StreamBuffer::getSize( int dimensionality, RTsize* sz )
{
    if( dimensionality > 0 )
        sz[0] = m_width;
    if( dimensionality > 1 )
        sz[1] = m_height;
    if( dimensionality > 2 )
        sz[2] = m_depth;
}

int StreamBuffer::getDimensionality()
{
    return m_ndims;
}

void StreamBuffer::bindSource( Buffer* source )
{
    if( source )
    {
        if( source->getFormat() != RT_FORMAT_FLOAT4 && source->getFormat() != RT_FORMAT_FLOAT3 )
            throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Unsupported stream source format" );
    }

    m_source = source;
}

Buffer* StreamBuffer::getSource()
{
    return m_source;
}

void* StreamBuffer::map( unsigned int level )
{
    if( level != 0 )
    {
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Mipmap stream buffers are not lwrrently supported" );
    }

    if( m_stream_is_mapped )
    {
        throw AlreadyMapped( RT_EXCEPTION_INFO, "StreamBuffer is already mapped" );
    }

    m_mutex.lock();
    m_update_ready     = false;
    m_stream_is_mapped = true;

    // Note: cannot map to host, or otherwise use the memory manager, in this function, which
    // may be called from the vca "client" thread while the vca "worker" thread is running.
    // Rely on a cached copy of the data instead.

    // If cache is empty, can't access its first element, but need to return a
    // non-null pointer anyway, so return a bogus one.
    void* host_ptr = m_stream_storage_host_cache.empty() ? (void*)this : &m_stream_storage_host_cache[0];

    return host_ptr;
}

void StreamBuffer::unmap( unsigned int level )
{
    if( level != 0 )
    {
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Mipmap stream buffers are not lwrrently supported" );
    }

    if( !m_stream_is_mapped )
    {
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "StreamBuffer is not mapped." );
    }

    m_stream_is_mapped = false;

    m_mutex.unlock();
}

void* StreamBuffer::map_aclwm( unsigned int level )
{
    if( level != 0 )
    {
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Mipmap stream buffers are not lwrrently supported" );
    }

    if( m_aclwm_storage_host )
    {
        throw AlreadyMapped( RT_EXCEPTION_INFO, "Buffer is already mapped" );
    }

    m_mutex.lock();

    RT_ASSERT( m_aclwm_storage_device );
    MemoryManager* mm = m_context->getMemoryManager();
    RT_ASSERT( !mm->isMappedToHost( m_aclwm_storage_device ) );
    m_aclwm_storage_host = mm->mapToHost( m_aclwm_storage_device, MAP_READ );

    return m_aclwm_storage_host;
}

void StreamBuffer::unmap_aclwm( unsigned int level )
{
    if( level != 0 )
    {
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Mipmap stream buffers are not lwrrently supported" );
    }

    if( !m_aclwm_storage_host )
    {
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Buffer is not mapped." );
    }

    m_aclwm_storage_host = nullptr;
    MemoryManager* mm    = m_context->getMemoryManager();
    RT_ASSERT( mm->isMappedToHost( m_aclwm_storage_device ) );
    mm->unmapFromHost( m_aclwm_storage_device );

    m_mutex.unlock();
}

void StreamBuffer::aclwmulateOnDevice( const LWDADevice* device,
                                       const float*      src_d,
                                       float*            aclwm_d,
                                       unsigned char*    output_d,
                                       int               npixels,
                                       int               nSrcChannels,
                                       int               nOutputChannels,
                                       int               pass,
                                       float             gamma )
{
    // Load the module lazily using the driver API
    if( !m_lwda_module.get() )
    {
        m_lwda_module   = lwca::Module::loadData( data::getStreamBufferKernelsSources()[1] );
        m_lwda_function = m_lwda_module.getFunction( "aclwmulateKernel" );
    }

    // Launch

    const int   block_size  = 512;
    const int   block_count = idivCeil( npixels, block_size );
    const float weight_src  = 1.0f / float( pass + 1 );
    const float weight_dst  = float( pass ) / float( pass + 1 );
    const void* params[]    = {&src_d,           &aclwm_d,    &output_d,   &npixels, &nSrcChannels,
                            &nOutputChannels, &weight_src, &weight_dst, &gamma};

    m_lwda_function.launchKernel( block_count, 1, 1, block_size, 1, 1, /*shmem bytes*/ 0, device->primaryStream(),
                                  const_cast<void**>( params ), /* extra*/ nullptr );
}

void StreamBuffer::fillFromSource( unsigned int max_subframes )
{
    std::lock_guard<std::mutex> lock( m_mutex );

    m_max_subframes = max_subframes;
    m_update_ready  = true;
    alloc();

    if( !m_source || !m_stream_storage_device )
        return;

    if( m_source->getMipLevelCount() != 1 )
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Mipmap stream buffers are not lwrrently supported" );

    // Accumulate and tonemap on primary device.
    LWDADevice* lwdaDevice = m_context->getDeviceManager()->primaryLWDADevice();
    RT_ASSERT( lwdaDevice );
    lwdaDevice->makeLwrrent();

    MemoryManager* mm = m_context->getMemoryManager();

    // This call pair is needed to make sure pinning works
    mm->syncAllMemoryBeforeLaunch();
    mm->releaseMemoryAfterLaunch();

    unsigned char* stream_device = (unsigned char*)( mm->pinToDevice( m_stream_storage_device, lwdaDevice ) );

    if( m_source->getWidth() != m_width ||    // make
        m_source->getHeight() != m_height ||  // James
        m_source->getDepth() != m_depth )     // happy :-)
    {
        lwca::memsetD8( (LWdeviceptr)stream_device, 0u, sizeBytes() );
        mm->unpinFromDevice( m_stream_storage_device, lwdaDevice );
        resetAclwm();
        return;
    }

    float* aclwm_device = (float*)( mm->pinToDevice( m_aclwm_storage_device, lwdaDevice ) );
    RT_ASSERT( stream_device && aclwm_device );

    // Clear aclwm buffer on first pass
    const size_t sizeOfAclwmStorageInBytes = getSrcNcomp() * sizeElems() * sizeof( float );
    if( m_naclwm == 0 )
    {
        lwca::memsetD8( (LWdeviceptr)aclwm_device, 0u, sizeOfAclwmStorageInBytes );
    }

    // Note: assuming here that the primary device is always a valid target for pinning, even for a
    // zero-copy source buffer.
    const float* src_device = (const float*)( mm->pinToDevice( m_source->getMBuffer(), lwdaDevice ) );
    RT_ASSERT( src_device );
    aclwmulateOnDevice( lwdaDevice, src_device, aclwm_device, stream_device, sizeElems(), getSrcNcomp(), m_elemsize,
                        m_naclwm, stream_attrib_gamma );

    // Copy stream data back to host so it can be safely mapped from another thread.
    RT_ASSERT( sizeBytes() == m_stream_storage_host_cache.size() );
    lwca::memcpyDtoH( &m_stream_storage_host_cache[0], (LWdeviceptr)stream_device, sizeBytes() );

    mm->unpinFromDevice( m_source->getMBuffer(), lwdaDevice );
    mm->unpinFromDevice( m_aclwm_storage_device, lwdaDevice );
    mm->unpinFromDevice( m_stream_storage_device, lwdaDevice );

    m_naclwm++;
}

void StreamBuffer::resetAclwm()
{
    m_naclwm       = 0;
    m_update_ready = false;
}

void StreamBuffer::updateReady( int* ready, unsigned int* subframe_count, unsigned int* max_subframes )
{
    std::lock_guard<std::mutex> lock( m_mutex );

    if( ready )
        *ready = (int)m_update_ready;

    if( m_update_ready )
    {
        if( subframe_count )
            *subframe_count = (unsigned)m_naclwm;
        if( max_subframes )
            *max_subframes = m_max_subframes;
    }
}

void StreamBuffer::markNotReady()
{
    m_update_ready = false;
}

size_t StreamBuffer::sizeBytes()
{
    return sizeElems() * m_elemsize;
}

size_t StreamBuffer::sizeElems()
{
    return m_width * m_height * m_depth;
}

void StreamBuffer::alloc()
{
    LWDADevice* lwdaDevice = m_context->getDeviceManager()->primaryLWDADevice();
    RT_ASSERT( lwdaDevice );
    DeviceSet devices;
    devices.insert( (Device*)lwdaDevice );
    devices.insert( (Device*)m_context->getDeviceManager()->cpuDevice() );

    // stream (output) storage on device and host
    {
        const size_t newsz = sizeBytes();

        BufferDimensions dimensions( RT_FORMAT_USER, /*element size*/ sizeof( unsigned char ), /*dims*/ 1, /*w,h,d*/ newsz, 1, 1 );
        if( m_stream_storage_device )
        {
            m_context->getMemoryManager()->changeSize( m_stream_storage_device, dimensions );
        }
        else
        {
            m_stream_storage_device =
                m_context->getMemoryManager()->allocateMBuffer( dimensions, MBufferPolicy::gpuLocal, devices );
        }

        if( newsz != m_stream_storage_host_cache.size() )
        {
            m_stream_storage_host_cache.resize( newsz );
            std::fill( m_stream_storage_host_cache.begin(), m_stream_storage_host_cache.end(), 0u );
        }
    }

    // aclwm storage on device
    {
        const size_t           newsz = getSrcNcomp() * sizeElems();
        const BufferDimensions dimensions( RT_FORMAT_USER, sizeof( float ), 1, newsz, 1, 1 );
        if( m_aclwm_storage_device )
        {
            // Note: check size here to avoid calling resetAclwm every time
            if( dimensions != m_aclwm_storage_device->getDimensions() )
            {
                m_context->getMemoryManager()->changeSize( m_aclwm_storage_device, dimensions );
                resetAclwm();
            }
        }
        else
        {
            m_aclwm_storage_device = m_context->getMemoryManager()->allocateMBuffer( dimensions, MBufferPolicy::gpuLocal, devices );
            resetAclwm();
        }
    }
}

int StreamBuffer::getSrcNcomp()
{
    if( !m_source )
        return 0;
    return m_source->getFormat() == RT_FORMAT_FLOAT3 ? 3 : 4;
}
