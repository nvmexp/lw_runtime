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

#include <LWCA/GraphicsResource.h>

#include <LWCA/Array.h>
#include <LWCA/ErrorCheck.h>
#include <LWCA/Stream.h>
#include <corelib/system/LwdaDriver.h>
#include <prodlib/exceptions/Assert.h>

using namespace optix;
using namespace optix::lwca;
using namespace corelib;

GraphicsResource::GraphicsResource()
    : m_resource( nullptr )
{
}

GraphicsResource::GraphicsResource( LWgraphicsResource resource )
    : m_resource( resource )
{
}

LWgraphicsResource GraphicsResource::get()
{
    return m_resource;
}

const LWgraphicsResource GraphicsResource::get() const
{
    return m_resource;
}

GraphicsResource GraphicsResource::GLRegisterBuffer( GLuint buffer, unsigned int flags, LWresult* returnResult )
{
    LWgraphicsResource result;
    CHECK( lwdaDriver().LwGraphicsGLRegisterBuffer( &result, buffer, flags ) );
    return GraphicsResource( result );
}

GraphicsResource GraphicsResource::GLRegisterImage( GLuint image, GLenum target, unsigned int flags, LWresult* returnResult )
{
    LWgraphicsResource result;
    CHECK( lwdaDriver().LwGraphicsGLRegisterImage( &result, image, target, flags ) );
    return GraphicsResource( result );
}

void GraphicsResource::unregister( LWresult* returnResult )
{
    RT_ASSERT( m_resource != nullptr );
    CHECK( lwdaDriver().LwGraphicsUnregisterResource( m_resource ) );
    m_resource = nullptr;
}

void GraphicsResource::setMapFlags( unsigned int flags, LWresult* returnResult )
{
    RT_ASSERT( m_resource != nullptr );
    CHECK( lwdaDriver().LwGraphicsResourceSetMapFlags( m_resource, flags ) );
}

void GraphicsResource::map( const Stream& hStream, LWresult* returnResult )
{
    RT_ASSERT( m_resource != nullptr );
    CHECK( lwdaDriver().LwGraphicsMapResources( 1, &m_resource, hStream.get() ) );
}

void GraphicsResource::map( unsigned int count, LWgraphicsResource* resources, const Stream& hStream, LWresult* returnResult )
{
    CHECK( lwdaDriver().LwGraphicsMapResources( count, resources, hStream.get() ) );
}

void GraphicsResource::unmap( const Stream& hStream, LWresult* returnResult )
{
    RT_ASSERT( m_resource != nullptr );
    CHECK( lwdaDriver().LwGraphicsUnmapResources( 1, &m_resource, hStream.get() ) );
}

void GraphicsResource::unmap( unsigned int count, LWgraphicsResource* resources, const Stream& hStream, LWresult* returnResult )
{
    CHECK( lwdaDriver().LwGraphicsUnmapResources( count, resources, hStream.get() ) );
}

LWdeviceptr GraphicsResource::getMappedPointer( size_t* pSize, LWresult* returnResult )
{
    RT_ASSERT( m_resource != nullptr );
    LWdeviceptr result;
    CHECK( lwdaDriver().LwGraphicsResourceGetMappedPointer( &result, pSize, m_resource ) );
    return result;
}

Array GraphicsResource::getMappedArray( unsigned int arrayIndex, unsigned int mipLevel, LWresult* returnResult )
{
    RT_ASSERT( m_resource != nullptr );
    LWarray result;
    CHECK( lwdaDriver().LwGraphicsSubResourceGetMappedArray( &result, m_resource, arrayIndex, mipLevel ) );
    return Array( result );
}

MipmappedArray GraphicsResource::getMappedMipmappedArray( LWresult* returnResult )
{
    RT_ASSERT( m_resource != nullptr );
    LWmipmappedArray result;
    CHECK( lwdaDriver().LwGraphicsResourceGetMappedMipmappedArray( &result, m_resource ) );
    return MipmappedArray( result );
}
