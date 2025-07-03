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

#include <lwca.h>

typedef unsigned int GLuint;
typedef unsigned int GLenum;

namespace optix {
namespace lwca {
class Stream;
class Array;
class MipmappedArray;

class GraphicsResource
{
  public:
    GraphicsResource();

    // Get the low-level graphics resource
    LWgraphicsResource       get();
    const LWgraphicsResource get() const;


    // Registers an OpenGL buffer object.
    static GraphicsResource GLRegisterBuffer( GLuint buffer, unsigned int flags, LWresult* returnResult = nullptr );

    // Register an OpenGL texture or renderbuffer object.
    static GraphicsResource GLRegisterImage( GLuint image, GLenum target, unsigned int flags, LWresult* returnResult = nullptr );

    // Unregisters a graphics resource for access by LWCA.
    void unregister( LWresult* returnResult = nullptr );

    // Set usage flags for mapping a graphics resource.
    void setMapFlags( unsigned int flags, LWresult* returnResult = nullptr );

    // Map graphics resource for access by LWCA.
    void map( const Stream& hStream, LWresult* returnResult = nullptr );

    // Map multiple graphics resources for access by LWCA.
    static void map( unsigned int count, LWgraphicsResource* resources, const Stream& hStream, LWresult* returnResult = nullptr );

    // Unmap graphics resource.
    void unmap( const Stream& hStream, LWresult* returnResult = nullptr );

    // Unmap multiple graphics resources.
    static void unmap( unsigned int count, LWgraphicsResource* resources, const Stream& hStream, LWresult* returnResult = nullptr );

    // Get a device pointer through which to access a mapped graphics resource. Size
    // is returned in pSize.
    LWdeviceptr getMappedPointer( size_t* pSize, LWresult* returnResult = nullptr );

    // Get an array through which to access a subresource of a mapped graphics resource.
    Array getMappedArray( unsigned int arrayIndex, unsigned int mipLevel, LWresult* returnResult = nullptr );

    // Get a mipmapped array through which to access a mapped graphics resource.
    MipmappedArray getMappedMipmappedArray( LWresult* returnResult = nullptr );

  private:
    LWgraphicsResource m_resource;
    explicit GraphicsResource( LWgraphicsResource resource );
};

}  // namespace lwca
}  // namespace optix
