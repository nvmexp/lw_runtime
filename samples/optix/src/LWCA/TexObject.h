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

namespace optix {
namespace lwca {

class Module;

class TexObject
{
  public:
    TexObject();

    // Get the low-level tex
    LWtexObject get() const;
    bool        isNull() const;

    // Creates a texture object.
    static TexObject create( const LWDA_RESOURCE_DESC&      pResDesc,
                             const LWDA_TEXTURE_DESC&       pTexDesc,
                             const LWDA_RESOURCE_VIEW_DESC* pResViewDesc,
                             LWresult*                      returnResult = nullptr );

    // Destroys a texture object.
    void destroy( LWresult* returnResult = nullptr );

    // Returns a texture object's resource descriptor.
    void getResourceDesc( LWDA_RESOURCE_DESC& pResDesc, LWresult* returnResult = nullptr ) const;

    // Returns a texture object's resource view descriptor.
    void getResourceViewDesc( LWDA_RESOURCE_VIEW_DESC& pResViewDesc, LWresult* returnResult = nullptr ) const;

    // Returns a texture object's texture descriptor.
    void getTextureDesc( LWDA_TEXTURE_DESC& pTexDesc, LWresult* returnResult = nullptr ) const;

  protected:
    explicit TexObject( LWtexObject texObject );

    LWtexObject m_texObject;
};

}  // namespace lwca
}  // namespace optix
